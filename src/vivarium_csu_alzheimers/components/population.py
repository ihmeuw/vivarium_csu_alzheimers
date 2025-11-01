import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health import utilities
from vivarium_public_health.population import ScaledPopulation
from vivarium_public_health.population.data_transformations import (
    load_population_structure,
)

from vivarium_csu_alzheimers.constants import data_keys
from vivarium_csu_alzheimers.constants.metadata import ARTIFACT_INDEX_COLUMNS


class AlzheimersPopulation(ScaledPopulation):
    """Adds new simulants to the simulation based on demography counts.

    The demography counts are created by the AlzheimersIncidence component, below.
    """

    @property
    def collect_metrics_priority(self) -> int:
        """We want to age simulants after metrics are collected, so use highest priority."""
        return 9

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.key_columns = builder.configuration.randomness.key_columns

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        if pop_data.user_data.get("sim_state") != "time_step":
            super().on_initialize_simulants(pop_data)
            return

        if "demographic_counts" in pop_data.user_data:
            # demographic_counts will be a pd.Series with demographic group index
            demographic_counts: pd.Series = pop_data.user_data["demographic_counts"]
        else:
            raise ValueError(
                "AlzPop requires demographic_counts in pop_data.user_data. "
                "Please provide it in the user_data dictionary."
            )

        # Verify pop_data and demographic_counts are compatible
        if not len(pop_data.index) == demographic_counts.sum():
            raise ValueError(
                "The index of pop_data does not match the index of demographic_counts. "
                "Please ensure they are the same."
            )

        # Update metadata for each demographic group
        new_simulants = pd.DataFrame(index=pop_data.index)
        groups_to_add = demographic_counts[demographic_counts > 0]
        start = pop_data.index.min()
        for idx, value in groups_to_add.items():
            sex, age_lower, age_upper = idx
            group_idx = pd.Index(list(range(start, start + value)))
            # This will always happen in the order of female then male and youngest to oldest
            new_simulants.loc[group_idx, "sex"] = sex
            age_draws = self.randomness["age_smoothing_age_bounds"].get_draw(
                group_idx, additional_key=f"{sex}_{age_lower}_{age_upper}"
            )
            new_simulants.loc[group_idx, "age"] = age_lower + age_draws * (
                age_upper - age_lower
            )
            start += value

        # Update additional population columns
        new_simulants["alive"] = "alive"
        new_simulants["entrance_time"] = pop_data.creation_time
        new_simulants["exit_time"] = pd.NaT

        self.population_view.update(new_simulants)
        # NOTE: This only works with key_columns because this component creates age and entrance_time
        self.register_simulants(new_simulants[self.key_columns])

    def on_time_step(self, event: Event) -> None:
        """Overrides aging to do nothing instead."""
        pass

    def on_collect_metrics(self, event: Event) -> None:
        """Ages simulants on the collect_metrics phase.

        We want to do this because otherwise aging happens prior to collecting metrics
        which means that simulants on age bin boundaries land in the older age bin
        than they should have.
        """
        population = self.population_view.get(event.index, query="alive == 'alive'")
        population["age"] += utilities.to_years(event.step_size)
        self.population_view.update(population)

    def _load_population_structure(self, builder: Builder) -> pd.DataFrame:
        """Overwriting this method to deal with multi-year population structure and custom age groups."""
        scaling_factor = self.get_data(builder, self.scaling_factor)
        # Population does not have under 5 age groups
        scaling_factor = scaling_factor[scaling_factor["age_start"] >= 5]
        population_structure = load_population_structure(builder)
        if not isinstance(scaling_factor, pd.DataFrame):
            raise ValueError(
                f"Scaling factor must be a pandas DataFrame. Provided value: {scaling_factor}"
            )
        # Coerce scaling factor to have same index as population structure
        scaling_factor = scaling_factor.drop(columns=["year_start", "year_end", "age_end"])
        scaling_factor = scaling_factor.set_index(
            [col for col in scaling_factor.columns if col != "value"]
        )
        population_structure = population_structure.set_index(
            [col for col in population_structure.columns if col != "value"]
        )
        scaled_population_structure = (population_structure * scaling_factor).reset_index()

        # HACK: address nans introduced by mismatch between
        # (age_start,age_end) in population_structure and scaling_factor
        scaled_population_structure["value"] = scaled_population_structure["value"].fillna(
            0.0
        )
        scaled_population_structure = scaled_population_structure.dropna()

        return scaled_population_structure


class AlzheimersIncidence(Component):
    """Class to handle the incidence of Alzheimer's disease.

    This is heavily based on the vivarium_public_health FertilityCrudeBirthRate
    class. It determines which simulants will be added to the simulation.

    Notes
    -----
    The AlzheimersPopulation component above ingests the demographic_counts created
    here and uses that to actually add simulants to the simulation.
    """

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.age_start = builder.configuration.population.initialization_age_min
        self.age_end = builder.configuration.population.initialization_age_max
        self.randomness = builder.randomness  # Manager
        # NOTE: All three of these methods are capping the upper age bound at 100
        self.bbbm_incidence_counts = self.load_bbbm_incidence_counts(builder)
        self.pop_structure = self.load_population_structure(builder)
        prevalence = self.load_prevalence(builder)
        start_year = builder.configuration.time.start.year
        # pop_structure starts in 2021, so adjust start year if necessary
        sub_pop = self.pop_structure.loc[
            self.pop_structure.index.get_level_values("year_start") == max(2021, start_year)
        ]
        # NOTE: we only have prevalence for a single year so the year_start/year_end will be difference
        # in the index levels but their structure is the same
        # Model scale = (population_size / (pop_structure * prevalence).sum())
        prevalence.index = prevalence.index.droplevel(["year_start", "year_end"])
        merged_df = pd.merge(
            sub_pop, prevalence, left_index=True, right_index=True, suffixes=("_pop", "_prev")
        )

        self.model_scale = builder.configuration.population.population_size / (
            (merged_df.value_pop * merged_df.value_prev).sum()
        )
        self.simulant_creator = builder.population.get_simulant_creator()

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        """Adds new simulants every time step based on the Crude Birth Rate
        and an assumption that birth is a Poisson process

        Parameters
        ----------
        event
            The event that triggered the function call.
        """

        step_size = utilities.to_years(event.step_size)
        pop_structure = self.pop_structure.copy()

        # Filter pop_structure by year_start based on event.time.year
        query_year = min(
            event.time.year, pop_structure.index.get_level_values("year_start").max()
        )
        pop_structure = pop_structure.loc[
            pop_structure.index.get_level_values("year_start") == query_year
        ]
        pop_structure.index = pop_structure.index.droplevel(["year_start", "year_end"])
        full_population_incident_cases = self.bbbm_incidence_counts.loc[
            self.bbbm_incidence_counts.index.get_level_values("year_start") == query_year
        ]
        full_population_incident_cases = full_population_incident_cases.droplevel(
            ["year_start", "year_end"]
        )
        # New simulants = model_scale * new_cases * step_size
        mean_incident_cases = self.model_scale * full_population_incident_cases * step_size
        simulants_to_add = pd.Series(0, index=mean_incident_cases.index)

        # Determine number of simulants to add for each demographic group
        for idx, value in mean_incident_cases.items():
            sex, age_lower, age_upper = idx
            r = np.random.RandomState(
                seed=self.randomness.get_seed(f"f{age_lower}_{sex}_alz_incidence")
            )
            simulants_to_add[(sex, age_lower, age_upper)] = r.poisson(value)
        total_simulants_to_add = simulants_to_add.sum()

        if total_simulants_to_add > 0:
            self.simulant_creator(
                total_simulants_to_add,
                {
                    "sim_state": "time_step",
                    "demographic_counts": simulants_to_add,
                },
            )

    ##################
    # Helper methods #
    ##################

    def load_bbbm_incidence_counts(self, builder: Builder) -> pd.Series:
        incidence_counts = builder.data.load(
            data_keys.ALZHEIMERS_CONSISTENT.SUSCEPTIBLE_TO_BBBM_TRANSITION_COUNT
        )
        # Updating age_end to match configuration since some simulants are living past 125
        incidence_counts.loc[incidence_counts["age_end"] == 125, "age_end"] = self.age_end
        incidence_counts = incidence_counts.set_index(ARTIFACT_INDEX_COLUMNS)
        incidence_counts = incidence_counts[["value"]].squeeze()
        return incidence_counts

    def load_population_structure(self, builder: Builder) -> pd.Series:
        pop_structure = builder.data.load(data_keys.POPULATION.STRUCTURE)
        pop_structure.loc[pop_structure["age_end"] == 125, "age_end"] = self.age_end
        pop_structure = pop_structure.set_index(ARTIFACT_INDEX_COLUMNS)["value"]
        return pop_structure

    def load_prevalence(self, builder: Builder) -> pd.Series:
        prevalence = builder.data.load(data_keys.ALZHEIMERS_CONSISTENT.AD_PREVALENCE)

        # Updating age_end to match configuration since some simulants are living past 125
        prevalence.loc[prevalence["age_end"] == 125, "age_end"] = self.age_end
        prevalence = prevalence.set_index(ARTIFACT_INDEX_COLUMNS).squeeze()
        return prevalence
