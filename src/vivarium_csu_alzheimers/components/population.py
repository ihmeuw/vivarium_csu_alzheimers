import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health import utilities
from vivarium_public_health.population import ScaledPopulation

from vivarium_csu_alzheimers.constants import data_keys
from vivarium_csu_alzheimers.constants.metadata import ARTIFACT_INDEX_COLUMNS


class AlzheimersPopulation(ScaledPopulation):
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


class AlzheimersIncidence(Component):
    """This is using FertilityCrudeBirthRate from Vivarium Public Health as a template."""

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.age_start = builder.configuration.population.initialization_age_min
        self.age_end = builder.configuration.population.initialization_age_max
        self.randomness = builder.randomness  # Manager
        # NOTE: All three of these methods are capping the upper age bound at 100
        self.incidence_rate = self.load_incidence_rate(builder)
        self.pop_structure = self.load_population_structure(builder)
        prevalence = self.load_prevalence(builder)

        # Model scale = (population_size / (pop_structure * prevalence).sum())
        # but use only one year of self.pop_structure * prevalence

        lvl = self.pop_structure.index.get_level_values('year_start')
        pop_2025 = self.pop_structure[lvl == 2025]

        lvl = prevalence.index.get_level_values('year_start')
        prev_2025 = prevalence[lvl == 2025]
        
        self.model_scale = builder.configuration.population.population_size / (
            (pop_2025 * prev_2025).sum()
        )
        assert self.model_scale > 0.0
        
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
        # TODO: get incidence rates and population for year in forecasted data if necessary
        pop_structure = self.pop_structure.copy()

        # select data for most appropriate year
        lvl = pop_structure.index.get_level_values('year_start')
        if event.time.year in lvl:
            target_year = event.time.year
        else:
            target_year = max(lvl)
        pop_structure = pop_structure[lvl == target_year]
        pop_structure.index = pop_structure.index.droplevel(["year_start", "year_end"])
        
        mean_incident_cases = (
            self.incidence_rate * pop_structure * step_size * self.model_scale
        )
        mean_incident_cases = mean_incident_cases.dropna() # Abie hacked around the youngest ages, which left some nans
        
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

    def load_incidence_rate(self, builder: Builder) -> pd.Series:
        incidence_rate = builder.data.load(data_keys.ALZHEIMERS.INCIDENCE_RATE)
        # FIXME: use "total-population incidence rate" here, i.e. incidence count / total population
        # not the "susceptible-population incidence rate" which is stored in the artifact
        
        incidence_rate.loc[incidence_rate["age_end"] == 125, "age_end"] = self.age_end
        incidence_rate = (
            incidence_rate[["sex", "age_start", "age_end", "value"]]
            .set_index(["sex", "age_start", "age_end"])
            .squeeze()
        )
        return incidence_rate

    def load_population_structure(self, builder: Builder) -> pd.Series:
        pop_structure = builder.data.load(data_keys.POPULATION.STRUCTURE)
        pop_structure.loc[pop_structure["age_end"] == 125, "age_end"] = self.age_end
        pop_structure = pop_structure.set_index(ARTIFACT_INDEX_COLUMNS)["value"]
        return pop_structure

    def load_prevalence(self, builder: Builder) -> pd.Series:
        prevalence = builder.data.load(data_keys.ALZHEIMERS.PREVALENCE_SCALE_FACTOR)
        prevalence.loc[prevalence["age_end"] == 125, "age_end"] = self.age_end
        prevalence = prevalence.set_index(ARTIFACT_INDEX_COLUMNS).squeeze()
        return prevalence
