import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium_public_health import utilities
from vivarium_public_health.population import ScaledPopulation


class AlzheimersPopulation(ScaledPopulation):
    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        if pop_data.user_data.get("sim_state") != "time_step":
            super().on_initialize_simulants(pop_data)
            return

        # get pop_data.user_data["demographic_counts"] if it exists
        if "demographic_counts" in pop_data.user_data:
            # demographic_counts will be a pd.Series
            demographic_counts = pop_data.user_data["demographic_counts"]
        else:
            # throw an error if it does not exist
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
        for idx, row in groups_to_add.items():
            sex, age_lower, age_upper = idx
            value = row
            group_idx = pd.Index(list(range(start, start + value)))
            # This will not work, but it is a placeholder for now
            new_simulants.loc[group_idx, "sex"] = sex
            age_draws = self.randomness["age_smoothing_age_bounds"].get_draw(
                group_idx, additional_key=f"{sex}_{age_lower}_{age_upper}"
            )
            new_simulants.loc[group_idx, "age"] = (age_lower + age_upper) * age_draws
            start = group_idx.max() + 1

        # Update additional population columns
        new_simulants["alive"] = "alive"
        new_simulants["entrance_time"] = pop_data.creation_time
        new_simulants["exit_time"] = pd.NaT
        breakpoint()

        self.population_view.update(new_simulants)


class AlzheimersIncidence(Component):
    """This is using FertilityCrudeBirthRate from Vivarium Public Health as a template."""

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        # todo: do we want this key to be configurable? probably not right now
        self.incidence_rate = builder.data.load(
            "cause.alzheimers_disease_and_other_dementias.incidence_rate"
        )
        self.clock = builder.time.clock()
        self.randomness = builder.randomness
        # TODO: compute model scale (population_size / (pop_structure * prevalence).sum())
        self.model_scale = builder.configuration.population.population_size / (
            builder.configuration.population.population_size * 0.25
        )
        self.simulant_creator = builder.population.get_simulant_creator()
        self.age_start = builder.configuration.population.initialization_age_min
        self.age_end = builder.configuration.population.initialization_age_max

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
        # TODO: get incidence rates for this year - currently dropping year_start and year_end
        incidence_rate = self.incidence_rate[
            ["sex", "age_start", "age_end", "value"]
        ].set_index(["sex", "age_start", "age_end"])
        mean_incident_cases = incidence_rate * step_size * self.model_scale
        simulants_to_add = pd.Series(0, index=mean_incident_cases.index)

        # Assume births occur as a Poisson process
        for idx, row in mean_incident_cases.iterrows():
            sex, age_lower, age_upper = idx
            value = row["value"]
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
