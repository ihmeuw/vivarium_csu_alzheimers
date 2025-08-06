from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from vivarium_public_health.population import ScaledPopulation


class AlzPop(ScaledPopulation):
    
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
        if not pop_data.index.equals(demographic_counts.index):
            raise ValueError(
                "The index of pop_data does not match the index of demographic_counts. "
                "Please ensure they are the same."
            )
        # TODO: add simulants from each demographic group based on the demographic_counts
        # TODO: uniformly at random sample age from the age range for each simulant
        pass


class AlzheimersIncidence(Component):

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        # todo: do we want this key to be configurable? probably not right now
        self.incidence_rate = builder.data.load("alzheimers.incidence_rate")
        self.clock = builder.time.clock()
        self.randomness = builder.randomness
        # todo compute model scale (population_size / true_population_size_with_alzheimers)
        self.model_scale = ...
        self.simulant_creator = builder.population.get_simulant_creator()
        self.age_start = builder.configuration.population.initialization_age_start
        self.age_end = builder.configuration.population.initialization_age_end

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
        # TODO: get incidence rates for this year
        incidence_rates = self.incidence_rate.loc[self.clock().year]
        step_size = utilities.to_years(event.step_size)
        mean_incident_cases = incidence_rates * step_size * self.model_scale
        simulants_to_add = pd.Series(0, index=mean_incident_cases.index)

        # Assume births occur as a Poisson process
        for age, sex, value in mean_incident_cases:
            r = np.random.RandomState(seed=self.randomness.get_seed(f"f{age}_{sex}_alz_incidence"))
            simulants_to_add[(age, sex)] = r.poisson(value)

        total_simulants_to_add = simulants_to_add.sum()

        if simulants_to_add > 0:
            self.simulant_creator(
                total_simulants_to_add,
                {
                    "sim_state": "time_step",
                    "demographic_counts": simulants_to_add,
                },
            )