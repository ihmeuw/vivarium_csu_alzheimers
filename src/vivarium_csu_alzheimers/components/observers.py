import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer
from vivarium.framework.time import get_time_stamp
from vivarium_public_health import ResultsStratifier as ResultsStratifier_
from vivarium_public_health.results import DiseaseObserver, PublicHealthObserver
from vivarium_public_health.utilities import to_years

from vivarium_csu_alzheimers.constants.data_values import (
    BBBM_AGE_MAX,
    BBBM_AGE_MIN,
    BBBM_TEST_RESULTS,
    BBBM_TIMESTEPS_UNTIL_RETEST,
    COLUMNS,
    TESTING_STATES,
)
from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL
from vivarium_csu_alzheimers.utilities import get_timedelta_from_step_size


class ResultsStratifier(ResultsStratifier_):
    def setup(self, builder: Builder) -> None:
        self.sim_start_date = get_time_stamp(builder.configuration.time.start)
        self.step_size = builder.configuration.time.step_size
        super().setup(builder)

    @staticmethod
    def get_age_bins(builder: Builder) -> pd.DataFrame:
        """Get the age bins for stratifying by age.

        Parameters
        ----------
        builder
            The builder object for the simulation.

        Returns
        -------
            The age bins for stratifying by age.
        """
        raw_age_bins = builder.data.load("population.age_bins")
        age_start = builder.configuration.population.initialization_age_min
        exit_age = builder.configuration.population.untracking_age

        age_start_mask = age_start < raw_age_bins["age_end"]
        exit_age_mask = raw_age_bins["age_start"] < exit_age if exit_age else True

        age_bins = raw_age_bins.loc[age_start_mask & exit_age_mask, :].copy()
        age_bins["age_group_name"] = (
            age_bins["age_group_name"].str.replace(" ", "_").str.lower()
        )
        # FIXME: MIC-4083 simulants can age past 125
        max_age = age_bins["age_end"].max()
        age_bins.loc[age_bins["age_end"] == max_age, "age_end"] = 200

        return age_bins

    def map_semester(self, pop: pd.DataFrame) -> pd.Series:
        def semesterize(date):
            day_number = (
                pd.Timestamp(date) - pd.Timestamp(year=date.year, month=1, day=1)
            ).days
            return "first" if day_number <= self.step_size else "second"

        return pop.squeeze(axis=1).dt.date.apply(semesterize)

    def map_treatment_durations(self, pop: pd.DataFrame) -> pd.Series:
        durations = pop.fillna(0.0)
        durations = durations.astype(int).squeeze(axis=1)
        return durations

    def register_stratifications(self, builder):
        super().register_stratifications(builder)
        builder.results.register_stratification(
            name="testing_state",
            categories=list(TESTING_STATES),
            requires_columns=[COLUMNS.TESTING_STATE],
        )
        builder.results.register_stratification(
            name="bbbm_test_results",
            categories=list(BBBM_TEST_RESULTS),
            requires_columns=[COLUMNS.BBBM_TEST_RESULT],
        )
        builder.results.register_stratification(
            name="semester",
            categories=["first", "second"],
            mapper=self.map_semester,
            is_vectorized=True,
            requires_columns=["event_time"],
        )
        builder.results.register_stratification(
            name="treatment_durations",
            categories=list(range(10)),
            mapper=self.map_treatment_durations,
            is_vectorized=True,
            requires_columns=[COLUMNS.TREATMENT_DURATION],
        )


class NewSimulantsObserver(Observer):
    """Observer to count the number of new simulants added to the population"""

    def setup(self, builder: Builder) -> None:
        """Set up the observer component."""
        self.clock = builder.time.clock()

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="counts_new_simulants",
            requires_columns=["entrance_time"],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.count_new_simulants,
        )

    def count_new_simulants(self, sims: pd.DataFrame) -> float:
        """Counts the number of new simulants added to the population this time step."""
        new = sims["entrance_time"] == self.clock()
        return new.sum()


class BaselineTestingObserver(PublicHealthObserver):
    """Observer to track baseline testing for Alzheimer's disease."""

    def register_observations(self, builder: Builder) -> None:
        pop_filter = (
            'alive == "alive" and tracked == True '
            f'and {COLUMNS.PREVIOUS_DISEASE_STATE} == "{ALZHEIMERS_DISEASE_MODEL.BBBM_STATE}" '
            f'and {COLUMNS.DISEASE_STATE} != "{ALZHEIMERS_DISEASE_MODEL.BBBM_STATE}" '
            f'and {COLUMNS.BBBM_TEST_RESULT} != "{BBBM_TEST_RESULTS.POSITIVE}"'
        )
        self.register_adding_observation(
            builder=builder,
            name="counts_baseline_tests_among_eligible",
            pop_filter=pop_filter,
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
        )

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("testing", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("baseline_testing", index=results.index)


class BBBMTestingObserver(PublicHealthObserver):
    """Observer to track BBBM testing metrics."""

    @property
    def columns_required(self) -> list[str]:
        return [
            "alive",
            "tracked",
            COLUMNS.BBBM_TEST_DATE,
            COLUMNS.DISEASE_STATE,
            COLUMNS.AGE,
            COLUMNS.BBBM_TEST_RESULT,
            COLUMNS.ENTRANCE_TIME,
            COLUMNS.BBBM_TEST_EVER_ELIGIBLE,
        ]

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.sim_start_date = builder.configuration.time.start

    def register_observations(self, builder: Builder) -> None:
        # TODO: clarify whether the default pop_filter to PublicHealthObserver
        #   should include alive == "alive" (it currently doesn't)
        pop_filter = 'alive == "alive" and tracked == True'

        self.register_adding_observation(
            builder=builder,
            name="counts_bbbm_tests",
            pop_filter=pop_filter,
            requires_columns=["alive", "tracked", COLUMNS.BBBM_TEST_DATE],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.count_bbbm_tests,
        )
        self.register_adding_observation(
            builder=builder,
            name="counts_newly_eligible_for_bbbm_testing",
            pop_filter=pop_filter,
            requires_columns=self.columns_required,
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.count_newly_eligible_simulants,
        )
        self.register_adding_observation(
            builder=builder,
            name="person_time_eligible_for_bbbm_testing",
            pop_filter=pop_filter,
            requires_columns=[
                "alive",
                "tracked",
                COLUMNS.DISEASE_STATE,
                COLUMNS.AGE,
                COLUMNS.BBBM_TEST_DATE,
                COLUMNS.BBBM_TEST_RESULT,
            ],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.aggregate_eligible_person_time,
        )
        self.register_adding_observation(
            builder=builder,
            name="person_time_ever_eligible_for_bbbm_testing",
            pop_filter=pop_filter,
            requires_columns=["alive", "tracked", COLUMNS.BBBM_TEST_EVER_ELIGIBLE],
            additional_stratifications=self.configuration.include
            + [ALZHEIMERS_DISEASE_MODEL.NAME],
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.aggregate_ever_eligible_person_time,
        )

    ###############
    # Aggregators #
    ###############

    def count_bbbm_tests(self, pop: pd.DataFrame) -> float:
        """Counts the number of BBBM tests conducted only on this time step and not from a
        simulants testing history to not count negative tests from history."""
        return sum(pop[COLUMNS.BBBM_TEST_DATE] == self.clock() + self.step_size())

    def count_newly_eligible_simulants(self, pop: pd.DataFrame) -> float:
        """Counts the number of simulants eligible for BBBM testing this time step.

        There are three groups of newly eligible simulants:
        - those who enter the simulation and are eligible
        - those who become eligible on this time step
        - those who got tested exactly three years ago and are eligible for a retest
        """
        eligible_state = pop[COLUMNS.DISEASE_STATE] == ALZHEIMERS_DISEASE_MODEL.BBBM_STATE
        eligible_age = (pop[COLUMNS.AGE] >= BBBM_AGE_MIN) & (pop[COLUMNS.AGE] < BBBM_AGE_MAX)
        eligible_test_result = pop[COLUMNS.BBBM_TEST_RESULT] != BBBM_TEST_RESULTS.POSITIVE
        eligible_baseline = eligible_state & eligible_age & eligible_test_result

        if not eligible_baseline.any():
            return 0.0

        sim_start_date = get_time_stamp(self.sim_start_date)
        step_size_in_years = self.step_size().days / 365.25

        new_entrants = pd.Series(False, index=pop.index)

        # Handle initialized population
        if self.clock() == sim_start_date:
            # everyone is a new entrant (either from sim initialization or introduced on first time step)
            new_entrants = pd.Series(
                pop[COLUMNS.ENTRANCE_TIME] < sim_start_date, index=pop.index
            )

        new_entrants |= pop[COLUMNS.ENTRANCE_TIME] == self.clock()
        # NOTE: this is not precisely handling the ages of simulants who entered
        # on simulation initialization
        aged_in = (pop[COLUMNS.AGE] >= BBBM_AGE_MIN) & (
            pop[COLUMNS.AGE] < BBBM_AGE_MIN + step_size_in_years
        )
        retest = pop[
            COLUMNS.BBBM_TEST_DATE
        ] == self.clock() + self.step_size() - get_timedelta_from_step_size(
            self.step_size().days, BBBM_TIMESTEPS_UNTIL_RETEST
        )

        return sum(eligible_baseline & (new_entrants | aged_in | retest))

    def aggregate_eligible_person_time(self, pop: pd.DataFrame) -> float:
        # Dates are finnicky in the pop_filter, so we just filter here to eligibility
        eligible_state = pop[COLUMNS.DISEASE_STATE] == ALZHEIMERS_DISEASE_MODEL.BBBM_STATE
        eligible_age = (pop[COLUMNS.AGE] >= BBBM_AGE_MIN) & (pop[COLUMNS.AGE] < BBBM_AGE_MAX)
        event_time = self.clock() + self.step_size()
        eligible_history = (pop[COLUMNS.BBBM_TEST_DATE].isna()) | (
            pop[COLUMNS.BBBM_TEST_DATE]
            <= event_time
            - get_timedelta_from_step_size(self.step_size().days, BBBM_TIMESTEPS_UNTIL_RETEST)
        )
        eligible_results = pop[COLUMNS.BBBM_TEST_RESULT] != BBBM_TEST_RESULTS.POSITIVE

        eligible = eligible_state & eligible_age & eligible_history & eligible_results

        return sum(eligible) * to_years(self.step_size())

    def aggregate_ever_eligible_person_time(self, pop: pd.DataFrame) -> float:
        return sum(pop[COLUMNS.BBBM_TEST_EVER_ELIGIBLE]) * to_years(self.step_size())

    ##############
    # Formatting #
    ##############

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("testing", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("bbbm_testing", index=results.index)


class TreatmentObserver(DiseaseObserver):
    @property
    def columns_required(self) -> list[str]:
        return super().columns_required + [
            COLUMNS.WAITING_FOR_TREATMENT_EVENT_TIME,
            COLUMNS.TREATMENT_DURATION,
        ]

    def __init__(self) -> None:
        super().__init__("treatment")

    def setup(self, builder):
        super().setup(builder)
        self.clock = builder.time.clock()
        self.sim_start_time = pd.Timestamp(
            month=builder.configuration.time.start.month,
            day=builder.configuration.time.start.day,
            year=builder.configuration.time.start.year,
        )

    def register_observations(self, builder):
        super().register_observations(builder)
        self.register_adding_observation(
            builder=builder,
            name="treatment_duration",
            pop_filter='alive == "alive" and tracked==True',
            requires_columns=[
                COLUMNS.WAITING_FOR_TREATMENT_EVENT_TIME,
            ],
            additional_stratifications=self.configuration.include + ["treatment_durations"],
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.count_treatment_durations,
        )

    def register_disease_state_stratification(self, builder: Builder) -> None:
        """Register the disease state stratification.

        This is a near copy/paste of the default VPH DiseaseObserver method except
        that it removes the postive_test TransientState from the list of categories.
        """
        categories = [
            state.state_id
            for state in self.disease_model.states
            if state.state_id != "positive_test"
        ]
        builder.results.register_stratification(
            self.disease,
            categories,
            requires_columns=[self.disease],
        )

    def register_transition_stratification(self, builder: Builder) -> None:
        """Register the transition stratification.

        This is a near copy/paste of the default VPH DiseaseObserver method.
        The only differences:
          - Add the transition categories that start in susceptible (these get lost
            when passing through the positve_test TransientState).
          - Remove the positve_test TransientState from the list of transitions.
        """
        transitions = [
            str(transition)
            for transition in self.disease_model.transition_names
            if "positive_test" not in str(transition)
        ] + [
            "no_transition",
            "susceptible_to_treatment_to_waiting_for_treatment",
            "susceptible_to_treatment_to_no_effect_never_treated",
        ]
        # manually append 'no_transition' as an excluded transition
        excluded_categories = (
            builder.configuration.stratification.excluded_categories.to_dict().get(
                self.transition_stratification_name, []
            )
        ) + ["no_transition"]
        builder.results.register_stratification(
            self.transition_stratification_name,
            categories=transitions,
            excluded_categories=excluded_categories,
            mapper=self.map_transitions,
            requires_columns=[self.disease, self.previous_state_column_name],
            is_vectorized=True,
        )

    def count_treatment_durations(self, pop: pd.DataFrame) -> float:
        """Aggregate the total treatment durations for simulants in the population."""
        # Handle first time step where we have initial population and time step observations
        if self.clock() == self.sim_start_time:
            treatment_durations = pop[COLUMNS.WAITING_FOR_TREATMENT_EVENT_TIME].notna()
        else:
            treatment_durations = (
                pop[COLUMNS.WAITING_FOR_TREATMENT_EVENT_TIME]
                == self.clock() + self.step_size()
            )
        return sum(treatment_durations)

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        """Rename the appropriate column to 'sub_entity'.

        The primary thing this method does is rename the appropriate column
        (either the transition stratification name of the disease name, depending
        on the measure) to 'sub_entity'. We do this here instead of the
        'get_sub_entity_column' method simply because we do not want the original
        column at all. If we keep it here and then return it as the sub-entity
        column later, the final results would have both.

        Parameters
        ----------
        measure
            The measure.
        results
            The results to format.

        Returns
        -------
            The formatted results.
        """
        results = results.reset_index()
        if "transition_count_" in measure:
            sub_entity = self.transition_stratification_name
        if "person_time_" in measure:
            sub_entity = self.disease
        # Handle treatment_duration measure
        if measure == "treatment_duration":
            sub_entity = "treatment_durations"
        results.rename(columns={sub_entity: "sub_entity"}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the 'measure' column values."""
        if "transition_count_" in measure:
            measure_name = "transition_count"
        if "person_time_" in measure:
            measure_name = "person_time"
        if measure == "treatment_duration":
            measure_name = "treatment_duration_count"
        return pd.Series(measure_name, index=results.index)
