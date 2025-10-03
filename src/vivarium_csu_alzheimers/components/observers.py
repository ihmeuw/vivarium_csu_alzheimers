import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer
from vivarium.framework.time import get_time_stamp
from vivarium_public_health import ResultsStratifier as ResultsStratifier_
from vivarium_public_health.results import PublicHealthObserver

from vivarium_csu_alzheimers.constants.data_values import (
    BBBM_AGE_MAX,
    BBBM_AGE_MIN,
    BBBM_OLD_TIME,
    BBBM_TEST_RESULTS,
    COLUMNS,
    TESTING_STATES,
)
from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL


class ResultsStratifier(ResultsStratifier_):
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

    def register_stratifications(self, builder):
        super().register_stratifications(builder)
        builder.results.register_stratification(
            name="event_year",
            categories=[str(year) for year in range(self.start_year, self.end_year + 2)],
            excluded_categories=[str(self.end_year + 1)],
            mapper=self.map_year,
            is_vectorized=True,
            requires_columns=["event_time"],
        )
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


class NewSimulantsObserver(Observer):
    """Observer to count the number of new simulants added to the population"""

    def setup(self, builder: Builder) -> None:
        """Set up the observer component."""
        self.clock = builder.time.clock()

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="new_simulants",
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
            name="baseline_test_counts_among_eligible",
            pop_filter=pop_filter,
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
        )

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("testing", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("baseline_testing", index=results.index)


class BBBMTestCountObserver(PublicHealthObserver):
    """Observer to track BBBM testing counts."""

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.BBBM_TEST_DATE]

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

    def register_observations(self, builder: Builder) -> None:
        # TODO: clarify whether the default pop_filter to PublicHealthObserver
        #   should include alive == "alive" (it currently doesn't)
        pop_filter = 'alive == "alive" and tracked == True'
        self.register_adding_observation(
            builder=builder,
            name="bbbm_test_counts",
            pop_filter=pop_filter,
            requires_columns=self.columns_required,
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.count_bbbm_tests,
        )

    def count_bbbm_tests(self, pop: pd.DataFrame) -> float:
        return sum(pop[COLUMNS.BBBM_TEST_DATE] == self.clock() + self.step_size())

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("testing", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("bbbm_testing", index=results.index)


class BBBMTestEligibilityObserver(PublicHealthObserver):
    """Observer to track BBBM testing eligible simulant counts."""

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.DISEASE_STATE,
            COLUMNS.AGE,
            COLUMNS.BBBM_TEST_RESULT,
            COLUMNS.ENTRANCE_TIME,
            COLUMNS.BBBM_TEST_DATE,
        ]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

    def register_observations(self, builder: Builder) -> None:
        pop_filter = 'alive == "alive" and tracked == True'
        self.register_adding_observation(
            builder=builder,
            name="bbbm_test_eligibility_counts",
            pop_filter=pop_filter,
            requires_columns=self.columns_required,
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.count_eligible_simulants,
        )

    def count_eligible_simulants(self, pop: pd.DataFrame) -> float:
        """Counts the number of simulants eligible for BBBM testing this time step.

        There are three groups of eligible simulants:
        - those who enter the simulation and are already eligible
        - those who become eligible on a time step
        - those who got tested exactly three years ago and are eligible for a retest
        """
        eligible_state = pop[COLUMNS.DISEASE_STATE] == ALZHEIMERS_DISEASE_MODEL.BBBM_STATE
        eligible_age = (pop[COLUMNS.AGE] >= BBBM_AGE_MIN) & (pop[COLUMNS.AGE] < BBBM_AGE_MAX)
        eligible_test_result = pop[COLUMNS.BBBM_TEST_RESULT] != BBBM_TEST_RESULTS.POSITIVE
        eligible_baseline = eligible_state & eligible_age & eligible_test_result

        if not eligible_baseline.any():
            return 0.0

        sim_start_date = get_time_stamp(self.config.time.start)
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
        retest = (
            pop[COLUMNS.BBBM_TEST_DATE] == self.clock() + self.step_size() - BBBM_OLD_TIME
        )

        return sum(eligible_baseline & (new_entrants | aged_in | retest))

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("testing", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("bbbm_testing", index=results.index)
