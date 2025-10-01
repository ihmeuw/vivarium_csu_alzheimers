import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer
from vivarium_public_health import ResultsStratifier as ResultsStratifier_
from vivarium_public_health.results import DiseaseObserver, PublicHealthObserver

from vivarium_csu_alzheimers.constants.data_values import (
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


class TestingDiseaseObserver(DiseaseObserver):
    """Class to deal with the transient state in the disease model. Simulants immediately
    transition from susceptible to transient to positive or negative states. This means
    that this transitions go from susceptible to positive or negative states even though
    simulants transition through the transient state and this must be correctly mapped.

    """

    def __init__(self) -> None:
        super().__init__("testing_for_alzheimers")

    def register_transition_stratification(self, builder: Builder) -> None:
        # Hardcoding transitions to deal with the transient state since the
        # transient state is never a value in the current or previous state columns.
        transitions = [
            "susceptible_to_testing_for_alzheimers_to_positive_test_for_alzheimers",
            "susceptible_to_testing_for_alzheimers_to_negative_test_for_alzheimers",
            "negative_test_for_alzheimers_to_susceptible_to_testing_for_alzheimers",
            "no_transition",
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

    @property
    def columns_required(self) -> list[str]:
        return [
            COLUMNS.ALIVE,
            COLUMNS.TRACKED,
            COLUMNS.DISEASE_STATE,
            COLUMNS.PREVIOUS_DISEASE_STATE,
            COLUMNS.TESTING_STATE,
            COLUMNS.BBBM_TEST_RESULT,
        ]

    def register_observations(self, builder: Builder) -> None:
        # Register stratification
        builder.results.register_stratification(
            name="testing_state",
            categories=list(TESTING_STATES),
            requires_columns=[COLUMNS.TESTING_STATE],
        )

        # Register observation
        pop_filter = (
            'alive == "alive" and tracked == True '
            f'and {COLUMNS.PREVIOUS_DISEASE_STATE} == "{ALZHEIMERS_DISEASE_MODEL.BBBM_STATE}" '
            f'and {COLUMNS.DISEASE_STATE} != "{ALZHEIMERS_DISEASE_MODEL.BBBM_STATE}" '
            f'and {COLUMNS.BBBM_TEST_RESULT} != "{BBBM_TEST_RESULTS.POSITIVE}"'
        )
        self.register_adding_observation(
            builder=builder,
            name="baseline_testing_counts",
            pop_filter=pop_filter,
            requires_columns=self.columns_required,
            additional_stratifications=["testing_state"] + self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
        )

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("testing", index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        return pd.Series("baseline_testing", index=results.index)
