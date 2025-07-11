from collections.abc import Callable

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import Machine, State, TransientState, Transition
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.disease.transition import RateTransition

from vivarium_csu_alzheimers.constants.models import (
    ALZHEIMERS_DISEASE_MODEL,
    TESTING_ALZHEIMERS_DISEASE_MODEL,
)

# Placehodler constants for the testing model. These will be replaced by real data later.
TESTING_RATE = 0.75
POSTIVE_TEST_RATE = 0.9


class TestingState(TransientState):
    def add_transition(
        self,
        output_state: State,
        **kwargs,
    ) -> Transition:
        transition = TestingTransition(
            self,
            output_state,
            **kwargs,
        )
        self.transition_set.append(transition)
        return transition


class TestingTransition(RateTransition):
    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> list[str]:
        return ["age", "alive", ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_MODEL_NAME]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.alzheimers_column = ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_MODEL_NAME
        self.alzheimers_susceptible_state = ALZHEIMERS_DISEASE_MODEL.SUSCEPTIBLE_TO_ALZHEIMERS

    ###################
    # Pipeline methods#
    ###################

    def compute_transition_rate(self, index) -> pd.Series:
        transition_rate = pd.Series(0.0, index=index)
        sub_pop = self.population_view.get(
            index,
            query=f"(alive == 'alive') & ({self.alzheimers_column} != '{self.alzheimers_susceptible_state}' & (age > 50))",
        ).index

        transition_rate.loc[sub_pop] = self.lookup_tables["transition_rate"](sub_pop)
        return transition_rate


class TestingForAlzheimers(Component):
    """A class to hold the testing model for Alzheimer's disease. This class includes
    the different states of the testing process and how simulants can transition between them.
    """

    @property
    def sub_components(self) -> list[Component]:
        return [self.testing_model]

    def __init__(self):
        super().__init__()
        self.testing_model = self._create_testing_model()

    def _create_testing_model(self):
        susceptible = SusceptibleState(
            TESTING_ALZHEIMERS_DISEASE_MODEL.SUSCEPTIBLE_TO_TESTING
        )
        testing = TestingState(TESTING_ALZHEIMERS_DISEASE_MODEL.TESTING_STATE)
        positive = DiseaseState(
            TESTING_ALZHEIMERS_DISEASE_MODEL.POSIITIVE_STATE,
            prevalence=0.0,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        negative = DiseaseState(
            TESTING_ALZHEIMERS_DISEASE_MODEL.NEGATIVE_STATE,
            prevalence=0.0,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )

        # Add transitions between states
        susceptible.add_transition(
            output_state=testing,
            probability_function=lambda index: pd.Series(TESTING_RATE, index=index),
        )
        testing.add_transition(
            output_state=positive,
            transition_rate=lambda index: pd.Series(POSTIVE_TEST_RATE, index=index),
        )
        testing.add_transition(
            output_state=negative,
            transition_rate=lambda index: pd.Series(1 - POSTIVE_TEST_RATE, index=index),
        )
        negative.add_transition(
            output_state=susceptible,
            probability_function=lambda index: pd.Series(1.0, index=index),
        )

        return DiseaseModel(
            TESTING_ALZHEIMERS_DISEASE_MODEL.TESTING_FOR_ALZHEIMERS_MODEL_NAME,
            initial_state=susceptible,
            states=[susceptible, testing, positive, negative],
        )
