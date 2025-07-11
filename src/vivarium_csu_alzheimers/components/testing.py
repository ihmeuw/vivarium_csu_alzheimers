from collections.abc import Callable

import pandas as pd
from vivarium import Component
from vivarium.framework.state_machine import Machine, State, TransientState

from vivarium_csu_alzheimers.constants.models import TESTING_ALZHEIMERS_DISEASE_MODEL

# Placehodler constants for the testing model. These will be replaced by real data later.
TESTING_RATE = 0.25
POSTIVE_TEST_RATE = 0.9


class TestingForAlzheimers(Component):
    """A class to hold the testing model for Alzheimer's disease. This class includes
    the different states of the testing process and how simulants can transition between them.
    NOTE: This testing model will test simulants that are susceptible to Alzheimer's disease and
    they could test positive or negative. I did not implement this in a way that makes all tests
    for simulants who are susceptible to Alzheimer's disease to be negative since we likely will
    not have a susceptible state in the final model.
    """

    @property
    def sub_components(self) -> list[Component]:
        return [self.testing_model]

    def __init__(self) -> None:
        super().__init__()
        self.testing_model = self._create_testing_model()

    def _create_testing_model(self) -> Machine:
        susceptible = State(
            TESTING_ALZHEIMERS_DISEASE_MODEL.SUSCEPTIBLE_TO_TESTING,
            allow_self_transition=True,
        )
        testing = TransientState(
            TESTING_ALZHEIMERS_DISEASE_MODEL.TESTING_STATE,
        )
        positive = State(TESTING_ALZHEIMERS_DISEASE_MODEL.POSIITIVE_STATE)
        negative = State(TESTING_ALZHEIMERS_DISEASE_MODEL.NEGATIVE_STATE)

        # Add transitions between states
        susceptible.add_transition(
            output_state=testing,
            probability_function=lambda index: pd.Series(TESTING_RATE, index=index),
        )
        testing.add_transition(
            output_state=positive,
            probability_function=lambda index: pd.Series(POSTIVE_TEST_RATE, index=index),
        )
        testing.add_transition(
            output_state=negative,
            probability_function=lambda index: pd.Series(1 - TESTING_RATE, index=index),
        )
        negative.add_transition(
            output_state=susceptible,
            probability_function=lambda index: pd.Series(1.0, index=index),
        )

        return Machine(
            TESTING_ALZHEIMERS_DISEASE_MODEL.TESTING_FOR_ALZHEIMERS_MODEL_NAME,
            states=[susceptible, testing, positive, negative],
            initial_state=susceptible,
        )
