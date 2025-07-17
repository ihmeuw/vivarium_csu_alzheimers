import pandas as pd
from vivarium import Component
from vivarium.framework.state_machine import State, TransientState
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.disease.transition import TransitionString

from vivarium_csu_alzheimers.constants.models import TESTING_ALZHEIMERS_DISEASE_MODEL

# Placehodler constants for the testing model. These will be replaced by real data later.
TESTING_RATE = 0.4  # per time step
POSITIVE_TEST_RATE = 0.9
POSITIVE_TEST_TRANSITION_RATE = TESTING_RATE * POSITIVE_TEST_RATE
NEGATIVE_TEST_TRANSITION_RATE = TESTING_RATE * (1 - POSITIVE_TEST_RATE)


class TestingState(TransientState):
    """A transient state that will be compatible with the vivarium public health disease model."""

    def get_transition_names(self) -> list[str]:
        transitions = []
        for trans in self.transition_set.transitions:
            init_state = trans.input_state.name.split(".")[1]
            end_state = trans.output_state.name.split(".")[1]
            transitions.append(TransitionString(f"{init_state}_TO_{end_state}"))
        return transitions


class TestingForAlzheimers(Component):
    """A class to hold the testing model for Alzheimer's disease. This class includes
    the different states of the testing process and how simulants can transition between them.
    NOTE: This testing model will test simulants that are susceptible to Alzheimer's disease and
    they could test positive or negative. I did not implement this in a way that makes all tests
    for simulants who are susceptible to Alzheimer's disease to be negative since we likely will
    not have a susceptible state in the final model. I also could not get a TransientState to work
    within a disease model so I had skip the Transient state and go straight to postive and negative
    states.

    """

    @property
    def sub_components(self) -> list[Component]:
        return [self.testing_model]

    def __init__(self) -> None:
        super().__init__()
        self.testing_model = self._create_testing_model()

    def _create_testing_model(self) -> DiseaseModel:
        susceptible = SusceptibleState(
            TESTING_ALZHEIMERS_DISEASE_MODEL.SUSCEPTIBLE_TO_TESTING,
            allow_self_transition=True,
        )
        # TODO: can't get TransientState to work in a DiseaseModel
        testing = TestingState(TESTING_ALZHEIMERS_DISEASE_MODEL.TESTING_STATE)
        positive = DiseaseState(
            TESTING_ALZHEIMERS_DISEASE_MODEL.POSITIVE_STATE,
            prevalence=0.0,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        duration = pd.Timedelta(days=(365 * 2))  # 2 years
        negative = DiseaseState(
            TESTING_ALZHEIMERS_DISEASE_MODEL.NEGATIVE_STATE,
            prevalence=0.0,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
            get_data_functions={"dwell_time": lambda _, __: duration},
        )

        # Add transitions between states
        susceptible.add_transition(
            output_state=testing,
            probability_function=lambda index: pd.Series(TESTING_RATE, index=index),
        )
        testing.add_transition(
            output_state=positive,
            probability_function=lambda index: pd.Series(POSITIVE_TEST_RATE, index=index),
        )
        testing.add_transition(
            output_state=negative,
            probability_function=lambda index: pd.Series(1 - POSITIVE_TEST_RATE, index=index),
        )
        negative.add_dwell_time_transition(output=susceptible)

        return DiseaseModel(
            TESTING_ALZHEIMERS_DISEASE_MODEL.TESTING_FOR_ALZHEIMERS_MODEL_NAME,
            initial_state=susceptible,
            states=[susceptible, testing, positive, negative],
        )
