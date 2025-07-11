import pandas as pd
from vivarium import Component
from vivarium.framework.state_machine import State, TransientState
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL

# Constants for the Alzheimer's disease model
# These values are placeholders and should be replaced with actual data from research later
PREVALENCE = 1 / 6
EXCESS_MORTALITY_RATE = 0.01
DISABILITY_WEIGHT = 0.2
TRANSITION_RATE = 0.75


class Alzheimers(Component):
    """A class to hold the Alzheimer's disease model. This class includes the
    different states of the disease and how simulants can transition between them.

    """

    @property
    def sub_components(self) -> list[Component]:
        return [self.disease_model]

    def __init__(self):
        super().__init__()
        self.disease_model = self._create_disease_model()

    def _create_disease_model(self):
        susceptible = SusceptibleState(ALZHEIMERS_DISEASE_MODEL.SUSCEPTIBLE_TO_ALZHEIMERS)
        state_1 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_FIRST_STATE,
            prevalence=PREVALENCE,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        state_2 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_SECOND_STATE,
            prevalence=PREVALENCE,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        state_3 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_THIRD_STATE,
            prevalence=PREVALENCE,
            disability_weight=DISABILITY_WEIGHT,
            excess_mortality_rate=0.0,
        )
        state_4 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_FOURTH_STATE,
            prevalence=PREVALENCE,
            disability_weight=DISABILITY_WEIGHT,
            excess_mortality_rate=0.0,
        )
        state_5 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_FIFTH_STATE,
            prevalence=PREVALENCE,
            disability_weight=DISABILITY_WEIGHT,
            excess_mortality_rate=EXCESS_MORTALITY_RATE,
        )

        # Add transitions between states
        susceptible.add_transition(
            output_state=state_1,
            probability_function=lambda index: pd.Series(TRANSITION_RATE, index=index),
        )
        state_1.add_transition(
            output_state=state_2,
            probability_function=lambda index: pd.Series(TRANSITION_RATE, index=index),
        )
        state_2.add_transition(
            output_state=state_3,
            probability_function=lambda index: pd.Series(TRANSITION_RATE, index=index),
        )
        state_3.add_transition(
            output_state=state_4,
            probability_function=lambda index: pd.Series(TRANSITION_RATE, index=index),
        )
        state_4.add_transition(
            output_state=state_5,
            probability_function=lambda index: pd.Series(TRANSITION_RATE, index=index),
        )
        # TODO: do we need an "end" state?

        return DiseaseModel(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_MODEL_NAME,
            initial_state=susceptible,
            states=[susceptible, state_1, state_2, state_3, state_4, state_5],
        )
