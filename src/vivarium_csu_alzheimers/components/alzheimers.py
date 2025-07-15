import pandas as pd
from vivarium import Component
from vivarium.framework.state_machine import State, TransientState
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL

# Constants for the Alzheimer's disease model
# These values are placeholders and should be replaced with actual data from research later
PREVALENCE = 1 / 6
EXCESS_MORTALITY_RATE = 0.035
DISABILITY_WEIGHT = 0.2
TRANSITION_RATE = 0.25


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
        susceptible = SusceptibleState(
            ALZHEIMERS_DISEASE_MODEL.SUSCEPTIBLE_TO_ALZHEIMERS,
            allow_self_transition=True,
        )
        state_1 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_FIRST_STATE,
            allow_self_transition=True,
            prevalence=PREVALENCE,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        state_2 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_SECOND_STATE,
            allow_self_transition=True,
            prevalence=PREVALENCE,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        state_3 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_THIRD_STATE,
            allow_self_transition=True,
            prevalence=PREVALENCE,
            disability_weight=DISABILITY_WEIGHT,
            excess_mortality_rate=0.0,
        )
        state_4 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_FOURTH_STATE,
            allow_self_transition=True,
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
        susceptible.add_proportion_transition(
            output=state_1,
            # TODO: this should be incidence right?
            proportion=TRANSITION_RATE,
        )
        state_1.add_rate_transition(
            output=state_2,
            transition_rate=TRANSITION_RATE,
        )
        state_2.add_rate_transition(
            output=state_3,
            transition_rate=TRANSITION_RATE,
        )
        state_3.add_rate_transition(
            output=state_4,
            transition_rate=TRANSITION_RATE,
        )
        state_4.add_rate_transition(
            output=state_5,
            transition_rate=TRANSITION_RATE,
        )
        # TODO: do we need an "end" state?

        return DiseaseModel(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_MODEL_NAME,
            initial_state=susceptible,
            states=[susceptible, state_1, state_2, state_3, state_4, state_5],
        )
