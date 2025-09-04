import pandas as pd
from vivarium import Component
from vivarium.framework.state_machine import State, TransientState
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_csu_alzheimers.constants.data_keys import ALZHEIMERS
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

    def setup(self, builder):
        # Load artifact data for disease model
        self.bbbm_prevalence = builder.data.load(ALZHEIMERS.BBBM_CONDITIONAL_PREVALANCE)
        self.mci_prevalence = builder.data.load(ALZHEIMERS.MCI_CONDITIONAL_PREVALENCE)
        self.alzheimers_prevalence = builder.data.load(ALZHEIMERS.PREVALENCE)
        # TODO: bbbm disability weight
        self.mci_disability_weight = builder.data.load(ALZHEIMERS.MCI_DISABILITY_WEIGHT)
        self.alzheimers_disability_weight = builder.data.load(ALZHEIMERS.DISABILIITY_WEIGHT)
        self.alzheimers_emr = builder.data.load(ALZHEIMERS.EMR)
        # TODO: load bbbm excess mortality rate = 0.0
        # TODO: load mci excess mortality rate = 0.0
        # TODO: load transition rates
        self.csmr = builder.data.load(ALZHEIMERS.CSMR)  

    def _create_disease_model(self):
        bbbm_state = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.BBBM_STATE,
            allow_self_transition=True,
            prevalence=self.bbbm_prevalence,
            disability_weight=DISABILITY_WEIGHT,
            excess_mortality_rate=0.0,
        )
        mci_state = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.MCI_STATE,
            allow_self_transition=True,
            prevalence=self.mci_prevalence,
            disability_weight=self.mci_disability_weight,
            excess_mortality_rate=0.0,
        )
        alzheimers_state = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_DISEASE_STATE,
            prevalence=self.alzheimers_prevalence,
            disability_weight=self.alzheimers_disability_weight,
            excess_mortality_rate=self.alzheimers_emr,
        )

        # Add transitions between states
        bbbm_state.add_rate_transition(
            output=mci_state,
            # TODO: update to get_data_functions
            transition_rate=TRANSITION_RATE,
        )
        mci_state.add_rate_transition(
            output=alzheimers_state,
            # TODO: update to get_data_functions
            transition_rate=TRANSITION_RATE,
        )

        return DiseaseModel(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_MODEL_NAME,
            initial_state=bbbm_state,
            states=[bbbm_state, mci_state, alzheimers_state],
        )
