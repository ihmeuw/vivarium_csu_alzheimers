from typing import Callable
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import Trigger
from vivarium.types import DataInput
from vivarium_public_health.disease import BaseDiseaseState, DiseaseModel, DiseaseState, RateTransition

from vivarium_csu_alzheimers.constants.data_keys import ALZHEIMERS, POPULATION
from vivarium_csu_alzheimers.constants import data_values
from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL


class BBBMTransitionRate(RateTransition):
    def _probability(self, index: pd.Index) -> pd.Series:
        super()._probability(index)

class BBBMDiseaseState(DiseaseState):
    def add_rate_transition(
        self,
        output: BaseDiseaseState,
        get_data_functions: dict[str, Callable] = None,
        triggered=Trigger.NOT_TRIGGERED,
        transition_rate: DataInput | None = None,
        rate_type: str = "transition_rate",
    ) -> RateTransition:
        """Method to instantiate BBBMTransitionRate and add it to the disease state."""
        transition = BBBMTransitionRate(
            input_state=self,
            output_state=output,
            get_data_functions=get_data_functions,
            triggered=triggered,
            transition_rate=transition_rate,
            rate_type=rate_type,
        )
        self.add_transition(transition)
        return transition
    

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
        # self.bbbm_prevalence = builder.data.load(ALZHEIMERS.BBBM_CONDITIONAL_PREVALANCE)
        self.mci_prevalence = builder.data.load(ALZHEIMERS.MCI_CONDITIONAL_PREVALENCE)
        self.alzheimers_prevalence = builder.data.load(ALZHEIMERS.PREVALENCE)
        self.mci_disability_weight = builder.data.load(ALZHEIMERS.MCI_DISABILITY_WEIGHT)
        self.alzheimers_disability_weight = builder.data.load(ALZHEIMERS.DISABILITY_WEIGHT)
        self.alzheimers_emr = builder.data.load(ALZHEIMERS.EMR)
        # TODO: load transition rates
        self.csmr = builder.data.load(ALZHEIMERS.CSMR)
        self.mci_to_alzheimers_rate = self.get_mci_to_alzheimers_rate(builder)

    def _create_disease_model(self):
        bbbm_state = BBBMDiseaseState(
            ALZHEIMERS_DISEASE_MODEL.BBBM_STATE,
            allow_self_transition=True,
            prevalence=lambda builder: builder.data.load(ALZHEIMERS.BBBM_CONDITIONAL_PREVALANCE),
            disability_weight=data_values.DW_BBBM,
            excess_mortality_rate=data_values.EMR_BBBM,
        )
        mci_state = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.MCI_STATE,
            allow_self_transition=True,
            prevalence=lambda builder: builder.data.load(ALZHEIMERS.MCI_CONDITIONAL_PREVALENCE),
            disability_weight=lambda builder: builder.data.load(ALZHEIMERS.MCI_DISABILITY_WEIGHT),
            excess_mortality_rate=data_values.EMC_MCI,
        )
        alzheimers_state = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_DISEASE_STATE,
            prevalence=lambda builder: builder.data.load(ALZHEIMERS.PREVALENCE),
            disability_weight=lambda builder: builder.data.load(ALZHEIMERS.DISABILITY_WEIGHT),
            excess_mortality_rate=lambda builder: builder.data.load(ALZHEIMERS.EMR),
        )

        # Add transitions between states
        # TODO: write Transition (RateTranstion)
        bbbm_state.add_rate_transition(
            output=mci_state,
            # TODO: update to use hazard rate calculation
            transition_rate=0.25,  # placeholder value
        )
        mci_state.add_rate_transition(
            output=alzheimers_state,
            # TODO: update to calculate rate
            transition_rate=self.get_mci_to_alzheimers_rate,
        )

        return DiseaseModel(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_MODEL_NAME,
            initial_state=bbbm_state,
            states=[bbbm_state, mci_state, alzheimers_state],
        )

    def get_mci_to_alzheimers_rate(self, builder: Builder) -> pd.DataFrame:
        # 1/dealth_mci - (acmr - csmr_c543 + emr_mci)
        acmr = builder.data.load(POPULATION.ACMR)
        alz_csmr = builder.data.load(ALZHEIMERS.CSMR)
        m_mci = acmr - alz_csmr + data_values.EMC_MCI
        transition_rate = 1 / data_values.MCI_AVG_DURATION - m_mci
        return transition_rate
