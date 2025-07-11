import pandas as pd
from vivarium import Component
from vivarium.framework.state_machine import Machine, State, TransientState
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_csu_alzheimers.constants.models import TESTING_ALZHEIMERS_DISEASE_MODEL


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
