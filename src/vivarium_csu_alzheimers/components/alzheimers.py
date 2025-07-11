from vivarium import Component
from vivarium.framework.state_machine import State, TransientState
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL


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
            # get_data_functions={
            #     "prevalence": lambda *_: 1.0,
            #     "disability_weight": lambda *_: 0.0,
            #     "excess_mortality_rate": lambda *_: 0.0,
            #     # Add a dummy dwell time so we can overwrite it later
            #     "dwell_time": lambda *_: 0.0,
            # },
        )
        state_2 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_SECOND_STATE,
            # get_data_functions={
            #     "prevalence": lambda *_: 0.0,
            #     "disability_weight": lambda *_: 0.0,
            #     "excess_mortality_rate": lambda *_: 0.0,
            #     "dwell_time": lambda builder, cause: builder.time.step_size()(),
            # },
        )
        state_3 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_THIRD_STATE,
            # get_data_functions={
            #     "prevalence": lambda *_: 0.0,
            #     "disability_weight": lambda *_: 0.0,
            #     "excess_mortality_rate": lambda *_: 0.0,
            #     "dwell_time": lambda builder, cause: builder.time.step_size()(),
            # },
        )
        state_4 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_FOURTH_STATE,
            # get_data_functions={
            #     "prevalence": lambda *_: 0.0,
            #     "disability_weight": lambda *_: 0.0,
            #     "excess_mortality_rate": lambda *_: 0.0,
            #     "dwell_time": lambda builder, cause: builder.time.step_size()(),
            # },
        )
        state_5 = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_FIFTH_STATE,
            # get_data_functions={
            #     "prevalence": lambda *_: 0.0,
            #     "disability_weight": lambda *_: 0.0,
            #     "excess_mortality_rate": lambda *_: 0.0,
            #     "dwell_time": lambda builder, cause: builder.time.step_size()(),
            # },
        )

        # Add transitions between states
        susceptible.add_transition(
            output_state=state_1,
            probability_function=lambda *_: 0.75,
        )
        state_1.add_transition(
            output_state=state_2,
            probability_function=lambda *_: 0.75,
        )
        state_2.add_transition(
            output_state=state_3,
            probability_function=lambda *_: 0.75,
        )
        state_3.add_transition(
            output_state=state_4,
            probability_function=lambda *_: 0.75,
        )
        state_4.add_transition(
            output_state=state_5,
            probability_function=lambda *_: 0.75,
        )
        # TODO: do we need an "end" state?

        return DiseaseModel(
            ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_MODEL_NAME,
            initial_state=susceptible,
            states=[susceptible, state_1, state_2, state_3, state_4, state_5],
        )
