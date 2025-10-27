import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.values import list_combiner, union_post_processor
from vivarium_public_health.disease import (
    BaseDiseaseState,
    DiseaseModel,
    DiseaseState,
    RateTransition,
)

from vivarium_csu_alzheimers.constants.data_keys import ALZHEIMERS_CONSISTENT as ALZHEIMERS
from vivarium_csu_alzheimers.constants.data_values import (
    BBBM_AVG_DURATION,
    BBBM_HAZARD_DIST,
    COLUMNS,
    DW_BBBM,
    EMR_BBBM,
    EMR_MCI,
    EMR_MILD,
    MCI_AVG_DURATION,
)
from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL
from vivarium_csu_alzheimers.data.mci_hazard import hazard


class BBBMTransitionRate(RateTransition):
    @property
    def columns_required(self) -> list[str]:
        return super().columns_required + [COLUMNS.BBBM_ENTRANCE_TIME]

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.step_size = builder.configuration.time.step_size / 365.0
        # Code below is copy/paste from super().setup but need bbbm entrance time column
        paf = builder.lookup.build_table(0)
        self.joint_paf = builder.value.register_value_producer(
            f"{self.transition_rate_pipeline_name}.paf",
            source=lambda index: [paf(index)],
            component=self,
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
        self.transition_rate = builder.value.register_rate_producer(
            self.transition_rate_pipeline_name,
            source=self.compute_transition_rate,
            component=self,
            required_resources=["alive", self.joint_paf, COLUMNS.BBBM_ENTRANCE_TIME],
        )
        self.rate_conversion_type = self.configuration["rate_conversion_type"]

    def compute_transition_rate(self, index: pd.Index) -> pd.Series:
        entrance_time = self.population_view.get(index)["bbbm_entrance_time"]
        current_time = self.clock()
        time_diff_numeric_years = (current_time - entrance_time).dt.total_seconds() / (
            365.0 * 24 * 3600
        )

        return hazard(time_diff_numeric_years, BBBM_HAZARD_DIST)


class BBBMDiseaseState(DiseaseState):
    def add_bbbm_transition(self, output: BaseDiseaseState) -> BBBMTransitionRate:
        """Method to instantiate BBBMTransitionRate and add it to the disease state."""
        transition = BBBMTransitionRate(
            input_state=self, output_state=output, transition_rate=0.0
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

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.BBBM_ENTRANCE_TIME]

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.ENTRANCE_TIME]

    @property
    def initialization_requirements(self):
        return [COLUMNS.ENTRANCE_TIME, self.randomness]

    def __init__(self):
        super().__init__()
        self.disease_model = self._create_disease_model()

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)
        self.step_size = builder.configuration.time.step_size / 365.0

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize BBBM entrance time for new simulants."""
        new_simulants = pd.DataFrame(index=pop_data.index)
        if pop_data.user_data.get("sim_state") == "time_step":
            new_simulants[COLUMNS.BBBM_ENTRANCE_TIME] = pop_data.creation_time
        else:
            draws = self.randomness.get_draw(
                pop_data.index, additional_key="bbbm_entrance_time"
            )
            bbbm_entrance_time = pd.to_datetime(
                draws * (BBBM_AVG_DURATION) * 365.0 * -1.0,
                yearfirst=True,
                unit="D",
                origin=pop_data.creation_time + pd.Timedelta(days=self.step_size * 365.0),
            )
            new_simulants[COLUMNS.BBBM_ENTRANCE_TIME] = bbbm_entrance_time

        self.population_view.update(new_simulants)

    def _create_disease_model(self) -> DiseaseModel:
        bbbm_state = BBBMDiseaseState(
            ALZHEIMERS_DISEASE_MODEL.BBBM_STATE,
            allow_self_transition=True,
            prevalence=lambda builder: builder.data.load(
                ALZHEIMERS.BBBM_CONDITIONAL_PREVALENCE
            ),
            disability_weight=DW_BBBM,
            excess_mortality_rate=EMR_BBBM,
        )
        mci_state = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.MCI_STATE,
            allow_self_transition=True,
            prevalence=lambda builder: builder.data.load(
                ALZHEIMERS.MCI_CONDITIONAL_PREVALENCE
            ),
            disability_weight=lambda builder: builder.data.load(
                ALZHEIMERS.MCI_DISABILITY_WEIGHT
            ),
            excess_mortality_rate=EMR_MCI,
        )
        mild_dementia_state = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.MILD_DEMENTIA_STATE,
            allow_self_transition=True,
            prevalence=lambda builder: builder.data.load(ALZHEIMERS.MILD_DEMENTIA_CONDITIONAL_PREVALENCE),
            disability_weight=lambda builder: builder.data.load(ALZHEIMERS.MILD_DEMENTIA_DISABILITY_WEIGHT),
            excess_mortality_rate=EMR_MILD,
        )
        moderate_dementia_state = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.MODERATE_DEMENTIA_STATE,
            allow_self_transition=True,
            prevalence=lambda builder: builder.data.load(ALZHEIMERS.MODERATE_DEMENTIA_CONDITIONAL_PREVALENCE),
            disability_weight=lambda builder: builder.data.load(ALZHEIMERS.MODERATE_DEMENTIA_DISABILITY_WEIGHT),
            excess_mortality_rate=lambda builder: builder.data.load(ALZHEIMERS.EMR),
        )
        severe_dementia_state = DiseaseState(
            ALZHEIMERS_DISEASE_MODEL.SEVERE_DEMENTIA_STATE,
            allow_self_transition=True,
            prevalence=lambda builder: builder.data.load(ALZHEIMERS.SEVERE_DEMENTIA_CONDITIONAL_PREVALENCE),
            disability_weight=lambda builder: builder.data.load(ALZHEIMERS.SEVERE_DEMENTIA_DISABILITY_WEIGHT),
            excess_mortality_rate=lambda builder: builder.data.load(ALZHEIMERS.EMR),
        )

        # AD progression transitions
        bbbm_state.add_bbbm_transition(output=mci_state)
        mci_state.add_rate_transition(
            output=mild_dementia_state,
            transition_rate=1/MCI_AVG_DURATION,
        )
        mild_dementia_state.add_rate_transition(
            output=moderate_dementia_state,
            transition_rate=lambda builder: builder.data.load(ALZHEIMERS.MILD_TO_MODERATE_DEMENTIA_TRANSITION_RATE),
        )
        moderate_dementia_state.add_rate_transition(
            output=severe_dementia_state,
            transition_rate=lambda builder: builder.data.load(ALZHEIMERS.MODERATE_TO_SEVERE_DEMENTIA_TRANSITION_RATE),
        )

        return DiseaseModel(
            ALZHEIMERS_DISEASE_MODEL.NAME,
            initial_state=bbbm_state,
            states=[bbbm_state, mci_state, mild_dementia_state, moderate_dementia_state, severe_dementia_state],
        )
