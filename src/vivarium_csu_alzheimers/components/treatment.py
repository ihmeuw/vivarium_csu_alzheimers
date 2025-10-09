from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium.framework.state_machine import Transition
from vivarium_public_health.disease import (
    DiseaseModel,
    DiseaseState,
    SusceptibleState,
    TransientDiseaseState,
)

from vivarium_csu_alzheimers.constants import scenarios
from vivarium_csu_alzheimers.constants.data_values import (
    BBBM_TEST_RESULTS,
    COLUMNS,
    DWELL_TIME_AWAITING_EFFECT_TIMESTEPS,
    DWELL_TIME_FULL_EFFECT_LONG_TIMESTEPS,
    DWELL_TIME_FULL_EFFECT_SHORT_TIMESTEPS,
    DWELL_TIME_WANING_EFFECT_LONG_TIMESTEPS,
    DWELL_TIME_WANING_EFFECT_SHORT_TIMESTEPS,
    LOCATION_TREATMENT_PROBS,
    TREATMENT_COMPLETION_PROBABILITY,
)
from vivarium_csu_alzheimers.utilities import get_timedelta_from_step_size


class TreatmentModel(DiseaseModel):
    """Alzheimer's treatment disease model."""

    @property
    def name(self) -> str:
        """Need to override the default name for DiseaseObserver to work."""
        return f"disease_model.{self.cause}"

    @property
    def time_step_priority(self) -> int:
        """We want treatment to occur after testing updates."""
        return 7


class Treatment(Component):
    """Alzheimer's treatment model component."""

    @property
    def sub_components(self) -> list[Component]:
        return [self.disease_model]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [self.randomness]

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.TREATMENT_PROPENSITY]

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.BBBM_TEST_RESULT]

    def __init__(self):
        super().__init__()
        self.step_size = 182  # days
        self.disease_model = self._create_treatment_mode()

    def setup(self, builder) -> None:
        # Check that step size set in init is correct
        if builder.configuration.time.step_size != self.step_size:
            raise ValueError(
                f"The step size set in the Treatment.__init__ method ({self.step_size}) "
                f"does not match the model configuration time step size ({builder.configuration.time.step_size})."
            )
        self.location = Path(builder.configuration.input_data.artifact_path).stem
        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream(self.name)
        self.scenario = scenarios.INTERVENTION_SCENARIOS[
            builder.configuration.intervention.scenario
        ]

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize treatment propensity for new simulants."""
        propensity = self.randomness.get_draw(
            pop_data.index, additional_key=COLUMNS.TREATMENT_PROPENSITY
        )
        propensity.name = COLUMNS.TREATMENT_PROPENSITY
        self.population_view.update(propensity)

    def _create_treatment_mode(self) -> TreatmentModel:

        # states
        susceptible = SusceptibleState("susceptible", allow_self_transition=True)
        positive_test = TransientDiseaseState("positive_test")
        start_treatment = DiseaseState(
            "start_treatment",
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_AWAITING_EFFECT_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        full_effect_long = DiseaseState(
            "full_effect_long",
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_FULL_EFFECT_LONG_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        full_effect_short = DiseaseState(
            "full_effect_short",
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_FULL_EFFECT_SHORT_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        waning_effect_long = DiseaseState(
            "waning_effect_long",
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_WANING_EFFECT_LONG_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        waning_effect_short = DiseaseState(
            "waning_effect_short",
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_WANING_EFFECT_SHORT_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        no_effect_after_short = DiseaseState(
            "no_effect_after_short",
            allow_self_transition=True,
            prevalence=0.0,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        no_effect_after_long = DiseaseState(
            "no_effect_after_long",
            allow_self_transition=True,
            prevalence=0.0,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        no_effect_never_treated = DiseaseState(
            "no_effect_never_treated",
            allow_self_transition=True,
            prevalence=0.0,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        # transitions
        susceptible.add_transition(
            output_state=positive_test, probability_function=self.has_positive_results
        )
        positive_test.add_transition(
            output_state=start_treatment, probability_function=self.start_treatment_probs
        )
        positive_test.add_transition(
            output_state=no_effect_never_treated,
            probability_function=self.decline_treatment_probs,
        )
        start_treatment.add_proportion_transition(
            full_effect_long, proportion=TREATMENT_COMPLETION_PROBABILITY
        )
        full_effect_long.add_transition(output_state=waning_effect_long)
        waning_effect_long.add_transition(output_state=no_effect_after_long)
        start_treatment.add_proportion_transition(
            full_effect_short, proportion=(1 - TREATMENT_COMPLETION_PROBABILITY)
        )
        full_effect_short.add_transition(output_state=waning_effect_short)
        waning_effect_short.add_transition(output_state=no_effect_after_short)

        return TreatmentModel(
            "treatment",
            initial_state=susceptible,
            states=[
                susceptible,
                positive_test,
                start_treatment,
                full_effect_long,
                full_effect_short,
                waning_effect_long,
                waning_effect_short,
                no_effect_after_long,
                no_effect_after_short,
                no_effect_never_treated,
            ],
            cause_specific_mortality_rate=0.0,
        )

    def has_positive_results(self, index: pd.Index[int]) -> pd.Series[float]:
        """Returns 1 if the bbbm test result is positive, 0 otherwise."""
        pop = self.population_view.subview(COLUMNS.BBBM_TEST_RESULT).get(index)
        is_positive = pd.Series(0.0, index=index)
        is_positive[pop[COLUMNS.BBBM_TEST_RESULT] == BBBM_TEST_RESULTS.POSITIVE] = 1.0
        return is_positive

    def start_treatment_probs(self, index: pd.Index[int]) -> pd.Series[float]:
        """Returns 1 if the propensity is less that treatment probability, 0 otherwise."""
        pop = self.population_view.subview(COLUMNS.TREATMENT_PROPENSITY).get(index)
        event_date = self.clock() + pd.Timedelta(days=self.step_size)
        probs = pd.Series(0.0, index=index)

        if not self.scenario.treatment:
            return probs

        TREATMENT_PROBS = LOCATION_TREATMENT_PROBS[self.location]
        if isinstance(TREATMENT_PROBS, float):
            treatment_prob = TREATMENT_PROBS
        elif event_date < TREATMENT_PROBS[0][0]:
            # Before the first defined time point, return 0
            treatment_prob = 0.0
        elif event_date > TREATMENT_PROBS[-1][0]:
            # Everything after the defined time point is a constant rate
            treatment_prob = TREATMENT_PROBS[-1][1]
        else:
            # interpolate
            timestamps = [ts.value for ts, _ in TREATMENT_PROBS]
            rates = [rate for _, rate in TREATMENT_PROBS]
            treatment_prob = np.interp(event_date.value, timestamps, rates)

        probs[pop[COLUMNS.TREATMENT_PROPENSITY] < treatment_prob] = 1.0
        return probs

    def decline_treatment_probs(self, index: pd.Index[int]) -> pd.Series[float]:
        """Returns the inverse of the start treatment probabilities."""
        start_treatment_probs = self.start_treatment_probs(index)
        probs = 1 - start_treatment_probs
        return probs


# TODO
class RiskEffect:
    ...

    # mostly the same as vph's, but if the exposure is waning, we need to interpolate
    # This effect should be targeting the pipeline transition
    # need to modify the RiskEffect.get_relative_risk_source. This is the thing
    # that would return a callable that returns 1 if untreated or 0.4-0.6 (full treatment)
    # or waning 0.4-0.6 - 1 (depending on waning).

    # OR register a modifier to the pipeline.
