from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium_public_health import (
    DiseaseModel,
    DiseaseState,
    RiskEffect,
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
from vivarium_csu_alzheimers.constants.models import TREATMENT_DISEASE_MODEL
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

    @property
    def columns_created(self) -> list[str]:
        """Override because we create the column in Treatment."""
        return []

    @property
    def columns_required(self) -> list[str]:
        """Need to add the column that we removed from column_created here."""
        cols = super().columns_required
        cols += [COLUMNS.TREATMENT_STATE]
        return cols

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Typical DiseaseModel initialization except we do not initialize the state column.

        We skip the super's (Machine's) on_initialize_simulants call because we do
        not want to initialize the disease states here since we are handling that
        in the Treatment component.
        """
        if pop_data.user_data.get("age_end", self.configuration_age_end) == 0:
            initialization_table_name = "birth_prevalence"
        else:
            initialization_table_name = "prevalence"

        for state in self.states:
            state.lookup_tables["initialization_weights"] = state.lookup_tables[
                initialization_table_name
            ]


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
        return [
            COLUMNS.TREATMENT_PROPENSITY,
            COLUMNS.TREATMENT_STATE,
            "waiting_for_treatment_event_time",
            "waiting_for_treatment_event_count",
            "no_effect_never_treated_event_time",
            "no_effect_never_treated_event_count",
        ]

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
        # register an exposure pipeline that just turns around
        builder.value.register_value_producer(
            f"{COLUMNS.TREATMENT_STATE}.exposure",
            source=self.get_treatment_states,
            component=self,
            required_resources=[COLUMNS.TREATMENT_STATE],
        )

    def get_treatment_states(self, index: pd.Index) -> pd.Series:
        return self.population_view.subview(COLUMNS.TREATMENT_STATE).get(index).squeeze()

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize treatment propensity for new simulants."""
        # Create the propensity column
        propensity = self.randomness.get_draw(
            pop_data.index, additional_key=COLUMNS.TREATMENT_PROPENSITY
        )
        propensity.name = COLUMNS.TREATMENT_PROPENSITY
        # We need the propensity column to exist in the population view in the
        # `start_treatment_probs` call below, so update now.
        self.population_view.update(propensity)

        # Create the waiting_for_treatment and no_effect_never_treated event time/count columns
        # HACK: We do this here rather than in the DiseaseState for these two classes
        #   because we need innitialized simulants to be treated as well as tested.
        #   It would be better to do this in the TreatmentModel itself, but it's not
        #   trivial to handle since we need to know the location to apply the appropriate
        #   treatment probabilities which requires access to the builder.
        #   Note that we are only special-casing the waiting_for_treatment and
        #   no_effect_never_treated states here because it's critical that we not
        #   skip potential treatment for initialized simulants; all other states
        #   in the disease model (aside from Susceptible) are downstream of starting
        #   treatment and since no simulants are initialized having already started
        #   treatment, they can be handled in the normal way.
        positive_results = self.has_positive_results(pop_data.index)
        positive_results_idx = positive_results[positive_results == 1].index
        start_treatment_probs = self.start_treatment_probs(positive_results_idx)
        # During initialization, we can only update the entire pop_data and so we cannot
        # just get the positive_results idx here.
        start_treatment_idx = (start_treatment_probs[start_treatment_probs == 1]).index
        decline_treatment_idx = positive_results_idx.difference(start_treatment_idx)
        event_time = pop_data.creation_time + get_timedelta_from_step_size(self.step_size)
        update = pd.DataFrame(
            data={
                COLUMNS.TREATMENT_STATE: f"{TREATMENT_DISEASE_MODEL.SUSCEPTIBLE_STATE}_to_treatment",
                "waiting_for_treatment_event_time": pd.NaT,
                "waiting_for_treatment_event_count": 0,
                "no_effect_never_treated_event_time": pd.NaT,
                "no_effect_never_treated_event_count": 0,
            },
            index=pop_data.index,
        )
        update.loc[
            update.index.isin(start_treatment_idx),
            [
                COLUMNS.TREATMENT_STATE,
                "waiting_for_treatment_event_time",
                "waiting_for_treatment_event_count",
            ],
        ] = [TREATMENT_DISEASE_MODEL.WAITING_FOR_TREATMENT_STATE, event_time, 1]
        update.loc[
            update.index.isin(decline_treatment_idx),
            [
                COLUMNS.TREATMENT_STATE,
                "no_effect_never_treated_event_time",
                "no_effect_never_treated_event_count",
            ],
        ] = [TREATMENT_DISEASE_MODEL.NO_EFFECT_NEVER_TREATED_STATE, event_time, 1]

        self.population_view.update(update)

    def _create_treatment_mode(self) -> TreatmentModel:

        # states
        susceptible = SusceptibleState(
            TREATMENT_DISEASE_MODEL.NAME, allow_self_transition=True
        )
        positive_test = TransientDiseaseState(
            TREATMENT_DISEASE_MODEL.POSITIVE_TEST_TRANSIENT_STATE
        )
        waiting_for_treatment = PositiveTestDecisionState(
            TREATMENT_DISEASE_MODEL.WAITING_FOR_TREATMENT_STATE,
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_AWAITING_EFFECT_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        full_effect_long = DiseaseState(
            TREATMENT_DISEASE_MODEL.FULL_EFFECT_LONG_STATE,
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_FULL_EFFECT_LONG_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        full_effect_short = DiseaseState(
            TREATMENT_DISEASE_MODEL.FULL_EFFECT_SHORT_STATE,
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_FULL_EFFECT_SHORT_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        waning_effect_long = DiseaseState(
            TREATMENT_DISEASE_MODEL.WANING_EFFECT_LONG_STATE,
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_WANING_EFFECT_LONG_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        waning_effect_short = DiseaseState(
            TREATMENT_DISEASE_MODEL.WANING_EFFECT_SHORT_STATE,
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_WANING_EFFECT_SHORT_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        no_effect_after_short = DiseaseState(
            TREATMENT_DISEASE_MODEL.NO_EFFECT_AFTER_SHORT_STATE,
            allow_self_transition=True,
            prevalence=0.0,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        no_effect_after_long = DiseaseState(
            TREATMENT_DISEASE_MODEL.NO_EFFECT_AFTER_LONG_STATE,
            allow_self_transition=True,
            prevalence=0.0,
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        no_effect_never_treated = PositiveTestDecisionState(
            TREATMENT_DISEASE_MODEL.NO_EFFECT_NEVER_TREATED_STATE,
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
            output_state=waiting_for_treatment,
            probability_function=self.start_treatment_probs,
        )
        positive_test.add_transition(
            output_state=no_effect_never_treated,
            probability_function=self.decline_treatment_probs,
        )
        waiting_for_treatment.add_proportion_transition(
            full_effect_long, proportion=TREATMENT_COMPLETION_PROBABILITY
        )
        full_effect_long.add_transition(output_state=waning_effect_long)
        waning_effect_long.add_transition(output_state=no_effect_after_long)
        waiting_for_treatment.add_proportion_transition(
            full_effect_short, proportion=(1 - TREATMENT_COMPLETION_PROBABILITY)
        )
        full_effect_short.add_transition(output_state=waning_effect_short)
        waning_effect_short.add_transition(output_state=no_effect_after_short)

        return TreatmentModel(
            TREATMENT_DISEASE_MODEL.NAME,
            initial_state=susceptible,
            states=[
                susceptible,
                positive_test,
                waiting_for_treatment,
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
        event_date = self.clock() + get_timedelta_from_step_size(self.step_size)
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


class PositiveTestDecisionState(DiseaseState):
    """Override initialization of the columns so that Treatment can handle it."""

    @property
    def columns_created(self):
        # Remove the columns created during DiseaseState since they will be
        # created in Treatment
        return []

    @property
    def columns_required(self):
        # Need to add the columns that we removed from column_created here
        cols = super().columns_required
        cols += [self.event_count_column, self.event_time_column]
        return cols

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Adds this state's columns to the simulation state table."""
        for transition in self.transition_set:
            if transition.start_active:
                transition.set_active(pop_data.index)


class TreatmentRiskEffect(RiskEffect):
    """Risk effect for Alzheimer's treatment."""

    @property
    def name(self) -> str:
        return f"risk_effect.{self.risk}_on_{self.target}"

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Overwrites the paf to be 0 (because treatment is the intervention)."""
        defaults = super().configuration_defaults
        defaults[self.name]["data_sources"]["population_attributable_fraction"] = 0.0
        return defaults

    @property
    def columns_required(self) -> list[str]:
        return [
            f"{TREATMENT_DISEASE_MODEL.WANING_EFFECT_LONG_STATE}_event_time",
            f"{TREATMENT_DISEASE_MODEL.WANING_EFFECT_SHORT_STATE}_event_time",
        ]

    def __init__(self, target: str):
        super().__init__(risk="treatment.treatment", target=target)

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

    def get_distribution_type(self, builder: Builder) -> str:
        """Returns the type of distribution for the exposure.

        We overwrite this just to prevent runtime errors - we don't actually need it.
        """
        return "polytomous"

    def load_relative_risk(
        self,
        builder: Builder,
        configuration=None,
    ) -> str | float | pd.DataFrame:
        return builder.data.load("treatment.relative_risk")["value"][0]

    def get_relative_risk_source(self, builder: Builder) -> Callable[[pd.Index], pd.Series]:
        """Modifies the relative risk based on treatment exposure.

        Notes
        -----
        The unaffected source should be 1.

        The affected source should be between the relative risk loaded from the data
        and 1, depending on whether the simulant is in full effect, waning, or none.
        """

        def generate_relative_risk(index: pd.Index) -> pd.Series:
            rr = self.lookup_tables["relative_risk"](index)
            if len(rr.unique()) != 1:
                raise NotImplementedError("Only a single relative risk value is supported.")
            rr_min = rr.iloc[0]

            exposure = self.exposure(index)
            relative_risk = pd.Series(index=index, dtype=float)

            full_effect_states = [
                TREATMENT_DISEASE_MODEL.FULL_EFFECT_LONG_STATE,
                TREATMENT_DISEASE_MODEL.FULL_EFFECT_SHORT_STATE,
            ]
            waning_states = [
                TREATMENT_DISEASE_MODEL.WANING_EFFECT_LONG_STATE,
                TREATMENT_DISEASE_MODEL.WANING_EFFECT_SHORT_STATE,
            ]
            affected_states = full_effect_states + waning_states

            # Unaffected relative risks are 1
            relative_risk[~exposure.isin(affected_states)] = 1.0

            # Modify relative risks to be the minimum rr value for fully affected states
            relative_risk[exposure.isin(full_effect_states)] = rr_min

            # Modify relative risks to be interpolated values for waning states
            self._interpolate_rr(
                relative_risk,
                rr_min,
                exposure,
                TREATMENT_DISEASE_MODEL.WANING_EFFECT_SHORT_STATE,
            )
            self._interpolate_rr(
                relative_risk,
                rr_min,
                exposure,
                TREATMENT_DISEASE_MODEL.WANING_EFFECT_LONG_STATE,
            )

            if relative_risk.isna().any():
                raise ValueError("NaN values found in relative risk.")

            return relative_risk

        return generate_relative_risk

    def _interpolate_rr(
        self,
        relative_risk: pd.Series[float],
        rr_min: float,
        exposure: pd.Series[str],
        waning_state: str,
    ) -> None:
        waning_mask = exposure == waning_state
        if waning_mask.any():
            event_date = self.clock() + self.step_size()
            waning_start_date = pd.to_datetime(
                (
                    self.population_view.subview(f"{waning_state}_event_time")
                    .get(waning_mask[waning_mask].index)
                    .squeeze()
                )
            )
            dwell_time = {
                TREATMENT_DISEASE_MODEL.WANING_EFFECT_SHORT_STATE: DWELL_TIME_WANING_EFFECT_SHORT_TIMESTEPS,
                TREATMENT_DISEASE_MODEL.WANING_EFFECT_LONG_STATE: DWELL_TIME_WANING_EFFECT_LONG_TIMESTEPS,
            }[waning_state]
            waning_end_date = waning_start_date + get_timedelta_from_step_size(
                self.step_size().days, dwell_time
            )
            # Linearly interpolate between the source rr and 1 based on where the
            # event date is between the waning start date and the waning end date
            relative_risk[waning_mask] = rr_min + (1.0 - rr_min) * (
                (event_date - waning_start_date) / (waning_end_date - waning_start_date)
            )
