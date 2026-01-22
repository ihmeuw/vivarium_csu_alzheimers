from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import Timedelta
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
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
    DWELL_TIME_TREATMENT_EFFECT_TIMESTEPS,
    DWELL_TIME_WANING_EFFECT_TIMESTEPS,
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
        return super().columns_required + [COLUMNS.TREATMENT_STATE]

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
            COLUMNS.WAITING_FOR_TREATMENT_EVENT_TIME,
            COLUMNS.WAITING_FOR_TREATMENT_EVENT_COUNT,
            COLUMNS.NO_EFFECT_NEVER_TREATED_EVENT_TIME,
            COLUMNS.NO_EFFECT_NEVER_TREATED_EVENT_COUNT,
            COLUMNS.TREATMENT_DURATION,
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
        builder.value.register_value_modifier(
            "treatment_effect.dwell_time",
            modifier=self.get_treatment_effect_duration,
            component=self,
            required_resources=[COLUMNS.TREATMENT_DURATION, "treatment_effect.dwell_time"],
        )
        builder.value.register_value_modifier(
            "waning_effect.dwell_time",
            modifier=self.get_waning_effect_duration,
            component=self,
            required_resources=[COLUMNS.TREATMENT_DURATION, "waning_effect.dwell_time"],
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

        # Initialize the treatment state column as well as treatment decision
        # event time and count columns
        # HACK: We need to manually do this here rather than relying on the TreatmentModel
        #   and DiseaseState classes because vivarium simulations by default do not
        #   include newly-initialized simulants when making decisions for a given
        #   time step, i.e. simulants need to be both tested and treated during
        #   initialization (without this they would be tested on initialization but
        #   not run through the treatment logic until the following time step).
        #
        #   NOTE: We do this here in Treatment rather than in the TreatmentModel
        #   Because we need to know the location to apply the appropriate
        #   treatment probabilities which requires access to the builder.
        #
        #   NOTE: We are only special-casing the waiting_for_treatment and
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
                COLUMNS.WAITING_FOR_TREATMENT_EVENT_TIME: pd.NaT,
                COLUMNS.WAITING_FOR_TREATMENT_EVENT_COUNT: 0,
                COLUMNS.NO_EFFECT_NEVER_TREATED_EVENT_TIME: pd.NaT,
                COLUMNS.NO_EFFECT_NEVER_TREATED_EVENT_COUNT: 0,
                COLUMNS.TREATMENT_DURATION: np.nan,
            },
            index=pop_data.index,
        )
        update.loc[
            start_treatment_idx,
            [
                COLUMNS.TREATMENT_STATE,
                COLUMNS.WAITING_FOR_TREATMENT_EVENT_TIME,
                COLUMNS.WAITING_FOR_TREATMENT_EVENT_COUNT,
            ],
        ] = [TREATMENT_DISEASE_MODEL.WAITING_FOR_TREATMENT_STATE, event_time, 1]
        # Update treatment duration for simulants waiting for treatment
        update.loc[
            start_treatment_idx, COLUMNS.TREATMENT_DURATION
        ] = self.get_treatment_duration(start_treatment_idx)

        update.loc[
            update.index.isin(decline_treatment_idx),
            [
                COLUMNS.TREATMENT_STATE,
                COLUMNS.NO_EFFECT_NEVER_TREATED_EVENT_TIME,
                COLUMNS.NO_EFFECT_NEVER_TREATED_EVENT_COUNT,
            ],
        ] = [TREATMENT_DISEASE_MODEL.NO_EFFECT_NEVER_TREATED_STATE, event_time, 1]

        self.population_view.update(update)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        # TODO: confirm this happens after disease state is updated to waiting for treatment
        waiting_for_treatment_idx = pop.index[
            pop[COLUMNS.TREATMENT_STATE]
            == TREATMENT_DISEASE_MODEL.WAITING_FOR_TREATMENT_STATE
        ]
        pop.loc[
            waiting_for_treatment_idx, COLUMNS.TREATMENT_DURATION
        ] = self.get_treatment_duration(waiting_for_treatment_idx)
        self.population_view.update(pop)

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
        treatment_effect = DiseaseState(
            TREATMENT_DISEASE_MODEL.TREATMENT_EFFECT,
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_TREATMENT_EFFECT_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        waning_effect = DiseaseState(
            TREATMENT_DISEASE_MODEL.WANING_EFFECT,
            allow_self_transition=True,
            prevalence=0.0,
            dwell_time=get_timedelta_from_step_size(
                self.step_size, DWELL_TIME_WANING_EFFECT_TIMESTEPS
            ),
            disability_weight=0.0,
            excess_mortality_rate=0.0,
        )
        no_effect_after_treatment = DiseaseState(
            TREATMENT_DISEASE_MODEL.NO_EFFECT_AFTER_TREATMENT,
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
            treatment_effect, proportion=TREATMENT_COMPLETION_PROBABILITY
        )
        treatment_effect.add_transition(output_state=waning_effect)
        waning_effect.add_transition(output_state=no_effect_after_treatment)

        return TreatmentModel(
            TREATMENT_DISEASE_MODEL.NAME,
            initial_state=susceptible,
            states=[
                susceptible,
                positive_test,
                waiting_for_treatment,
                treatment_effect,
                waning_effect,
                no_effect_after_treatment,
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

    def get_treatment_duration(self, waiting_for_treatment: pd.Index) -> pd.Series:
        """Returns the treatment duration for each simulant in months."""
        months_of_treatment = pd.Series(index=waiting_for_treatment)
        # First determine which simulants get full vs short treatment
        treatment_draws = self.randomness.get_draw(
            waiting_for_treatment, additional_key="treatment_duration_draws"
        )
        short_treatment_idx = treatment_draws.index[treatment_draws > 0.9]
        # Get treatment duration for short treatment simulants
        months_of_treatment.loc[short_treatment_idx] = self.randomness.choice(
            short_treatment_idx,
            choices=list(range(1, 9)),
            additional_key="short_treatment_duration",
        )
        months_of_treatment.loc[waiting_for_treatment.difference(short_treatment_idx)] = 9
        return months_of_treatment

    def get_treatment_effect_duration(
        self, index: pd.Index, target: pd.Series[float]
    ) -> pd.Series[float]:
        """Returns the treatment effect duration for each simulant. Scale effect duration by dwell time.

        Parameters
        ----------
        index
            Index of simulants to calculate duration for
        target
            Dwell time in days

        Returns
        -------
        pd.Series[float]
            Modified dwell time in days (as float)
        """
        treatment_length = (
            self.population_view.subview(COLUMNS.TREATMENT_DURATION).get(index).squeeze()
        )
        # Treatment length is in months, target is dwell time in days (float)
        effect_duration = (treatment_length / 9.0) * target
        # Round to nearest timestep
        effect_duration = (effect_duration / self.step_size).round() * self.step_size
        return effect_duration

    def get_waning_effect_duration(
        self, index: pd.Index, target: pd.Series[float]
    ) -> pd.Series[float]:
        """Returns the waning effect duration for each simulant. Scale effect duration by dwell time.

        Parameters
        ----------
        index
            Index of simulants to calculate duration for
        target
            Dwell time in days

        Returns
        -------
        pd.Series[float]
            Modified dwell time in days (as float)
        """
        treatment_length = (
            self.population_view.subview(COLUMNS.TREATMENT_DURATION).get(index).squeeze()
        )
        # Treatment length is in months, target is dwell time in days (float)
        effect_duration = (treatment_length / 9.0) * target
        # Round to nearest timestep
        effect_duration = (effect_duration / self.step_size).round() * self.step_size
        return effect_duration


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
        return super().columns_required + [self.event_count_column, self.event_time_column]

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
        return [f"{TREATMENT_DISEASE_MODEL.WANING_EFFECT}_event_time"]

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

            # TODO: update to combine states, check artifact
            affected_states = [
                TREATMENT_DISEASE_MODEL.TREATMENT_EFFECT,
                TREATMENT_DISEASE_MODEL.WANING_EFFECT,
            ]

            # Unaffected relative risks are 1
            relative_risk[~exposure.isin(affected_states)] = 1.0

            # Modify relative risks to be the minimum rr value for fully affected states
            relative_risk[exposure.isin(full_effect_states)] = rr_min

            # Modify relative risks to be interpolated values for waning states
            self._interpolate_rr(
                relative_risk,
                rr_min,
                exposure,
                TREATMENT_DISEASE_MODEL.WANING_EFFECT,
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
            # TODO: interpolate remaining time
            dwell_time = {
                TREATMENT_DISEASE_MODEL.WANING_EFFECT_SHORT_STATE: DWELL_TIME_WANING_EFFECT_SHORT_TIMESTEPS,
                TREATMENT_DISEASE_MODEL.WANING_EFFECT: DWELL_TIME_WANING_EFFECT_TIMESTEPS,
            }[waning_state]
            waning_end_date = waning_start_date + get_timedelta_from_step_size(
                self.step_size().days, dwell_time
            )
            # Linearly interpolate between the source rr and 1 based on where the
            # event date is between the waning start date and the waning end date
            relative_risk[waning_mask] = rr_min + (1.0 - rr_min) * (
                (event_date - waning_start_date) / (waning_end_date - waning_start_date)
            )
