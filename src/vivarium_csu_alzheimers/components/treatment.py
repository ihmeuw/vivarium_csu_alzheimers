from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.engine.framework.event import Event
from vivarium.engine.framework.population import SimulantData
from vivarium.public_health import (
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
    TREATMENT_COMPLETION_PROBABILITY,
    TREATMENT_FULL_DURATION,
    TREATMENT_PROBS_RAMP,
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

    def __init__(
        self,
        cause: str,
        initial_state_source: Callable[[pd.Index[int]], pd.Series[str]],
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        cause
            The name of the cause of disease.
        initial_state_source
            A callable that maps the index of the simulants being initialized to
            the states they should start in.
        kwargs
            Additional keyword arguments for DiseaseModel.
        """
        super().__init__(cause, **kwargs)
        self.initial_state_source = initial_state_source

    def setup(self, builder: Builder) -> None:
        """Typical DiseaseModel setup except for how the initial states are chosen.

        The code below is copy/paste from DiseaseModel.setup and Machine.setup except
        that the state initializer takes the states chosen by the Treatment component
        (see Treatment.get_initial_treatment_states) rather than sampling them from
        the states' prevalence.
        """
        self.randomness = builder.randomness.get_stream(self.name)
        builder.population.register_initializer(
            initializer=self.initialize_state,
            columns=self.state_column,
            required_resources=[COLUMNS.TREATMENT_PROPENSITY, COLUMNS.BBBM_TEST_RESULT],
        )

        self.csmr_table = self.build_lookup_table(builder, "cause_specific_mortality_rate")

        builder.value.register_attribute_modifier(
            "cause_specific_mortality_rate",
            self.adjust_cause_specific_mortality_rate,
            required_resources=["age", "sex"],
        )

    def initialize_state(self, pop_data: SimulantData) -> None:
        initial_states = self.initial_state_source(pop_data.index)
        self.population_view.initialize(initial_states.rename(self.state_column))


class Treatment(Component):
    """Alzheimer's treatment model component."""

    @property
    def sub_components(self) -> list[Component]:
        return [self.disease_model]

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
        builder.population.register_initializer(
            initializer=self.initialize_propensity,
            columns=[COLUMNS.TREATMENT_PROPENSITY],
            required_resources=[self.randomness],
        )
        builder.population.register_initializer(
            initializer=self.initialize_treatment_duration,
            columns=[COLUMNS.TREATMENT_DURATION],
            required_resources=[COLUMNS.TREATMENT_STATE, self.randomness],
        )
        # register an exposure pipeline that just turns around
        builder.value.register_attribute_producer(
            f"{COLUMNS.TREATMENT_STATE}.exposure",
            source=self.get_treatment_states,
            required_resources=[COLUMNS.TREATMENT_STATE],
        )
        builder.value.register_attribute_modifier(
            "treatment_effect.dwell_time",
            modifier=self.modify_dwell_time,
            required_resources=[COLUMNS.TREATMENT_DURATION],
        )
        builder.value.register_attribute_modifier(
            "waning_effect.dwell_time",
            modifier=self.modify_dwell_time,
            required_resources=[COLUMNS.TREATMENT_DURATION],
        )

    def get_treatment_states(self, index: pd.Index) -> pd.Series:
        return self.population_view.get(index, COLUMNS.TREATMENT_STATE)

    def initialize_propensity(self, pop_data: SimulantData) -> None:
        """Initialize the treatment propensity for new simulants."""
        propensity = self.randomness.get_draw(
            pop_data.index, additional_key=COLUMNS.TREATMENT_PROPENSITY
        )
        self.population_view.initialize(propensity.rename(COLUMNS.TREATMENT_PROPENSITY))

    def initialize_treatment_duration(self, pop_data: SimulantData) -> None:
        """Initialize the treatment duration of simulants who start out in treatment."""
        treatment_states = self.population_view.get(pop_data.index, COLUMNS.TREATMENT_STATE)
        durations = pd.Series(np.nan, index=pop_data.index, name=COLUMNS.TREATMENT_DURATION)
        waiting_for_treatment_idx = treatment_states.index[
            treatment_states == TREATMENT_DISEASE_MODEL.WAITING_FOR_TREATMENT_STATE
        ]
        if not waiting_for_treatment_idx.empty:
            durations.loc[waiting_for_treatment_idx] = self.get_treatment_duration(
                waiting_for_treatment_idx
            )
        self.population_view.initialize(durations)

    def get_initial_treatment_states(self, index: pd.Index[int]) -> pd.Series[str]:
        """Chooses the treatment states that simulants are initialized into.

        HACK: We need to manually do this here rather than relying on the TreatmentModel
        and DiseaseState classes because vivarium simulations by default do not
        include newly-initialized simulants when making decisions for a given
        time step, i.e. simulants need to be both tested and treated during
        initialization (without this they would be tested on initialization but
        not run through the treatment logic until the following time step).

        Notes
        -----
        We do this here in Treatment rather than in the TreatmentModel because we
        need to know the location to apply the appropriate treatment probabilities
        which requires access to the builder.

        We are only special-casing the waiting_for_treatment and
        no_effect_never_treated states here because it's critical that we not
        skip potential treatment for initialized simulants; all other states
        in the disease model (aside from Susceptible) are downstream of starting
        treatment and since no simulants are initialized having already started
        treatment, they can be handled in the normal way. The corresponding event
        times and counts are initialized by PositiveTestDecisionState, below.

        Parameters
        ----------
        index
            The index of the simulants being initialized.

        Returns
        -------
            The treatment state of each simulant being initialized.
        """
        states = pd.Series(
            f"{TREATMENT_DISEASE_MODEL.SUSCEPTIBLE_STATE}_to_treatment", index=index
        )
        positive_results = self.has_positive_results(index)
        positive_results_idx = positive_results[positive_results == 1].index
        start_treatment_probs = self.start_treatment_probs(positive_results_idx)
        start_treatment_idx = (start_treatment_probs[start_treatment_probs == 1]).index
        decline_treatment_idx = positive_results_idx.difference(start_treatment_idx)
        states.loc[start_treatment_idx] = TREATMENT_DISEASE_MODEL.WAITING_FOR_TREATMENT_STATE
        states.loc[
            decline_treatment_idx
        ] = TREATMENT_DISEASE_MODEL.NO_EFFECT_NEVER_TREATED_STATE
        return states

    def on_time_step_cleanup(self, event: Event) -> None:
        """Set treatment duration for simulants in waiting_for_treatment state.

        This runs after the state machine transitions (priority 8) to ensure
        all simulants who entered waiting_for_treatment during this time step
        have their treatment duration set.
        """
        waiting_for_treatment_idx = self.population_view.get_filtered_index(
            event.index,
            query=f"{COLUMNS.TREATMENT_STATE} == "
            f"'{TREATMENT_DISEASE_MODEL.WAITING_FOR_TREATMENT_STATE}'",
        )
        if not waiting_for_treatment_idx.empty:
            durations = self.get_treatment_duration(waiting_for_treatment_idx)
            self.population_view.update(
                COLUMNS.TREATMENT_DURATION,
                lambda _: durations.rename(COLUMNS.TREATMENT_DURATION),
                index=waiting_for_treatment_idx,
            )

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
            initial_state_source=self.get_initial_treatment_states,
            residual_state=susceptible,
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
        test_results = self.population_view.get(index, COLUMNS.BBBM_TEST_RESULT)
        is_positive = pd.Series(0.0, index=index)
        is_positive[test_results == BBBM_TEST_RESULTS.POSITIVE] = 1.0
        return is_positive

    def start_treatment_probs(self, index: pd.Index[int]) -> pd.Series[float]:
        """Returns 1 if the propensity is less that treatment probability, 0 otherwise."""
        propensity = self.population_view.get(index, COLUMNS.TREATMENT_PROPENSITY)
        event_date = self.clock() + get_timedelta_from_step_size(self.step_size)
        probs = pd.Series(0.0, index=index)

        if not self.scenario.treatment:
            return probs

        if event_date < TREATMENT_PROBS_RAMP[0][0]:
            # Before the first defined time point, return 0
            treatment_prob = 0.0
        elif event_date > TREATMENT_PROBS_RAMP[-1][0]:
            # Everything after the defined time point is a constant rate
            treatment_prob = TREATMENT_PROBS_RAMP[-1][1]
        else:
            # interpolate
            timestamps = [ts.value for ts, _ in TREATMENT_PROBS_RAMP]
            rates = [rate for _, rate in TREATMENT_PROBS_RAMP]
            treatment_prob = np.interp(event_date.value, timestamps, rates)

        probs[propensity < treatment_prob] = 1.0
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
        short_treatment_idx = treatment_draws.index[treatment_draws < 0.1]
        # Get treatment duration for short treatment simulants
        months_of_treatment.loc[short_treatment_idx] = self.randomness.choice(
            short_treatment_idx,
            choices=list(range(1, 9)),
            additional_key="short_treatment_duration",
        )
        months_of_treatment.loc[waiting_for_treatment.difference(short_treatment_idx)] = 9
        return months_of_treatment

    def modify_dwell_time(
        self, index: pd.Index, target: pd.Series[float]
    ) -> pd.Series[float]:
        """Returns the modified dwell time for treatment and waning effect states.

        Parameters
        ----------
        index
            Index of simulants to calculate duration for
        target
            Dwell time in days

        Returns
        -------
            Modified dwell time in days
        """
        treatment_length = self.population_view.get(index, COLUMNS.TREATMENT_DURATION)
        # Treatment length is in months, target is dwell time in days (float)
        effect_duration = (treatment_length / TREATMENT_FULL_DURATION) * target
        # Round to nearest timestep
        effect_duration = (effect_duration / self.step_size).round() * self.step_size
        return effect_duration


class PositiveTestDecisionState(DiseaseState):
    """A treatment state that simulants can be initialized directly into.

    The Treatment component decides during initialization which simulants start out
    waiting for treatment and which have declined it. This state records the
    corresponding event time and count rather than the empty values the base state
    would provide.
    """

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.step_size = builder.configuration.time.step_size

    def get_initial_event_times(self, pop_data: SimulantData) -> pd.DataFrame:
        """Records an event for simulants initialized into this state.

        Notes
        -----
        The treatment state column is already a required resource of this state's
        initializer (DiseaseState requires its model's state column), so it is
        guaranteed to have been initialized by the Treatment component by now.
        """
        pop_update = super().get_initial_event_times(pop_data)
        treatment_states = self.population_view.get(pop_data.index, self.model)
        initialized_in_state = treatment_states == self.state_id
        pop_update.loc[
            initialized_in_state, self.event_time_column
        ] = pop_data.creation_time + get_timedelta_from_step_size(self.step_size)
        pop_update.loc[initialized_in_state, self.event_count_column] = 1
        return pop_update


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
            rr = self.relative_risk_table(index)
            if len(rr.unique()) != 1:
                raise NotImplementedError("Only a single relative risk value is supported.")
            rr_min = rr.iloc[0]

            exposure = self.population_view.get(index, self.exposure_name)
            relative_risk = pd.Series(index=index, dtype=float)

            affected_states = [
                TREATMENT_DISEASE_MODEL.TREATMENT_EFFECT,
                TREATMENT_DISEASE_MODEL.WANING_EFFECT,
            ]

            # Unaffected relative risks are 1
            relative_risk[~exposure.isin(affected_states)] = 1.0

            # Modify relative risks to be the minimum rr value for fully affected states
            relative_risk[exposure == TREATMENT_DISEASE_MODEL.TREATMENT_EFFECT] = rr_min

            # Modify relative risks to be interpolated values for waning states
            self._interpolate_rr(
                relative_risk,
                rr_min,
                exposure,
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
    ) -> None:
        waning_mask = exposure == TREATMENT_DISEASE_MODEL.WANING_EFFECT
        if waning_mask.any():
            event_date = self.clock() + self.step_size()
            waning_index = waning_mask[waning_mask].index
            waning_start_date = pd.to_datetime(
                self.population_view.get(
                    waning_index,
                    f"{TREATMENT_DISEASE_MODEL.WANING_EFFECT}_event_time",
                )
            )

            # Dwell times are number of days in waning effect
            dwell_times = self.population_view.get(
                waning_index, f"{TREATMENT_DISEASE_MODEL.WANING_EFFECT}.dwell_time"
            )
            dwell_times = dwell_times / (self.step_size() / pd.Timedelta(days=1))
            waning_end_date = waning_start_date + get_timedelta_from_step_size(
                self.step_size().days, dwell_times
            )

            # Linearly interpolate between the source rr and 1 based on where the
            # event date is between the waning start date and the waning end date
            relative_risk[waning_mask] = rr_min + (1.0 - rr_min) * (
                (event_date - waning_start_date) / (waning_end_date - waning_start_date)
            )
