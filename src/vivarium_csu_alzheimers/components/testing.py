from __future__ import annotations

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource

from vivarium_csu_alzheimers.constants import scenarios
from vivarium_csu_alzheimers.constants.data_keys import TESTING_RATES
from vivarium_csu_alzheimers.constants.data_values import (
    BBBM_AGE_MAX,
    BBBM_AGE_MIN,
    BBBM_OLD_TIME,
    BBBM_POSITIVE_DIAGNOSIS_PROBABILITY,
    BBBM_TEST_RESULTS,
    BBBM_TESTING_RATES,
    COLUMNS,
    TESTING_STATES,
)
from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL


class Testing(Component):
    """Marks simulants as having been tested if they meet the eligibility criteria."""

    @property
    def time_step_priority(self) -> int:
        return 6

    @property
    def columns_created(self) -> list[str]:
        return [
            COLUMNS.TESTING_PROPENSITY,
            COLUMNS.TESTING_STATE,
            COLUMNS.BBBM_TEST_DATE,
            COLUMNS.BBBM_TEST_RESULT,
        ]

    @property
    def columns_required(self) -> list[str]:
        return [COLUMNS.DISEASE_STATE, COLUMNS.AGE]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [COLUMNS.DISEASE_STATE, self.randomness, COLUMNS.AGE]

    def setup(self, builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)
        self.csf_testing_rate = builder.data.load(TESTING_RATES.CSF)["value"].item()
        self.pet_testing_rate = builder.data.load(TESTING_RATES.PET)["value"].item()
        self.scenario = scenarios.INTERVENTION_SCENARIOS[
            builder.configuration.intervention.scenario
        ]

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize testing propensity and testing history for new simulants."""
        pop = self.population_view.subview(
            [
                COLUMNS.DISEASE_STATE,
                COLUMNS.AGE,
            ]
        ).get(pop_data.index)

        # Initialize columns
        pop[COLUMNS.TESTING_PROPENSITY] = self.randomness.get_draw(
            pop_data.index, additional_key=COLUMNS.TESTING_PROPENSITY
        )
        pop[COLUMNS.TESTING_STATE] = TESTING_STATES.NOT_TESTED
        pop[COLUMNS.BBBM_TEST_DATE] = pd.NaT
        pop[COLUMNS.BBBM_TEST_RESULT] = pd.NA

        self._update_baseline_testing(pop)
        self._update_bbbm_testing(pop, event_time=pop_data.creation_time)

        self.population_view.update(pop)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        self._update_baseline_testing(pop)
        self._update_bbbm_testing(pop, event_time=event.time)
        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def _update_baseline_testing(self, pop: pd.DataFrame) -> None:

        # Define eligibility
        eligible_state = pop[COLUMNS.DISEASE_STATE].isin(
            [
                ALZHEIMERS_DISEASE_MODEL.MCI_STATE,
                ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_DISEASE_STATE,
            ]
        )
        eligible_csf_propensity = pop[COLUMNS.TESTING_PROPENSITY] < self.csf_testing_rate
        eligible_pet_propensity = (
            pop[COLUMNS.TESTING_PROPENSITY] >= self.csf_testing_rate
        ) & (pop[COLUMNS.TESTING_PROPENSITY] < self.csf_testing_rate + self.pet_testing_rate)
        eligible_untested = ~pop[COLUMNS.TESTING_STATE].isin(
            [TESTING_STATES.CSF, TESTING_STATES.PET]
        )
        eligible_bbbm_results = pop[COLUMNS.BBBM_TEST_RESULT] != BBBM_TEST_RESULTS.POSITIVE

        eligible_baseline_testing = eligible_state & eligible_untested & eligible_bbbm_results

        # Update tested status with those who had CSF tests
        pop.loc[
            eligible_baseline_testing & eligible_csf_propensity,
            COLUMNS.TESTING_STATE,
        ] = TESTING_STATES.CSF

        # Update testing status with those who had PET tests
        pop.loc[
            eligible_baseline_testing & eligible_pet_propensity,
            COLUMNS.TESTING_STATE,
        ] = TESTING_STATES.PET

    def _update_bbbm_testing(self, pop: pd.DataFrame, event_time: pd.Timestamp) -> None:

        if not self.scenario.bbbm_testing:
            return

        testing_rate = self._get_bbbm_testing_rate(event_time)
        if testing_rate == 0.0:
            return

        # Define eligibility
        eligible_state = pop[COLUMNS.DISEASE_STATE] == ALZHEIMERS_DISEASE_MODEL.BBBM_STATE
        eligible_age = (pop[COLUMNS.AGE] >= BBBM_AGE_MIN) & (pop[COLUMNS.AGE] < BBBM_AGE_MAX)
        eligible_history = (pop[COLUMNS.BBBM_TEST_DATE].isna()) | (
            pop[COLUMNS.BBBM_TEST_DATE] <= event_time - BBBM_OLD_TIME
        )
        eligible_results = pop[COLUMNS.BBBM_TEST_RESULT] != "positive"
        eligible_propensity = pop[COLUMNS.TESTING_PROPENSITY] < testing_rate

        # Calculate test results
        tested_mask = (
            eligible_state
            & eligible_age
            & eligible_history
            & eligible_results
            & eligible_propensity
        )

        test_results = self.randomness.choice(
            index=pop[tested_mask].index,
            choices=["positive", "negative"],
            p=[BBBM_POSITIVE_DIAGNOSIS_PROBABILITY, 1 - BBBM_POSITIVE_DIAGNOSIS_PROBABILITY],
            additional_key="bbbm_test_result",
        )

        # Update BBBM-specific columns for those who had BBBM tests
        pop.loc[tested_mask, COLUMNS.TESTING_STATE] = TESTING_STATES.BBBM
        pop.loc[tested_mask, COLUMNS.BBBM_TEST_RESULT] = test_results
        pop.loc[tested_mask, COLUMNS.BBBM_TEST_DATE] = event_time

    def _get_bbbm_testing_rate(self, event_time: pd.Timestamp) -> float:
        """Gets the BBBM testing rate for a given timestamp using piecewise linear interpolation."""

        if event_time < BBBM_TESTING_RATES[0][0]:
            # Before the first defined time point, return 0
            return 0.0

        if event_time > BBBM_TESTING_RATES[-1][0]:
            # Everything after the defined time point is a constant rate
            return BBBM_TESTING_RATES[-1][1]

        # Linearly interpolate everything else
        timestamps = [ts.value for ts, _ in BBBM_TESTING_RATES]
        rates = [rate for _, rate in BBBM_TESTING_RATES]

        return np.interp(event_time.value, timestamps, rates)
