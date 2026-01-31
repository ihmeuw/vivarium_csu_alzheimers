from __future__ import annotations

import numpy as np
import pandas as pd
from vivarium import Component
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium.types import Time

from vivarium_csu_alzheimers.constants import scenarios
from vivarium_csu_alzheimers.constants.data_keys import TESTING_RATES
from vivarium_csu_alzheimers.constants.data_values import (
    BBBM_AGE_MAX,
    BBBM_AGE_MIN,
    BBBM_POSITIVE_DIAGNOSIS_PROBABILITY,
    BBBM_TEST_RESULTS,
    BBBM_TESTING_RATES,
    BBBM_TESTING_START_DATE,
    COLUMNS,
    TESTING_STATES,
    TIME_STEPS_UNTIL_NEXT_BBBM_TEST,
)
from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL
from vivarium_csu_alzheimers.utilities import get_timedelta_from_step_size


class Testing(Component):
    """Marks simulants as having been tested if they meet the eligibility criteria."""

    @property
    def time_step_priority(self) -> int:
        """We want testing to occur after disease state updates."""
        return 6

    @property
    def columns_created(self) -> list[str]:
        return [
            COLUMNS.TESTING_PROPENSITY,
            COLUMNS.TESTING_STATE,
            COLUMNS.BBBM_TEST_DATE,
            COLUMNS.BBBM_TEST_RESULT,
            COLUMNS.BBBM_TEST_EVER_ELIGIBLE,
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
        self.step_size = builder.configuration.time.step_size

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
        pop[COLUMNS.BBBM_TEST_RESULT] = BBBM_TEST_RESULTS.NOT_TESTED
        pop[COLUMNS.BBBM_TEST_EVER_ELIGIBLE] = False
        pop[COLUMNS.BBBM_TEST_DATE] = pd.NaT

        # Update to reflect history
        event_time = pop_data.creation_time + get_timedelta_from_step_size(self.step_size)
        if self.scenario.bbbm_testing:
            bbbm_eligible_mask = self._get_bbbm_eligible_simulants(pop, event_time)
            testing_rate = self._get_bbbm_testing_rate(event_time)
            test_history_mask = bbbm_eligible_mask & (
                pop[COLUMNS.TESTING_PROPENSITY] < testing_rate
            )
            pop[COLUMNS.BBBM_TEST_DATE] = self._generate_bbbm_testing_data(
                pop, test_history_mask, event_time
            )

        self._update_baseline_testing(pop)

        bbbm_tested_now_mask = pop[COLUMNS.BBBM_TEST_DATE] == event_time
        self._update_bbbm_testing(
            pop,
            bbbm_tested_now_mask,
            event_time=event_time,
        )

        self.population_view.update(pop)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index)
        self._update_baseline_testing(pop)
        eligible_mask = self._get_bbbm_eligible_simulants(pop, event.time)
        self._update_bbbm_testing(pop, eligible_mask, event_time=event.time)
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
        eligible_test_propensity = pop[COLUMNS.TESTING_PROPENSITY] < (
            self.csf_testing_rate + self.pet_testing_rate
        )
        eligible_untested = ~pop[COLUMNS.TESTING_STATE].isin(
            [TESTING_STATES.CSF, TESTING_STATES.PET]
        )
        eligible_bbbm_results = pop[COLUMNS.BBBM_TEST_RESULT] != BBBM_TEST_RESULTS.POSITIVE

        eligible_baseline_testing = (
            eligible_state
            & eligible_test_propensity
            & eligible_untested
            & eligible_bbbm_results
        )

        # Update tested status
        pop.loc[eligible_baseline_testing, COLUMNS.TESTING_STATE] = self.randomness.choice(
            index=pop[eligible_baseline_testing].index,
            choices=[TESTING_STATES.CSF, TESTING_STATES.PET],
            p=[self.csf_testing_rate, self.pet_testing_rate],
            additional_key=COLUMNS.TESTING_STATE,
        )

    def _update_bbbm_testing(
        self, pop: pd.DataFrame, eligible: pd.Series, event_time: pd.Timestamp
    ) -> None:

        if not self.scenario.bbbm_testing:
            return

        # Update the ever-eligible column
        pop.loc[eligible, COLUMNS.BBBM_TEST_EVER_ELIGIBLE] = True

        # Calculate test results
        testing_rate = self._get_bbbm_testing_rate(event_time)
        if testing_rate == 0.0:
            return
        tested_mask = eligible & (pop[COLUMNS.TESTING_PROPENSITY] < testing_rate)

        test_results = self.randomness.choice(
            index=pop[tested_mask].index,
            choices=[BBBM_TEST_RESULTS.POSITIVE, BBBM_TEST_RESULTS.NEGATIVE],
            p=[BBBM_POSITIVE_DIAGNOSIS_PROBABILITY, 1 - BBBM_POSITIVE_DIAGNOSIS_PROBABILITY],
            additional_key="bbbm_test_result",
        )

        # Update BBBM-specific columns for those who had BBBM tests
        pop.loc[tested_mask, COLUMNS.TESTING_STATE] = TESTING_STATES.BBBM
        pop.loc[tested_mask, COLUMNS.BBBM_TEST_RESULT] = test_results
        # Choose next test date for negative tests
        negative_test = (test_results == BBBM_TEST_RESULTS.NEGATIVE) & tested_mask
        time_steps_until_next_test = self.randomness.choice(
            index=pop[negative_test].index,
            choices=TIME_STEPS_UNTIL_NEXT_BBBM_TEST,
            additional_key="bbbm_time_until_next_test",
        )
        pop.loc[
            negative_test, COLUMNS.BBBM_TEST_DATE
        ] = event_time + get_timedelta_from_step_size(
            self.step_size, time_steps_until_next_test
        )

    def _get_bbbm_eligible_simulants(
        self, pop: pd.DataFrame, event_time: Time
    ) -> pd.Series[bool]:
        eligible_state = pop[COLUMNS.DISEASE_STATE] == ALZHEIMERS_DISEASE_MODEL.BBBM_STATE
        eligible_age = (pop[COLUMNS.AGE] >= BBBM_AGE_MIN) & (pop[COLUMNS.AGE] < BBBM_AGE_MAX)
        eligible_history = (pop[COLUMNS.BBBM_TEST_DATE].isna()) | (
            pop[COLUMNS.BBBM_TEST_DATE] <= event_time
        )
        eligible_results = pop[COLUMNS.BBBM_TEST_RESULT] != BBBM_TEST_RESULTS.POSITIVE
        return eligible_state & eligible_age & eligible_history & eligible_results

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

    def _generate_bbbm_testing_data(
        self, simulants: pd.DataFrame, eligible_sims: pd.Series, time_of_event: Time
    ) -> pd.Series[Time]:
        """Generates BBBM test data for new simulants next BBBM test 0 to 4.5 into the future.
        This will sample the 3-5 year range for time until the next test. It will then sample
        how far along that simulant is in that range to determine the test date. For example,
        if a simulant is intering the simulation on 2030-01-01, and CRN samples that their test
        will be 4 years after their previous test, then then will take a sample between 0 and 4
        years. If that sample is 1 year, their COLUMNS.BBBM_TEST_DATE will be 2033-01-01.
        """

        test_dates = simulants[COLUMNS.BBBM_TEST_DATE].copy()
        if not self.scenario.bbbm_testing or time_of_event < BBBM_TESTING_START_DATE:
            return test_dates

        time_steps_until_next_test = self.randomness.choice(
            index=simulants[eligible_sims].index,
            choices=TIME_STEPS_UNTIL_NEXT_BBBM_TEST[:-1],
            additional_key="bbbm_time_until_next_test_history",
        )
        interval_choices = time_steps_until_next_test.apply(
            lambda x: list(np.arange(0.0, x + 1.0, 1.0))
        )
        time_into_test_interval = self.randomness.choice(
            index=simulants[eligible_sims].index,
            choices=interval_choices,
            additional_key="bbbm_time_into_test_interval_history",
        )
        steps_until_next_test = time_steps_until_next_test - time_into_test_interval
        test_dates.loc[eligible_sims] = time_of_event + get_timedelta_from_step_size(
            self.step_size, steps_until_next_test
        )

        return test_dates
