from __future__ import annotations

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
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
    COLUMNS,
    TESTING_STATES,
)
from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL

FAKE_BBBM_TESTING_RATE = 0.3


class Testing(Component):
    """Marks simulants as having been tested if they meet the eligibility criteria."""

    @property
    def columns_created(self) -> list[str]:
        return [
            COLUMNS.TESTED_STATUS,
            COLUMNS.TESTING_PROPENSITY,
            COLUMNS.BBBM_TEST_DATE,
            COLUMNS.BBBM_TEST_RESULT,
        ]

    @property
    def columns_required(self) -> list[str]:
        return [
            ALZHEIMERS_DISEASE_MODEL.MODEL_NAME,
            COLUMNS.AGE,
        ]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [ALZHEIMERS_DISEASE_MODEL.MODEL_NAME, self.randomness, COLUMNS.AGE]

    def setup(self, builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)
        self.csf_testing_rate = builder.data.load(TESTING_RATES.CSF)["value"].item()
        self.pet_testing_rate = builder.data.load(TESTING_RATES.PET)["value"].item()
        # FIXME: replace with linear scale-ups
        self.bbbm_testing_rate = FAKE_BBBM_TESTING_RATE
        self.scenario = scenarios.INTERVENTION_SCENARIOS[
            builder.configuration.intervention.scenario
        ]
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize testing propensity and testing history for new simulants."""
        pop = self.population_view.subview(
            [
                ALZHEIMERS_DISEASE_MODEL.MODEL_NAME,
                COLUMNS.AGE,
            ]
        ).get(pop_data.index)

        # Initialize columns
        pop[COLUMNS.TESTED_STATUS] = TESTING_STATES.NOT_TESTED
        pop[COLUMNS.TESTING_PROPENSITY] = self.randomness.get_draw(
            pop_data.index, additional_key=COLUMNS.TESTING_PROPENSITY
        )
        pop[COLUMNS.BBBM_TEST_DATE] = pd.NaT
        pop[COLUMNS.BBBM_TEST_RESULT] = pd.NA

        self._update_baseline_testing(pop)
        # FIXME: is this the correct event time? does it matter on initialization?
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

    def _update_baseline_testing(self, pop: pd.DataFrame) -> pd.DataFrame:

        # Define eligibility
        eligible_state = pop[ALZHEIMERS_DISEASE_MODEL.MODEL_NAME].isin(
            [
                ALZHEIMERS_DISEASE_MODEL.MCI_STATE,
                ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_DISEASE_STATE,
            ]
        )
        eligible_csf_propensity = pop[COLUMNS.TESTING_PROPENSITY] < self.csf_testing_rate
        eligible_pet_propensity = (
            pop[COLUMNS.TESTING_PROPENSITY] >= self.csf_testing_rate
        ) & (pop[COLUMNS.TESTING_PROPENSITY] < self.csf_testing_rate + self.pet_testing_rate)
        eligible_untested = ~pop[COLUMNS.TESTED_STATUS].isin(
            [TESTING_STATES.CSF, TESTING_STATES.PET]
        )
        eligible_bbbm_results = pop[COLUMNS.BBBM_TEST_RESULT] != BBBM_TEST_RESULTS.POSITIVE

        # Update tested status with those who had CSF tests
        pop.loc[
            eligible_state
            & eligible_csf_propensity
            & eligible_untested
            & eligible_bbbm_results,
            COLUMNS.TESTED_STATUS,
        ] = TESTING_STATES.CSF

        # Update testing status with those who had PET tests
        pop.loc[
            eligible_state
            & eligible_pet_propensity
            & eligible_untested
            & eligible_bbbm_results,
            COLUMNS.TESTED_STATUS,
        ] = TESTING_STATES.PET

        return pop

    def _update_bbbm_testing(
        self, pop: pd.DataFrame, event_time: pd.Timestamp
    ) -> pd.DataFrame:

        if not self.scenario.bbbm_testing:
            return pop

        # Define eligibility
        eligible_state = (
            pop[ALZHEIMERS_DISEASE_MODEL.MODEL_NAME] == ALZHEIMERS_DISEASE_MODEL.BBBM_STATE
        )
        eligible_age = (pop[COLUMNS.AGE] >= BBBM_AGE_MIN) & (pop[COLUMNS.AGE] < BBBM_AGE_MAX)
        # FIXME: clarify < or <=
        eligible_history = (pop[COLUMNS.BBBM_TEST_DATE].isna()) | (
            pop[COLUMNS.BBBM_TEST_DATE] < event_time - BBBM_OLD_TIME
        )
        eligible_results = pop[COLUMNS.BBBM_TEST_RESULT] != "positive"
        eligible_propensity = pop[COLUMNS.TESTING_PROPENSITY] < self.bbbm_testing_rate

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
        pop.loc[tested_mask, COLUMNS.TESTED_STATUS] = TESTING_STATES.BBBM
        pop.loc[tested_mask, COLUMNS.BBBM_TEST_RESULT] = test_results
        pop.loc[tested_mask, COLUMNS.BBBM_TEST_DATE] = event_time

        return pop
