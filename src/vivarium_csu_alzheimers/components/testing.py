from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource

from vivarium_csu_alzheimers.constants.data_keys import TESTING_RATES
from vivarium_csu_alzheimers.constants.data_values import COLUMNS, TESTING_STATES
from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL


class Testing(Component):
    """Marks simulants as having been tested if they meet the eligibility criteria."""

    @property
    def columns_created(self) -> list[str]:
        return [COLUMNS.TESTING_PROPENSITY, COLUMNS.TESTED_STATUS]

    @property
    def columns_required(self) -> list[str]:
        return [ALZHEIMERS_DISEASE_MODEL.MODEL_NAME]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [ALZHEIMERS_DISEASE_MODEL.MODEL_NAME, self.randomness]

    def setup(self, builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)
        self.csf_testing_rate = builder.data.load(TESTING_RATES.CSF)["value"].item()
        self.pet_testing_rate = builder.data.load(TESTING_RATES.PET)["value"].item()

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize testing propensity and testing history for new simulants."""
        pop = self.population_view.subview(ALZHEIMERS_DISEASE_MODEL.MODEL_NAME).get(
            pop_data.index
        )
        pop[COLUMNS.TESTING_PROPENSITY] = self.randomness.get_draw(
            pop_data.index, additional_key=COLUMNS.TESTING_PROPENSITY
        )
        pop[COLUMNS.TESTED_STATUS] = TESTING_STATES.NOT_TESTED
        self._update_baseline_testing(pop)

    def on_time_step(self, event: Event) -> None:
        self._update_baseline_testing(self.population_view.get(event.index))

    ##################
    # Helper methods #
    ##################

    def _update_baseline_testing(self, df: pd.DataFrame) -> None:

        # Define eligibility
        eligible_state_idx = df[
            df[ALZHEIMERS_DISEASE_MODEL.MODEL_NAME].isin(
                [
                    ALZHEIMERS_DISEASE_MODEL.MCI_STATE,
                    ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_DISEASE_STATE,
                ]
            )
        ].index
        eligible_csf_propensity_idx = df[
            df[COLUMNS.TESTING_PROPENSITY] < self.csf_testing_rate
        ].index
        eligible_pet_propensity_idx = df[
            (df[COLUMNS.TESTING_PROPENSITY] >= self.csf_testing_rate)
            & (df[COLUMNS.TESTING_PROPENSITY] < self.csf_testing_rate + self.pet_testing_rate)
        ].index
        # TODO: Need to implement bbbm test result logic here (never received a positive bbbm test)
        eligible_untested_idx = df[
            df[COLUMNS.TESTED_STATUS] == TESTING_STATES.NOT_TESTED
        ].index
        # Update tested status with those who had CSF tests
        df.loc[
            eligible_state_idx.intersection(eligible_csf_propensity_idx).intersection(
                eligible_untested_idx
            ),
            COLUMNS.TESTED_STATUS,
        ] = TESTING_STATES.CSF
        # Update testing status with those who had PET tests
        df.loc[
            eligible_state_idx.intersection(eligible_pet_propensity_idx).intersection(
                eligible_untested_idx
            ),
            COLUMNS.TESTED_STATUS,
        ] = TESTING_STATES.PET

        self.population_view.update(df)
