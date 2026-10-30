from typing import NamedTuple

from vivarium_csu_alzheimers.constants.data_values import BBBM_TESTING_RAMPS

#############
# Scenarios #
#############


class InterventionScenario:
    def __init__(
        self,
        name: str,
        bbbm_testing_ramp: str | None = None,
        treatment: bool = False,
        zero_impact_treatment: bool = False,
    ):
        self.name = name
        self.bbbm_testing_ramp = bbbm_testing_ramp
        self.treatment = treatment
        # Treatment is assigned but has no effect on outcomes
        self.zero_impact_treatment = zero_impact_treatment

    @property
    def bbbm_testing(self) -> bool:
        return self.bbbm_testing_ramp is not None


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    SLOW_TESTING_ZERO_IMPACT_TREATMENT: InterventionScenario = InterventionScenario(
        "slow_testing_zero_impact_treatment",
        bbbm_testing_ramp=BBBM_TESTING_RAMPS.SLOW,
        treatment=True,
        zero_impact_treatment=True,
    )
    INTERMEDIATE_TESTING_ZERO_IMPACT_TREATMENT: InterventionScenario = InterventionScenario(
        "intermediate_testing_zero_impact_treatment",
        bbbm_testing_ramp=BBBM_TESTING_RAMPS.INTERMEDIATE,
        treatment=True,
        zero_impact_treatment=True,
    )
    ACCELERATED_TESTING_ZERO_IMPACT_TREATMENT: InterventionScenario = InterventionScenario(
        "accelerated_testing_zero_impact_treatment",
        bbbm_testing_ramp=BBBM_TESTING_RAMPS.ACCELERATED,
        treatment=True,
        zero_impact_treatment=True,
    )
    SLOW_TESTING_WITH_TREATMENT: InterventionScenario = InterventionScenario(
        "slow_testing_with_treatment",
        bbbm_testing_ramp=BBBM_TESTING_RAMPS.SLOW,
        treatment=True,
    )
    INTERMEDIATE_TESTING_WITH_TREATMENT: InterventionScenario = InterventionScenario(
        "intermediate_testing_with_treatment",
        bbbm_testing_ramp=BBBM_TESTING_RAMPS.INTERMEDIATE,
        treatment=True,
    )
    ACCELERATED_TESTING_WITH_TREATMENT: InterventionScenario = InterventionScenario(
        "accelerated_testing_with_treatment",
        bbbm_testing_ramp=BBBM_TESTING_RAMPS.ACCELERATED,
        treatment=True,
    )

    def __getitem__(self, item) -> InterventionScenario:
        return self._asdict()[item.upper()]


INTERVENTION_SCENARIOS = __InterventionScenarios()
