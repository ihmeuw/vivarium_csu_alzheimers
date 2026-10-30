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
    ):
        self.name = name
        self.bbbm_testing_ramp = bbbm_testing_ramp
        self.treatment = treatment

    @property
    def bbbm_testing(self) -> bool:
        return self.bbbm_testing_ramp is not None


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    # Temporarily assign intermediate as only scenario
    BBBM_TESTING: InterventionScenario = InterventionScenario(
        "bbbm_testing",
        bbbm_testing_ramp=BBBM_TESTING_RAMPS.INTERMEDIATE,
    )
    BBBM_TESTING_AND_TREATMENT: InterventionScenario = InterventionScenario(
        "bbbm_testing_and_treatment",
        bbbm_testing_ramp=BBBM_TESTING_RAMPS.INTERMEDIATE,
        treatment=True,
    )

    def __getitem__(self, item) -> InterventionScenario:
        return self._asdict()[item.upper()]


INTERVENTION_SCENARIOS = __InterventionScenarios()
