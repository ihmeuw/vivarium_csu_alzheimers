from typing import NamedTuple

#############
# Scenarios #
#############


class InterventionScenario:
    def __init__(
        self,
        name: str,
        bbbm_testing: bool = False,
        treatment: bool = False,
    ):
        self.name = name
        self.bbbm_testing = bbbm_testing
        self.treatment = treatment


class __InterventionScenarios(NamedTuple):
    BASELINE: InterventionScenario = InterventionScenario("baseline")
    BBBM_TESTING: InterventionScenario = InterventionScenario(
        "bbbm_testing",
        bbbm_testing=True,
    )
    BBBM_TESTING_AND_TREATMENT: InterventionScenario = InterventionScenario(
        "treatment",
        bbbm_testing=True,
        treatment=True,
    )

    def __getitem__(self, item) -> InterventionScenario:
        for scenario in self:
            if scenario.name == item:
                return scenario
        raise KeyError(item)


INTERVENTION_SCENARIOS = __InterventionScenarios()
