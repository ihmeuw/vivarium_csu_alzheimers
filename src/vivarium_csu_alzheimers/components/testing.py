import pandas as pd
from vivarium import Component
from vivarium_public_health.disease import (
    BaseDiseaseState,
    DiseaseModel,
    DiseaseState,
    ProportionTransition,
    RateTransition,
)

class TestingForAlzheimers(Component):
    """A class to hold the testing model for Alzheimer's disease. This class includes
    the different states of the testing process and how simulants can transition between them.
    """

    def __init__(self, cause: str, underlying_ad_model_name: str):
        """
        Initializes the testing model.

        Parameters
        ----------
        cause
            The name of the testing model itself (e.g., 'alzheimers_testing').
        underlying_ad_model_name
            The name of the main Alzheimer's disease model. This is used to
            find the simulants' AD status.
        """
        super().__init__()
        self.underlying_ad_model_name = underlying_ad_model_name

        # Define the states
        eligible = DiseaseState(f"eligible_for_{self.cause.name}")
        testing = DiseaseState(f"testing_for_{self.cause.name}", is_transient=True)
        positive = DiseaseState(f"tested_positive_for_{self.cause.name}")
        negative = DiseaseState(f"tested_negative_for_{self.cause.name}")

        # Define the transitions between states
        # 1. Eligible simulants get tested at a given rate.
        eligible.add_transition(
            RateTransition(
                input_state=eligible,
                output_state=testing,
                get_data_functions={
                    "transition_rate": self.get_testing_rate,
                },
            )
        )
        # 2. From the transient 'testing' state, determine the result.
        testing.add_transition(
            ProportionTransition(
                input_state=testing,
                output_state=positive,
                probability_func=self._probability_positive,
            )
        )
        # 3. The remaining simulants tested receive a negative result.
        testing.add_transition(negative)

        self.states = [eligible, testing, positive, negative]
        self.initial_state = eligible

    def setup(self, builder):
        """Standard vivarium setup method."""
        super().setup(builder)
        self.sensitivity = builder.configuration[self.cause.name].sensitivity
        self.specificity = builder.configuration[self.cause.name].specificity

        self.ad_state = builder.value.get_value(f"{self.underlying_ad_model_name}")

    def get_testing_rate(self, builder, cause: str):
        """
        Gets the rate at which simulants are tested.

        NOTE: This is the location to modify if you want to use an artifact-based
        age-specific testing rate instead of a single value from the config.
        """
        return builder.configuration[self.cause.name].test_rate

    def _probability_positive(self, index: pd.Index) -> pd.Series:
        """
        Calculates the probability of a positive test result based on the
        underlying AD status and the test's sensitivity and specificity.
        """
        pop = self.population_view.get(index)
        ad_status = self.ad_state(index)

        # Default probability is 0
        prob_positive = pd.Series(0.0, index=index)

        # Simulants with AD test positive based on sensitivity
        has_ad_mask = ad_status == self.underlying_ad_model_name
        prob_positive[has_ad_mask] = self.sensitivity

        # Simulants without AD test positive based on (1 - specificity)
        no_ad_mask = ad_status != self.underlying_ad_model_name
        prob_positive[no_ad_mask] = 1 - self.specificity

        return prob_positive

