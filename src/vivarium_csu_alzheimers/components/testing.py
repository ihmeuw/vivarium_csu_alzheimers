import pandas as pd
from vivarium import Component
from vivarium.framework.state_machine import Machine, State, TransientState


class TestingForAlzheimers(Component):
    """A class to hold the testing model for Alzheimer's disease. This class includes
    the different states of the testing process and how simulants can transition between them.
    """

    @property
    def sub_components(self) -> list[Component]:
        return [self.machine]

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
        self.cause = cause
        self.underlying_ad_model_name = underlying_ad_model_name
        self.machine = self._create_machine()

    def _create_machine(self):
        # Define the states
        eligible = State(f"eligible_for_{self.cause}")
        testing = TransientState(f"testing_for_{self.cause}")
        positive = State(f"tested_positive_for_{self.cause}")
        negative = State(f"tested_negative_for_{self.cause}")

        # Define the transitions between states
        # 1. Eligible simulants get tested at a given rate.
        eligible.add_transition(
            output_state=testing,
            probability_function=lambda index: pd.Series(0.1, index=index),  # 10% chance
        )
        # 2. From the transient 'testing' state, determine the result.
        testing.add_transition(
            output_state=positive,
            probability_function=self._probability_positive,
        )
        # 3. The remaining simulants tested receive a negative result.
        testing.add_transition(
            output_state=negative,
            probability_function=self._probability_negative,
        )

        return Machine(
            f"testing_for_{self.cause}",
            states=[eligible, testing, positive, negative],
            initial_state=eligible,
        )

    def setup(self, builder):
        """Standard vivarium setup method."""
        super().setup(builder)
        self.sensitivity = builder.configuration[self.cause].sensitivity
        self.specificity = builder.configuration[self.cause].specificity

        # Access the AD state column - DiseaseModel creates a state column
        self.ad_population_view = builder.population.get_view([self.underlying_ad_model_name])

    def _get_testing_probability(self, index: pd.Index) -> pd.Series:
        """
        Gets the probability at which simulants are tested.

        NOTE: This is a simplified implementation. In a real scenario, this might
        depend on age, risk factors, or other characteristics.
        """
        # For now, use a constant probability. In a real model, this would be
        # more complex and potentially based on data from the builder.
        return pd.Series(0.1, index=index)  # 10% chance of being tested per time step

    def _probability_positive(self, index: pd.Index) -> pd.Series:
        """
        Calculates the probability of a positive test result based on the
        underlying AD status and the test's sensitivity and specificity.
        """
        # Get the AD status from the population view
        pop_data = self.ad_population_view.get(index)
        ad_status = pop_data[self.underlying_ad_model_name]

        print(ad_status.value_counts())
        breakpoint()
        # Default probability is 0
        prob_positive = pd.Series(0.0, index=index)

        # Simulants with AD (any state other than susceptible) test positive based on sensitivity
        susceptible_state = f"susceptible_to_{self.underlying_ad_model_name}"
        has_ad_mask = ad_status != susceptible_state
        prob_positive[has_ad_mask] = self.sensitivity

        # Simulants without AD (susceptible state) test positive based on (1 - specificity)
        no_ad_mask = ad_status == susceptible_state
        prob_positive[no_ad_mask] = 1 - self.specificity

        return prob_positive

    def _probability_negative(self, index: pd.Index) -> pd.Series:
        """
        Calculates the probability of a negative test result.
        This is simply 1 - probability_positive.
        """
        return 1 - self._probability_positive(index)
