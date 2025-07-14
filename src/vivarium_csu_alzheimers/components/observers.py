from layered_config_tree import LayeredConfigTree
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.results import Observer
from vivarium_public_health.results import ResultsStratifier as ResultsStratifier_

from vivarium_csu_alzheimers.constants.models import TESTING_ALZHEIMERS_DISEASE_MODEL


class ResultsStratifier(ResultsStratifier_):
    def register_stratifications(self, builder: Builder) -> None:
        super().register_stratifications(builder)
        builder.results.register_stratification(
            "testing",
            categories=["positive", "negative", "susceptible"],
            excluded_categories=["susceptible"],
            mapper=self.map_testing_states,
            is_vectorized=True,
            requires_columns=[TESTING_ALZHEIMERS_DISEASE_MODEL.TESTING_FOR_ALZHEIMERS_MODEL_NAME],
        )
    
    def map_testing_states(self, pop: pd.DataFrame) -> str:
        _state_mapper = {
            TESTING_ALZHEIMERS_DISEASE_MODEL.SUSCEPTIBLE_TO_TESTING: "susceptible",
            TESTING_ALZHEIMERS_DISEASE_MODEL.POSIITIVE_STATE: "positive",
            TESTING_ALZHEIMERS_DISEASE_MODEL.NEGATIVE_STATE: "negative",
        }
        test_states = pop.squeeze(axis=1)
        return test_states.map(_state_mapper)


class TestingObserver(Observer):
    """An observer to track the testing process for Alzheimer's disease."""

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="testing",
            when="collect_metrics",
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            to_observe=self.to_observe,
        )

    def to_observe(self, event: Event) -> bool:
        """Determine if the event should be observed."""
        return True