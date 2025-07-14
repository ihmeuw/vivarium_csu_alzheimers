import pandas as pd
from layered_config_tree import LayeredConfigTree
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
            requires_columns=[
                TESTING_ALZHEIMERS_DISEASE_MODEL.TESTING_FOR_ALZHEIMERS_MODEL_NAME
            ],
        )

    def map_testing_states(self, pop: pd.DataFrame) -> str:
        _state_mapper = {
            TESTING_ALZHEIMERS_DISEASE_MODEL.SUSCEPTIBLE_TO_TESTING: "susceptible",
            TESTING_ALZHEIMERS_DISEASE_MODEL.POSIITIVE_STATE: "positive",
            TESTING_ALZHEIMERS_DISEASE_MODEL.NEGATIVE_STATE: "negative",
        }
        test_states = pop.squeeze(axis=1)
        return test_states.map(_state_mapper)
