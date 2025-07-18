from typing import NamedTuple

from vivarium_csu_alzheimers.constants import data_keys

###########################
# Disease Model variables #
###########################


class __AlzheimersDiseaseModel(NamedTuple):
    ALZHEIMERS_MODEL_NAME: str = data_keys.ALZHEIMERS.name
    SUSCEPTIBLE_TO_ALZHEIMERS: str = f"susceptible_to_{ALZHEIMERS_MODEL_NAME}"
    ALZHEIMERS_FIRST_STATE: str = "alzheimers_first_state"
    ALZHEIMERS_SECOND_STATE: str = "alzheimers_second_state"
    ALZHEIMERS_THIRD_STATE: str = "alzheimers_third_state"
    ALZHEIMERS_FOURTH_STATE: str = "alzheimers_fourth_state"
    ALZHEIMERS_FIFTH_STATE: str = "alzheimers_fifth_state"

    def __iter__(self):
        """Allow iteration over the named tuple field values."""
        for field in self._fields:
            yield getattr(self, field)


ALZHEIMERS_DISEASE_MODEL = __AlzheimersDiseaseModel()


class __TestingAlzheimersDiseaseModel(NamedTuple):
    TESTING_FOR_ALZHEIMERS_MODEL_NAME: str = "testing_for_alzheimers"
    SUSCEPTIBLE_TO_TESTING: str = "susceptible_to_testing_for_alzheimers"
    TESTING_STATE: str = "testing_for_alzheimers"
    POSITIVE_STATE: str = "positive_test_for_alzheimers"
    NEGATIVE_STATE: str = "negative_test_for_alzheimers"

    def __iter__(self):
        """Allow iteration over the named tuple field values."""
        for field in self._fields:
            yield getattr(self, field)


TESTING_ALZHEIMERS_DISEASE_MODEL = __TestingAlzheimersDiseaseModel()
