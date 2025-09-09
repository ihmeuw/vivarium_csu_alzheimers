from typing import NamedTuple

from vivarium_csu_alzheimers.constants import data_keys

###########################
# Disease Model variables #
###########################


class __AlzheimersDiseaseModel(NamedTuple):
    ALZHEIMERS_MODEL_NAME: str = data_keys.ALZHEIMERS.log_name
    BBBM_STATE: str = "alzheimers_blood_based_biomarker_state"
    MCI_STATE: str = "alzheimers_mild_cognitive_impairment_state"
    ALZHEIMERS_DISEASE_STATE: str = "alzheimers_disease_state"

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
