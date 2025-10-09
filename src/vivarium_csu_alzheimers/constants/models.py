from typing import NamedTuple

from vivarium_csu_alzheimers.constants import data_keys

###########################
# Disease Model variables #
###########################


class __AlzheimersDiseaseModel(NamedTuple):
    NAME: str = data_keys.ALZHEIMERS.log_name
    BBBM_STATE: str = "alzheimers_blood_based_biomarker_state"
    MCI_STATE: str = "alzheimers_mild_cognitive_impairment_state"
    ALZHEIMERS_DISEASE_STATE: str = "alzheimers_disease_state"

    def __iter__(self):
        """Allow iteration over the named tuple field values."""
        for field in self._fields:
            yield getattr(self, field)


ALZHEIMERS_DISEASE_MODEL = __AlzheimersDiseaseModel()


class __TreatmentDiseaseModel(NamedTuple):
    NAME: str = "treatment"
    SUSCEPTIBLE_STATE: str = "susceptible"
    POSITIVE_TEST_TRANSIENT_STATE: str = "positive_test"
    START_TREATMENT_STATE: str = "start_treatment"
    FULL_EFFECT_LONG_STATE: str = "full_effect_long"
    FULL_EFFECT_SHORT_STATE: str = "full_effect_short"
    WANING_EFFECT_LONG_STATE: str = "waning_effect_long"
    WANING_EFFECT_SHORT_STATE: str = "waning_effect_short"
    NO_EFFECT_AFTER_SHORT_STATE: str = "no_effect_after_short"
    NO_EFFECT_AFTER_LONG_STATE: str = "no_effect_after_long"
    NO_EFFECT_NEVER_TREATED_STATE: str = "no_effect_never_treated"

    def __iter__(self):
        """Allow iteration over the named tuple field values."""
        for field in self._fields:
            yield getattr(self, field)


TREATMENT_DISEASE_MODEL = __TreatmentDiseaseModel()
