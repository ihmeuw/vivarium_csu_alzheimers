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
    WAITING_FOR_TREATMENT_STATE: str = "waiting_for_treatment"
    TREATMENT_EFFECT: str = "treatment_effect"
    WANING_EFFECT: str = "waning_effect"
    NO_EFFECT_AFTER_TREATMENT: str = "no_effect_after_treatment"
    NO_EFFECT_NEVER_TREATED_STATE: str = "no_effect_never_treated"

    def __iter__(self):
        """Allow iteration over the named tuple field values."""
        for field in self._fields:
            yield getattr(self, field)


TREATMENT_DISEASE_MODEL = __TreatmentDiseaseModel()
