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
