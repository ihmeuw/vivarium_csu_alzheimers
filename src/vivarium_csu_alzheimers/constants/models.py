from typing import NamedTuple

from vivarium_csu_alzheimers.constants import data_keys

###########################
# Disease Model variables #
###########################

class __ALZHEIMERS_DISEASE_MODEL(NamedTuple):
    ALZHEIMERS_DISEASE: str = data_keys.ALZHEIMERS.name
    SUSCEPTIBLE_TO_ALZHEIMERS: str = f"susceptible_to_{ALZHEIMERS_DISEASE}"
    ALZHEIMERS_FIRST_STATE: str = "alzheimers_first_state"
    ALZHEIMERS_SECOND_STATE: str = "alzheimers_second_state"
    ALZHEIMERS_THIRD_STATE: str = "alzheimers_third_state"
    ALZHEIMERS_FOURTH_STATE: str = "alzheimers_fourth_state"
    ALZHEIMERS_FIFTH_STATE: str = "alzheimers_fifth_state"
