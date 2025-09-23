import pandas as pd
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import utilities as vi_utils
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

import pdb


@vi_utils.cache
def load_raw_incidence(entity, location: str) -> pd.DataFrame:
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        "cause_id",
        entity.gbd_id,
        source=gbd_constants.SOURCES.COMO,
        location_id=location_id,
        year_id=2021,
        release_id=gbd_constants.RELEASE_IDS.GBD_2021,
        measure_id=vi_globals.MEASURES["Incidence rate"],
        metric_id=vi_globals.METRICS["Rate"],
    )
    return data


@vi_utils.cache
def load_incidence_dismod(location: str) -> pd.DataFrame:
    """
    Gets total population incidence rate from Dis-Mod using get_draws

    Note from get_draws docs https://scicomp-docs.ihme.washington.edu/get_draws/current/sources.html#epi
    "For a Dismod-MR model type which has both prevalence and incidence columns, the incidence
    column will be interpreted as hazard and converted to incidence via the equation
    incidence = hazard * (1 - prevalence)."

    Abie interpreted hazard as = incident cases count / (total population years * (1 - prevalence)),
    so incidence would = incident cases count / total population years after cancelling.
    """
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        source=gbd_constants.SOURCES.EPI,
        gbd_id_type="modelable_entity_id",
        gbd_id=24351,  # Unadjusted dementia (post-mortality) -  from DisMod, post-mortality modeling
        release_id=16,  # GBD 2023
        year_id=2023,
        location_id=location_id,
        measure_id=vi_globals.MEASURES["Incidence rate"],
    )
    return data


@vi_utils.cache
def load_prevalence_dismod(location: str) -> pd.DataFrame:
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        source=gbd_constants.SOURCES.EPI,
        gbd_id_type="modelable_entity_id",
        gbd_id=24351,  # Unadjusted dementia (post-mortality) - Dementia prevalence from DisMod, post-mortality modeling
        release_id=16,  # GBD 2023
        year_id=2023,
        location_id=location_id,
        measure_id=vi_globals.MEASURES["Prevalence"],
    )
    return data
