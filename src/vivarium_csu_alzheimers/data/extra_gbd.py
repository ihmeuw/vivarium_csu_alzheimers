import pandas as pd
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import utilities as vi_utils
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data


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
    return load_dementia_dismod(location, "Incidence rate")


@vi_utils.cache
def load_prevalence_dismod(location: str) -> pd.DataFrame:
    return load_dementia_dismod(location, "Prevalence")


@vi_utils.cache
def load_emr_dismod(location: str) -> pd.DataFrame:
    return load_dementia_dismod(location, "Excess mortality rate")


def load_dementia_dismod(location, measure_name):
    location_id = utility_data.get_location_id(location)
    data = vi_utils.get_draws(
        source=gbd_constants.SOURCES.EPI,
        gbd_id_type="modelable_entity_id",
        gbd_id=24351,  # Unadjusted dementia (post-mortality) -  from DisMod, post-mortality modeling
        release_id=16,  # GBD 2023
        year_id=2023,
        location_id=location_id,
        measure_id=vi_globals.MEASURES[measure_name],
        downsample=True,
        n_draws=500,
    )
    return data


