import pandas as pd
from vivarium_gbd_access import constants as gbd_constants
from vivarium_gbd_access import utilities as vi_utils
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data


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


