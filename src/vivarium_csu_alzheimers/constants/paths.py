from pathlib import Path

import vivarium_csu_alzheimers
from vivarium_csu_alzheimers.constants import metadata

BASE_DIR = Path(vivarium_csu_alzheimers.__file__).resolve().parent

ARTIFACT_ROOT = Path(
    f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}/artifacts/"
)

FORECAST_NC_DATA_FILEPATHS_DICT = {
    "population": "/mnt/share/forecasting/data/9/future/population/20240320_daly_capstone_resubmission_squeeze_soft_round_shifted_hiv_shocks_covid_all_who_reagg/population_agg.nc",
    #     'births': '/mnt/share/forecasting/data/9/future/live_births/20231204_ref/live_births.nc',
    #     'deaths': '/snfs1/Project/forecasting/results/7/future/death/20240320_daly_capstone_resubmission_squeeze_soft_round_shifted_hiv_shocks_covid_all_who_reagg_num/_all.nc',
    "mortality": "/snfs1/Project/forecasting/results/7/future/death/20240320_daly_capstone_resubmission_squeeze_soft_round_shifted_hiv_shocks_covid_all_who_reagg/_all.nc",
    #     'migration': '/mnt/share/forecasting/data/6/future/migration/20230605_loc_intercept_shocks_pg_21LOCS_ATTENUATED/migration.nc',
}
