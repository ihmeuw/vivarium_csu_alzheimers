from typing import NamedTuple

import pandas as pd

####################
# Project metadata #
####################

PROJECT_NAME = "vivarium_csu_alzheimers"
CLUSTER_PROJECT = "proj_simscience_prod"

CLUSTER_QUEUE = "all.q"
MAKE_ARTIFACT_MEM = 10  # GB
MAKE_ARTIFACT_CPU = 1
MAKE_ARTIFACT_RUNTIME = "3:00:00"
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    "Sweden",
    "United States of America",
    "United Kingdom",
    "China",
    "Brazil",
    "Japan",
    "Germany",
    "Israel",
    "Taiwan",
    "Spain",
]

ARTIFACT_INDEX_COLUMNS = [
    "sex",
    "age_start",
    "age_end",
    "year_start",
    "year_end",
]

DRAW_COUNT = 1000
ARTIFACT_COLUMNS = pd.Index([f"draw_{i}" for i in range(DRAW_COUNT)])


FORECAST_NC_FNAME_DICT = {
    "population": "/mnt/share/forecasting/data/9/future/population/20240320_daly_capstone_resubmission_squeeze_soft_round_shifted_hiv_shocks_covid_all_who_reagg/population_agg.nc",
    #     'births': '/mnt/share/forecasting/data/9/future/live_births/20231204_ref/live_births.nc',
    #     'deaths': '/snfs1/Project/forecasting/results/7/future/death/20240320_daly_capstone_resubmission_squeeze_soft_round_shifted_hiv_shocks_covid_all_who_reagg_num/_all.nc',
    "mortality": "/snfs1/Project/forecasting/results/7/future/death/20240320_daly_capstone_resubmission_squeeze_soft_round_shifted_hiv_shocks_covid_all_who_reagg/_all.nc",
    #     'migration': '/mnt/share/forecasting/data/6/future/migration/20230605_loc_intercept_shocks_pg_21LOCS_ATTENUATED/migration.nc',
}


class __Scenarios(NamedTuple):
    baseline: str = "baseline"
    # TODO - add scenarios here


SCENARIOS = __Scenarios()
