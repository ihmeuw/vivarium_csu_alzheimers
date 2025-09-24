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
    "Taiwan (Province of China)",
    "Spain",
]

ARTIFACT_INDEX_COLUMNS = [
    "sex",
    "age_start",
    "age_end",
    "year_start",
    "year_end",
]

DRAW_COUNT = 500
ARTIFACT_COLUMNS = pd.Index([f"draw_{i}" for i in range(DRAW_COUNT)])


class __Scenarios(NamedTuple):
    baseline: str = "baseline"
    # TODO - add scenarios here


SCENARIOS = __Scenarios()
