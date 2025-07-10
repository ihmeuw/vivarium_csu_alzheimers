from pathlib import Path

import vivarium_csu_alzheimers
from vivarium_csu_alzheimers.constants import metadata

BASE_DIR = Path(vivarium_csu_alzheimers.__file__).resolve().parent

ARTIFACT_ROOT = Path(
    f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}/artifacts/"
)
