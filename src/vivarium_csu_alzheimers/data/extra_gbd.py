import pandas as pd
from vivarium_gbd_access import utilities as vi_utils
from vivarium_gbd_access.constants import SEX
from vivarium_gbd_access.gbd import base_data
from vivarium_gbd_access.gbd.aggregation import aggregate_to_requested_location
from vivarium_gbd_access.gbd.demographics import get_age_group_id
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import utility_data

from vivarium_csu_alzheimers.constants.metadata import DRAW_COUNT

DEMENTIA_ME_ID = 24351  # Unadjusted dementia (post-mortality), from DisMod post-mortality modeling
GBD_2023_RELEASE_ID = 16
DEMENTIA_SEX_IDS = SEX.MALE + SEX.FEMALE


@vi_utils.cache
def load_incidence_dismod(location: str) -> pd.DataFrame:
    """
    Gets total population incidence rate from Dis-Mod.

    Note from the get_draws docs https://scicomp-docs.ihme.washington.edu/get_draws/current/sources.html#epi
    which described the DisMod behavior this model relies on:
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


@aggregate_to_requested_location
def _get_dementia_dismod_draws(
    location_id: int,
    year_id: int,
    measure_id: int,
    release_id: int = GBD_2023_RELEASE_ID,
    sex_id: list[int] = DEMENTIA_SEX_IDS,
    n_draws: int = DRAW_COUNT,
) -> pd.DataFrame:
    """Pull DisMod draws for the dementia modelable entity, aggregating location if needed.

    DisMod draw files hold only the locations the model was actually run for, which are
    most detailed; a national aggregate such as the USA (location_id 102) is not among
    them and the underlying call raises rather than returning nothing.

    ``gbd.get_modelable_entity_draws`` is the equivalent library getter, but it accepts
    neither ``measure_id`` nor ``n_draws``, both of which this model needs -- the
    artifact carries exactly 500 draws. So its dimension choices are mirrored here instead.
    """
    data = base_data.get_model_estimates(
        modelable_entity_id=DEMENTIA_ME_ID,
        estimates="draws",
        release_id=release_id,
        year_id=year_id,
        location_id=location_id,
        measure_id=measure_id,
        sex_id=sex_id,
        age_group_id=get_age_group_id(release_id),
        n_draws=n_draws,
    )
    # An epi model is linked to a cause, so its metadata carries a cause_id. These draws
    # describe the modelable entity rather than the cause, and the downstream reshape
    # would otherwise push cause_id into the index. gbd.get_modelable_entity_draws drops
    # it for the same reason.
    return data.drop(columns="cause_id", errors="ignore")


def _zero_fill_missing_age_groups(
    data: pd.DataFrame, release_id: int, sex_id: list[int]
) -> pd.DataFrame:
    """Fill missing values with 0.

    The dementia DisMod model only returns data from age group 40-44 upward. 
    """
    expected_age_groups = get_age_group_id(release_id)
    draw_cols = [col for col in data.columns if str(col).startswith("draw_")]
    key_cols = ["age_group_id", "sex_id"]

    grid = pd.MultiIndex.from_product(
        [expected_age_groups, sex_id], names=key_cols
    ).to_frame(index=False)
    filled = grid.merge(data, on=key_cols, how="left")
    filled[draw_cols] = filled[draw_cols].fillna(0.0)

    # The synthetic rows carry no id metadata. Every such column is constant across a
    # single location/measure/year pull, so copy that value rather than leaving NaN --
    # the downstream reshape drops these levels but fails if they are missing.
    for col in [c for c in data.columns if c not in draw_cols + key_cols]:
        values = data[col].dropna().unique()
        if len(values) == 1:
            filled[col] = values[0]

    return filled[list(data.columns)]


def load_dementia_dismod(location, measure_name):
    location_id = utility_data.resolve_location(location)
    data = _get_dementia_dismod_draws(
        location_id=location_id,
        year_id=2023,
        measure_id=vi_globals.MEASURES[measure_name],
    )
    return _zero_fill_missing_age_groups(
        data, release_id=GBD_2023_RELEASE_ID, sex_id=DEMENTIA_SEX_IDS
    )
