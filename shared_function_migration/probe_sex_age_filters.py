"""Does passing sex_id/age_group_id to get_model_estimates change the values?

The pre-migration get_draws call passed neither; the migrated call passes both.
This isolates that one difference: pull Israel DisMod prevalence twice -- once with
the filters our code passes, once with them omitted -- and compare the rows the two
have in common.

If the shared rows are identical, the filters only select and the ~0.5x artifact
discrepancy is not caused by them. If they differ, the filters are implicated.

Run with the artifact env:
    /ihme/homes/sbachmei/miniconda3/envs/vivarium_csu_alzheimers_artifact/bin/python \
        /mnt/share/homes/sbachmei/scratch/alz/probe_sex_age_filters.py
"""

import numpy as np
import pandas as pd
from vivarium_gbd_access.constants import SEX
from vivarium_gbd_access.gbd import base_data
from vivarium_gbd_access.gbd.demographics import get_age_group_id

ME = 24351  # Unadjusted dementia (post-mortality), DisMod
RELEASE = 16  # GBD 2023
YEAR = 2023
MEASURE = 5  # Prevalence
LOCATION = 85  # Israel -- most-detailed, so no aggregation decorator involvement
N_DRAWS = 500

COMMON = dict(
    modelable_entity_id=ME,
    estimates="draws",
    release_id=RELEASE,
    year_id=YEAR,
    location_id=LOCATION,
    measure_id=MEASURE,
    n_draws=N_DRAWS,
)
KEY_COLS = ["sex_id", "age_group_id"]


def describe(label: str, data: pd.DataFrame) -> list[str]:
    draws = [c for c in data.columns if str(c).startswith("draw_")]
    print(f"\n{label}")
    print(f"   rows       : {len(data)}")
    print(f"   draw cols  : {len(draws)}")
    print(f"   sex_id     : {sorted(data['sex_id'].unique())}")
    print(f"   age groups : {len(data['age_group_id'].unique())} "
          f"-> {sorted(data['age_group_id'].unique())}")
    for col in ("model_version_id", "metric_id"):
        if col in data:
            print(f"   {col:11s}: {sorted(data[col].unique())}")
    return draws


print("=" * 72)
print("A: WITH the filters our migrated code passes")
with_filters = base_data.get_model_estimates(
    sex_id=SEX.MALE + SEX.FEMALE,
    age_group_id=get_age_group_id(RELEASE),
    **COMMON,
).drop(columns="cause_id", errors="ignore")
draws = describe("A (sex_id=[1,2], age_group_id=standard)", with_filters)

print("=" * 72)
print("B: WITHOUT them, matching the pre-migration get_draws call")
without_filters = base_data.get_model_estimates(**COMMON).drop(
    columns="cause_id", errors="ignore"
)
describe("B (no sex_id, no age_group_id)", without_filters)

print("\n" + "=" * 72)
a = with_filters.set_index(KEY_COLS).sort_index()
b = without_filters.set_index(KEY_COLS).sort_index()
only_a = sorted(set(a.index) - set(b.index))
only_b = sorted(set(b.index) - set(a.index))
shared = sorted(set(a.index) & set(b.index))

print(f"rows only in A (filtered)   : {len(only_a)}  {only_a[:6]}")
print(f"rows only in B (unfiltered) : {len(only_b)}  {only_b[:6]}")
print(f"shared rows                 : {len(shared)}")

av = np.asarray(a.loc[shared, draws], dtype=float)
bv = np.asarray(b.loc[shared, draws], dtype=float)
identical = bool(np.array_equal(av, bv, equal_nan=True))
max_abs = float(np.nanmax(np.abs(av - bv))) if av.size else 0.0

print("\n" + "=" * 72)
print("VERDICT")
print(f"   shared rows identical : {identical}")
print(f"   max abs difference    : {max_abs:.6g}")
if identical:
    print("\n   -> The filters only SELECT rows; they do not change values.")
    print("      The ~0.5x artifact discrepancy is NOT caused by sex_id/age_group_id,")
    print("      and the Slack reply stands as written.")
else:
    print("\n   -> The filters DO change values. They are implicated in the")
    print("      discrepancy and the Slack reply needs correcting.")
if only_b:
    print(f"\n   NOTE: the unfiltered pull carries {len(only_b)} extra rows that the")
    print("   filtered one drops. Those would have reached reshape_to_vivarium_format")
    print("   in the baseline build. Inspect them if the verdict above is 'identical'")
    print("   but the artifact still differs -- extra rows can shift downstream joins.")
