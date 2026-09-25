"""Replay the exact January 2026 get_draws call and compare against the January artifact.

The MCVE notebook's `old` side omits `downsample=True, n_draws=500`, so it is not
byte-identical to the call that built the artifact. This closes that gap: it replays
the January kwargs verbatim, from the source at commit 4f75828 (2025-11-12):

    data = vi_utils.get_draws(
        source=gbd_constants.SOURCES.EPI,   # == "epi"
        gbd_id_type="modelable_entity_id",
        gbd_id=24351,
        release_id=16,
        year_id=2023,
        location_id=location_id,
        measure_id=vi_globals.MEASURES[measure_name],
        downsample=True,
        n_draws=500,
    )

`vi_utils.get_draws` was a pure pass-through to `get_draws.api.get_draws` (verified at
v4.2.6 and on main -- it warns and returns), so calling the central-comp function
directly is equivalent.

Excess mortality rate is the measure compared, because `load_emr` applies no
post-processing: the artifact value IS the raw draw value. Prevalence and incidence are
multiplied by an AD-proportion frame in the artifact and so cannot be compared directly.

Run with an env that has get_draws:
    /ihme/code/central_comp/miniconda/envs/gbd_midnight/bin/python probe_exact_january_call.py
"""

import numpy as np
import pandas as pd
from get_draws.api import get_draws

# Verbatim from commit 4f75828.
ME = 24351
RELEASE = 16
YEAR = 2023
LOCATION = 85  # Israel -- most-detailed, so no location aggregation
MEASURE_EMR = 9  # vi_globals.MEASURES["Excess mortality rate"]
N_DRAWS = 500

JAN_ARTIFACT = (
    "/mnt/share/homes/sbachmei/repos/vivarium_csu_alzheimers/"
    "src/vivarium_csu_alzheimers/artifacts/israel.hdf"
)
ARTIFACT_KEY = "/cause/alzheimers/excess_mortality_rate"

AGE_MAP = {13: 40.0, 14: 45.0, 15: 50.0, 16: 55.0, 17: 60.0, 18: 65.0,
           19: 70.0, 20: 75.0, 30: 80.0, 31: 85.0, 32: 90.0, 235: 95.0}
SEX_FEMALE = 2


def main() -> None:
    print(f"Replaying the January call verbatim: ME={ME} release={RELEASE} "
          f"year={YEAR} location={LOCATION} measure={MEASURE_EMR} "
          f"downsample=True n_draws={N_DRAWS}\n")

    data = get_draws(
        source="epi",  # gbd_constants.SOURCES.EPI
        gbd_id_type="modelable_entity_id",
        gbd_id=ME,
        release_id=RELEASE,
        year_id=YEAR,
        location_id=LOCATION,
        measure_id=MEASURE_EMR,
        downsample=True,
        n_draws=N_DRAWS,
    )
    draws = [c for c in data.columns if str(c).startswith("draw_")]
    mv = sorted(data["model_version_id"].unique()) if "model_version_id" in data else "n/a"
    print(f"returned {len(data)} rows, {len(draws)} draws, model_version_id={mv}")

    today = data[data["sex_id"] == SEX_FEMALE].set_index("age_group_id")[draws].mean(axis=1)
    # Only mappable age groups: an unmapped id survives the rename as a bare integer
    # and can collide with an artifact age_start (age_group_id 10 is ages 25-29,
    # age_start 10.0 is ages 10-15).
    today = today[today.index.isin(AGE_MAP)].rename(index=AGE_MAP).sort_index()

    frame = pd.read_hdf(JAN_ARTIFACT, ARTIFACT_KEY).xs("Female", level="sex")
    january = frame.groupby(level="age_start").mean().mean(axis=1)

    ages = sorted(set(today.index) & set(january.index))
    print(f"\n{'age':>5s} {'get_draws today':>18s} {'January artifact':>18s} {'ratio':>9s}")
    for age in ages:
        t, j = today[age], january[age]
        print(f"{age:5.0f} {t:18.8g} {j:18.8g} {t / j:9.5f}")

    ratios = np.array([today[a] / january[a] for a in ages])
    print(f"\nratio: min={ratios.min():.6f} max={ratios.max():.6f}")
    if np.allclose(ratios, 1.0, atol=1e-3):
        print("\nVERDICT: identical. The exact January call, run today against the same\n"
              "         get_draws line, reproduces the artifact's EMR -- so the call and\n"
              "         the code path are confirmed equivalent, and EMR data is unchanged.")
    else:
        print("\nVERDICT: differs. The exact January call no longer reproduces the\n"
              "         artifact's EMR.")


if __name__ == "__main__":
    main()
