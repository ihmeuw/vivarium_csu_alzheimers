"""Fit one sex of one artifact with the branch sampler and save the chains, gate or no gate.

For looking into a fit that the convergence gate refused: the pickle holds every chain,
the leapfrog step counts and divergences, the diagnostics table and the observations.
Read it with notebooks/inspect_chains.py.

usage: python notebooks/fit_one_sex.py <artifact.hdf> <Male|Female> <out.pkl> [--keep-zero-obs] [--no-smoothness] [--se-floor X]
                             [--seed N] [--warmup N]

--keep-zero-obs  keep the all-zero GBD cells in the likelihood (the pre-2026-10-26 behaviour)
--no-smoothness  disable the age-smoothness priors
--se-floor X     use max(standard_error, X) for every observation
--seed N         MCMC seed (default: the module's MCMC_SEED)
--warmup N       warmup iterations per chain (default: the module's NUM_WARMUP)
--samples N      samples per chain (default: the module's NUM_SAMPLES_PER_CHAIN)
--min-age A      youngest knot age (default 30)
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd
from vivarium.artifact import Artifact

from vivarium_csu_alzheimers.data import consistent_rates as cr

src, sex, out = sys.argv[1:4]
opts = sys.argv[4:]
drop_zero = "--drop-zero-obs" in opts
se_floor = float(opts[opts.index("--se-floor") + 1]) if "--se-floor" in opts else None
if "--seed" in opts:
    cr.MCMC_SEED = int(opts[opts.index("--seed") + 1])
if "--warmup" in opts:
    cr.NUM_WARMUP = int(opts[opts.index("--warmup") + 1])
if "--samples" in opts:
    cr.NUM_SAMPLES_PER_CHAIN = int(opts[opts.index("--samples") + 1])
art = Artifact(src)
min_age = int(opts[opts.index("--min-age") + 1]) if "--min-age" in opts else 30
ages = np.arange(min_age, 101, 5)
load_key = {
    "p_dementia": "cause.alzheimers.prevalence",
    "i_dementia": "cause.alzheimers.population_incidence_rate",
    "f": "cause.alzheimers.excess_mortality_rate",
    "m_all": "cause.all_causes.cause_specific_mortality_rate",
}
df_data = pd.concat(
    [
        cr.transform_to_data(
            "p_dementia", art.load(load_key["p_dementia"]), sex, ages, [2023]
        ),
        cr.transform_to_data(
            "i_dementia", art.load(load_key["i_dementia"]), sex, ages, [2023]
        ),
        cr.transform_to_data("f", art.load(load_key["f"]), sex, ages, [2023]),
        cr.transform_to_data("m_all", art.load(load_key["m_all"]), sex, ages, [2025]),
    ]
)
young_ages = []
if "--keep-zero-obs" not in opts:
    df_data, young_ages = cr.split_unmodelled_ages(df_data)
    ages = np.array([a for a in ages if a not in young_ages])
if "--no-smoothness" in opts:
    cr.AGE_SMOOTHNESS = {}
if se_floor is not None:
    df_data["standard_error"] = df_data["standard_error"].clip(lower=se_floor)
print(
    f"{len(df_data)} observations; knots {ages.min()}-{ages.max()} smoothness={cr.AGE_SMOOTHNESS} "
    f"se_floor={se_floor} "
    f"seed={cr.MCMC_SEED} warmup={cr.NUM_WARMUP}",
    flush=True,
)
model = cr.BBBM_AD_Model(ages, [2025], sex)
start = time.time()
try:
    model.fit(ages, df_data)
    status = "passed"
except RuntimeError as e:
    status = str(e)
result = dict(
    status=status,
    minutes=(time.time() - start) / 60,
    ages=ages,
    sex=sex,
    by_chain=model.samples_by_chain,
    extra=model.extra_fields,
    diagnostics=model.diagnostics,
    data=df_data,
    settings=dict(
        seed=cr.MCMC_SEED,
        chains=cr.NUM_CHAINS,
        warmup=cr.NUM_WARMUP,
        samples=cr.NUM_SAMPLES_PER_CHAIN,
        sigma=cr.ODE_SIGMA,
        target_accept=cr.TARGET_ACCEPT_PROB,
        young_ages=young_ages,
        se_floor=se_floor,
        smoothness=dict(cr.AGE_SMOOTHNESS),
    ),
)
with open(out, "wb") as f:
    pickle.dump(result, f)
print(status)
print(f"{result['minutes']:.1f} min")
