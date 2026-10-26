"""Describe a saved one-sex fit chain by chain.

usage: python notebooks/inspect_chains.py <fit.pkl> [<reference.hdf>]

Prints, for each parameter, the per-chain posterior mean at ages 40-95 against the
pooled mean, the per-chain leapfrog steps and the per-chain log density, and, if a
reference artifact is given, the pooled mean against that artifact's consistent key.
"""
import pickle
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

res = pickle.load(open(sys.argv[1], "rb"))
ref = sys.argv[2] if len(sys.argv) > 2 else None
ages = res["ages"]
sex = res["sex"]
print(res["status"][:300])
print(f"{res['minutes']:.1f} min; settings {res['settings']}")

by_chain = res["by_chain"]
extra = res["extra"]
n_chains = next(iter(by_chain.values())).shape[0]
steps = extra["num_steps"].reshape(n_chains, -1)
div = extra["diverging"].reshape(n_chains, -1)
print("\nper-chain leapfrog steps/iter (mean, 90th pct) and divergences:")
for c in range(n_chains):
    print(
        f"  chain {c}: {steps[c].mean():6.0f} {np.percentile(steps[c], 90):6.0f}  div {div[c].sum()}"
    )

sel = (ages >= 40) & (ages <= 95)
for name in ["p", "delta_BBBM", "delta_MCI", "h_S_to_BBBM", "i", "f", "m"]:
    if name not in by_chain:
        continue
    v = by_chain[name][:, :, :, 0]  # (chains, samples, ages)
    pooled = v.reshape(-1, v.shape[-1]).mean(axis=0)
    tab = pd.DataFrame(
        {f"chain{c}/pooled": v[c].mean(axis=0)[sel] / pooled[sel] for c in range(n_chains)},
        index=pd.Index(ages[sel], name="age"),
    )
    tab["pooled mean"] = pooled[sel]
    tab["pooled cv"] = v.reshape(-1, v.shape[-1]).std(axis=0)[sel] / pooled[sel]
    # per-chain within-sd vs between-chain sd, as a crude r_hat proxy per age
    within = v.std(axis=1).mean(axis=0)[sel]
    between = v.mean(axis=1).std(axis=0)[sel]
    tab["between/within sd"] = between / within
    print(f"\n== {name}")
    print(tab.round(3).to_string())

if ref:
    from vivarium.artifact import Artifact

    art = Artifact(ref)
    keys = {
        "p": "cause.alzheimers_consistent.prevalence_any",
        "delta_BBBM": "cause.alzheimers_consistent.bbbm_conditional_prevalence",
        "delta_MCI": "cause.alzheimers_consistent.mci_conditional_prevalence",
        "h_S_to_BBBM": "cause.alzheimers_consistent.population_incidence_any",
        "i": "cause.alzheimers_consistent.population_incidence_dementia",
        "f": "cause.alzheimers_consistent.excess_mortality_rate",
    }
    print(f"\n== pooled mean / reference ({ref}) by age, ages 40-95")
    tab = {}
    for name, key in keys.items():
        df = art.load(key).xs(sex, level="sex")
        d = df[[c for c in df.columns if str(c).startswith("draw_")]].mean(axis=1)
        d.index = d.index.get_level_values("age_start")
        v = by_chain[name][:, :, :, 0]
        pooled = pd.Series(v.reshape(-1, v.shape[-1]).mean(axis=0), index=ages)
        tab[name] = (pooled / d.reindex(ages))[sel]
    print(pd.DataFrame(tab).round(3).to_string())
