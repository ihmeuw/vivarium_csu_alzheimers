"""Build notebooks/consistent_rates_refit250_summary.ipynb: all ten locations at a glance."""
import sys
from pathlib import Path

import nbformat as nbf

out = Path(sys.argv[1])
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "name": "consistent-rates-sampler",
    "display_name": "Python (consistent-rates-sampler)",
    "language": "python",
}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(
    """# Consistent rates, all ten locations: main's sampler vs the branch sampler

One row per location. For each location three artifacts are compared:

- **main** — built by `main` (PR #82, 250 draws) on 2026-10-23: one NUTS chain, 500 warmup,
  250 consecutive iterations written as the draws, no convergence check
  (`artifact_comparisons/main250/`, see `PROVENANCE.txt` there).
- **refit** — the same inputs refit with this branch's sampler
  (`notebooks/refit_consistent_rates.py` → `artifact_comparisons/refit250/`): 4 chains,
  1000 warmup, 1250 samples each, convergence gate, thinned to 250 near-independent draws.
- **archive** — the model10.0 artifact used for the published runs (old sampler, 500 draws,
  inputs of January 2026), for reference only.

`consistent_rates_old_vs_new.ipynb` has the same comparison in depth for one location."""
)

code(
    """import re
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vivarium.artifact import Artifact

from vivarium_csu_alzheimers.data import consistent_rates as cr

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
print(subprocess.run(["date"], capture_output=True, text=True).stdout.strip())"""
)

md("## Parameters")
code(
    """REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
MAIN_DIR = REPO / "artifact_comparisons" / "main250"
REFIT_DIR = REPO / "artifact_comparisons" / "refit250"
ARCHIVE_DIR = Path("/mnt/share/homes/abie/projects/2025/vivarium_csu_alzheimers")
ARCHIVE_NAMES = {"taiwan": "taiwan_(province_of_china)"}  # file name differs in the archive

LOCATIONS = sorted(p.stem for p in MAIN_DIR.glob("*.hdf"))
AGES_OF_INTEREST = (65, 90)  # the sim's testing/treatment window
KEYS = [
    "cause.alzheimers_consistent.prevalence_any",
    "cause.alzheimers_consistent.population_incidence_any",
    "cause.alzheimers_consistent.bbbm_conditional_prevalence",
    "cause.alzheimers_consistent.mci_conditional_prevalence",
    "cause.alzheimers_consistent.dementia_conditional_prevalence",
    "cause.alzheimers_consistent.excess_mortality_rate",
    "cause.alzheimers_disease_and_other_dementias.cause_specific_mortality_rate",
]
short = lambda k: k.split(".")[-1]

arts = {}
for loc in LOCATIONS:
    paths = {
        "main": MAIN_DIR / f"{loc}.hdf",
        "refit": REFIT_DIR / f"{loc}.hdf",
        "archive": ARCHIVE_DIR / f"{ARCHIVE_NAMES.get(loc, loc)}.hdf",
    }
    arts[loc] = {label: Artifact(p) for label, p in paths.items() if p.exists()}
    print(f"{loc:26s} " + ", ".join(f"{k}={'yes' if k in arts[loc] else 'no':3s}" for k in paths))
STYLE = {"main": dict(color="C0", ls="--"), "refit": dict(color="C1", ls="-"), "archive": dict(color="C2", ls=":")}"""
)

md("## Helpers")
code(
    '''def draws(art, key):
    df = art.load(key)
    return df[[c for c in df.columns if str(c).startswith("draw_")]]


def summarize(df, age_max=95):
    v = df.to_numpy()
    out = pd.DataFrame(
        {"mean": v.mean(axis=1), "lo": np.quantile(v, 0.025, axis=1), "hi": np.quantile(v, 0.975, axis=1),
         "sd": v.std(axis=1)},
        index=df.index,
    ).reset_index()
    out = out[out.age_start < age_max + 1]
    return out.set_index(["sex", "age_start"]).sort_index()


def lag1_autocorr(df):
    x = df.to_numpy()
    r = [np.corrcoef(row[:-1], row[1:])[0, 1] for row in x if row.std() > 0]
    return float(np.nanmedian(r)) if r else np.nan


def in_window(s):
    a = s.index.get_level_values("age_start")
    return s[(a >= AGES_OF_INTEREST[0]) & (a < AGES_OF_INTEREST[1])]


def rel_diff(loc, key, label, ref):
    """Mean |relative difference| of stratum means, label vs ref, ages of interest, by sex."""
    if label not in arts[loc] or ref not in arts[loc]:
        return pd.Series(np.nan, index=["Female", "Male"])
    a = summarize(draws(arts[loc][ref], key))["mean"]
    b = summarize(draws(arts[loc][label], key))["mean"]
    rel = in_window((b / a - 1).dropna())
    return rel.abs().groupby(level="sex").mean()'''
)

md(
    """## 1. Did every chain converge? (the gate's receipt)

Per location and sex: leapfrog steps per iteration and wall time from the refit log, and
the worst `r_hat` / smallest effective sample size over all knots from the
`mcmc_diagnostics` key. The build gate refuses `r_hat > MAX_RHAT`, `ESS < MIN_ESS` or more
than `MAX_DIVERGENCE_FRACTION` divergent transitions, so every refit artifact that exists
passed it; the numbers show by how much."""
)
code(
    """pat = re.compile(
    r"consistent rates (\\w+): (\\d+) chains x (\\d+) samples in ([\\d.]+) min; (\\d+) leapfrog "
    r"steps/iter; divergent ([\\d.]+)%; max r_hat ([\\d.]+); min ESS (\\d+)"
)
rows = []
for loc in LOCATIONS:
    log = REFIT_DIR / f"{loc}.log"
    found = pat.findall(log.read_text()) if log.exists() else []
    for sex, chains, n, minutes, steps, div, rhat, ess in found:
        rows.append(dict(location=loc, sex=sex, minutes=float(minutes), steps_per_iter=int(steps),
                         divergent_pct=float(div), max_r_hat=float(rhat), min_ess=int(ess)))
gate = pd.DataFrame(rows).set_index(["location", "sex"]).sort_index()
display(gate)
print(f"gate: MAX_RHAT={cr.MAX_RHAT}, MIN_ESS={cr.MIN_ESS}, MAX_DIVERGENCE_FRACTION={cr.MAX_DIVERGENCE_FRACTION}")
print(f"total wall time: {gate.minutes.sum() / 60:.1f} h; "
      f"worst r_hat {gate.max_r_hat.max():.3f}; smallest ESS {gate.min_ess.min()}")"""
)

code(
    """# Which parameter and age carries the worst r_hat / smallest ESS, per location
rows = []
for loc in LOCATIONS:
    if "refit" not in arts[loc] or cr.MCMC_DIAGNOSTICS_KEY not in arts[loc]["refit"].keys:
        continue
    d = arts[loc]["refit"].load(cr.MCMC_DIAGNOSTICS_KEY).reset_index()
    worst_r = d.loc[d.r_hat.idxmax()]
    worst_e = d.loc[d.n_eff.idxmin()]
    rows.append(dict(location=loc,
                     worst_r_hat=round(worst_r.r_hat, 3), at_r=f"{worst_r.sex} {worst_r.parameter} age {worst_r.age_start}",
                     min_ess=int(worst_e.n_eff), at_ess=f"{worst_e.sex} {worst_e.parameter} age {worst_e.age_start}"))
display(pd.DataFrame(rows).set_index("location"))"""
)

md(
    """## 2. Are the draws independent?

Median over strata of the lag-1 autocorrelation of `draw_i`, `draw_{i+1}` for
`prevalence_any`. Consecutive MCMC iterations (main, archive) are strongly correlated;
thinned, interleaved draws (refit) should sit near 0."""
)
code(
    """key = "cause.alzheimers_consistent.prevalence_any"
ac = pd.DataFrame({loc: {label: lag1_autocorr(draws(a, key)) for label, a in arts[loc].items()} for loc in LOCATIONS}).T
display(ac.round(3))"""
)

md(
    """## 3. How far do the refit's rates sit from main's, at the ages the sim acts on?

Mean |relative difference| of the stratum posterior means over ages 65–90, by sex.
`refit vs main` isolates the sampler change on identical inputs. `main vs archive` shows
how much the January-to-October input drift plus the old sampler's chain-to-chain noise
moved the same keys, which is the yardstick."""
)
code(
    """tabs = {}
for label, ref in [("refit", "main"), ("main", "archive"), ("refit", "archive")]:
    t = pd.concat({loc: pd.concat({short(k): rel_diff(loc, k, label, ref) for k in KEYS}, names=["key", "sex"])
                   for loc in LOCATIONS}, names=["location"])
    tabs[f"{label} vs {ref}"] = t
table = pd.concat(tabs, axis=1)
print("refit vs main — mean |relative difference| of stratum means, ages 65–90:")
display(table["refit vs main"].unstack("sex").unstack("key").round(3))
print("same, main vs archive (input drift + old sampler noise):")
display(table["main vs archive"].unstack("sex").unstack("key").round(3))"""
)

code(
    """# Summary across keys and sexes: one number per location and comparison
summary = table.groupby(level="location").mean()
summary["worst key, refit vs main"] = table["refit vs main"].groupby(level="location").idxmax().map(lambda t: f"{t[1]} {t[2]}")
display(summary.round(3))"""
)

md(
    """## 4. The sim-relevant rates by age, every location

`prevalence_any` sizes the simulated population and `population_incidence_any` (the S→BBBM
hazard) is the inflow. Solid = refit, dashed = main, dotted = archive; one panel per
location, Female and Male in different colours. Shaded = ages 65–90."""
)
code(
    """def grid(key, logy=False, gbd_key=None):
    n = len(LOCATIONS)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.2), sharey=True)
    for ax, loc in zip(axes.ravel(), LOCATIONS):
        for sex, color in [("Female", "C3"), ("Male", "C0")]:
            if gbd_key and "main" in arts[loc]:
                g = summarize(draws(arts[loc]["main"], gbd_key)).xs(sex)
                ax.plot(g.index, g["mean"], color="k", lw=0.8, alpha=0.6)
            for label, art in arts[loc].items():
                if key not in art.keys:
                    continue
                s = summarize(draws(art, key)).xs(sex)
                ax.plot(s.index, s["mean"], color=color, ls=STYLE[label]["ls"], lw=1.2,
                        label=f"{sex} {label}" if loc == LOCATIONS[0] else None)
                if label == "refit":
                    ax.fill_between(s.index, s.lo, s.hi, color=color, alpha=0.12)
        ax.set_title(loc, fontsize=9); ax.grid(alpha=0.3)
        ax.axvspan(*AGES_OF_INTEREST, color="grey", alpha=0.07, lw=0)
        if logy:
            ax.set_yscale("log")
    for ax in axes.ravel()[len(LOCATIONS):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(short(key) + (" (black = GBD input)" if gbd_key else ""))
    fig.tight_layout()
    return fig

grid("cause.alzheimers_consistent.prevalence_any");
grid("cause.alzheimers_consistent.population_incidence_any", logy=True);
grid("cause.alzheimers_consistent.excess_mortality_rate", logy=True, gbd_key="cause.alzheimers.excess_mortality_rate");"""
)

code(
    """# Dementia prevalence implied by the fit (prevalence_any × dementia share) against the GBD input
n = len(LOCATIONS)
fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.2), sharey=True)
for ax, loc in zip(axes.ravel(), LOCATIONS):
    for sex, color in [("Female", "C3"), ("Male", "C0")]:
        g = summarize(draws(arts[loc]["main"], "cause.alzheimers.prevalence")).xs(sex)
        ax.plot(g.index, g["mean"], color="k", lw=0.8, alpha=0.6)
        for label, art in arts[loc].items():
            dem = draws(art, "cause.alzheimers_consistent.prevalence_any") * draws(art, "cause.alzheimers_consistent.dementia_conditional_prevalence")
            s = summarize(dem).xs(sex)
            ax.plot(s.index, s["mean"], color=color, ls=STYLE[label]["ls"], lw=1.2,
                    label=f"{sex} {label}" if loc == LOCATIONS[0] else None)
    ax.set_title(loc, fontsize=9); ax.grid(alpha=0.3); ax.axvspan(*AGES_OF_INTEREST, color="grey", alpha=0.07, lw=0)
for ax in axes.ravel()[len(LOCATIONS):]:
    ax.axis("off")
axes[0, 0].legend(fontsize=7)
fig.suptitle("dementia prevalence implied by the fit (black = GBD input)"); fig.tight_layout()"""
)

md(
    """## 5. New BBBM entrants per year

`susceptible_to_bbbm_transition_count` summed over ages (mean of draws), which with the
population size sets how many simulants enter each year."""
)
code(
    """n = len(LOCATIONS)
fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6.2))
for ax, loc in zip(axes.ravel(), LOCATIONS):
    for label, art in arts[loc].items():
        tc = art.load("cause.alzheimers_consistent.susceptible_to_bbbm_transition_count")
        tc = tc[[c for c in tc.columns if str(c).startswith("draw_")]].mean(axis=1).groupby(["sex", "year_start"]).sum()
        for sex, color in [("Female", "C3"), ("Male", "C0")]:
            s = tc.xs(sex)
            ax.plot(s.index, s.values, color=color, ls=STYLE[label]["ls"], lw=1.2,
                    label=f"{sex} {label}" if loc == LOCATIONS[0] else None)
    ax.set_title(loc, fontsize=9); ax.grid(alpha=0.3)
for ax in axes.ravel()[len(LOCATIONS):]:
    ax.axis("off")
axes[0, 0].legend(fontsize=7)
fig.suptitle("new BBBM entrants per year (sum over ages)"); fig.tight_layout()"""
)

md(
    """## Take-aways

*(filled in after the refits ran; see the executed notebook)*"""
)

nb["cells"] = cells
nbf.write(nb, out)
print("wrote", out)
