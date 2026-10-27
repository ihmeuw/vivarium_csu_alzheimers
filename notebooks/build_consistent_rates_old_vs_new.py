"""Build notebooks/consistent_rates_old_vs_new.ipynb (nbformat JSON) for the sampler branch."""
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
    """# Consistent rates in the artifact: old sampler vs new sampler

Compares the `cause.alzheimers_consistent.*` keys (and the keys derived from them) between
two artifacts for one location:

- **old** — the artifact that `main` builds today (PR #82, 250 draws): one NUTS chain,
  knots sampled on [0, 1], 500 warmup iterations, then 250 consecutive iterations written
  as `draw_0..draw_249`, no convergence check.
- **new** — the same raw GBD inputs refit with the sampler on branch
  `abie/consistent-rates-sampler`: latent-scale knots from age 40 (the first age with GBD
  dementia data; ages 30 and 35 are written as zero prevalence and zero inflow), a light
  curvature prior on the log S→BBBM hazard, several chains, convergence gate, draws
  thinned across chains, `cause.alzheimers_consistent.mcmc_diagnostics` written.

Optionally more artifacts (`EXTRA`) — by default the archived model10.0 artifact (old
sampler, 500 draws, inputs pulled in January 2026), so that input drift and sampler
change can be told apart.

Modelled on the V&V notebooks in `vivarium_research_alzheimers/verification_and_validation`
and on `generate_and_test_consistent_rates.ipynb`: age on the x-axis, one panel per sex,
GBD input in black, old dashed, new solid, 95% UI bands.

**How to use:** set `LOCATION`, `OLD_ARTIFACT`, `NEW_ARTIFACT` below, then Run All.
Kernel: `Python (consistent-rates-sampler)` (the worktree's `.venv`)."""
)

code(
    """from pathlib import Path
import subprocess, json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from vivarium.artifact import Artifact  # vivarium-suite (this branch)
except ImportError:  # legacy vivarium 3.x env
    from vivarium import Artifact

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print(subprocess.run(["date"], capture_output=True, text=True).stdout.strip())"""
)

md("## Parameters")
code(
    """LOCATION = "united_states_of_america"

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

# Old: an artifact built by `main` (one chain, 250 draws). The ten in artifact_comparisons/main250
# were built 2026-10-23 from main at PR #82; see PROVENANCE.txt there. Kept out of git.
OLD_ARTIFACT = REPO / "artifact_comparisons" / "main250" / f"{LOCATION}.hdf"

# New: the same inputs refit with the new sampler. Produce it with
#   python notebooks/refit_consistent_rates.py <old.hdf> artifact_comparisons/refit250/<LOCATION>.hdf
# (see notebooks/README.md). Kept out of git.
NEW_ARTIFACT = REPO / "artifact_comparisons" / "refit250" / f"{LOCATION}.hdf"

# Optional extra references, label -> path. Empty dict to skip.
ARCHIVE_DIR = Path("/mnt/share/homes/abie/projects/2025/vivarium_csu_alzheimers")  # = git-lfs archive on main
EXTRA = {
    "archive (model10.0)": ARCHIVE_DIR / f"{LOCATION}.hdf",
}
EXTRA = {k: v for k, v in EXTRA.items() if Path(v).exists()}

AGES_OF_INTEREST = (65, 90)  # the sim's testing/treatment window; used for summary tables

for label, path in [("old", OLD_ARTIFACT), ("new", NEW_ARTIFACT), *EXTRA.items()]:
    print(f"{label:22s} {path}  exists={Path(path).exists()}")"""
)

code(
    """arts = {"old": Artifact(OLD_ARTIFACT), "new": Artifact(NEW_ARTIFACT)}
assert "cause.alzheimers_consistent.mcmc_diagnostics" in arts["new"].keys, (
    "NEW_ARTIFACT has no mcmc_diagnostics key: the refit has not finished writing it"
)
arts.update({k: Artifact(v) for k, v in EXTRA.items()})
STYLE = {"old": dict(color="C0", ls="--"), "new": dict(color="C1", ls="-")}
for i, k in enumerate(EXTRA):
    STYLE[k] = dict(color=f"C{i + 2}", ls=":")

CONSISTENT_KEYS = [k for k in sorted(arts["new"].keys)
                   if k.startswith("cause.alzheimers_consistent.") and "mcmc_diagnostics" not in k]
print("\\n".join(CONSISTENT_KEYS))"""
)

md("## Helpers")
code(
    '''def draws(art, key):
    df = art.load(key)
    return df[[c for c in df.columns if str(c).startswith("draw_")]]


def summarize(df, age_max=95):
    """mean / 2.5% / 97.5% per stratum, indexed by (sex, age_start)."""
    v = df.to_numpy()
    out = pd.DataFrame(
        {"mean": v.mean(axis=1), "lo": np.quantile(v, 0.025, axis=1), "hi": np.quantile(v, 0.975, axis=1),
         "sd": v.std(axis=1)},
        index=df.index,
    ).reset_index()
    out = out[out.age_start < age_max + 1]
    return out.set_index(["sex", "age_start"]).sort_index()


def lag1_autocorr(df):
    """Median over strata of corr(draw_i, draw_{i+1}); ~0 for independent draws."""
    x = df.to_numpy()
    r = [np.corrcoef(row[:-1], row[1:])[0, 1] for row in x if row.std() > 0]
    return float(np.nanmedian(r)) if r else np.nan


def age_panels(key, gbd=None, logy=False, title=None, ymin=None):
    """One panel per sex: GBD input (black), each artifact's consistent key (band + line)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, sex in zip(axes, ["Male", "Female"]):
        if gbd is not None:
            g = summarize(gbd).xs(sex)
            ax.plot(g.index, g["mean"], "k.-", label="GBD input", zorder=5)
            ax.fill_between(g.index, g.lo, g.hi, color="k", alpha=0.08)
        for label, art in arts.items():
            if key not in art.keys:
                continue
            s = summarize(draws(art, key)).xs(sex)
            ax.plot(s.index, s["mean"], label=label, **STYLE[label])
            ax.fill_between(s.index, s.lo, s.hi, color=STYLE[label]["color"], alpha=0.15)
        ax.set_title(sex)
        ax.set_xlabel("age_start")
        ax.axvspan(*AGES_OF_INTEREST, color="grey", alpha=0.07, lw=0)
        if logy:
            ax.set_yscale("log")
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title or key.replace("cause.alzheimers_consistent.", "~."))
    fig.tight_layout()
    return fig


def rel_diff_table(key, ref="old"):
    """Mean |relative difference| of stratum means vs `ref`, over AGES_OF_INTEREST, by sex."""
    base = summarize(draws(arts[ref], key))["mean"]
    rows = {}
    for label, art in arts.items():
        if label == ref or key not in art.keys:
            continue
        other = summarize(draws(art, key))["mean"]
        rel = (other / base - 1).dropna()
        rel = rel[(rel.index.get_level_values("age_start") >= AGES_OF_INTEREST[0])
                  & (rel.index.get_level_values("age_start") < AGES_OF_INTEREST[1])]
        rows[label] = rel.abs().groupby(level="sex").mean()
    return pd.DataFrame(rows)'''
)

md(
    """## 1. What the new sampler reports about itself

`cause.alzheimers_consistent.mcmc_diagnostics` holds per-knot `n_eff` (effective sample
size, bulk) and `r_hat` (split-chain) from the run that produced the draws. The build gate
on this branch refuses to write an artifact with any knot below `MIN_ESS` or above
`MAX_RHAT`, so a shipped artifact should look clean here; the table is the receipt."""
)
code(
    """from vivarium_csu_alzheimers.data import consistent_rates as cr

if cr.MCMC_DIAGNOSTICS_KEY in arts["new"].keys:
    diag = arts["new"].load(cr.MCMC_DIAGNOSTICS_KEY)
    summary = diag.groupby(["sex", "parameter"]).agg(
        n_eff_min=("n_eff", "min"), n_eff_median=("n_eff", "median"),
        r_hat_max=("r_hat", "max"), r_hat_median=("r_hat", "median"),
    )
    display(summary.round(3))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, sex in zip(axes, ["Male", "Female"]):
        d = diag.xs(sex, level="sex").reset_index()
        for i, (param, g) in enumerate(d.groupby("parameter")):
            ax.scatter(np.full(len(g), i) + np.random.uniform(-0.15, 0.15, len(g)), g.n_eff, s=12)
        ax.set_xticks(range(d.parameter.nunique()))
        ax.set_xticklabels(sorted(d.parameter.unique()), rotation=45)
        ax.axhline(cr.MIN_ESS, color="r", ls="--", lw=1, label=f"MIN_ESS = {cr.MIN_ESS}")
        ax.set_yscale("log"); ax.set_title(f"{sex}: ESS per knot"); ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
else:
    print("new artifact has no mcmc_diagnostics key")"""
)

md(
    """## 2. Are the artifact draws independent?

The old artifact's draws are consecutive MCMC iterations, so `draw_i` and `draw_{i+1}` are
strongly correlated (lag-1 autocorrelation ~0.9 for the archived USA artifact). The new
sampler thins evenly along each chain and interleaves chains, so neighbouring draws should
be close to independent. Anything near 0 is good; the GBD-sourced keys are shown for
reference (their draws are independent by construction)."""
)
code(
    """keys_for_autocorr = CONSISTENT_KEYS + ["population.scaling_factor", "cause.alzheimers.prevalence"]
ac = pd.DataFrame(
    {label: {k.replace("cause.alzheimers_consistent.", "~."): lag1_autocorr(draws(art, k))
             for k in keys_for_autocorr if k in art.keys}
     for label, art in arts.items()}
)
display(ac.round(3))
ax = ac.plot.barh(figsize=(8, 0.4 * len(ac) + 1), color=[STYLE[c]["color"] for c in ac.columns])
ax.set_xlabel("lag-1 autocorrelation of draws (median over strata)"); ax.axvline(0, color="k", lw=0.8)
ax.grid(alpha=0.3)"""
)

code(
    """# A trace of one stratum makes the difference visible: Female prevalence_any at age 75.
key, sex, age = "cause.alzheimers_consistent.prevalence_any", "Female", 75
fig, axes = plt.subplots(len(arts), 1, figsize=(11, 1.9 * len(arts)), sharex=True, sharey=True)
for ax, (label, art) in zip(np.atleast_1d(axes), arts.items()):
    row = draws(art, key).xs(sex, level="sex")
    row = row[row.index.get_level_values("age_start") == age].to_numpy().ravel()
    ax.plot(row, lw=0.7, **{k: v for k, v in STYLE[label].items() if k == "color"})
    ax.set_ylabel(label, rotation=0, ha="right"); ax.grid(alpha=0.3)
np.atleast_1d(axes)[-1].set_xlabel("draw index")
fig.suptitle(f"{key.split('.')[-1]} — {sex}, age {age}: value by draw index")
fig.tight_layout()"""
)

md(
    """## 3. Rates by age: GBD input vs old vs new

The consistent model's job is to reproduce the GBD dementia prevalence, incidence and
excess mortality (black) while adding the pre-dementia BBBM and MCI states. Where a GBD
counterpart exists it is drawn in black; where none exists (the conditional shares and the
S→BBBM hazard) only the fitted curves are shown. Grey band = ages the sim's testing and
treatment act on."""
)
code(
    """# GBD dementia prevalence should match the dementia-phase part of the consistent model:
# prevalence_any * dementia_conditional_prevalence.
gbd_prev = draws(arts["old"], "cause.alzheimers.prevalence")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, sex in zip(axes, ["Male", "Female"]):
    g = summarize(gbd_prev).xs(sex)
    ax.plot(g.index, g["mean"], "k.-", label="GBD dementia prevalence", zorder=5)
    for label, art in arts.items():
        dem = draws(art, "cause.alzheimers_consistent.prevalence_any") * draws(art, "cause.alzheimers_consistent.dementia_conditional_prevalence")
        s = summarize(dem).xs(sex)
        ax.plot(s.index, s["mean"], label=f"{label}: prevalence_any × dementia share", **STYLE[label])
        ax.fill_between(s.index, s.lo, s.hi, color=STYLE[label]["color"], alpha=0.15)
    ax.set_title(sex); ax.set_xlabel("age_start"); ax.grid(alpha=0.3); ax.axvspan(*AGES_OF_INTEREST, color="grey", alpha=0.07, lw=0)
axes[0].legend(fontsize=8)
fig.suptitle("Dementia prevalence: GBD vs implied by the consistent rates"); fig.tight_layout()"""
)

code(
    """age_panels("cause.alzheimers_consistent.population_incidence_dementia",
           gbd=draws(arts["old"], "cause.alzheimers.population_incidence_rate"), logy=True,
           title="Dementia incidence (per total population): GBD vs implied by the consistent rates");"""
)
code(
    """age_panels("cause.alzheimers_consistent.population_incidence_any",
           gbd=draws(arts["old"], "cause.alzheimers.population_incidence_rate"), logy=True, ymin=1e-5,
           title="S→BBBM hazard (population_incidence_any); GBD dementia incidence shown for scale");"""
)
code(
    """age_panels("cause.alzheimers_consistent.excess_mortality_rate",
           gbd=draws(arts["old"], "cause.alzheimers.excess_mortality_rate"), logy=True, ymin=1e-2,
           title="Dementia excess mortality: GBD (DisMod) vs consistent");"""
)
code(
    """age_panels("cause.alzheimers_consistent.prevalence_any", title="prevalence_any = share of population in any AD state (BBBM+MCI+dementia)")
age_panels("cause.alzheimers_consistent.bbbm_conditional_prevalence")
age_panels("cause.alzheimers_consistent.mci_conditional_prevalence")
age_panels("cause.alzheimers_consistent.dementia_conditional_prevalence")
age_panels("cause.alzheimers_disease_and_other_dementias.cause_specific_mortality_rate", logy=True, ymin=1e-7, title="Derived CSMR = EMR × prevalence_any × dementia share")
age_panels("cause.alzheimers_consistent.ode_errors", title="ODE consistency error (log-ratio units; the penalty's scale is ODE_SIGMA)");"""
)

md(
    """## 4. Stratum means, old vs new

Each point is one (sex, age) stratum's posterior mean. Points on the diagonal are
unchanged. The archive panel (if present) shows the old sampler on the January 2026
inputs, so it mixes input drift with the old sampler's chain-to-chain noise."""
)
code(
    """keys_scatter = [k for k in CONSISTENT_KEYS if "ode_errors" not in k and "transition_count" not in k]
others = [k for k in arts if k != "old"]
fig, axes = plt.subplots(1, len(others), figsize=(5.2 * len(others), 4.6), squeeze=False)
for ax, label in zip(axes[0], others):
    for sex, marker in [("Male", "o"), ("Female", "s")]:
        xs, ys = [], []
        for k in keys_scatter:
            if k not in arts[label].keys:
                continue
            a = summarize(draws(arts["old"], k))["mean"].xs(sex)
            b = summarize(draws(arts[label], k))["mean"].xs(sex)
            sel = (a.index >= AGES_OF_INTEREST[0]) & (a.index < AGES_OF_INTEREST[1])
            xs += list(a[sel]); ys += list(b[sel])
        ax.scatter(xs, ys, s=14, marker=marker, alpha=0.7, label=sex)
    lim = (1e-5, 1.5)
    ax.plot(lim, lim, "k-", lw=0.8); ax.plot(lim, [l * 1.1 for l in lim], "k:", lw=0.6); ax.plot(lim, [l / 1.1 for l in lim], "k:", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("old (stratum mean)"); ax.set_ylabel(f"{label} (stratum mean)")
    ax.set_title(f"{label} vs old, ages {AGES_OF_INTEREST[0]}–{AGES_OF_INTEREST[1]}; dotted = ±10%"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()"""
)

code(
    """tables = {k.replace("cause.alzheimers_consistent.", "~."): rel_diff_table(k) for k in keys_scatter}
table = pd.concat(tables, names=["key", "sex"])
print(f"Mean |relative difference| of stratum means vs old, ages {AGES_OF_INTEREST[0]}–{AGES_OF_INTEREST[1]}:")
display(table.unstack("sex").round(3))"""
)

code(
    """# The same table with the archive as the reference. If "new" sits closer to the archive than
# "old" does, the old sampler's chain for this location wandered, not the inputs.
if any("archive" in k for k in arts):
    ref = next(k for k in arts if "archive" in k)
    tables = {k.replace("cause.alzheimers_consistent.", "~."): rel_diff_table(k, ref=ref) for k in keys_scatter}
    table_archive = pd.concat(tables, names=["key", "sex"])
    print(f"Mean |relative difference| of stratum means vs {ref}, ages {AGES_OF_INTEREST[0]}–{AGES_OF_INTEREST[1]}:")
    display(table_archive.unstack("sex").round(3))"""
)

md(
    """## 5. Uncertainty: posterior spread old vs new

Ratio of the posterior standard deviation (new / old) by age. A badly mixed chain
understates spread (it barely moves), so ratios above 1 for the Female side are expected
if the old Female chain was stuck."""
)
code(
    """fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, sex in zip(axes, ["Male", "Female"]):
    for k in keys_scatter:
        a = summarize(draws(arts["old"], k)).xs(sex)["sd"]
        b = summarize(draws(arts["new"], k)).xs(sex)["sd"]
        ax.plot(a.index, (b / a).replace([np.inf, -np.inf], np.nan), marker=".", lw=1, label=k.split(".")[-1])
    ax.axhline(1, color="k", lw=0.8); ax.set_yscale("log"); ax.set_title(f"{sex}: posterior sd, new / old")
    ax.set_xlabel("age_start"); ax.grid(alpha=0.3); ax.axvspan(*AGES_OF_INTEREST, color="grey", alpha=0.07, lw=0)
axes[1].legend(fontsize=7, loc="upper right")
fig.tight_layout()"""
)

md(
    """## 6. What the simulation actually consumes

`population.scaling_factor` (= prevalence_any) sizes the simulated AD population, and
`cause.alzheimers_consistent.susceptible_to_bbbm_transition_count` is the number of new
BBBM entrants per year. Summed over ages, by sex and year."""
)
code(
    """fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for label, art in arts.items():
    tc = art.load("cause.alzheimers_consistent.susceptible_to_bbbm_transition_count")
    tc = tc[[c for c in tc.columns if str(c).startswith("draw_")]].mean(axis=1).groupby(["sex", "year_start"]).sum()
    for sex, ls_sex in [("Male", "-"), ("Female", "--")]:
        s = tc.xs(sex)
        axes[0].plot(s.index, s.values, color=STYLE[label]["color"], ls=ls_sex, label=f"{label} {sex}")
    sf = summarize(draws(art, "population.scaling_factor"))
    for sex, ls_sex in [("Male", "-"), ("Female", "--")]:
        s = sf.xs(sex)
        axes[1].plot(s.index, s["mean"], color=STYLE[label]["color"], ls=ls_sex, label=f"{label} {sex}")
axes[0].set_title("new BBBM entrants per year (sum over ages, mean of draws)"); axes[0].set_xlabel("year")
axes[1].set_title("population.scaling_factor by age"); axes[1].set_xlabel("age_start")
for ax in axes:
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout()"""
)

md(
    """## Take-aways (USA, main's 250-draw inputs of 2026-10-23; written 2026-10-26)

- **The gate works.** The first version of this sampler, with knots from age 30, refused
  the USA Female fit on these inputs: one chain of four ran at the NUTS tree-depth cap in a
  second mode that the exact-zero GBD cells at ages 30 and 35 create (no AD states, or AD
  states that are all pre-dementia). Fitting from age 40 and smoothing the S→BBBM hazard
  across age removed both problems: 63–83 leapfrog steps per iteration, no divergences,
  every knot with ESS ≥ 780 and r_hat ≤ 1.010, 31 minutes for both sexes.
- **Main's Female chain is stuck.** Its draws have lag-1 autocorrelation 0.85–0.91, and its
  `prevalence_any` has a bump at 70–75 and a dip at 80–85 that neither the refit nor a
  converged chain of the old model (2026-10-15) shows. Male is stable: the refit is within
  1–11% of main at ages 65–90.
- **Rates the simulation uses (Female, 65–90):** `prevalence_any` and the S→BBBM hazard
  move by 19% and 55% on average against main, almost all of it at 70–85. In a
  2,000-simulant baseline smoke run (draw 0, 2022–2040) that gives 25% more new simulants,
  22% more BBBM person-time and 17% more MCI person-time for Female, with dementia
  person-time unchanged (+1%) because the fit is pinned to GBD dementia prevalence. Male
  totals stay within 5%. See `artifact_comparisons/smoke/`.
- **Ages 30 and 35** are written as zero prevalence and zero inflow. The archived USA
  artifact sent 3 of 721,000 new BBBM entrants in below age 40, so the simulation does not
  notice.
- **Unchanged by any of this:** every fit sits 10–28% below GBD dementia prevalence at
  ages 65–95 while matching incidence and excess mortality. That is a property of the
  model (the consistency constraint plus fixed BBBM and MCI durations), not of the sampler.

## Earlier result (USA, archived model10.0 inputs vs a refit of them; written 2026-10-15)

- **The archived Female chain never converged.** Its 500 draws have lag-1 autocorrelation
  ~0.9 and an effective sample size of 3–6 per knot, with 5 divergent transitions. The new
  sampler (4 chains × 1250, sigma 0.02) and an independent chain of the *old* sampler
  (seed 1, sigma 0.005) agree with each other within 0.3–4% at ages 65–90 for both sexes,
  and both disagree with the archived Female rates by 7–65% (S→BBBM hazard worst). The
  published model12.2 USA runs used those archived Female rates.
- **Male rates are stable.** New vs archive 1–8%, inside the old sampler's own
  chain-to-chain noise.
- **Sigma 0.02 vs 0.005 does not move the sim-relevant rates**: new (0.02) vs seed 1
  (0.005) is within 0.3–4%. It does make the posterior sampleable: ~130–170 leapfrog steps
  per iteration instead of 1023, every knot ESS > 320, r_hat ≤ 1.012, no divergences.
- **Uncertainty was understated for Female**: the new posterior sd is 2–5× the archived
  chain's (which barely moved) and 1.05–1.2× the seed-1 chain's.
- **Draws are now near-independent** (lag-1 autocorrelation ~0.01 vs 0.85–0.89), so
  `input_draw_number` picks genuinely different posterior samples.
- **Unchanged by any of this**: both old and new fits sit ~25% below GBD dementia
  prevalence above age 80 while matching GBD incidence and EMR closely. That is a property
  of the model (the consistency constraint plus a fixed BBBM/MCI duration), not of the
  sampler, and is worth a separate look."""
)

nb["cells"] = cells
nbf.write(nb, out)
print("wrote", out)
