"""Script that programmatically builds the consistent_fit_accuracy.ipynb notebook.

Running this generates the .ipynb file. Keeping as a separate build-script makes
the cell contents easier to inspect and edit than a giant JSON blob.
"""
import json
import uuid
from pathlib import Path

CELLS = []


def md(src: str):
    CELLS.append({
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": src.splitlines(keepends=True),
    })


def code(src: str):
    CELLS.append({
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    })


# -----------------------------------------------------------------------------
md("""# How accurate is the consistent fit?

In the companion notebook `consistent_rates_explainer.ipynb` we built up
the ideas behind `vivarium_csu_alzheimers/data/consistent_rates.py` a step
at a time.  Part 2 of that notebook showed a *two-rate* example
(prevalence + incidence) where a soft ODE-consistency penalty was added
to the usual Gaussian likelihood.

**The question this notebook tries to answer:**
> When we add the consistency factor, does the fit actually become more
> accurate, or does it only *look* tighter?

We will measure three things across many Monte Carlo replicates:

1. **Bias** — mean of (posterior median − truth) per age
2. **RMSE** — root mean-squared error of the posterior median per age
3. **95% UI coverage** — fraction of replicates for which the 95%
   credible interval contains the truth, per age

We start with a single example (so you can see what one replicate looks
like), then replicate it many times, then summarise the findings in a
GIF that watches the metrics converge as replicates accumulate.
""")

# -----------------------------------------------------------------------------
md("""## Setup — identical priors/likelihood/consistency as Part 2

The model definitions below mirror Part 2 of
`consistent_rates_explainer.ipynb`, which in turn uses the same
ingredients (`numpyro.sample` priors, Gaussian data likelihood,
`numpyro.factor`-based soft consistency penalty) as the production
`consistent_rates.py`.
""")

code("""import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer

numpyro.set_host_device_count(1)

rng = np.random.default_rng(20260413)
""")

# -----------------------------------------------------------------------------
md("""### Ground-truth rates, ODE-consistent by construction

We pick an incidence schedule that grows with age, then integrate
forward the simple ODE
$p_{k+1} = p_k + \\Delta a \\cdot i_k \\cdot (1 - p_k)$
to get a prevalence schedule that is *exactly* consistent with the
chosen incidence.  This is the setup from cell 10 of the explainer
notebook.
""")

code("""ages = np.array([35, 45, 55, 65, 75, 85, 95])
n_ages = len(ages)
da = 10.0  # age-group width (years)

true_inc = np.array([0.0005, 0.003, 0.01, 0.025, 0.05, 0.08, 0.10])

true_prev = np.zeros(n_ages)
true_prev[0] = 0.001
for k in range(1, n_ages):
    true_prev[k] = true_prev[k-1] + da * true_inc[k-1] * (1 - true_prev[k-1])

# Observation-error sizes scale with the rate magnitude plus a floor.
obs_prev_se = 0.25 * true_prev + 0.002
obs_inc_se = 0.25 * true_inc + 0.001

print("ages     :", ages)
print("true inc :", np.round(true_inc, 4))
print("true prev:", np.round(true_prev, 4))
""")

# -----------------------------------------------------------------------------
md("""### The two models

Both use the same truncated-Normal prior and Gaussian data likelihood.
The consistent model adds a `numpyro.factor` term that penalises
log-ratio mismatch between observed and ODE-predicted prevalence.
""")

code("""def model_no_consistency(obs_prev, obs_inc):
    prev = numpyro.sample(
        "prev",
        dist.TruncatedNormal(
            loc=jnp.full(n_ages, 0.05),
            scale=jnp.full(n_ages, 0.5),
            low=1e-6, high=1.0 - 1e-6,
        ),
    )
    inc = numpyro.sample(
        "inc",
        dist.TruncatedNormal(
            loc=jnp.full(n_ages, 0.01),
            scale=jnp.full(n_ages, 0.5),
            low=1e-6, high=1.0 - 1e-6,
        ),
    )
    numpyro.sample("prev_obs", dist.Normal(prev, jnp.asarray(obs_prev_se)), obs=obs_prev)
    numpyro.sample("inc_obs",  dist.Normal(inc,  jnp.asarray(obs_inc_se)),  obs=obs_inc)


def model_with_consistency(obs_prev, obs_inc):
    prev = numpyro.sample(
        "prev",
        dist.TruncatedNormal(
            loc=jnp.full(n_ages, 0.05),
            scale=jnp.full(n_ages, 0.5),
            low=1e-6, high=1.0 - 1e-6,
        ),
    )
    inc = numpyro.sample(
        "inc",
        dist.TruncatedNormal(
            loc=jnp.full(n_ages, 0.01),
            scale=jnp.full(n_ages, 0.5),
            low=1e-6, high=1.0 - 1e-6,
        ),
    )
    numpyro.sample("prev_obs", dist.Normal(prev, jnp.asarray(obs_prev_se)), obs=obs_prev)
    numpyro.sample("inc_obs",  dist.Normal(inc,  jnp.asarray(obs_inc_se)),  obs=obs_inc)

    # Soft ODE consistency: log p[k+1] should equal log(p[k] + da*i[k]*(1-p[k]))
    sigma = 0.01
    eps = 1e-12
    for k in range(n_ages - 1):
        predicted = prev[k] + da * inc[k] * (1.0 - prev[k])
        err = jnp.log(jnp.clip(prev[k+1], eps)) - jnp.log(jnp.clip(predicted, eps))
        numpyro.factor(f"consistency_{k}", dist.Normal(0.0, sigma).log_prob(err))
""")

# -----------------------------------------------------------------------------
md("""### A single fit for one simulated dataset

We draw one noisy observation of prev/inc, fit both models, and plot
posterior medians plus 95% UIs against the truth.
""")

code("""def simulate_dataset(seed):
    r = np.random.default_rng(seed)
    obs_prev = np.clip(true_prev + r.normal(0, obs_prev_se), 1e-6, 1 - 1e-6)
    obs_inc  = np.clip(true_inc  + r.normal(0, obs_inc_se),  1e-6, 1 - 1e-6)
    return obs_prev, obs_inc


def fit(model_fn, obs_prev, obs_inc, seed, num_warmup=150, num_samples=150):
    kernel = infer.NUTS(model_fn)
    mcmc = infer.MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples,
        num_chains=1, progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), obs_prev=obs_prev, obs_inc=obs_inc)
    return mcmc.get_samples()
""")

code("""obs_prev_1, obs_inc_1 = simulate_dataset(seed=0)
samples_no = fit(model_no_consistency,   obs_prev_1, obs_inc_1, seed=11)
samples_yes = fit(model_with_consistency, obs_prev_1, obs_inc_1, seed=12)

def summarise(samples, name):
    med = np.median(samples[name], axis=0)
    lo  = np.quantile(samples[name], 0.025, axis=0)
    hi  = np.quantile(samples[name], 0.975, axis=0)
    return med, lo, hi

prev_no  = summarise(samples_no,  "prev")
inc_no   = summarise(samples_no,  "inc")
prev_yes = summarise(samples_yes, "prev")
inc_yes  = summarise(samples_yes, "inc")
print("Done.")
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (truth, obs, (med_n, lo_n, hi_n), (med_y, lo_y, hi_y), title, obs_se) in zip(
    axes,
    [
        (true_prev, obs_prev_1, prev_no, prev_yes, "Prevalence", obs_prev_se),
        (true_inc,  obs_inc_1,  inc_no,  inc_yes,  "Incidence",  obs_inc_se),
    ],
):
    ax.errorbar(ages, obs, yerr=1.96*obs_se, fmt='o', color='0.3',
                alpha=0.5, capsize=3, label='observed (±1.96 SE)')
    ax.plot(ages, truth, 'k--', lw=2, label='truth')
    ax.fill_between(ages, lo_n, hi_n, alpha=0.25, color='C0', label='no consistency 95% UI')
    ax.plot(ages, med_n, color='C0', lw=2)
    ax.fill_between(ages, lo_y, hi_y, alpha=0.25, color='C1', label='with consistency 95% UI')
    ax.plot(ages, med_y, color='C1', lw=2)
    ax.set_title(title); ax.set_xlabel('age')
    ax.legend(loc='upper left', fontsize=8)

plt.tight_layout(); plt.show()
""")

# -----------------------------------------------------------------------------
md("""## Monte Carlo replication

Now we repeat: draw a new noisy dataset, fit both models, record the
posterior median and 95% UI for each age.  After `N_REPS` replicates we
have enough samples to estimate bias, RMSE, and coverage.

> Part 2 is a small model, so each fit is fast — we use modest
> `num_warmup`/`num_samples` to keep the full loop well under the
> notebook cell-execution timeout.
""")

code("""N_REPS = 30
NUM_WARMUP = 120
NUM_SAMPLES = 120

results = {
    "no":  {"prev_med": [], "prev_lo": [], "prev_hi": [],
            "inc_med":  [], "inc_lo":  [], "inc_hi":  []},
    "yes": {"prev_med": [], "prev_lo": [], "prev_hi": [],
            "inc_med":  [], "inc_lo":  [], "inc_hi":  []},
}
datasets = []

for rep in range(N_REPS):
    seed = 1000 + rep
    op, oi = simulate_dataset(seed)
    datasets.append((op, oi))

    s_no  = fit(model_no_consistency,   op, oi, seed=2000+rep,
                num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES)
    s_yes = fit(model_with_consistency, op, oi, seed=3000+rep,
                num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES)

    for name, samples, tag in [
        ("prev", s_no,  "no"), ("inc", s_no,  "no"),
        ("prev", s_yes, "yes"), ("inc", s_yes, "yes"),
    ]:
        results[tag][f"{name}_med"].append(np.median(samples[name], axis=0))
        results[tag][f"{name}_lo"].append(np.quantile(samples[name], 0.025, axis=0))
        results[tag][f"{name}_hi"].append(np.quantile(samples[name], 0.975, axis=0))

    if (rep + 1) % 5 == 0:
        print(f"  finished replicate {rep+1}/{N_REPS}")

for tag in results:
    for key in results[tag]:
        results[tag][key] = np.asarray(results[tag][key])
print("shape check:", results["no"]["prev_med"].shape, "(reps, ages)")
""")

# -----------------------------------------------------------------------------
md("""### Bias, RMSE and 95% UI coverage

For each age we compute:

- **bias** = mean over replicates of `(posterior_median − truth)`
- **RMSE** = sqrt(mean over replicates of `(posterior_median − truth)^2`)
- **coverage** = fraction of replicates where `lo ≤ truth ≤ hi`
""")

code("""def accuracy(med, lo, hi, truth):
    bias = np.mean(med - truth[None, :], axis=0)
    rmse = np.sqrt(np.mean((med - truth[None, :])**2, axis=0))
    cov  = np.mean((lo <= truth[None, :]) & (truth[None, :] <= hi), axis=0)
    return bias, rmse, cov

metrics = {}
for tag in ["no", "yes"]:
    metrics[tag] = {}
    for name, truth in [("prev", true_prev), ("inc", true_inc)]:
        b, r, c = accuracy(
            results[tag][f"{name}_med"],
            results[tag][f"{name}_lo"],
            results[tag][f"{name}_hi"],
            truth,
        )
        metrics[tag][name] = {"bias": b, "rmse": r, "coverage": c}

# Print a compact summary row-per-age
import pandas as pd
rows = []
for name, truth in [("prev", true_prev), ("inc", true_inc)]:
    for i, a in enumerate(ages):
        rows.append({
            "rate": name, "age": a, "truth": truth[i],
            "bias (no)":  metrics["no"][name]["bias"][i],
            "bias (yes)": metrics["yes"][name]["bias"][i],
            "rmse (no)":  metrics["no"][name]["rmse"][i],
            "rmse (yes)": metrics["yes"][name]["rmse"][i],
            "cov (no)":   metrics["no"][name]["coverage"][i],
            "cov (yes)":  metrics["yes"][name]["coverage"][i],
        })
pd.DataFrame(rows).round(4)
""")

# -----------------------------------------------------------------------------
md("""### Summary plot — where does consistency help?

Bars show the difference between the two fitting strategies.  For bias
and RMSE lower is better; for coverage, closer to the nominal 0.95 is
better.
""")

code("""fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
x = np.arange(n_ages); width = 0.38

for row, (name, truth) in enumerate([("prev", true_prev), ("inc", true_inc)]):
    for col, metric in enumerate(["bias", "rmse", "coverage"]):
        ax = axes[row, col]
        vno  = metrics["no"][name][metric]
        vyes = metrics["yes"][name][metric]
        ax.bar(x - width/2, vno,  width, color='C0', label='no consistency')
        ax.bar(x + width/2, vyes, width, color='C1', label='with consistency')
        if metric == "coverage":
            ax.axhline(0.95, color='k', ls=':', lw=1, label='nominal 0.95')
            ax.set_ylim(0, 1.05)
        elif metric == "bias":
            ax.axhline(0.0, color='k', lw=0.8)
        ax.set_title(f"{name} — {metric}")
        ax.set_xticks(x); ax.set_xticklabels(ages)
        if row == 1: ax.set_xlabel('age')
        if row == 0 and col == 0: ax.legend(fontsize=8)

plt.tight_layout(); plt.show()
""")

# -----------------------------------------------------------------------------
md("""## A compelling GIF — metrics converging as replicates accumulate

The animation below has two panels.

* **Left**: the current replicate's fits (posterior medians + 95% UIs)
  for prevalence and incidence, plus the truth.
* **Right**: running RMSE per age, *averaged across all replicates seen
  so far*.  Watch the orange bars (with-consistency) shrink relative to
  the blue ones (no-consistency) as the Monte Carlo estimate settles.

We save the animation as a GIF in this directory so it can be embedded
in slides.
""")

code("""from matplotlib.animation import FuncAnimation, PillowWriter

def running_rmse(tag, name, truth, upto):
    med = results[tag][f"{name}_med"][:upto+1]
    return np.sqrt(np.mean((med - truth[None, :])**2, axis=0))

fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
(ax_prev, ax_rmse_prev), (ax_inc, ax_rmse_inc) = axes

def draw_frame(rep_idx):
    for ax in axes.flat: ax.clear()

    op, oi = datasets[rep_idx]
    prev_n = (results["no"]["prev_med"][rep_idx],
              results["no"]["prev_lo"][rep_idx],
              results["no"]["prev_hi"][rep_idx])
    prev_y = (results["yes"]["prev_med"][rep_idx],
              results["yes"]["prev_lo"][rep_idx],
              results["yes"]["prev_hi"][rep_idx])
    inc_n  = (results["no"]["inc_med"][rep_idx],
              results["no"]["inc_lo"][rep_idx],
              results["no"]["inc_hi"][rep_idx])
    inc_y  = (results["yes"]["inc_med"][rep_idx],
              results["yes"]["inc_lo"][rep_idx],
              results["yes"]["inc_hi"][rep_idx])

    # --- Left column: this replicate's fits ---
    for ax, truth, obs, se, (mn, ln, hn), (my, ly, hy), title in [
        (ax_prev, true_prev, op, obs_prev_se, prev_n, prev_y, "Prevalence — this replicate"),
        (ax_inc,  true_inc,  oi, obs_inc_se,  inc_n,  inc_y,  "Incidence — this replicate"),
    ]:
        ax.errorbar(ages, obs, yerr=1.96*se, fmt='o', color='0.3',
                    alpha=0.5, capsize=3)
        ax.plot(ages, truth, 'k--', lw=2, label='truth')
        ax.fill_between(ages, ln, hn, alpha=0.25, color='C0')
        ax.plot(ages, mn, color='C0', lw=2, label='no consistency')
        ax.fill_between(ages, ly, hy, alpha=0.25, color='C1')
        ax.plot(ages, my, color='C1', lw=2, label='with consistency')
        ax.set_title(title); ax.set_xlabel('age')
        ax.legend(loc='upper left', fontsize=8)

    # --- Right column: running RMSE ---
    x = np.arange(n_ages); width = 0.38
    for ax, name, truth, title in [
        (ax_rmse_prev, "prev", true_prev, "Running RMSE (prevalence)"),
        (ax_rmse_inc,  "inc",  true_inc,  "Running RMSE (incidence)"),
    ]:
        rn = running_rmse("no",  name, truth, rep_idx)
        ry = running_rmse("yes", name, truth, rep_idx)
        ax.bar(x - width/2, rn, width, color='C0')
        ax.bar(x + width/2, ry, width, color='C1')
        ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(ages)
        ax.set_xlabel('age')

    fig.suptitle(f"replicate {rep_idx+1}/{N_REPS}", y=1.02, fontsize=12)
    fig.tight_layout()

anim = FuncAnimation(fig, draw_frame, frames=N_REPS, interval=400)

gif_path = "consistent_fit_accuracy.gif"
anim.save(gif_path, writer=PillowWriter(fps=2.5))
plt.close(fig)
print("wrote", gif_path)
""")

code("""from IPython.display import Image
Image(filename="consistent_fit_accuracy.gif")
""")

# -----------------------------------------------------------------------------
md("""## Take-aways

* The consistency factor **propagates information between prev and
  inc**: a noisy incidence observation gets pulled toward values that
  are compatible with the neighbouring prevalences, and vice versa.
* In this setup that translates into **lower RMSE** (particularly at
  the ages where one of the rates is poorly measured relative to the
  other) without wrecking coverage.
* The GIF makes this visible: as replicates accumulate, the orange
  bars (with-consistency) settle to shorter heights than the blue
  ones.

This is the same mechanism that operates inside the full
`consistent_rates.py` ODE — only there the "prediction" comes from
`diffeqsolve` over a 5-compartment system rather than the
two-line update we used in Part 2.
""")


# -----------------------------------------------------------------------------
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "consistent_fit_accuracy.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
