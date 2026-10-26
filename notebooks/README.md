# Consistent-rates notebooks

Two notebooks compare the `cause.alzheimers_consistent.*` keys that `main`'s sampler
writes with the keys that the sampler on this branch writes from the same inputs.

- `consistent_rates_old_vs_new.ipynb`: one location in depth (default: USA). Plots by
  age, chain diagnostics, draw autocorrelation, stratum-mean differences, posterior spread.
- `consistent_rates_refit250_summary.ipynb`: all ten locations at a glance. Gate receipts
  (`r_hat`, ESS, divergences, run time), draw autocorrelation, relative differences at
  ages 65–90, and the sim-relevant rates by age.

Each notebook is generated from its `build_*.py` script. Edit the builder and re-run it;
do not edit the notebook by hand.

## What the branch changes in the sampler

Compared with `main`, `consistent_rates.py` on this branch:

- samples the knots on an unbounded latent scale (logit for fractions, log for hazards)
  instead of a normal truncated to [0, 1];
- fits knots from the first age with GBD dementia data (40) and writes the younger ages as
  zero AD prevalence and zero inflow, instead of feeding the all-zero GBD cells at 30 and
  35 to the likelihood with standard error 1e-8 (a hard wall that made one chain in four
  sit at the tree-depth cap and biased the age-40 knot);
- adds a light curvature prior on the log S→BBBM hazard across age knots (second
  differences, sd 0.25), which removes a zigzag that chains explored with opposite phase;
- uses `ODE_SIGMA = 0.02` for the consistency penalty (main: 0.005, which NUTS cannot
  sample to convergence);
- runs 4 chains of 1000 warmup and 1250 samples with target acceptance 0.95 and a dense
  mass matrix, refuses the build unless every knot has `r_hat <= 1.02`, `ESS >= 100` and
  divergences stay below 1%, thins to `NUM_DRAWS` draws interleaved across chains, and
  writes the per-knot diagnostics to `cause.alzheimers_consistent.mcmc_diagnostics`.

`fit_one_sex.py` fits one sex of one artifact and keeps every chain in a pickle even when
the gate refuses the fit; `inspect_chains.py` describes that pickle chain by chain.

## Setup (no cluster, no GBD access needed)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --override overrides.txt -e ".[test]" \
    --extra-index-url https://artifactory.ihme.washington.edu/artifactory/api/pypi/pypi-shared/simple \
    "vivarium-inputs>=9.0.0,<10.0.0" \
    "jax<0.7.2" numpyro diffrax interpax "pyarrow<21" matplotlib nbconvert nbformat ipykernel \
    "black==22.3.0" "isort==5.13.2"
.venv/bin/python -m ipykernel install --user --name consistent-rates-sampler \
    --display-name "Python (consistent-rates-sampler)"
```

`overrides.txt` holds the three pins of `[tool.uv] override-dependencies` in
`pyproject.toml` (`pandas<3.0.0`, `sqlalchemy>=2.0.0,<3.0.0`, `numpy<2.0.0`).
`vivarium-inputs` comes from the IHME package index. `consistent_rates.py` imports it for
`NUM_DRAWS`, and nothing in the refit calls GBD. Black and isort are the versions the
repo's lint extra pins.

## Inputs: artifacts built by `main`

`artifact_comparisons/main250/<location>.hdf` are the ten artifacts that `main` builds
(one NUTS chain, 250 draws). The copies here were built on 2026-10-23 from `main` at
PR #82 (see `PROVENANCE.txt` in that directory). To rebuild them, run `make_artifacts`
in the artifact environment on the cluster.

`artifact_comparisons/` is for these scratch artifacts; keep it out of git (it is in
`.git/info/exclude` on the machine this was set up on; add it to yours).

## Refit with this branch's sampler

```bash
for loc in brazil china germany israel japan spain sweden taiwan united_kingdom united_states_of_america; do
    .venv/bin/python notebooks/refit_consistent_rates.py \
        artifact_comparisons/main250/$loc.hdf artifact_comparisons/refit250/$loc.hdf \
        > artifact_comparisons/refit250/$loc.log 2>&1
done
```

The script copies the input artifact and replaces the consistent-rates keys on the copy,
which is what `make_artifacts` does after it loads the GBD keys. It raises if any chain
fails the convergence gate in `consistent_rates.py` (`MAX_RHAT`, `MIN_ESS`,
`MAX_DIVERGENCE_FRACTION`); that is the point. One location takes about an hour on two
CPU cores. To run two locations at once on two cores, set
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"` so that
each process keeps to one core.

## Run the notebooks

Open a notebook with the `Python (consistent-rates-sampler)` kernel, set the paths in
the Parameters cell if they differ, and Run All. Or from the command line:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 notebooks/consistent_rates_refit250_summary.ipynb
```
