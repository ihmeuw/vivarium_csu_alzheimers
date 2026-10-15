# Consistent-rates notebooks

`consistent_rates_old_vs_new.ipynb` compares the `cause.alzheimers_consistent.*` keys of an
archived (model10.0) artifact with a refit of the same inputs using the sampler on this
branch. It is generated from `build_consistent_rates_old_vs_new.py`; edit the builder and
re-run it rather than editing the notebook by hand.

## Setup (no cluster, no GBD access needed)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --override overrides.txt -e ".[test]" \
    "jax<0.7.2" numpyro diffrax interpax "pyarrow<21" matplotlib nbconvert nbformat ipykernel
.venv/bin/python -m ipykernel install --user --name consistent-rates-sampler \
    --display-name "Python (consistent-rates-sampler)"
```

where `overrides.txt` contains the same three pins as `[tool.uv] override-dependencies`
in `pyproject.toml` (`pandas<3.0.0`, `sqlalchemy>=2.0.0,<3.0.0`, `numpy<2.0.0`).
The `data` extra is not required for this; it needs the IHME package index.

## Produce the "new" artifact

```bash
git lfs pull   # if src/vivarium_csu_alzheimers/artifacts/*.hdf are still pointers
.venv/bin/python notebooks/refit_consistent_rates.py \
    src/vivarium_csu_alzheimers/artifacts/united_states_of_america.hdf \
    artifact_comparisons/united_states_of_america.new_sampler.hdf
```

`artifact_comparisons/` is for these scratch artifacts; keep it out of git (it is in
`.git/info/exclude` on the machine this was set up on; add it to yours).

The refit raises if any chain fails the convergence gate in `consistent_rates.py`
(`MAX_RHAT`, `MIN_ESS`, `MAX_DIVERGENCE_FRACTION`); that is the point.

## Run the notebook

Open `consistent_rates_old_vs_new.ipynb` with the `Python (consistent-rates-sampler)`
kernel, set `LOCATION` / paths in the Parameters cell, Run All.
