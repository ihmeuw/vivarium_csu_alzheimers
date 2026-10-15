"""Sampler-level guarantees for the consistent-rates fit.

The archived (model10.0) fit ran one NUTS chain on [0, 1]-truncated knots and wrote its
500 consecutive iterations straight into draw_0..draw_499. Diagnostics on the archived
USA artifact (Oct 2026 audit) showed lag-1 autocorrelation ~0.9, effective sample sizes
of 3-6 for the Female chain, and no convergence check. These tests pin the pieces that
fix that: bounded latent-scale transforms, interleaved thinning across chains, and a
gate that fails the build when chains have not converged.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("numpyro")

vi_globals = pytest.importorskip("vivarium_inputs.globals")
from vivarium_csu_alzheimers.data import consistent_rates as cr


def _chain_samples(n_chains: int = 4, n_samples: int = 20, n_ages: int = 3) -> dict:
    """value = chain * 1000 + sample index, so provenance is readable off the value."""
    arr = np.zeros((n_chains, n_samples, n_ages, 1))
    for c in range(n_chains):
        for s in range(n_samples):
            arr[c, s] = c * 1000 + s
    return {"p": arr}


def test_thin_interleaves_chains_and_returns_requested_count():
    thinned = cr.thin_samples(_chain_samples(), 8)
    assert thinned["p"].shape == (8, 3, 1)
    chains = (thinned["p"][:, 0, 0] // 1000).astype(int)
    assert chains.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    within = (thinned["p"][:, 0, 0] % 1000).astype(int)
    assert within.tolist() == [0, 0, 0, 0, 19, 19, 19, 19]


def test_thin_requires_enough_pooled_samples():
    with pytest.raises(ValueError):
        cr.thin_samples(_chain_samples(n_chains=1, n_samples=5), 10)


def test_convergence_gate_passes_good_chains():
    good = {"p": {"r_hat": np.array([[1.0], [1.005]]), "n_eff": np.array([[400.0], [150.0]])}}
    cr.check_convergence(good, divergence_fraction=0.0, sex="Male")


def test_convergence_gate_raises_on_bad_rhat():
    bad = {"p": {"r_hat": np.array([[1.0], [1.3]]), "n_eff": np.array([[400.0], [150.0]])}}
    with pytest.raises(RuntimeError, match="p"):
        cr.check_convergence(bad, divergence_fraction=0.0, sex="Male")


def test_convergence_gate_raises_on_low_ess():
    bad = {"h_S_to_BBBM": {"r_hat": np.array([[1.0]]), "n_eff": np.array([[12.0]])}}
    with pytest.raises(RuntimeError, match="h_S_to_BBBM"):
        cr.check_convergence(bad, divergence_fraction=0.0, sex="Female")


def test_convergence_gate_raises_on_divergences():
    good = {"p": {"r_hat": np.array([[1.0]]), "n_eff": np.array([[400.0]])}}
    with pytest.raises(RuntimeError, match="diverg"):
        cr.check_convergence(good, divergence_fraction=0.05, sex="Female")


def test_convergence_gate_ignores_nan_from_constant_sites():
    constant = {"p": {"r_hat": np.array([[np.nan]]), "n_eff": np.array([[np.nan]])}}
    cr.check_convergence(constant, divergence_fraction=0.0, sex="Male")


def test_transforms_are_bounded_and_start_at_initial_rate():
    shape = (2, 1)
    for name in cr.FRACTION_PARAMS + cr.HAZARD_PARAMS:
        start = np.asarray(cr.transform_knots(name, cr.initial_latent(name, shape)))
        np.testing.assert_allclose(start, cr.INITIAL_RATE, rtol=1e-6)
        huge = np.asarray(cr.transform_knots(name, np.full(shape, 5.0)))
        tiny = np.asarray(cr.transform_knots(name, np.full(shape, -5.0)))
        assert np.all(huge > 0) and np.all(tiny > 0)
        if name in cr.FRACTION_PARAMS:
            assert np.all(huge < 1.0)


def test_get_rate_writes_draw_count_draws():
    model = cr.BBBM_AD_Model(ages=np.array([30, 35]), years=[2025], sex="Male")
    model.samples = {"p": np.random.rand(vi_globals.NUM_DRAWS, 2, 1)}
    df = model.get_rate("p", 2025)
    assert df.shape == (2, vi_globals.NUM_DRAWS)
    assert list(df.columns) == list(vi_globals.DRAW_COLUMNS)


def test_diagnostics_table_has_one_row_per_param_and_stratum():
    summary = {
        "p": {
            "r_hat": np.array([[1.0], [1.01]]),
            "n_eff": np.array([[300.0], [200.0]]),
            "mean": np.array([[0.1], [0.2]]),
            "std": np.array([[0.01], [0.02]]),
        }
    }
    df = cr.diagnostics_table(summary, ages=[30, 35], years=[2025], sex="Male", params=["p"])
    assert len(df) == 2
    assert {"r_hat", "n_eff", "mean", "sd"} <= set(df.columns)
    assert df.index.names == ["sex", "parameter", "age_start", "year_start"]
