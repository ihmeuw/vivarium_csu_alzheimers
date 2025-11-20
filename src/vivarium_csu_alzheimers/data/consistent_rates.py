"""NumPyro/JAX implementation of DisMod-AT, refactored from notebook 2024_07_28a_dismod_ipd_ai_refactor.ipynb
"""
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from numpyro import distributions as dist
from numpyro import infer
from vivarium.framework.artifact import Artifact

from vivarium_csu_alzheimers.constants import data_keys
from vivarium_csu_alzheimers.constants.data_values import (
    BBBM_AVG_DURATION,
    MCI_AVG_DURATION,
)

n_mcmc_samples = (
    500  # 500 for a real run, perhaps something smaller for iterative development
)


def generate_consistent_rates(art: Artifact):
    """Generates consistent rates for sim of BBBM-AD

    Parameters
    ----------
    art
        The artifact to read from and write to.

    """
    load_key = {
        "p_ad_dementia": data_keys.ALZHEIMERS.AD_DEMENTIA_PREVALENCE,
        "p_mild": data_keys.ALZHEIMERS.MILD_DEMENTIA_PREVALENCE,
        "p_moderate": data_keys.ALZHEIMERS.MODERATE_DEMENTIA_PREVALENCE,
        "p_severe": data_keys.ALZHEIMERS.SEVERE_DEMENTIA_PREVALENCE,
        "i_ad_dementia": data_keys.ALZHEIMERS.AD_DEMENTIA_INCIDENCE_RATE_TOTAL_POPULATION,
        "f": data_keys.ALZHEIMERS.EMR_DISMOD,
        "m_all": data_keys.POPULATION.ACMR,
    }

    save_key = {
        "p_ad": data_keys.ALZHEIMERS_CONSISTENT.AD_PREVALENCE,
        "i_ad": data_keys.ALZHEIMERS_CONSISTENT.MILD_DEMENTIA_INCIDENCE_RATE_TOTAL_POPULATION,
        "h_S_to_BBBM": data_keys.ALZHEIMERS_CONSISTENT.BBBM_AD_INCIDENCE_RATE,
        "h_mild_to_moderate": data_keys.ALZHEIMERS_CONSISTENT.MILD_TO_MODERATE_DEMENTIA_TRANSITION_RATE,
        "h_moderate_to_severe": data_keys.ALZHEIMERS_CONSISTENT.MODERATE_TO_SEVERE_DEMENTIA_TRANSITION_RATE,
        "f_mild": data_keys.ALZHEIMERS_CONSISTENT.EMR_MILD,
        "f_moderate": data_keys.ALZHEIMERS_CONSISTENT.EMR_MODERATE,
        "f_severe": data_keys.ALZHEIMERS_CONSISTENT.EMR_SEVERE,
        "delta_BBBM": data_keys.ALZHEIMERS_CONSISTENT.BBBM_CONDITIONAL_PREVALENCE,
        "delta_MCI": data_keys.ALZHEIMERS_CONSISTENT.MCI_CONDITIONAL_PREVALENCE,
        "delta_mild": data_keys.ALZHEIMERS_CONSISTENT.MILD_DEMENTIA_CONDITIONAL_PREVALENCE,
        "delta_moderate": data_keys.ALZHEIMERS_CONSISTENT.MODERATE_DEMENTIA_CONDITIONAL_PREVALENCE,
        "delta_severe": data_keys.ALZHEIMERS_CONSISTENT.SEVERE_DEMENTIA_CONDITIONAL_PREVALENCE,
        "ode_errors": "cause.alzheimers_consistent.ode_errors",  # HACK: store this for development/debug purposes
    }

    # fit model separately for Male and Female
    model_dict = {}
    ages = np.arange(30, 101, 5)
    for sex in ["Male", "Female"]:
        # ETL the data
        severity_sum = (
            art.load(load_key["p_mild"])
            + art.load(load_key["p_moderate"])
            + art.load(load_key["p_severe"])
        )
        severity_sum += 1e-8  # offset denominator to avoid divide by zero

        df_data = pd.concat(
            [
                transform_to_data(
                    "p_ad_dementia", art.load(load_key["p_ad_dementia"]), sex, ages, [2023]
                ),
                transform_to_data(
                    "frac_mild",
                    art.load(load_key["p_mild"]) / severity_sum,
                    sex,
                    ages,
                    [2021],
                ),
                transform_to_data(
                    "frac_moderate",
                    art.load(load_key["p_moderate"]) / severity_sum,
                    sex,
                    ages,
                    [2021],
                ),
                transform_to_data(
                    "frac_severe",
                    art.load(load_key["p_severe"]) / severity_sum,
                    sex,
                    ages,
                    [2021],
                ),
                transform_to_data(
                    "i_ad_dementia", art.load(load_key["i_ad_dementia"]), sex, ages, [2023]
                ),
                transform_to_data("f", art.load(load_key["f"]), sex, ages, [2023]),
                transform_to_data("m_all", art.load(load_key["m_all"]), sex, ages, [2025]),
            ]
        )
        # create and fit the model
        model_dict[sex] = BBBM_AD_Model(ages, [2025], sex)
        model_dict[sex].fit(ages, df_data)

    # store consistent rates in artifact
    for rate_type in save_key.keys():
        # generate data for this rate_type
        df_out = []
        for model in model_dict.values():
            df_out.append(model.get_rate(rate_type, 2025))
        df_out = pd.concat(df_out)

        # store generated data in artifact
        rate_name = save_key[rate_type]
        write_or_replace(art, rate_name, df_out)


class BBBM_AD_Model:
    """Class to create and fit a consistent model of BBBM-AD using MCMC with Numpyro"""

    def __init__(self, ages, years, sex):
        self.ages = ages
        self.years = years
        self.sex = sex

    def fit(self, ages, df_data):

        # copy this into shorter variable names for convenience
        years = self.years
        ages = self.ages

        def numpyro_model():
            knot_val_dict = {}
            for param in [
                "p_ad",
                "delta_BBBM",
                "delta_MCI",
                "delta_mild",
                "delta_moderate",
                "delta_severe",
                "h_S_to_BBBM",
                "i_ad",
                "h_mild_to_moderate",
                "h_moderate_to_severe",
                "f_mild",
                "f_moderate",
                "f_severe",
                "m",
            ]:
                knot_val_dict[param] = numpyro.sample(
                    f"{param}",
                    dist.TruncatedNormal(
                        loc=jnp.zeros((len(ages), len(years))),
                        scale=jnp.ones((len(ages), len(years))),
                        low=0.0,
                        high=1.0,
                    ),
                )

            # smooth some of the knot values
            # add_age_smoothness_factors(knot_val_dict, "delta_BBBM")
            # add_age_smoothness_factors(knot_val_dict, "delta_MCI")
            # add_age_smoothness_factors(knot_val_dict, "delta_moderate")
            # add_age_smoothness_factors(knot_val_dict, "delta_mild")
            # add_age_smoothness_factors(knot_val_dict, "delta_severe")

            add_age_smoothness_factors(
                knot_val_dict, "h_S_to_BBBM", sigma_first=10, sigma_second=0.25
            )
            add_age_smoothness_factors(knot_val_dict, "h_mild_to_moderate")
            add_age_smoothness_factors(knot_val_dict, "h_moderate_to_severe")
            add_age_smoothness_factors(knot_val_dict, "f_mild")
            add_age_smoothness_factors(knot_val_dict, "f_moderate")
            add_age_smoothness_factors(knot_val_dict, "f_severe")

            add_age_monotone_increasing_factor(knot_val_dict, "f_mild")
            add_age_monotone_increasing_factor(knot_val_dict, "f_moderate")
            add_age_monotone_increasing_factor(knot_val_dict, "f_severe")

            add_rate_ordering_factor(knot_val_dict, "f_moderate", smaller_name="f_mild")
            add_rate_ordering_factor(knot_val_dict, "f_severe", smaller_name="f_moderate")

            p_ad = at_param(
                f"p_ad",
                ages,
                years,
                knot_val_dict["p_ad"],
            )
            delta_BBBM = at_param(
                f"delta_BBBM",
                ages,
                years,
                knot_val_dict["delta_BBBM"],
            )
            delta_MCI = at_param(
                f"delta_MCI",
                ages,
                years,
                knot_val_dict["delta_MCI"],
            )
            delta_mild = at_param(
                f"delta_mild",
                ages,
                years,
                knot_val_dict["delta_mild"],
            )
            delta_moderate = at_param(
                f"delta_moderate",
                ages,
                years,
                knot_val_dict["delta_moderate"],
            )
            delta_severe = at_param(
                f"delta_severe",
                ages,
                years,
                knot_val_dict["delta_severe"],
            )

            # deltas-sum-to-one constraint
            # Add a normal penalty for difference between sum and 1.0
            knot_sum = (
                knot_val_dict["delta_BBBM"]
                + knot_val_dict["delta_MCI"]
                + knot_val_dict["delta_mild"]
                + knot_val_dict["delta_moderate"]
                + knot_val_dict["delta_severe"]
            )
            delta_sum = dist.Normal(1, 0.005).log_prob(knot_sum).sum()
            numpyro.factor(f"sum_to_one_factor", delta_sum)

            def p_ad_dementia(a, t):
                return p_ad(a, t) * (1 - delta_BBBM(a, t) - delta_MCI(a, t))

            data_model(
                "p_ad_dementia", p_ad_dementia, df_data.query("measure == 'p_ad_dementia'")
            )

            def frac_mild(a, t):
                return delta_mild(a, t) / (1 - delta_BBBM(a, t) - delta_MCI(a, t))

            data_model("frac_mild", frac_mild, df_data.query("measure == 'frac_mild'"))

            def frac_moderate(a, t):
                return delta_moderate(a, t) / (1 - delta_BBBM(a, t) - delta_MCI(a, t))

            data_model(
                "frac_moderate", frac_moderate, df_data.query("measure == 'frac_moderate'")
            )

            def frac_severe(a, t):
                return delta_severe(a, t) / (1 - delta_BBBM(a, t) - delta_MCI(a, t))

            data_model("frac_severe", frac_severe, df_data.query("measure == 'frac_severe'"))

            h_S_to_BBBM = at_param(
                f"h_S_to_BBBM",
                ages,
                years,
                knot_val_dict["h_S_to_BBBM"],
            )

            i_ad = at_param(f"i_ad", ages, years, knot_val_dict["i_ad"])
            data_model("i_ad", i_ad, df_data.query('measure == "i_ad_dementia"'))

            h_mild_to_moderate = at_param(
                f"h_mild_to_moderate",
                ages,
                years,
                knot_val_dict["h_mild_to_moderate"],
            )
            h_moderate_to_severe = at_param(
                f"h_moderate_to_severe",
                ages,
                years,
                knot_val_dict["h_moderate_to_severe"],
            )

            f_mild = at_param(
                f"f_mild",
                ages,
                years,
                knot_val_dict["f_mild"],
            )
            f_moderate = at_param(
                f"f_moderate",
                ages,
                years,
                knot_val_dict["f_moderate"],
            )
            f_severe = at_param(
                f"f_severe",
                ages,
                years,
                knot_val_dict["f_severe"],
            )

            def f(a, t):
                return (
                    f_mild(a, t) * frac_mild(a, t)
                    + f_moderate(a, t) * frac_moderate(a, t)
                    + f_severe(a, t) * frac_severe(a, t)
                )

            data_model("f", f, df_data.query('measure == "f"'))

            m = at_param(f"m", ages, years, knot_val_dict["m"])

            def m_all(a, t):
                # Population all-cause mortality: m * (1 - p_dementia) + (m + f) * p_dementia = m + f * p_dementia  # TODO: update eqn
                return m(a, t) + p_ad_dementia(a, t) * (
                    frac_mild(a, t) * f_mild(a, t)
                    + frac_moderate(a, t) * f_moderate(a, t)
                    + frac_severe(a, t) * f_severe(a, t)
                )

            data_model("m_all", m_all, df_data.query('measure == "m_all"'))

            include_consistency_constraints = True
            if include_consistency_constraints:
                sigma = 0.02  # TODO: consider effect of making this larger --- does it lead to more model uncertainty?

                def odf_function(t, y, args):
                    (
                        S,
                        BBBM,
                        MCI,
                        D_AD_mild,
                        D_AD_moderate,
                        D_AD_severe,
                        new_D_AD_mild,
                    ) = y
                    (
                        h_S_to_BBBM,
                        h_BBBM_to_MCI,
                        h_MCI_to_mild,
                        h_mild_to_moderate,
                        h_moderate_to_severe,
                        f_mild,
                        f_moderate,
                        f_severe,
                        f,
                        m,
                    ) = args

                    # fmt: off
                    return (
                        0 - m * S                                                               - h_S_to_BBBM * S,
                        0 - m * BBBM                                     - h_BBBM_to_MCI * BBBM + h_S_to_BBBM * S,
                        0 - m * MCI                - h_MCI_to_mild * MCI + h_BBBM_to_MCI * BBBM,
                        0 - (m+f_mild) * D_AD_mild + h_MCI_to_mild * MCI      - h_mild_to_moderate * D_AD_mild,
                        0 - (m+f_moderate) * D_AD_moderate                    + h_mild_to_moderate * D_AD_mild  \
                                                                                    - h_moderate_to_severe * D_AD_moderate,
                        0 - (m+f_severe) * D_AD_severe                              + h_moderate_to_severe * D_AD_moderate,
                        h_MCI_to_mild * MCI,
                    )
                    # fmt: on

                def ode_consistency_factor(at):
                    h_BBBM_to_MCI = 1 / BBBM_AVG_DURATION
                    h_MCI_to_mild = 1 / MCI_AVG_DURATION

                    a, t = at
                    dt = 5
                    term = ODETerm(odf_function)
                    solver = Dopri5()
                    saveat = SaveAt(t0=False, t1=True)

                    solution = diffeqsolve(
                        term,
                        solver,
                        t0=t,
                        t1=t + dt,
                        dt0=0.5,
                        y0=(
                            1 - p_ad(a, t),
                            p_ad(a, t) * delta_BBBM(a, t),
                            p_ad(a, t) * delta_MCI(a, t),
                            p_ad(a, t) * delta_mild(a, t),
                            p_ad(a, t) * delta_moderate(a, t),
                            p_ad(a, t) * delta_severe(a, t),
                            0,
                        ),
                        saveat=saveat,
                        args=[
                            h_S_to_BBBM(a, t),
                            h_BBBM_to_MCI,
                            h_MCI_to_mild,
                            h_mild_to_moderate(a, t),
                            h_moderate_to_severe(a, t),
                            f_mild(a, t),
                            f_moderate(a, t),
                            f_severe(a, t),
                            f(a, t),
                            m(a, t),
                        ],
                    )
                    (
                        S,
                        BBBM,
                        MCI,
                        D_mild,
                        D_moderate,
                        D_severe,
                        new_D_due_to_AD,
                    ) = solution.ys
                    # Numerical stability for log terms
                    eps = 1e-8
                    denom_alive = S + BBBM + MCI + D_mild + D_moderate + D_severe
                    denom_ad = BBBM + MCI + D_mild + D_moderate + D_severe
                    # Clip to avoid log(0)
                    r_bbbm = jnp.clip(BBBM / (denom_ad + eps), eps)
                    r_mci = jnp.clip(MCI / (denom_ad + eps), eps)
                    r_mild = jnp.clip(D_mild / (denom_ad + eps), eps)
                    r_moderate = jnp.clip(D_moderate / (denom_ad + eps), eps)
                    r_severe = jnp.clip(D_severe / (denom_ad + eps), eps)
                    r_prev_ad_dementia = jnp.clip(
                        (D_mild + D_moderate + D_severe) / (denom_alive + eps), eps
                    )
                    r_inc_ad_dementia = jnp.clip(
                        new_D_due_to_AD / (dt * (denom_alive + eps)), eps
                    )

                    sq_difference = 0.0
                    sq_difference += (
                        10
                        * (
                            jnp.log(r_bbbm)
                            - jnp.log(jnp.clip(delta_BBBM(a + dt, t + dt), eps))
                        )
                        ** 2
                    )
                    sq_difference += (
                        10
                        * (jnp.log(r_mci) - jnp.log(jnp.clip(delta_MCI(a + dt, t + dt), eps)))
                        ** 2
                    )
                    sq_difference += (
                        jnp.log(r_mild) - jnp.log(jnp.clip(delta_mild(a + dt, t + dt), eps))
                    ) ** 2
                    sq_difference += (
                        jnp.log(r_moderate)
                        - jnp.log(jnp.clip(delta_moderate(a + dt, t + dt), eps))
                    ) ** 2
                    sq_difference += (
                        jnp.log(r_severe)
                        - jnp.log(jnp.clip(delta_severe(a + dt, t + dt), eps))
                    ) ** 2
                    sq_difference += (
                        10
                        * (
                            jnp.log(r_prev_ad_dementia)
                            - jnp.log(jnp.clip(p_ad_dementia(a + dt, t + dt), eps))
                        )
                        ** 2
                    )
                    sq_difference += (
                        10
                        * (
                            jnp.log(r_inc_ad_dementia)
                            - jnp.log(jnp.clip(i_ad(a + dt / 2, t + dt / 2), eps))
                        )
                        ** 2
                    )
                    return jnp.sqrt(sq_difference)

                # Vectorize the ode_consistency_factor function
                ode_consistency_factors = jax.vmap(ode_consistency_factor)

                # Create a mesh grid of ages and years
                age_mesh, year_mesh = jnp.meshgrid(
                    jnp.array([a for a in ages if a >= 45]), jnp.array(years)
                )
                at_list = jnp.stack([age_mesh.ravel(), year_mesh.ravel()], axis=-1)

                # Compute ODE errors for all age-time combinations at once
                ode_errors = numpyro.deterministic(
                    f"ode_errors", ode_consistency_factors(at_list)
                )

                # Add a normal penalty for difference between solution and params
                log_pr = dist.Normal(0, sigma).log_prob(ode_errors).sum()
                numpyro.factor(f"ode_consistency_factor", log_pr)

        sampler = infer.MCMC(
            infer.NUTS(
                numpyro_model,
                init_strategy=numpyro.infer.init_to_value(
                    values={
                        f"p_ad": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"delta_BBBM": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"delta_MCI": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"delta_mild": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"delta_moderate": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"delta_severe": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"h_S_to_BBBM": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"i_ad": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"h_mild_to_moderate": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"h_moderate_to_severe": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"f_mild": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"f_moderate": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"f_severe": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"f": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"m": jnp.ones([len(ages), len(years)]) * 0.05,
                    }
                ),
            ),
            num_warmup=n_mcmc_samples,
            num_samples=n_mcmc_samples,
            num_chains=1,
            progress_bar=True,
        )

        sampler.run(
            jax.random.PRNGKey(0),
        )
        self.samples = sampler.get_samples()

    def get_rate(self, param, year):
        assert hasattr(self, "samples"), "Must run fit() first"

        rate_table = []
        for i, a in enumerate(self.ages):
            for j, t in enumerate(self.years):
                if year != t:
                    continue
                rate = self.samples[param][:, i, j]

                row = dict(
                    age_start=a,
                    age_end=a + 5,
                    year_start=t,
                    year_end=t + 1,
                    sex=self.sex,
                )
                for k, r_k in enumerate(rate):
                    row[f"draw_{k}"] = float(r_k)
                rate_table.append(row)
        return pd.DataFrame(rate_table).set_index(
            ["sex", "age_start", "age_end", "year_start", "year_end"]
        )


def add_age_smoothness_factors(
    knot_val_dict, name, sigma_first=0.2, sigma_second=0.1, EPS=1e-8
):
    """Add first- and second-difference smoothness priors along age."""
    vals = knot_val_dict[name]  # shape (n_age, n_year)

    # Work on the log scale to respect positivity and relative changes.
    log_vals = jnp.log(vals + EPS)

    # First differences along age (axis 0).
    d1 = jnp.diff(log_vals, axis=0)  # shape (n_age-1, n_year)

    # Second differences along age: diff of first differences.
    d2 = jnp.diff(d1, axis=0)  # shape (n_age-2, n_year)

    # Penalize both sets of differences toward 0.
    numpyro.factor(
        f"{name}_age_first_diff",
        dist.Normal(0.0, sigma_first).log_prob(d1).sum(),
    )
    numpyro.factor(
        f"{name}_age_second_diff",
        dist.Normal(0.0, sigma_second).log_prob(d2).sum(),
    )


def add_age_monotone_increasing_factor(
    knot_val_dict: dict[str, jnp.ndarray],
    name: str,
    sigma: float = 0.1,
    use_log: bool = True,
    EPS=1e-8,
):
    """
    Encourage the age-pattern of `knot_val_dict[name]` to be non-decreasing
    in age (axis=0), for each year independently.

    Monotonicity is enforced on the log scale by default:
        log(vals(age+1)) - log(vals(age)) >= 0

    Any negative differences are penalized quadratically.

    Parameters
    ----------
    name : str
        Key in knot_val_dict.
    knot_val_dict : dict[str, jnp.ndarray]
        Dict of parameter fields, each shape (n_age, n_year).
    sigma : float
        Strength/scale of the penalty. Smaller = stronger belief that
        the curve is monotone increasing.
    use_log : bool
        If True, impose monotonicity on log(vals); otherwise on vals.
    """
    vals = knot_val_dict[name]  # shape (n_age, n_year)

    if use_log:
        vals = jnp.log(vals + EPS)

    # First differences along age: age_{a+1} - age_a
    diffs = jnp.diff(vals, axis=0)  # shape (n_age-1, n_year)

    # Amount of violation: how far below zero each difference is.
    # If diffs >= 0 → violations = 0 (no penalty)
    # If diffs < 0  → violations = -diffs > 0
    violations = jnp.maximum(-diffs, 0.0)

    # Simple quadratic penalty: ~ Normal(0, sigma) on violations,
    # but without bothering with constants.
    penalty = -0.5 * jnp.square(violations / sigma).sum()

    numpyro.factor(f"{name}_age_monotone_increasing", penalty)


def add_rate_ordering_factor(
    knot_val_dict: dict[str, jnp.ndarray],
    larger_name: str,
    smaller_name: str,
    sigma: float = 0.1,
    use_log: bool = True,
    margin: float = 0.0,
    EPS=1e-8,
):
    """
    Encourage knot_val_dict[larger_name] >= knot_val_dict[smaller_name]
    (optionally on the log scale), pointwise over age & year.

    Any violations are penalized quadratically.

    Parameters
    ----------
    larger_name : str
        Key for the field that should be >= the other (e.g. "f_severe").
    smaller_name : str
        Key for the field that should be <= the other (e.g. "f_mild").
    knot_val_dict : dict[str, jnp.ndarray]
        Dict of parameter fields, each shape (n_age, n_year).
    sigma : float
        Scale of the penalty. Smaller = stronger belief in the inequality.
    use_log : bool
        If True, apply inequality in log-space (i.e., multiplicative order).
    margin : float
        Optional strictness margin in the same scale you’re working in:
        - If use_log=True, margin is in log units (e.g. log(1.1) for 10%).
        - If use_log=False, margin is in raw units.
        The prior then prefers larger >= smaller + margin.
    """
    a = knot_val_dict[larger_name]
    b = knot_val_dict[smaller_name]

    if use_log:
        a = jnp.log(a + EPS)
        b = jnp.log(b + EPS)

    # We want: a >= b + margin
    diffs = a - (b + margin)  # shape (n_age, n_year)

    # Violations where a < b + margin
    violations = jnp.maximum(-diffs, 0.0)

    # Quadratic penalty on violations
    penalty = -0.5 * jnp.square(violations / sigma).sum()

    numpyro.factor(f"{larger_name}_ge_{smaller_name}", penalty)


def at_param(name: str, ages, years, knot_val) -> Callable:
    """Create an age- and time-specific rate function for a DisMod model.

    This function generates a piecewise-constant 2d-interpolated rate function
    (the prior on knot values is passed as input).

    It uses `searchsorted` as an efficient piecewise constant
    interpolation method.

    Parameters
    ----------
    name : str
        The name of the rate parameter.
    ages : array-like
    years : array-like
    knot_val : array-like of priors with rows for ages and columns for years

    Returns
    -------
    Callable
        A function `f(a, t)` that takes array-like values age `a`
        and time `t` as inputs and returns the interpolated rate value
        specific to the given age and time.

    Notes
    -----
    - For constant interpolation, the `searchsorted` method is set to
      'scan', which is allegedly efficient for GPU computation.

    """

    def f(a: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        method = "scan"  # 'scan_unrolled' can be more performant on GPU at the
        # expense of additional compile time
        a_index = jnp.searchsorted(
            jnp.asarray(ages[1:]),  # Start from ages[1]
            # so that ages less than ages[1] map to
            # index 0
            jnp.asarray(a),
            method=method,
            side="right",
        )
        t_index = jnp.searchsorted(
            jnp.asarray(years[1:]),  # Start from years[1]
            # so that years less than years[1] map to
            # index 0
            jnp.asarray(t),
            method=method,
            side="right",
        )
        return knot_val[a_index, t_index]

    return f


def data_model(
    name: str, f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], df_data: pd.DataFrame
):
    assert len(df_data) > 0, f"{name} should have some data"

    ages = jnp.array(0.5 * (df_data.age_start + df_data.age_end))
    years = jnp.array(0.5 * (df_data.year_start + df_data.year_end))
    rate_obs_loc = jnp.array(df_data["mean"])
    rate_obs_scale = jnp.array(df_data["standard_error"])

    rate_pred = f(ages, years)

    rate_obs = numpyro.sample(
        f"{name}_obs", dist.Normal(loc=rate_pred, scale=rate_obs_scale), obs=rate_obs_loc
    )
    return rate_obs


def transform_to_data(
    param: str, df_in: pd.DataFrame, sex: str, ages: Iterable[int], years: Iterable[int]
) -> pd.DataFrame:
    """Convert artifact data to a format suitable for DisMod-AT-NumPyro."""
    t = df_in.loc[sex]
    results = []  # fill with rows of data, then convert to a dataframe

    for a in ages:
        for y in years:
            # Note: age_end and year_end here are meta fields used only for midpoints
            # computation downstream; they do not affect which rows are selected from
            # the artifact (selection is done via the query below).
            row = {
                "age_start": a + 2.5,
                "age_end": a + 2.5,
                "year_start": y + 0.5,
                "year_end": y + 0.5,
                "sex": sex,
                "measure": param,
            }
            tt = t.query(
                "age_start <= @a and @a < age_end and year_start <= @y and @y < year_end"
            )
            assert len(tt) == 1
            row["mean"] = np.mean(tt.iloc[0])
            row["standard_error"] = (
                np.std(tt.iloc[0]) + 1e-8  # small epsilon to avoid zero standard errors
            )

            results.append(row)

    return pd.DataFrame(results)


def write_or_replace(art: Artifact, key: str, data: pd.DataFrame):
    # Use artifact containment check for compatibility
    if key in art:
        art.replace(key, data)
    else:
        art.write(key, data)


def generate_consistent_data_for_population_components(art):
    """use the cause.alzheimers_consistent.population_incidence_any and
    the time-varying population to create a
    SUSCEPTIBLE_TO_BBBM_TRANSITION_COUNT value

    also copy data_keys.ALZHEIMERS_CONSISTENT.AD_PREVALENCE to
    data_keys.POPULATION.SCALING_FACTOR because that is where the
    Population component expects to find it

    """
    prevalence = art.load(data_keys.ALZHEIMERS_CONSISTENT.AD_PREVALENCE)
    write_or_replace(art, data_keys.POPULATION.SCALING_FACTOR, prevalence)

    pop = art.load(data_keys.POPULATION.STRUCTURE)

    transition_rate = art.load(data_keys.ALZHEIMERS_CONSISTENT.BBBM_AD_INCIDENCE_RATE)

    df = pd.merge(
        pop.reset_index().drop(["location"], axis=1),
        transition_rate.reset_index().drop(["year_start", "year_end"], axis=1),
        on=["sex", "age_start", "age_end"],
        suffixes=("", "_rate"),
    ).set_index(["sex", "age_start", "age_end", "year_start", "year_end"])

    df = pd.merge(
        df.reset_index(),
        prevalence.reset_index().drop(["year_start", "year_end"], axis=1),
        on=["sex", "age_start", "age_end"],
        suffixes=("", "_prev"),
    ).set_index(["sex", "age_start", "age_end", "year_start", "year_end"])

    for i in range(min(n_mcmc_samples, 1_000)):
        df[f"draw_{i}"] *= df[f"draw_{i}_rate"] * (1 - df[f"draw_{i}_prev"])
        del df[f"draw_{i}_rate"]
        del df[f"draw_{i}_prev"]

    write_or_replace(
        art, data_keys.ALZHEIMERS_CONSISTENT.SUSCEPTIBLE_TO_BBBM_TRANSITION_COUNT, df
    )


def generate_consistent_data_for_disease_components(art):
    ad_prevalence = art.load(data_keys.ALZHEIMERS_CONSISTENT.AD_PREVALENCE)
    emr_mild = art.load(data_keys.ALZHEIMERS_CONSISTENT.EMR_MILD)
    emr_moderate = art.load(data_keys.ALZHEIMERS_CONSISTENT.EMR_MODERATE)
    emr_severe = art.load(data_keys.ALZHEIMERS_CONSISTENT.EMR_SEVERE)
    delta_mild = art.load(
        data_keys.ALZHEIMERS_CONSISTENT.MILD_DEMENTIA_CONDITIONAL_PREVALENCE
    )
    delta_moderate = art.load(
        data_keys.ALZHEIMERS_CONSISTENT.MODERATE_DEMENTIA_CONDITIONAL_PREVALENCE
    )
    delta_severe = art.load(
        data_keys.ALZHEIMERS_CONSISTENT.SEVERE_DEMENTIA_CONDITIONAL_PREVALENCE
    )

    csmr = ad_prevalence * (
        emr_mild * delta_mild + emr_moderate * delta_moderate + emr_severe * delta_severe
    )
    write_or_replace(art, data_keys.ALZHEIMERS_CONSISTENT.CSMR, csmr)


if __name__ == "__main__":
    fname = "sweden.hdf"
    print("updating", fname)
    art = Artifact(fname)
    generate_consistent_rates(art)
    generate_consistent_data_for_population_components(art)
    generate_consistent_data_for_disease_components(art)
