"""NumPyro/JAX implementation of DisMod-AT, refactored from notebook 2024_07_28a_dismod_ipd_ai_refactor.ipynb
"""
from typing import Callable, Iterable

import interpax
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from numpyro import distributions as dist
from numpyro import infer
from vivarium.framework.artifact import Artifact

def transform_to_data(param: str, df_in: pd.DataFrame, sex: str, ages: Iterable[int], years: Iterable[int]) -> pd.DataFrame:
    """Convert artifact data to a format suitable for DisMod-AT-NumPyro."""
    t = df_in.loc[sex]
    results = []  # fill with rows of data, then convert to a dataframe

    for a in ages:
        for y in years:
            # Note: age_end and year_end here are meta fields used only for midpoints
            # computation downstream; they do not affect which rows are selected from
            # the artifact (selection is done via the query below).
            row = {
                "age_start": a,
                "age_end": a,
                "year_start": y,
                "year_end": y,
                "sex": sex,
                "measure": param,
            }
            tt = t.query(
                "age_start <= @a and @a < age_end and year_start <= @y and @y < year_end"
            )
            assert len(tt) == 1
            row["mean"] = np.mean(tt.iloc[0])
            row["standard_error"] = (
                np.std(tt.iloc[0])
                + 1e-8  # small epsilon to avoid zero standard errors
            )

            results.append(row)

    return pd.DataFrame(results)


def transform_to_prior(df, sex, ages, years, location):
    """Convert artifact data to a format suitable for DisMod-AT-NumPyro."""
    t = df.loc[(location, sex)]
    mu = pd.DataFrame(index=ages, columns=years, dtype=float)
    s2 = pd.DataFrame(index=ages, columns=years, dtype=float)

    for a in ages:
        for y in years:
            tt = t.query(
                "age_start <= @a and @a < age_end and year_start <= @y and @y < year_end"
            )
            assert len(tt) == 1
            mu.loc[a, y] = np.mean(tt.iloc[0])
            s2.loc[a, y] = np.var(tt.iloc[0])

    return mu, s2


def at_param(name: str, ages, years, knot_val, method: str = "constant") -> Callable:
    """Create an age- and time-specific rate function for a DisMod model.

    This function generates a 2d-interpolated rate function
    (the prior on knot values is defined elsewhere).

    It uses `searchsorted` as an efficient piecewise constant
    interpolation method.

    Parameters
    ----------
    name : str
        The name of the rate parameter.
    ages : array-like
    years : array-like
    knot_val : array-like with rows for ages and columns for years
    method : str, interpolation method of "constant" or "linear"

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

    if method == "constant":

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

    elif method == "linear":
        f = interpax.Interpolator2D(ages, years, knot_val, method="linear", extrap=True)
    else:
        assert 0, f'Method "{method}" unrecognized, should be "constant" or "linear"'
    return f


def data_model(name: str, f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], df_data: pd.DataFrame):
    assert len(df_data) > 0, f'{name} should have some data'

    ages = jnp.array(0.5 * (df_data.age_start + df_data.age_end))
    years = jnp.array(0.5 * (df_data.year_start + df_data.year_end))
    rate_obs_loc = jnp.array(df_data["mean"])
    rate_obs_scale = jnp.array(df_data["standard_error"])

    rate_pred = f(ages, years)

    rate_obs = numpyro.sample(
        f"{name}_obs", dist.Normal(loc=rate_pred, scale=rate_obs_scale), obs=rate_obs_loc
    )
    return rate_obs


def ode_model(group: str,
              p: Callable, i: Callable, delta_BBBM: Callable, delta_MCI: Callable,
              h_S_to_BBBM: Callable, f: Callable, m: Callable,
              sigma: float, ages, years):
    def dismod_f(t, y, args):
        S, BBBM, MCI, D, new_D = y
        h_S_to_BBBM, h_BBBM_to_MCI, h_MCI_to_dementia, f, m = args
        return (
            0 - m * S                                                       - h_S_to_BBBM * S,
            0 - m * BBBM                             - h_BBBM_to_MCI * BBBM + h_S_to_BBBM * S,
            0 - m * MCI    - h_MCI_to_dementia * MCI + h_BBBM_to_MCI * BBBM,
            0 - (m+f) * D  + h_MCI_to_dementia * MCI,
            h_MCI_to_dementia * MCI,
        )

    def ode_consistency_factor(at):
        h_BBBM_to_MCI = 1/5 # FIXME
        h_MCI_to_dementia = 1/5 # FIXME
        
        a, t = at
        dt = 5
        term = ODETerm(dismod_f)
        solver = Dopri5()
        saveat = SaveAt(t0=False, t1=True)

        y0 = (
            1 - p(a, t),
            p(a, t) * delta_BBBM(a, t), 
            p(a, t) * delta_MCI(a, t),
            p(a, t) * (1 - delta_BBBM(a, t) - delta_MCI(a, t)),
            0
        )
        solution = diffeqsolve(
            term,
            solver,
            t0=t,
            t1=t + dt,
            dt0=0.5,
            y0=y0,
            saveat=saveat,
            args=[
                h_S_to_BBBM(a, t),
                h_BBBM_to_MCI,
                h_MCI_to_dementia,
                f(a, t),
                m(a, t)
            ],
        )

        S, BBBM, MCI, D, new_D = solution.ys
        # Numerical stability for log terms
        eps = 1e-12
        denom_alive = S + BBBM + MCI + D
        denom_noS = BBBM + MCI + D
        # Clip to avoid log(0)
        r_bbbm = jnp.clip(BBBM / (denom_noS + eps), eps)
        r_mci = jnp.clip(MCI / (denom_noS + eps), eps)
        r_prev = jnp.clip(D / (denom_alive + eps), eps)
        r_inc = jnp.clip(new_D / (dt * (denom_alive + eps)), eps)

        difference = 0.0
        difference += jnp.log(r_bbbm) - jnp.log(jnp.clip(delta_BBBM(a + dt, t + dt), eps))
        difference += jnp.log(r_mci) - jnp.log(jnp.clip(delta_MCI(a + dt, t + dt), eps))
        difference += jnp.log(r_prev) - jnp.log(jnp.clip(p(a + dt, t + dt), eps))
        difference += jnp.log(r_inc) - jnp.log(jnp.clip(i(a + dt/2, t + dt/2), eps))
        return difference

    # Vectorize the ode_consistency_factor function
    ode_consistency_factors = jax.vmap(ode_consistency_factor)

    # Create a mesh grid of ages and years
    age_mesh, year_mesh = jnp.meshgrid(jnp.array(ages[:-1]),
                                       jnp.array(years[:-1]))
    at_list = jnp.stack([age_mesh.ravel(), year_mesh.ravel()], axis=-1)

    # Compute ODE errors for all age-time combinations at once
    ode_errors = numpyro.deterministic(
        f"ode_errors_{group}", ode_consistency_factors(at_list)
    )

    # Add a normal penalty for difference between solution and params
    log_pr = dist.Normal(0, sigma).log_prob(ode_errors).sum()
    numpyro.factor(f"ode_consistency_factor_{group}", log_pr)


class ConsistentModel:
    def __init__(self, sex, ages, years):
        self.sex = sex
        self.ages = ages
        self.years = years

    def fit(self, df_data):
        # expect this to take about 2 minutes to run
        group = ""
        ages, years = self.ages, self.years
        location = ""
        sex = self.sex

        def model():
            knot_val_dict = {}
            for param in ["p", "delta_BBBM", "delta_MCI", "h_S_to_BBBM", "i", "f", "m"]:
                knot_val_dict[param] = numpyro.sample(
                    f"{param}_{group}",
                    dist.TruncatedNormal(
                        loc=jnp.zeros((len(ages), len(years))),
                        scale=jnp.ones((len(ages), len(years))),
                        low=0.0,
                        high=1.0,
                    ),
                )

            p = at_param(
                f"p",
                ages,
                [2025],
                knot_val_dict["p"],
                method="constant",
            )
            delta_BBBM = at_param(
                f"delta_BBBM",
                ages,
                [2025],
                knot_val_dict["delta_BBBM"],
                method="constant",
            )
            delta_MCI = at_param(
                f"delta_MCI",
                ages,
                [2025],
                knot_val_dict["delta_MCI"],
                method="constant",
            )

            def p_dementia(a, t):
                return p(a, t) * jnp.clip(1 - delta_BBBM(a, t) - delta_MCI(a, t), 0, 1)

            data_model("p_dementia", p_dementia, df_data.query("measure == 'p_dementia'"))

            h_S_to_BBBM = at_param(
                f"h_S_to_BBBM",
                ages,
                [2025],
                knot_val_dict["h_S_to_BBBM"],
                method="constant",
            )

            i = at_param(
                f"i", ages, years, knot_val_dict["i"]
            )
            data_model("i", i, df_data.query('measure == "i_dementia"'))
            
            f = at_param(
                f"f", ages, years, knot_val_dict["f"],
            )
            data_model('f', f, df_data.query('measure == "f"'))
            m = at_param(
                f"m", ages, years, knot_val_dict["m"]
            )
            def m_all(a, t):
                # Population all-cause mortality: m * (1 - p_dementia) + (m + f) * p_dementia = m + f * p_dementia
                return m(a, t) + f(a, t) * p_dementia(a, t)
            data_model("m_all", m_all, df_data.query('measure == "m_all"'))

            include_consistency_constraints = True
            if include_consistency_constraints:
                ode_model(group, 
                          p, i, delta_BBBM, delta_MCI, h_S_to_BBBM, f, m,
                          sigma=0.01, ages=ages, years=years)


        sampler = infer.MCMC(
            infer.NUTS(
                model,
                init_strategy=numpyro.infer.init_to_value(
                    values={
                        f"p_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"delta_BBBM_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"delta_MCI_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"h_S_to_BBBM_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"i_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"f_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"m_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                    }
                ),
            ),
            num_warmup=1_000,
            num_samples=1_000,
            num_chains=1,
            progress_bar=True,
        )

        sampler.run(
            jax.random.PRNGKey(0),
        )
        self.samples = sampler.get_samples()

    def get_rate(self, param, year):
        assert hasattr(self, "samples"), "Must run fit() first"
        group = ""

        rate_table = []
        for i, a in enumerate(self.ages):
            for j, t in enumerate(self.years):
                if year != t:
                    continue
                rate = self.samples[f"{param}_{group}"][:, i, j]

                row = dict(
                    age_start=a,
                    age_end=a + 5,
                    year_start=t,
                    year_end=t + 1,
                    sex=self.sex,
                )
                for i, r_i in enumerate(rate):
                    row[f"draw_{i}"] = float(r_i)
                rate_table.append(row)
        return pd.DataFrame(rate_table).set_index(
            ["sex", "age_start", "age_end", "year_start", "year_end"]
        )


def generate_consistent_rates(art: Artifact, location: str):
    """Generates consistent rates for sim

    Parameters
    ----------
    art
        The artifact to read from and write to.
    location
        The location associated with the data to load and the artifact to
        write to.
    years
        The years to load data for.

    """
    # TODO: check if the consistent rates are already in the artifact, and if so, skip rest of this function
    ages = np.arange(5, 96, 5)
    years = [2025, 2030] # np.arange(2025, 2051, 5)
    sexes = ["Male", "Female"]
    load_key = {
        "p_dementia": "cause.alzheimers.prevalence",
        "i_dementia": "cause.alzheimers.population_incidence_rate",
        "f": "cause.alzheimers_disease_and_other_dementias.excess_mortality_rate",
        "m_all": "cause.all_causes.cause_specific_mortality_rate",
    }
    save_key = {
        "h_S_to_BBBM": "cause.alzheimers_consistent.population_incidence_rate",
        "p": "cause.alzheimers_consistent.prevalence",
        "i": "cause.alzheimers_consistent.incidence",
        "f": "cause.alzheimers_consistent.excess_mortality_rate",
        "delta_BBBM": "cause.alzheimers_consistent.bbbm_conditional_prevalence",
        "delta_MCI": "cause.alzheimers_consistent.mci_conditional_prevalence",
    }

    def etl_data(sex):
        df_data = pd.concat(
            [
                transform_to_data("p_dementia", art.load(load_key["p_dementia"]), sex, ages, [2023]),
                transform_to_data("i_dementia", art.load(load_key["i_dementia"]), sex, ages, [2023]),
                transform_to_data("f", art.load(load_key["f"]), sex, ages, [2021]),
                transform_to_data("m_all", art.load(load_key["m_all"]), sex, ages, years),
            ]
        )

        for key in load_key.keys():
            assert key in set(df_data.measure)

        return df_data

    def get_rates(model_dict, rate_type, year):
        df_out = []
        for model in model_dict.values():
            df_out.append(model.get_rate(rate_type, year))
        df_out = pd.concat(df_out)
        return df_out

    # fit model separately for Male and Female
    m = {}
    for sex in sexes:
        m[sex] = ConsistentModel(sex, ages, years)
        m[sex].fit(etl_data(sex))

    # store consistent rates in artifact
    for rate_type in save_key.keys():
        # generate data for this rate_type
        df_out = get_rates(m, rate_type, 2025)
        # store generated data in artifact
        rate_name = save_key[rate_type]
        write_or_replace(art, rate_name, df_out)


def write_or_replace(art: Artifact, key: str, data: pd.DataFrame):
    # Use artifact containment check for compatibility
    if key in art:
        art.replace(key, data)
    else:
        art.write(key, data)


if __name__ == "__main__":
    location = "United States of America"
    art = Artifact("united_states_of_america.hdf")
    generate_consistent_rates(art, location)
