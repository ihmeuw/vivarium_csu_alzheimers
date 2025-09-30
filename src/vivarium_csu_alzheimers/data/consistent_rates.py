"""NumPyro/JAX implementation of DisMod-AT, refactored from notebook 2024_07_28a_dismod_ipd_ai_refactor.ipynb
"""
from typing import Callable

import interpax
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from numpyro import distributions as dist
from numpyro import infer
from vivarium import Artifact

def transform_to_data(param, df_in, sex, ages, years):
    """Convert artifact data to a format suitable for DisMod-AT-NumPyro."""
    t = df_in.loc[sex]
    results = []  # fill with rows of data, then convert to a dataframe

    for a in ages:
        for y in years:
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
                + 0.00000001  # add a little bit to the s2 for everything, just to avoid zeros
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


def at_param(name: str, ages, years, knot_val, method="constant") -> Callable:
    """Create an age- and time-specific rate function for a DisMod model.

    This function generates a 2d-interpolated rate function
    with a TruncatedNormal prior.

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
    function
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


def data_model(name, f, df_data):  # FIXME: expose beta so it can be common across locations
    if len(df_data) == 0:
        return

    ages = jnp.array(0.5 * (df_data.age_start + df_data.age_end))
    years = jnp.array(0.5 * (df_data.year_start + df_data.year_end))
    rate_obs_loc = jnp.array(df_data["mean"])
    rate_obs_scale = jnp.array(df_data["standard_error"])

    # refactor the following to use jax.vmap and to include population weights
    rate_pred = jnp.zeros(len(df_data))
    n_points = 5
    for alpha in np.linspace(0, 1, n_points):
        ages = jnp.array(alpha * df_data.age_start + (1 - alpha) * df_data.age_end)
        rate_pred += f(ages, years)
    rate_pred /= n_points

    if len(df_data.filter(like="x_").columns) != 0:
        # include fixed effects
        X = jnp.array(df_data.filter(like="x_").fillna(0))
        beta = numpyro.sample(
            f"{name}_beta",
            dist.Normal(loc=jnp.zeros(len(X[0])), scale=1.0),
        )
        rate_pred = jnp.exp(jnp.dot(X, beta)) * rate_pred

    rate_obs = numpyro.sample(
        f"{name}_obs", dist.Normal(loc=rate_pred, scale=rate_obs_scale), obs=rate_obs_loc
    )
    return rate_obs


def at_param_w_data(param, ages, years, knot_val, df_data, method="constant"):
    rate_function = at_param(param, ages, years, knot_val, method)
    rate_data_obs = data_model(param, rate_function, df_data)
    return rate_function


def group_name(sex, location):
    return f"{sex}_{location}".replace(" ", "_").lower()


def single_location_model(
    group,
    sex,
    location,
    ages,
    years,
    knot_val_dict,
    df_data,
    include_consistency_constraints=True,
):
    group = group_name(sex, location)
    i = at_param_w_data(
        f"i_{group}", ages, years, knot_val_dict["i"], df_data[df_data.measure == "i"]
    )
    r = at_param_w_data(
        f"r_{group}", ages, years, knot_val_dict["r"], df_data[df_data.measure == "r"]
    )
    ti = at_param_w_data(
        f"ti_{group}", ages, years, knot_val_dict["ti"], df_data[df_data.measure == "ti"]
    )
    ts = at_param_w_data(
        f"ts_{group}", ages, years, knot_val_dict["ts"], df_data[df_data.measure == "ts"]
    )
    tf = at_param_w_data(
        f"tf_{group}", ages, years, knot_val_dict["tf"], df_data[df_data.measure == "tf"]
    )
    f = at_param_w_data(
        f"f_{group}", ages, years, knot_val_dict["f"], df_data[df_data.measure == "f"]
    )
    m = at_param_w_data(
        f"m_{group}", ages, years, knot_val_dict["m"], df_data[df_data.measure == "m"]
    )

    p = at_param_w_data(
        f"p_{group}",
        ages,
        years,
        knot_val_dict["p"],
        df_data[df_data.measure == "p"],
        method="constant",
    )

    tx = at_param_w_data(
        f"tx_{group}",
        ages,
        years,
        knot_val_dict["tx"],
        df_data[df_data.measure == "tx"],
        method="constant",
    )

    if include_consistency_constraints:
        ode_model(group, p, tx, i, r, ti, ts, tf, f, m, sigma=0.01, ages=ages, years=years)
    return dict(p=p, i=i, f=f, m=m, r=r, ti=ti, ts=ts, tf=tf)


def ode_model(group, p, tx, i, r, ti, ts, tf, f, m, sigma, ages, years):
    def dismod_f(t, y, args):
        S, C, T = y
        i, r, ti, ts, tf, f, m = args
        return (
            0 - m * S - i * S + r * C + ts * T,
            0 - m * C - f * C + i * S - r * C - ti * C + tf * T,
            0 - m * T - f * T + ti * C - ts * T - tf * T,
        )

    def ode_consistency_factor(at):
        a, t = at
        dt = 5
        term = ODETerm(dismod_f)
        solver = Dopri5()
        saveat = SaveAt(t0=False, t1=True)

        y0 = (1 - p(a, t), p(a, t) * (1 - tx(a, t)), p(a, t) * tx(a, t))
        solution = diffeqsolve(
            term,
            solver,
            t0=t,
            t1=t + dt,
            dt0=0.5,
            y0=y0,
            saveat=saveat,
            args=[i(a, t), r(a, t), ti(a, t), ts(a, t), tf(a, t), f(a, t), m(a, t)],
        )

        S, C, T = solution.ys
        difference = jnp.log((C + T) / (S + C + T)) - jnp.log(p(a + dt, t + dt))
        difference += jnp.log(T / (T + C)) - jnp.log(tx(a + dt, t + dt))
        return difference

    # Vectorize the ode_consistency_factor function
    ode_consistency_factors = jax.vmap(ode_consistency_factor)

    # Create a mesh grid of ages and years
    age_mesh, year_mesh = jnp.meshgrid(ages[:-1], years[:-1])
    at_list = jnp.stack([age_mesh.ravel(), year_mesh.ravel()], axis=-1)

    # Compute ODE errors for all age-time combinations at once
    ode_errors = numpyro.deterministic(
        f"ode_errors_{group}", ode_consistency_factors(at_list)
    )

    # Add a normal penalty for difference between solution and SC
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
            for param in ["p", "tx", "i", "f", "m", "r", "ti", "tf", "ts"]:
                knot_val_dict[param] = numpyro.sample(
                    f"{param}_{group}",
                    dist.TruncatedNormal(
                        loc=jnp.zeros((len(ages), len(years))),
                        scale=jnp.ones((len(ages), len(years))),
                        low=0.0,
                    ),
                )
            # TODO: consider moving knots of p to midpoints
            rate_functions = single_location_model(
                group,
                sex,
                location,
                ages,
                years,
                knot_val_dict,
                df_data,
                include_consistency_constraints=True,
            )

        sampler = infer.MCMC(
            infer.NUTS(
                model,
                init_strategy=numpyro.infer.init_to_value(
                    values={
                        f"p_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"tx_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"i_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"f_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"m_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"r_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"ti_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"ts_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
                        f"tf_{group}": jnp.ones([len(ages), len(years)]) * 0.05,
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
    years = np.arange(2025, 2051, 5)
    sexes = ["Male", "Female"]
    key = {
        "i": "cause.opioid_use_disorders.incidence_rate",
        "p": "cause.alzheimers.prevalence",
        "f": "cause.opioid_use_disorders.excess_mortality_rate",
        "m_all": "cause.all_causes.cause_specific_mortality_rate",
        "csmr_with": "cause.opioid_use_disorders.cause_specific_mortality_rate",
        "pop": "population.structure",
    }

    def etl_data(sex):
        df_data = pd.concat(
            [
                transform_to_data("p", art.load(key["p"]), sex, ages, [2023]),
                # transform_to_data("i", art.load(key["i"]), sex, ages, [2021]),
                # transform_to_data("f", art.load(key["f"]), sex, ages, [2021]),
                # transform_to_data(
                #     "m",
                #     art.load(key["m_all"]) - art.load(key["csmr_with"]),
                #     sex,
                #     ages,
                #     [2021],
                # ),
                # transform_to_data("ti", utils.generate_constant_data(0.0), sex, ages, [2021]), 
                # transform_to_data("ts", utils.generate_constant_data(0.0), sex, ages, [2021]), 
                # transform_to_data("tf", utils.generate_constant_data(1.0), sex, ages, [2021]), 
                # transform_to_data("tx", utils.generate_constant_data(0.0), sex, ages, [2021]), 
            ]
        )
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
        breakpoint()

        m[sex] = ConsistentModel(sex, ages, years)
        m[sex].fit(etl_data(sex))

    # store consistent rates in artifact
    for rate_type in ["i", "p_bbbm", "p_mci", "p_ad"]:
        # generate data for k
        df_out = get_rates(m, rate_type, 2020)
        # store generated data in artifact
        rate_name = key[rate_type]
        rate_name = rate_name.replace("alzheimers", "ad_consistent")
        write_or_replace(art, rate_name, df_out)


def write_or_replace(art, key, data):
    if key in art.keys:
        art.replace(key, data)
    else:
        art.write(key, data)


if __name__ == "__main__":
    location = "United States of America"
    art = Artifact("united_states_of_america.hdf")
    generate_consistent_rates(art, location)
