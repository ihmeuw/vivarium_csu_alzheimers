"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from gbd_mapping import causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import gbd
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import interface
from vivarium_inputs import utilities as vi_utils
from vivarium_inputs import utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_csu_alzheimers.constants import data_keys


def get_data(
    lookup_key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        data_keys.POPULATION.LOCATION: load_population_location,
        data_keys.POPULATION.STRUCTURE: load_population_structure,
        data_keys.POPULATION.AGE_BINS: load_age_bins,
        data_keys.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        data_keys.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        data_keys.POPULATION.ACMR: load_standard_data,
        # TODO - confirm population data with nathaniel
        data_keys.POPULATION.LIVE_BIRTH_RATE: load_standard_data,
        data_keys.ALZHEIMERS.PREVALENCE_SCALE_FACTOR: load_alzheimers_prevalence,
        data_keys.ALZHEIMERS.PREVALENCE: load_alzheimers_prevalence,
        data_keys.ALZHEIMERS.INCIDENCE_RATE: load_standard_data,
        data_keys.ALZHEIMERS.CSMR: load_standard_data,
        data_keys.ALZHEIMERS.EMR: load_standard_data,
        data_keys.ALZHEIMERS.DISABLIITY_WEIGHT: load_standard_data,
        data_keys.ALZHEIMERS.RESTRICTIONS: load_metadata,
    }
    return mapping[lookup_key](lookup_key, location, years)


def load_population_location(
    key: str, location: str, years: int | str | list[int] | None = None
) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f"Unrecognized key {key}")

    return location


def table_from_nc(fname_dict, param, loc_id, loc_name, age_mapping):
    ds = xr.open_dataset(
        fname_dict[param],
        engine="netcdf4",  # let xarray auto-detect; list here if you know it
        decode_cf=True,  # handle CF-conventions (time units, etc.)
    )

    # Select relevant part of FHS dataset
    var_name = param
    if param == "births":
        var_name = "population"
    elif param in ["migration", "mortality"]:
        var_name = "value"

    pop_ts = ds[var_name].sel(
        location_id=loc_id,
    )

    if param != "migration":
        pop_ts = pop_ts.isel(scenario=0)

    pop_ts = pop_ts.squeeze(drop=True)  # remove now-singleton dims

    df = pop_ts.to_dataframe(name="value").reset_index()

    # Transform to vivarium format
    # 1. Convert location_id to location name
    df["location"] = loc_name

    # 2. Convert sex_id to sex names
    sex_mapping = {
        1: "Male",
        2: "Female",
    }
    df["sex"] = df["sex_id"].map(sex_mapping)

    # 3. Convert age_group_id to age intervals
    if param != "births":
        age_bins = age_mapping.set_index("age_group_id")
        df["age_start"] = np.round(df["age_group_id"].map(age_bins["age_start"]), 3)
        df["age_end"] = np.round(df["age_group_id"].map(age_bins["age_end"]), 3)
        age_cols = ["age_start", "age_end"]
    else:
        age_cols = []

    # 4. Convert year_id to year intervals
    df["year_start"] = df["year_id"].map(int)
    df["year_end"] = df["year_id"].map(int) + 1

    # 5. Set index and unstack to get draw columns
    index_cols = (
        [
            "location",
            "sex",
        ]
        + age_cols
        + ["year_start", "year_end", "draw"]
    )
    df_indexed = df.dropna(subset=index_cols).set_index(index_cols)

    df_wide = df_indexed["value"].unstack(level="draw")

    # 6. Rename columns to draw_x format
    df_wide.columns = [f"draw_{col}" for col in df_wide.columns]

    return df_wide


def get_fname_dict():
    return {
        "population": "/mnt/share/forecasting/data/9/future/population/20240320_daly_capstone_resubmission_squeeze_soft_round_shifted_hiv_shocks_covid_all_who_reagg/population_agg.nc",
        #     'births': '/mnt/share/forecasting/data/9/future/live_births/20231204_ref/live_births.nc',
        #     'deaths': '/snfs1/Project/forecasting/results/7/future/death/20240320_daly_capstone_resubmission_squeeze_soft_round_shifted_hiv_shocks_covid_all_who_reagg_num/_all.nc',
        "mortality": "/snfs1/Project/forecasting/results/7/future/death/20240320_daly_capstone_resubmission_squeeze_soft_round_shifted_hiv_shocks_covid_all_who_reagg/_all.nc",
        #     'migration': '/mnt/share/forecasting/data/6/future/migration/20230605_loc_intercept_shocks_pg_21LOCS_ATTENUATED/migration.nc',
    }


def load_population_structure(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    


    return interface.get_population_structure(location, years)


def load_age_bins(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    return interface.get_age_bins()


def load_demographic_dimensions(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    return interface.get_demographic_dimensions(location, years)


def load_theoretical_minimum_risk_life_expectancy(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_standard_data(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)
    return interface.get_measure(entity, key.measure, location, years).droplevel(
        "location"
    )


def load_metadata(key: str, location: str, years: int | str | list[int] | None = None):
    key = EntityKey(key)
    entity = get_entity(key)
    entity_metadata = entity[key.measure]
    if hasattr(entity_metadata, "to_dict"):
        entity_metadata = entity_metadata.to_dict()
    return entity_metadata


def load_categorical_paf(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    try:
        risk = {
            # todo add keys as needed
            data_keys.KEYGROUP.PAF: data_keys.KEYGROUP,
        }[key]
    except KeyError:
        raise ValueError(f"Unrecognized key {key}")

    distribution_type = get_data(risk.DISTRIBUTION, location)

    if distribution_type != "dichotomous" and "polytomous" not in distribution_type:
        raise NotImplementedError(
            f"Unrecognized distribution {distribution_type} for {risk.name}. Only dichotomous and "
            f"polytomous are recognized categorical distributions."
        )

    exp = get_data(risk.EXPOSURE, location)
    rr = get_data(risk.RELATIVE_RISK, location)

    # paf = (sum_categories(exp * rr) - 1) / sum_categories(exp * rr)
    sum_exp_x_rr = (
        (exp * rr)
        .groupby(list(set(rr.index.names) - {"parameter"}))
        .sum()
        .reset_index()
        .set_index(rr.index.names[:-1])
    )
    paf = (sum_exp_x_rr - 1) / sum_exp_x_rr
    return paf


def _load_em_from_meid(location, meid, measure):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == vi_globals.MEASURES[measure]]
    data = vi_utils.normalize(data, fill_value=0)
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS)
    data = vi_utils.reshape(data)
    data = vi_utils.scrub_gbd_conventions(data, location)
    data = vi_utils.split_interval(
        data, interval_column="age", split_column_prefix="age"
    )
    data = vi_utils.split_interval(
        data, interval_column="year", split_column_prefix="year"
    )
    return vi_utils.sort_hierarchical_data(data).droplevel("location")


# TODO - add project-specific data functions here


def load_alzheimers_prevalence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """Need to return the "opposite" prevalence key we are trying to write to the artifact.
    Meaning, we want the prevalence scale factor, to be the normal GBD prevalence of Alzheimers to properly
    scale the population and "fertility" components of the model. We want the Alzheimers SI model to really
    only be an I model, so we will set the prevalence to 1, so all simulants are created with the disease.
    """
    prevalence = load_standard_data(data_keys.ALZHEIMERS.PREVALENCE, location, years)
    if key == data_keys.ALZHEIMERS.PREVALENCE_SCALE_FACTOR:
        return prevalence

    # Set prevalence to 1
    prevalence.loc[:, :] = 1.0
    return prevalence


def get_entity(key: str | EntityKey):
    # Map of entity types to their gbd mappings.
    type_map = {
        "cause": causes,
        "covariate": covariates,
        "risk_factor": risk_factors,
        "alternative_risk_factor": alternative_risk_factors,
    }
    key = EntityKey(key)
    return type_map[key.type][key.name]
