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
from gbd_mapping import causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium_gbd_access import gbd
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import interface
from vivarium_inputs import utilities as vi_utils
from vivarium_inputs import utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_csu_alzheimers.constants import data_keys
from vivarium_csu_alzheimers.constants import data_values
from vivarium_csu_alzheimers.constants.paths import FORECAST_NC_DATA_FILEPATHS_DICT
from vivarium_csu_alzheimers.data.extra_gbd import load_raw_incidence
from vivarium_csu_alzheimers.data.forecasts import table_from_nc


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
        data_keys.POPULATION.STRUCTURE: load_forecasted_population_structure,
        data_keys.POPULATION.AGE_BINS: load_age_bins,
        data_keys.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        data_keys.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        data_keys.POPULATION.ACMR: load_forecasted_mortality,
        data_keys.POPULATION.LIVE_BIRTH_RATE: load_standard_data,
        data_keys.POPULATION.SCALING_FACTOR: load_alzheimers_all_states_prevalence,
        data_keys.ALZHEIMERS.PREVALENCE: load_alzheimers_prevalence,
        data_keys.ALZHEIMERS.BBBM_CONDITIONAL_PREVALANCE: load_bbbm_conditional_prevalence,
        data_keys.ALZHEIMERS.MCI_CONDITIONAL_PREVALENCE: load_mci_conditional_prevalence,
        data_keys.ALZHEIMERS.INCIDENCE_RATE: load_standard_data,
        data_keys.ALZHEIMERS.BBBM_INCIDENCE_RATE: load_bbbm_incidence_rate,
        # MCI incidence rate caluclated during sim using mci_hazard.py and time in state
        data_keys.ALZHEIMERS.CSMR: load_standard_data,
        data_keys.ALZHEIMERS.EMR: load_standard_data,
        data_keys.ALZHEIMERS.DISABILIITY_WEIGHT: load_standard_data,
        data_keys.ALZHEIMERS.MCI_DISABILITY_WEIGHT: load_mci_disability_weight,
        data_keys.ALZHEIMERS.RESTRICTIONS: load_metadata,
        data_keys.ALZHEIMERS.TOTAL_POPULATION_INCIDENCE_RATE: load_alzheimers_total_population_incidence,
    }
    return mapping[lookup_key](lookup_key, location, years)


def load_population_location(
    key: str, location: str, years: int | str | list[int] | None = None
) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f"Unrecognized key {key}")

    return location


def load_forecast(param: str, location: str, years: int | str | list[int]) -> pd.DataFrame:
    loc_id = utility_data.get_location_id(location)
    age_mapping = get_data(data_keys.POPULATION.AGE_BINS, location, years)
    return table_from_nc(
        FORECAST_NC_DATA_FILEPATHS_DICT, param, loc_id, location, age_mapping
    )


def load_forecasted_population_structure(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    return load_forecast("population", location, years)


def load_forecasted_mortality(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    return load_forecast("mortality", location, years)


def load_age_bins(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    df = pd.DataFrame()
    df.index = pd.MultiIndex.from_frame(utility_data.get_age_bins().query("age_start >= 5.0"))
    return df


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
    return interface.get_measure(entity, key.measure, location, years).droplevel("location")


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
    data = vi_utils.split_interval(data, interval_column="age", split_column_prefix="age")
    data = vi_utils.split_interval(data, interval_column="year", split_column_prefix="year")
    return vi_utils.sort_hierarchical_data(data).droplevel("location")


# TODO - add project-specific data functions here


def load_alzheimers_prevalence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    return load_standard_data(data_keys.ALZHEIMERS.PREVALENCE, location, years)


def load_alzheimers_total_population_incidence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """Load raw Alzheimers incidence rates from GBD. The incidence rate we pull through vivarium framework
    is the incidence rate / the susceptible population. We want incidence rate / total population.
    """
    entity = get_entity(key)
    raw_incidence = load_raw_incidence(entity, location)
    incidence = reshape_to_vivarium_format(raw_incidence, location)
    incidence.index = incidence.index.droplevel(
        ["cause_id", "measure_id", "metric_id", "version_id"]
    )

    return incidence


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


def reshape_to_vivarium_format(df, location):
    df = vi_utils.reshape(df, value_cols=vi_globals.DRAW_COLUMNS)
    df = vi_utils.scrub_gbd_conventions(df, location)
    df = vi_utils.split_interval(df, interval_column="age", split_column_prefix="age")
    df = vi_utils.split_interval(df, interval_column="year", split_column_prefix="year")
    df = vi_utils.sort_hierarchical_data(df)
    df.index = df.index.droplevel("location")
    return df


def load_alzheimers_all_states_prevalence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    all_state_dur = load_alzheimers_duration_all_states(None, location, years)
    alz_dur = load_alzheimers_duration(None, location, years)
    alz_prev = get_data(data_keys.ALZHEIMERS.PREVALENCE, location, years)
    return (all_state_dur * alz_prev) / alz_dur


def load_alzheimers_duration(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    prev = get_data(data_keys.ALZHEIMERS.PREVALENCE, location, years)
    total_pop_inc = get_data(
        data_keys.ALZHEIMERS.TOTAL_POPULATION_INCIDENCE_RATE,
        location,
        years,
    )
    return prev / total_pop_inc


def load_alzheimers_duration_all_states(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    alz_dur = load_alzheimers_duration(None, location, years)
    return alz_dur + data_values.BBBM_AVG_DURATION + data_values.MCI_AVG_DURATION


def load_bbbm_conditional_prevalence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    # Prevalence of BBBM state among simulants in any state
    all_state_dur = load_alzheimers_duration_all_states(None, location, years)
    return data_values.BBBM_AVG_DURATION / all_state_dur


def load_mci_conditional_prevalence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    # Prevalence of MCI state among simulants in any state
    all_state_dur = load_alzheimers_duration_all_states(None, location, years)
    return data_values.MCI_AVG_DURATION / all_state_dur


def load_bbbm_incidence_rate(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    # calculation TBD from Nathaniel - uses ALZHEIMERS.INCIDENCE_RATE
    inc = get_data(data_keys.ALZHEIMERS.INCIDENCE_RATE, location, years)
    return inc


def load_mci_disability_weight(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    data = pd.read_csv("/ihme/epi/disability_weights/standard/dw_full.csv")
    motor_data = (
        data[data.healthstate.notnull() & data.healthstate.str.contains("motor")]
        .set_index("healthstate")
        .filter(like="draw")
    )
    motor_mild = motor_data.loc["motor_mild"]
    motor_cog_mild = motor_data.loc["motor_cog_mild"]
    df_dw_mci = (motor_cog_mild - motor_mild) / (1 - motor_mild)

    # use demography to cast draws to all ages/sexes
    demography = get_data(data_keys.POPULATION.DEMOGRAPHY, location)
    data = pd.DataFrame([df_dw_mci], index=demography.index)
    data.index = data.index.droplevel("location")
    return data
