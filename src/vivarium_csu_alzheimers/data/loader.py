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

import numpy as np
import pandas as pd
from gbd_mapping import causes, covariates, risk_factors
from vivarium.framework.artifact import EntityKey
from vivarium_inputs import globals as vi_globals
from vivarium_inputs import interface
from vivarium_inputs import utilities as vi_utils
from vivarium_inputs import utility_data
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_csu_alzheimers.constants import data_keys, data_values, metadata
from vivarium_csu_alzheimers.constants.metadata import ARTIFACT_COLUMNS
from vivarium_csu_alzheimers.constants.paths import (
    DEMENTIA_PROPORTIONS_PATH,
    FORECAST_NC_DATA_FILEPATHS_DICT,
)
from vivarium_csu_alzheimers.data.extra_gbd import (
    load_incidence_dismod,
    load_prevalence_dismod,
)
from vivarium_csu_alzheimers.data.forecasts import table_from_nc
from vivarium_csu_alzheimers.utilities import get_norm, get_random_variable_draws


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
        data_keys.ALZHEIMERS.PREVALENCE: load_prevalence,
        data_keys.ALZHEIMERS.BBBM_CONDITIONAL_PREVALANCE: load_bbbm_conditional_prevalence,
        data_keys.ALZHEIMERS.MCI_CONDITIONAL_PREVALENCE: load_mci_conditional_prevalence,
        data_keys.ALZHEIMERS.MCI_TO_DEMENTIA_TRANSITION_RATE: load_mci_to_dementia_transition_rate,
        data_keys.ALZHEIMERS.SUSCEPTIBLE_TO_BBBM_TRANSITION_COUNT: load_susceptible_to_bbbm_transition_count,
        # MCI incidence rate caluclated during sim using mci_hazard.py and time in state
        data_keys.ALZHEIMERS.CSMR: load_standard_data,
        data_keys.ALZHEIMERS.EMR: load_standard_data,
        data_keys.ALZHEIMERS.DISABILITY_WEIGHT: load_standard_data,
        data_keys.ALZHEIMERS.MCI_DISABILITY_WEIGHT: load_mci_disability_weight,
        data_keys.ALZHEIMERS.RESTRICTIONS: load_metadata,
        data_keys.ALZHEIMERS.INCIDENCE_RATE_TOTAL_POPULATION: load_alzheimers_incidence_total_population,
        data_keys.TESTING_RATES.CSF: load_csf_pet_testing_rates,
        data_keys.TESTING_RATES.PET: load_csf_pet_testing_rates,
    }
    mapped_value = mapping[lookup_key](lookup_key, location, years)

    # To avoid issues with irrelevant differences between the very young age
    # groups in GBD and FHS data, we will drop rows with age_start < 5 from
    # all age-specific DataFrames
    if isinstance(mapped_value, pd.DataFrame):
        df = mapped_value
        if "age_start" in df.index.names:
            df = df.query("age_start >= 5.0")
        return df
    else:
        return mapped_value


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
    df.index = pd.MultiIndex.from_frame(utility_data.get_age_bins())
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


def load_alzheimers_incidence_total_population(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """Load raw Alzheimers incidence rates from GBD. The incidence rate we pull through vivarium framework
    is the incidence rate / the susceptible population. We want incidence rate / total population.
    """

    raw_incidence = load_incidence_dismod(location)  # total population
    incidence = reshape_to_vivarium_format(raw_incidence, location)
    incidence.index = incidence.index.droplevel(
        ["measure_id", "metric_id", "model_version_id", "modelable_entity_id"]
    )

    return incidence * load_dementia_proportions(None, location, years)


def load_prevalence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:

    raw_prevalence = load_prevalence_dismod(location)  # total population
    prevalence = reshape_to_vivarium_format(raw_prevalence, location)
    prevalence.index = prevalence.index.droplevel(
        ["measure_id", "metric_id", "model_version_id", "modelable_entity_id"]
    )

    return prevalence * load_dementia_proportions(None, location, years)


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
    df = df[
        ~df.age_group_id.isin(metadata.UNEXPECTED_AGE_GROUPS)
    ]  # unexpected age groups cause vi_utils.scrub_gbd_conventions to fail
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
    """
    eq 1 in attention box:
    https://vivarium-research.readthedocs.io/en/latest/models/causes/alzheimers/presymptomatic_and_mci_gbd_2021/index.html#alzheimers-cause-state-data-including-susceptible-note
    """
    all_state_dur = load_all_states_duration(None, location, years)
    alz_dur = load_dementia_duration(None, location, years)
    alz_prev = get_data(data_keys.ALZHEIMERS.PREVALENCE, location, years)
    # bfill(limit=2) casts the 40-44 age group data to the 30-34 and 35-39 age group rows
    return ((all_state_dur * alz_prev) / alz_dur).bfill(limit=2).fillna(0)


def load_dementia_duration(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """delta_AD in
    https://vivarium-research.readthedocs.io/en/latest/models/causes/alzheimers/presymptomatic_and_mci_gbd_2021/index.html#alzheimers-cause-state-data-including-susceptible-note
    """
    prev = get_data(data_keys.ALZHEIMERS.PREVALENCE, location, years)
    total_pop_inc = get_data(
        data_keys.ALZHEIMERS.INCIDENCE_RATE_TOTAL_POPULATION,
        location,
        years,
    )
    return (prev / total_pop_inc).fillna(0)


def load_all_states_duration(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """delta_all_AD_states in
    https://vivarium-research.readthedocs.io/en/latest/models/causes/alzheimers/presymptomatic_and_mci_gbd_2021/index.html#alzheimers-cause-state-data-including-susceptible-note
    """
    alz_dur = load_dementia_duration(None, location, years)
    return alz_dur + data_values.BBBM_AVG_DURATION + data_values.MCI_AVG_DURATION


def load_bbbm_conditional_prevalence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """BBBM-AD initial prevalence in
    https://vivarium-research.readthedocs.io/en/latest/models/causes/alzheimers/presymptomatic_and_mci_gbd_2021/index.html#id6
    """
    all_state_dur = load_all_states_duration(None, location, years)
    return data_values.BBBM_AVG_DURATION / all_state_dur


def load_mci_conditional_prevalence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """MCI-AD initial prevalence in
    https://vivarium-research.readthedocs.io/en/latest/models/causes/alzheimers/presymptomatic_and_mci_gbd_2021/index.html#id6
    """
    all_state_dur = load_all_states_duration(None, location, years)
    return data_values.MCI_AVG_DURATION / all_state_dur


def transform_group_index_J_BBBM(DUR, W, N, index_p0):
    # transform index for current population age/year group (p0)
    # into indices needed to compute the J_BBBM equation

    (location, sex, age, _, year, _) = index_p0

    # I_BBBM (p)opulation age/year group 1 (p1)
    age_p1 = age + (N * W)  # age_start for group 1S
    # NB: don't do min(age_p1, 95), instead return 0 population
    year_p12 = round(year + DUR)  # year_start for groups 1 and 2
    year_p12 = min(year_p12, 2050)  # use 2050 data for future years

    # I_BBBM (p)opulation age/year group 2 (p2)
    age_p2 = age + ((N + 1) * W)
    # NB: don't do min(age_p2, 95), instead return 0 incidence

    # I_BBBM (i)ncidence age/year groups (i1, i2)
    age_i1 = age_p1
    age_i2 = age_p2
    year_i12 = 2023  # incidence uses 2023 data since model 5

    # gamma (m)ortality age/year group (m)
    age_m = age + (DUR / 2)
    age_m = W * round(age_m / W)  # round to nearest W
    age_m = min(age_m, 95)  # NB: for mortality, still using oldest age group
    year_m = round(year + (DUR / 2))
    year_m = min(year_m, 2050)

    return (
        (  # I_BBBM (p)opulation age/year group 1 (p1)
            location,
            sex,
            age_p1,
            age_p1 + 5 if age_p1 != 95 else 125,
            year_p12,
            year_p12 + 1,
        ),
        (  # I_BBBM (p)opulation age/year group 2 (p2)
            location,
            sex,
            age_p2,
            age_p2 + 5 if age_p2 != 95 else 125,
            year_p12,
            year_p12 + 1,
        ),
        (  # I_BBBM (i)ncidence age/year groups (i1)
            sex,
            age_i1,
            age_i1 + 5 if age_i1 != 95 else 125,
            year_i12,
            year_i12 + 1,
        ),
        (  # I_BBBM (i)ncidence age/year groups (i2)
            sex,
            age_i2,
            age_i2 + 5 if age_i2 != 95 else 125,
            year_i12,
            year_i12 + 1,
        ),
        (  # gamma (m)ortality age/year group (m)
            sex,
            age_m,
            age_m + 5 if age_m != 95 else 125,
            year_m,
            year_m + 1,
        ),
    )


def loc_any_group(df, index, val):
    # get row of dataframe (population, incidence or mortality) by index tuple.
    # if age group does not exist, set row to val

    age = index[-4]  # population df starts with location; incidence and mortality dfs don't
    if age > 95:
        # return zeros for all values
        df_row = df.iloc[0]  # get row with same shape as df
        df_row[:] = val  # set all values to val
        df_row.name = index  # set proper location/sex/age/year (for debugging)
        return df_row
    else:
        return df.loc[index]


def load_susceptible_to_bbbm_transition_count(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """I_BBBM_g,t + D_g,t in "Calculating entrance rate with presymptomatic and MCI stages" from
    https://vivarium-research.readthedocs.io/en/latest/models/other_models/alzheimers_population/index.html
    Link to WIP docs from open PR:
    https://vivarium-research--1768.org.readthedocs.build/en/1768/models/other_models/alzheimers_population/index.html#calculating-entrance-rate-with-presymptomatic-and-mci-stages
    """

    inc = get_data(data_keys.ALZHEIMERS.INCIDENCE_RATE_TOTAL_POPULATION, location, years)
    pop = get_data(data_keys.POPULATION.STRUCTURE, location, years)
    mort = load_background_mortality(None, location, years)

    DUR = (
        data_values.BBBM_AVG_DURATION + data_values.MCI_AVG_DURATION
    )  # total duration of pre-dementia AD
    W = data_values.GBD_AGE_GROUPS_WIDTH
    N = int(DUR / W)
    R = DUR % W  # remainder

    new_bbbm_people = pd.DataFrame(0, index=pop.index, columns=pop.columns)
    for index, _ in new_bbbm_people.iterrows():
        (index_p1, index_p2, index_i1, index_i2, index_m) = transform_group_index_J_BBBM(
            DUR, W, N, index
        )
        I_bbbm = (
            (1 - (R / W)) * loc_any_group(inc, index_i1, 0) * loc_any_group(pop, index_p1, 0)
        ) + ((R / W) * loc_any_group(inc, index_i2, 0) * loc_any_group(pop, index_p2, 0))
        gamma = DUR * mort.loc[index_m]
        gamma = 1 - np.exp(-gamma)
        new_bbbm_people.loc[index] = I_bbbm / (1 - gamma)

    new_bbbm_people.index = new_bbbm_people.index.droplevel("location")
    return new_bbbm_people


def load_mci_disability_weight(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """DW_MCI in
    https://vivarium-research.readthedocs.io/en/latest/models/causes/alzheimers/presymptomatic_and_mci_gbd_2021/index.html#id9
    """
    data = pd.read_csv(
        "/mnt/team/simulation_science/pub/models/vivarium_csu_alzheimers/data/dw_full.csv"
    )  # comes from /ihme/epi/disability_weights/standard/dw_full.csv
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
    data.columns = data.columns.str.replace("draw", "draw_")
    return data


def load_background_mortality(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """
    Mortality with emr = 0, such as for BBBM and MCI states
    """
    acmr = get_data(data_keys.POPULATION.ACMR, location, years)
    csmr = get_data(data_keys.ALZHEIMERS.CSMR, location, years).droplevel(
        ["year_start", "year_end"]
    )

    # for now, assume csmr is the same for all years based on docs
    csmr_all_years = pd.DataFrame(csmr, index=acmr.index)
    return acmr - csmr_all_years  #  emr_MCI = 0


def load_mci_to_dementia_transition_rate(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """i_AD in
    https://vivarium-research.readthedocs.io/en/latest/models/causes/alzheimers/presymptomatic_and_mci_gbd_2021/index.html#id5
    """
    return 1 / data_values.MCI_AVG_DURATION


def load_csf_pet_testing_rates(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    """Load the testing rates.

    NOTES
    -----
    Testing rates do not vary by age, sex, or year, and so we do not include
    demographics in the artifact. We simply return a single-row dataframe where
    the index is meaningless.
    """
    rate = data_values.CSF_PET_LOCATION_TESTING_RATES[location][key]
    dist = get_norm(
        mean=rate.mean,
        ninety_five_pct_confidence_interval=(rate.ci_lower, rate.ci_upper),
    )
    draws = get_random_variable_draws(ARTIFACT_COLUMNS, key, dist)
    return pd.DataFrame([draws], columns=ARTIFACT_COLUMNS)


def load_dementia_proportions(
    key: str, location: str, years: int | str | list[int] | None = None
) -> pd.DataFrame:
    df = pd.read_csv(DEMENTIA_PROPORTIONS_PATH)
    bins = load_age_bins(None, None).index.to_frame().reset_index(drop=True)
    merged = pd.merge(df, bins, on="age_group_name", how="left")
    merged = merged[
        (merged.type_label == "Alzheimer's disease")  # AD only, no mixed
        & (merged.age_group_name != "All Age")
    ]
    merged["year_start"] = 2023
    merged["year_end"] = 2024
    merged = (
        merged.set_index(["sex", "age_start", "age_end", "year_start", "year_end"])
        .drop(["age_group_name", "type_label", "age_group_id"], axis=1)
        .rename({"proportion": "value"}, axis=1)
    )

    # add <40 age groups
    for _, group in bins[bins.age_start < 40].iterrows():
        merged.loc[("Male", group.age_start, group.age_end, 2023, 2024)] = 0
        merged.loc[("Female", group.age_start, group.age_end, 2023, 2024)] = 0
    merged = vi_utils.sort_hierarchical_data(merged)

    # fake draws from value (for multiplication with incidence and prevalence)
    df = pd.DataFrame(columns=vi_globals.DRAW_COLUMNS, index=merged.index)
    for col in df.columns:
        df[col] = merged.value
    return df
