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

from vivarium_csu_alzheimers.components.testing import POSTIVE_TEST_RATE
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
        data_keys.POPULATION.LIVE_BIRTH_RATE: load_standard_data,
        # TODO - add appropriate mappings
        data_keys.ALZHEIMERS.PREVALENCE: load_standard_data,
        data_keys.ALZHEIMERS.INCIDENCE_RATE: load_standard_data,
        data_keys.ALZHEIMERS.REMISSION_RATE: make_rate_of_zero,
        data_keys.ALZHEIMERS.CSMR: load_standard_data,
        data_keys.ALZHEIMERS.EMR: load_standard_data,
        data_keys.ALZHEIMERS.DISABILITY_WEIGHT: load_standard_data,
        data_keys.ALZHEIMERS.RESTRICTIONS: load_metadata,
        data_keys.TESTING_FOR_ALZHEIMERS.PREVALENCE: load_testing_prevalence,
        data_keys.TESTING_FOR_ALZHEIMERS.INCIDENCE_RATE: load_testing_incidence_rate,
        data_keys.TESTING_FOR_ALZHEIMERS.REMISSION_RATE: make_rate_of_zero,
        data_keys.TESTING_FOR_ALZHEIMERS.CSMR: make_rate_of_zero,
        data_keys.TESTING_FOR_ALZHEIMERS.EMR: make_rate_of_zero,
        data_keys.TESTING_FOR_ALZHEIMERS.DISABILITY_WEIGHT: make_rate_of_zero,
        data_keys.TESTING_FOR_ALZHEIMERS.RESTRICTIONS: load_testing_restrictions,
        data_keys.HYPOTHETICAL_ALZHEIMERS_INTERVENTION.COVERAGE: load_intervention_coverage,
        data_keys.HYPOTHETICAL_ALZHEIMERS_INTERVENTION.DISTRIBUTION_TYPE: load_intervention_distribution,
        data_keys.HYPOTHETICAL_ALZHEIMERS_INTERVENTION.EXPOSURE_STANDARD_DEVIATION: load_intervention_exposure_standard_deviation,
        data_keys.HYPOTHETICAL_ALZHEIMERS_INTERVENTION.RELATIVE_RISK: load_intervention_relative_risk,
        data_keys.HYPOTHETICAL_ALZHEIMERS_INTERVENTION.PAF: load_intervention_paf,
        data_keys.HYPOTHETICAL_ALZHEIMERS_INTERVENTION.CATEGORIES: load_intervention_categories,
    }
    return mapping[lookup_key](lookup_key, location, years)


def load_population_location(
    key: str, location: str, years: int | str | list[int] | None = None
) -> str:
    if key != data_keys.POPULATION.LOCATION:
        raise ValueError(f"Unrecognized key {key}")

    return location


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


def make_rate_of_zero(
    key: str, location: str, years: int | str | list[int] | None = None
) -> float:
    return 0.0


def load_testing_restrictions(
    key: str, location: str, years: int | str | list[int] | None = None
) -> dict[str, Any]:
    return get_data(data_keys.ALZHEIMERS.RESTRICTIONS, location, years)


def load_testing_prevalence(
    key: str, location: str, years: int | str | list[int] | None = None
) -> float:
    return 0.75


def load_testing_incidence_rate(
    key: str, location: str, years: int | str | list[int] | None = None
) -> float:
    return 0.75


def load_intervention_coverage(
    key: str, location: str, years: int | str | list[int] | None = None
) -> float:
    # We want coverage to be probability of testing positive
    return POSTIVE_TEST_RATE


def load_intervention_distribution(
    key: str, location: str, years: int | str | list[int] | None = None
) -> float:
    return "dichotomous"


def load_intervention_exposure_standard_deviation(
    key: str, location: str, years: int | str | list[int] | None = None
) -> float:
    return 0.1


def load_intervention_relative_risk(
    key: str, location: str, years: int | str | list[int] | None = None
) -> float:
    return 0.5


def load_intervention_paf(
    key: str, location: str, years: int | str | list[int] | None = None
) -> float:
    return 1 - 0.6


def load_intervention_categories(
    key: str, location: str, years: int | str | list[int] | None = None
) -> dict[str, str]:
    return {
        "covered": "covered by hypothetical alzheimers intervention",
        "uncovered": "not covered by hypothetical alzheimers intervention",
    }
