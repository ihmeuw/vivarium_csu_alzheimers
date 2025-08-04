from typing import NamedTuple

from vivarium_public_health.utilities import TargetString

#############
# Data Keys #
#############

METADATA_LOCATIONS = "metadata.locations"


class __Population(NamedTuple):
    LOCATION: str = "population.location"
    STRUCTURE: str = "population.structure"
    AGE_BINS: str = "population.age_bins"
    DEMOGRAPHY: str = "population.demographic_dimensions"
    TMRLE: str = "population.theoretical_minimum_risk_life_expectancy"
    ACMR: str = "cause.all_causes.cause_specific_mortality_rate"
    LIVE_BIRTH_RATE: str = "covariate.live_births_by_sex.estimate"

    @property
    def name(self):
        return "population"

    @property
    def log_name(self):
        return "population"


POPULATION = __Population()


class __Alzheimers(NamedTuple):
    PREVALENCE: str = "cause.alzheimers_disease_and_other_dementias.prevalence"
    INCIDENCE_RATE: str = "cause.alzheimers_disease_and_other_dementias.incidence_rate"
    CSMR: str = (
        "cause.alzheimers_disease_and_other_dementias.cause_specific_mortality_rate"
    )
    EMR: str = "cause.alzheimers_disease_and_other_dementias.excess_mortality_rate"
    DISABLIITY_WEIGHT: str = (
        "cause.alzheimers_disease_and_other_dementias.disability_weight"
    )
    RESTRICTIONS: str = "cause.alzheimers_disease_and_other_dementias.restrictions"

    @property
    def name(self):
        return "alzheimers disease and other dementias"

    @property
    def log_name(self):
        return "alzheimers_disease_and_other_dementias"


ALZHEIMERS = __Alzheimers()

MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    # TODO: list all key groups here
    ALZHEIMERS,
]
