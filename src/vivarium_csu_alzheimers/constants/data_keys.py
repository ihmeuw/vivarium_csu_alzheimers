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
    # Keys that will be loaded into the artifact. must have a colon type declaration
    PREVALENCE: TargetString = TargetString(
        "cause.alzheimers_disease_and_other_dementias.prevalence"
    )
    INCIDENCE_RATE: TargetString = TargetString(
        "cause.alzheimers_disease_and_other_dementias.incidence_rate"
    )
    REMISSION_RATE: TargetString = TargetString(
        "cause.alzheimers_disease_and_other_dementias.remission_rate"
    )
    DISABILITY_WEIGHT: TargetString = TargetString(
        "cause.alzheimers_disease_and_other_dementias.disability_weight"
    )
    EMR: TargetString = TargetString(
        "cause.alzheimers_disease_and_other_dementias.excess_mortality_rate"
    )
    CSMR: TargetString = TargetString(
        "cause.alzheimers_disease_and_other_dementias.cause_specific_mortality_rate"
    )
    RESTRICTIONS: TargetString = TargetString(
        "cause.alzheimers_disease_and_other_dementias.restrictions"
    )

    @property
    def name(self):
        return "alzheimers_disease_and_other_dementias"

    @property
    def log_name(self):
        return "alzheimers disease and other dementias"


ALZHEIMERS = __Alzheimers()


class __TestingForAlzheimers(NamedTuple):
    PREVALENCE: TargetString = TargetString("cause.testing_for_alzheimers.prevalence")
    INCIDENCE_RATE: TargetString = TargetString("cause.testing_for_alzheimers.incidence_rate")
    REMISSION_RATE: TargetString = TargetString("cause.testing_for_alzheimers.remission_rate")
    DISABILITY_WEIGHT: TargetString = TargetString(
        "cause.testing_for_alzheimers.disability_weight"
    )
    EMR: TargetString = TargetString("cause.testing_for_alzheimers.excess_mortality_rate")
    CSMR: TargetString = TargetString(
        "cause.testing_for_alzheimers.cause_specific_mortality_rate"
    )
    RESTRICTIONS: TargetString = TargetString("cause.testing_for_alzheimers.restrictions")

    @property
    def name(self):
        return "testing_for_alzheimers"

    @property
    def log_name(self):
        return "testing for alzheimers"


TESTING_FOR_ALZHEIMERS = __TestingForAlzheimers()


class __HypotheticAlzheimersIntervention(NamedTuple):
    COVERAGE: TargetString = TargetString(
        "intervention.hypothetical_alzheimers_intervention.coverage"
    )
    EXPOSURE_STANDARD_DEVIATION: TargetString = TargetString(
        "intervention.hypothetical_alzheimers_intervention.exposure_standard_deviation"
    )
    DISTRIBUTION_TYPE: TargetString = TargetString(
        "intervention.hypothetical_alzheimers_intervention.distribution"
    )
    RELATIVE_RISK: TargetString = TargetString(
        "intervention.hypothetical_alzheimers_intervention.relative_risk"
    )
    PAF: TargetString = TargetString(
        "intervention.hypothetical_alzheimers_intervention.population_attributable_fraction"
    )

    @property
    def name(self):
        return "hypothetical_alzheimers_intervention"

    @property
    def log_name(self):
        return "hypothetical alzheimers intervention"


HYPOTHETICAL_ALZHEIMERS_INTERVENTION = __HypotheticAlzheimersIntervention()


MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    # TODO: list all key groups here
    ALZHEIMERS,
    TESTING_FOR_ALZHEIMERS,
    HYPOTHETICAL_ALZHEIMERS_INTERVENTION,
]
