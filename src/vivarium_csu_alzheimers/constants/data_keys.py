from typing import NamedTuple

#############
# Data Keys #
#############

METADATA_LOCATIONS = "metadata.locations"


class __Population(NamedTuple):
    LOCATION: str = "population.location"
    STRUCTURE: str = "population.structure"
    SCALING_FACTOR: str = "population.scaling_factor"
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
    AD_DEMENTIA_PREVALENCE: str = "cause.alzheimers.ad_dementia_prevalence"
    MIXED_DEMENTIA_PREVALENCE: str = "cause.alzheimers.mixed_dementia_prevalence"

    AD_DEMENTIA_INCIDENCE_RATE_TOTAL_POPULATION: str = (
        "cause.alzheimers.ad_dementia_population_incidence_rate"
    )
    MIXED_DEMENTIA_INCIDENCE_RATE_TOTAL_POPULATION: str = (
        "cause.alzheimers.mixed_dementia_population_incidence_rate"
    )

    MILD_DEMENTIA_PREVALENCE: str = (
        "sequela.mild_alzheimers_disease_and_other_dementias.prevalence"
    )
    MODERATE_DEMENTIA_PREVALENCE: str = (
        "sequela.moderate_alzheimers_disease_and_other_dementias.prevalence"
    )
    SEVERE_DEMENTIA_PREVALENCE: str = (
        "sequela.severe_alzheimers_disease_and_other_dementias.prevalence"
    )

    EMR_COMO: str = "cause.alzheimers_disease_and_other_dementias.excess_mortality_rate"
    EMR_DISMOD: str = "cause.dementia.excess_mortality_rate"

    MCI_DISABILITY_WEIGHT: str = "cause.alzheimers.mci_disability_weight"
    MILD_DEMENTIA_DISABILITY_WEIGHT: str = (
        "sequela.mild_alzheimers_disease_and_other_dementias.disability_weight"
    )
    MODERATE_DEMENTIA_DISABILITY_WEIGHT: str = (
        "sequela.moderate_alzheimers_disease_and_other_dementias.disability_weight"
    )
    SEVERE_DEMENTIA_DISABILITY_WEIGHT: str = (
        "sequela.severe_alzheimers_disease_and_other_dementias.disability_weight"
    )

    RESTRICTIONS: str = "cause.alzheimers_disease_and_other_dementias.restrictions"

    @property
    def name(self):
        return "alzheimers disease and other dementias"

    @property
    def log_name(self):
        return "alzheimers_disease_and_other_dementias"


ALZHEIMERS = __Alzheimers()


class __AlzheimersConsistent(NamedTuple):
    AD_PREVALENCE: str = "cause.alzheimers_consistent.alzheimers_prevalence"
    MIXED_DEMENTIA_PREVALENCE: str = "cause.alzheimers_consistent.mixed_dementia_prevalence"
    BBBM_CONDITIONAL_PREVALENCE: str = (
        "cause.alzheimers_consistent.bbbm_conditional_prevalence"
    )
    MCI_CONDITIONAL_PREVALENCE: str = "cause.alzheimers_consistent.mci_conditional_prevalence"
    MILD_DEMENTIA_CONDITIONAL_PREVALENCE: str = (
        "cause.alzheimers_consistent.mild_dementia_conditional_prevalence"
    )
    MODERATE_DEMENTIA_CONDITIONAL_PREVALENCE: str = (
        "cause.alzheimers_consistent.moderate_dementia_conditional_prevalence"
    )
    SEVERE_DEMENTIA_CONDITIONAL_PREVALENCE: str = (
        "cause.alzheimers_consistent.severe_dementia_conditional_prevalence"
    )

    SUSCEPTIBLE_TO_BBBM_TRANSITION_COUNT: str = (
        "cause.alzheimers_consistent.susceptible_to_bbbm_transition_count"
    )
    # BBBM to MCI transition rate caluclated during sim using mci_hazard.py and time in state
    # MCI to MILD transition rate is specified in data_values.py
    MILD_TO_MODERATE_DEMENTIA_TRANSITION_RATE: str = (
        "cause.alzheimers_consistent.mild_to_moderate_dementia_transition_rate"
    )
    MODERATE_TO_SEVERE_DEMENTIA_TRANSITION_RATE: str = (
        "cause.alzheimers_consistent.moderate_to_severe_dementia_transition_rate"
    )

    BBBM_AD_INCIDENCE_RATE: str = (
        "cause.alzheimers_consistent.susceptible_to_bbbm_ad_transition_rate"
    )
    MIXED_DEMENTIA_INCIDENCE_RATE_TOTAL_POPULATION: str = (
        "cause.alzheimers_consistent.mixed_dementia_population_incidence_rate"
    )
    MILD_DEMENTIA_INCIDENCE_RATE_TOTAL_POPULATION: str = (
        "cause.alzheimers_consistent.population_incidence_mild_dementia"
    )

    CSMR: str = "cause.alzheimers_disease_and_other_dementias.cause_specific_mortality_rate"
    EMR: str = "cause.alzheimers_consistent.excess_mortality_rate"
    MCI_DISABILITY_WEIGHT: str = "cause.alzheimers.mci_disability_weight"
    MILD_DEMENTIA_DISABILITY_WEIGHT: str = (
        "sequela.mild_alzheimers_disease_and_other_dementias.disability_weight"
    )
    MODERATE_DEMENTIA_DISABILITY_WEIGHT: str = (
        "sequela.moderate_alzheimers_disease_and_other_dementias.disability_weight"
    )
    SEVERE_DEMENTIA_DISABILITY_WEIGHT: str = (
        "sequela.severe_alzheimers_disease_and_other_dementias.disability_weight"
    )

    RESTRICTIONS: str = "cause.alzheimers_disease_and_other_dementias.restrictions"

    @property
    def name(self):
        return "alzheimers disease and other dementias"

    @property
    def log_name(self):
        return "alzheimers_disease_and_other_dementias"


ALZHEIMERS_CONSISTENT = __AlzheimersConsistent()


class __AlzheimersConsistent(NamedTuple):
    PREVALENCE_ANY: str = "cause.alzheimers_consistent.prevalence_any"
    PREVALENCE: str = "cause.alzheimers.prevalence"
    BBBM_CONDITIONAL_PREVALENCE: str = (
        "cause.alzheimers_consistent.bbbm_conditional_prevalence"
    )
    MCI_CONDITIONAL_PREVALENCE: str = "cause.alzheimers_consistent.mci_conditional_prevalence"
    DEMENTIA_CONDITIONAL_PREVALENCE: str = (
        "cause.alzheimers_consistent.dementia_conditional_prevalence"
    )
    MCI_TO_DEMENTIA_TRANSITION_RATE: str = "cause.alzheimers.mci_to_dementia_transition_rate"
    SUSCEPTIBLE_TO_BBBM_TRANSITION_COUNT: str = (
        "cause.alzheimers_consistent.susceptible_to_bbbm_transition_count"
    )
    # BBBM to MCI transition rate caluclated during sim using mci_hazard.py and time in state
    INCIDENCE_RATE_TOTAL_POPULATION: str = (
        "cause.alzheimers_consistent.population_incidence_dementia"
    )
    CSMR: str = "cause.alzheimers_disease_and_other_dementias.cause_specific_mortality_rate"
    EMR: str = "cause.alzheimers_consistent.excess_mortality_rate"
    DISABILITY_WEIGHT: str = "cause.alzheimers_disease_and_other_dementias.disability_weight"
    MCI_DISABILITY_WEIGHT: str = "cause.alzheimers.mci_disability_weight"
    RESTRICTIONS: str = "cause.alzheimers_disease_and_other_dementias.restrictions"

    @property
    def name(self):
        return "alzheimers disease and other dementias"

    @property
    def log_name(self):
        return "alzheimers_disease_and_other_dementias"


ALZHEIMERS_CONSISTENT = __AlzheimersConsistent()


class __TestingRates(NamedTuple):
    CSF: str = "testing_rates.csf"
    PET: str = "testing_rates.pet"

    @property
    def name(self):
        return "testing rates"

    @property
    def log_name(self):
        return self.name.replace(" ", "_")


TESTING_RATES = __TestingRates()


class __Treatment(NamedTuple):
    RR: str = "treatment.relative_risk"

    @property
    def name(self):
        return "treatment"

    @property
    def log_name(self):
        return self.name.replace(" ", "_")


TREATMENT = __Treatment()

MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    ALZHEIMERS,
    TESTING_RATES,
    TREATMENT,
]
