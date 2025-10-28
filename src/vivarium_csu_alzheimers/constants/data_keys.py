from typing import NamedTuple

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
    SCALING_FACTOR: str = "population.scaling_factor"

    @property
    def name(self):
        return "population"

    @property
    def log_name(self):
        return "population"


POPULATION = __Population()


class __Alzheimers(NamedTuple):
    PREVALENCE: str = "cause.alzheimers.prevalence"
    BBBM_CONDITIONAL_PREVALENCE: str = "cause.alzheimers.bbbm_conditional_prevalence"
    MCI_CONDITIONAL_PREVALENCE: str = "cause.alzheimers.mci_conditional_prevalence"
    MCI_TO_DEMENTIA_TRANSITION_RATE: str = "cause.alzheimers.mci_to_dementia_transition_rate"
    SUSCEPTIBLE_TO_BBBM_TRANSITION_COUNT: str = (
        "cause.alzheimers.susceptible_to_bbbm_transition_count"
    )
    # BBBM to MCI transition rate caluclated during sim using mci_hazard.py and time in state
    INCIDENCE_RATE_TOTAL_POPULATION: str = "cause.alzheimers.population_incidence_rate"
    CSMR: str = "cause.alzheimers_disease_and_other_dementias.cause_specific_mortality_rate"
    EMR: str = "cause.alzheimers.excess_mortality_rate"
    DISABILITY_WEIGHT: str = "cause.alzheimers_disease_and_other_dementias.disability_weight"
    MCI_DISABILITY_WEIGHT: str = "cause.alzheimers.mci_disability_weight"
    RESTRICTIONS: str = "cause.alzheimers_disease_and_other_dementias.restrictions"

    @property
    def name(self):
        return "alzheimers disease and other dementias"

    @property
    def log_name(self):
        return "alzheimers_disease_and_other_dementias"


ALZHEIMERS = __Alzheimers()

class __AlzheimersConsistent(NamedTuple):
    PREVALENCE_ANY: str = "cause.alzheimers_consistent.prevalence_any"
    PREVALENCE: str = "cause.alzheimers.prevalence"
    BBBM_CONDITIONAL_PREVALENCE: str = "cause.alzheimers_consistent.bbbm_conditional_prevalence"
    MCI_CONDITIONAL_PREVALENCE: str = "cause.alzheimers_consistent.mci_conditional_prevalence"
    DEMENTIA_CONDITIONAL_PREVALENCE: str = "cause.alzheimers_consistent.dementia_conditional_prevalence"
    MCI_TO_DEMENTIA_TRANSITION_RATE: str = "cause.alzheimers.mci_to_dementia_transition_rate"
    SUSCEPTIBLE_TO_BBBM_TRANSITION_COUNT: str = (
        "cause.alzheimers_consistent.susceptible_to_bbbm_transition_count"
    )
    # BBBM to MCI transition rate caluclated during sim using mci_hazard.py and time in state
    INCIDENCE_RATE_TOTAL_POPULATION: str = "cause.alzheimers_consistent.population_incidence_dementia"
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
