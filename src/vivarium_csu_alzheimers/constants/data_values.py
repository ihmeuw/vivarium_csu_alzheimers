from datetime import datetime
from typing import NamedTuple

import pandas as pd
import scipy

from vivarium_csu_alzheimers.constants.data_keys import TESTING_RATES
from vivarium_csu_alzheimers.constants.models import ALZHEIMERS_DISEASE_MODEL

############################
# Disease Model Parameters #
############################

REMISSION_RATE = 0.1
MEAN_SOJOURN_TIME = 10


##############################
# Screening Model Parameters #
##############################

PROBABILITY_ATTENDING_SCREENING_KEY = "probability_attending_screening"
PROBABILITY_ATTENDING_SCREENING_START_MEAN = 0.25
PROBABILITY_ATTENDING_SCREENING_START_STDDEV = 0.0025
PROBABILITY_ATTENDING_SCREENING_END_MEAN = 0.5
PROBABILITY_ATTENDING_SCREENING_END_STDDEV = 0.005

FIRST_SCREENING_AGE = 21
MID_SCREENING_AGE = 30
LAST_SCREENING_AGE = 65


###################################
# Scale-up Intervention Constants #
###################################
SCALE_UP_START_DT = datetime(2021, 1, 1)
SCALE_UP_END_DT = datetime(2030, 1, 1)
SCREENING_SCALE_UP_GOAL_COVERAGE = 0.50
SCREENING_SCALE_UP_DIFFERENCE = (
    SCREENING_SCALE_UP_GOAL_COVERAGE - PROBABILITY_ATTENDING_SCREENING_START_MEAN
)


# Gamma distribution parameters from Nathaniel's nb
# Fix mean at midpoint of interval [3.5, 4]
# Adjust variance until about 90% of probability lies in interval (variance = .03)

# Use method of moments to get shape and rate parameters
# FIXME: Fix values when we have a better hazard function
WEIBULL_SHAPE = 1.22
WEIBULL_SCALE = 6.76
BBBM_HAZARD_DIST = scipy.stats.weibull_min(WEIBULL_SHAPE, scale=WEIBULL_SCALE)
BBBM_AVG_DURATION = BBBM_HAZARD_DIST.mean()

MCI_AVG_DURATION = 3.85
DW_BBBM = 0
EMR_BBBM = 0
EMR_MCI = 0
GBD_AGE_GROUPS_WIDTH = 5


class __Columns(NamedTuple):
    DISEASE_STATE: str = ALZHEIMERS_DISEASE_MODEL.NAME
    PREVIOUS_DISEASE_STATE: str = "previous_" + ALZHEIMERS_DISEASE_MODEL.NAME
    BBBM_ENTRANCE_TIME: str = "bbbm_entrance_time"
    ENTRANCE_TIME: str = "entrance_time"
    TESTING_PROPENSITY: str = "testing_propensity"
    TESTING_STATE: str = "testing_state"
    BBBM_TEST_DATE: str = "bbbm_test_date"
    BBBM_TEST_RESULT: str = "bbbm_test_result"
    AGE: str = "age"
    ALIVE: str = "alive"
    TRACKED: str = "tracked"


COLUMNS = __Columns()


class __TestingStates(NamedTuple):
    NOT_TESTED: str = "not_tested"
    CSF: str = "csf"
    PET: str = "pet"
    BBBM: str = "bbbm"


TESTING_STATES = __TestingStates()


class TestingRates(NamedTuple):
    mean: float
    ci_lower: float
    ci_upper: float


CSF_PET_LOCATION_TESTING_RATES = {
    "United States of America": {
        TESTING_RATES.CSF: TestingRates(mean=0.108, ci_lower=0.054, ci_upper=0.161),
        TESTING_RATES.PET: TestingRates(mean=0.15, ci_lower=0.075, ci_upper=0.225),
    },
    "Germany": {
        TESTING_RATES.CSF: TestingRates(mean=0.187, ci_lower=0.094, ci_upper=0.281),
        TESTING_RATES.PET: TestingRates(mean=0.16, ci_lower=0.08, ci_upper=0.24),
    },
    "Spain": {
        TESTING_RATES.CSF: TestingRates(mean=0.246, ci_lower=0.123, ci_upper=0.369),
        TESTING_RATES.PET: TestingRates(mean=0.259, ci_lower=0.129, ci_upper=0.388),
    },
    "Sweden": {
        TESTING_RATES.CSF: TestingRates(mean=0.405, ci_lower=0.203, ci_upper=0.608),
        TESTING_RATES.PET: TestingRates(mean=0.045, ci_lower=0.023, ci_upper=0.068),
    },
    "United Kingdom": {
        TESTING_RATES.CSF: TestingRates(mean=0.095, ci_lower=0.047, ci_upper=0.142),
        TESTING_RATES.PET: TestingRates(mean=0.107, ci_lower=0.053, ci_upper=0.16),
    },
    "Japan": {
        TESTING_RATES.CSF: TestingRates(mean=0.133, ci_lower=0.067, ci_upper=0.2),
        TESTING_RATES.PET: TestingRates(mean=0.149, ci_lower=0.075, ci_upper=0.224),
    },
    "Israel": {
        TESTING_RATES.CSF: TestingRates(mean=0.133, ci_lower=0.067, ci_upper=0.2),
        TESTING_RATES.PET: TestingRates(mean=0.149, ci_lower=0.075, ci_upper=0.224),
    },
    "Taiwan (Province of China)": {
        TESTING_RATES.CSF: TestingRates(mean=0.133, ci_lower=0.067, ci_upper=0.2),
        TESTING_RATES.PET: TestingRates(mean=0.149, ci_lower=0.075, ci_upper=0.224),
    },
    "Brazil": {
        TESTING_RATES.CSF: TestingRates(mean=0.133, ci_lower=0.067, ci_upper=0.2),
        TESTING_RATES.PET: TestingRates(mean=0.149, ci_lower=0.075, ci_upper=0.224),
    },
    "China": {
        TESTING_RATES.CSF: TestingRates(mean=0.044, ci_lower=0.022, ci_upper=0.066),
        TESTING_RATES.PET: TestingRates(mean=0.061, ci_lower=0.03, ci_upper=0.091),
    },
}


BBBM_AGE_MIN = 60
BBBM_AGE_MAX = 80
BBBM_TIMESTEPS_UNTIL_RETEST = 6  # three years b/c time step is ~6 months
BBBM_POSITIVE_DIAGNOSIS_PROBABILITY = 0.9


class __BBBMTestResults(NamedTuple):
    NOT_TESTED: str = "not_tested"
    POSITIVE: str = "positive"
    NEGATIVE: str = "negative"


BBBM_TEST_RESULTS = __BBBMTestResults()

# bbbm testing rates are piecewise-linear starting at 2030 and maxing out in 2045
BBBM_TESTING_RATES = [
    (pd.Timestamp("2030-01-01"), 0.1),  # step increase from 0 in 2030
    (pd.Timestamp("2035-01-01"), 0.2),
    (pd.Timestamp("2040-01-01"), 0.4),
    (pd.Timestamp("2045-01-01"), 0.6),  # plateaus from here on out
]

TREATMENT_HAZARD_RATIO_MIN = 0.4
TREATMENT_HAZARD_RATIO_MAX = 0.6
