from datetime import datetime
from typing import NamedTuple

import scipy

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
    BBBM_ENTRANCE_TIME: str = "bbbm_entrance_time"
    ENTRANCE_TIME: str = "entrance_time"


COLUMNS = __Columns()
