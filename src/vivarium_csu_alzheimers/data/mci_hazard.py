# copied from Nathaniel's notebook
import numpy as np

from vivarium_csu_alzheimers.constants.data_values import BBBM_HAZARD_DIST


def hazard(t, dist):
    return np.exp(dist.logpdf(t) - dist.logsf(t))


gamma_hazard = lambda t: hazard(t, BBBM_HAZARD_DIST)
