# copied from Nathaniel's notebook
import numpy as np

from vivarium_csu_alzheimers.constants.data_values import BBBM_GAMMA_DIST


def hazard(t, dist):
    # hazard = (probability density) / (survival function)
    return np.exp(dist.logpdf(t) - dist.logsf(t))


gamma_hazard = lambda t: hazard(t, BBBM_GAMMA_DIST)
