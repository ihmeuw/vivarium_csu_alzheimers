# copied from Nathaniel's notebook
import numpy as np


def hazard(t, dist):
    return np.exp(dist.logpdf(t) - dist.logsf(t))
