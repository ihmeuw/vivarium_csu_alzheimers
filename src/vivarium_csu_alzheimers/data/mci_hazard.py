# copied from Nathaniel's notebook

from vivarium_csu_alzheimers.constants.data_values import BBBM_GAMMA_DIST


def hazard(t, dist):
    # hazard = (probability density) / (survival function)
    return dist.pdf(t) / dist.sf(t)


gamma_hazard = lambda t: hazard(t, BBBM_GAMMA_DIST)
