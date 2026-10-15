"""Refit the consistent rates of an existing artifact with the sampler on this branch.

Copies <src.hdf> to <out.hdf>, re-runs generate_consistent_rates and the three derived
steps on the copy (the raw GBD keys are left untouched), and prints the MCMC diagnostics.
This is what make_artifacts does after loading the GBD keys, so it lets you compare old
and new consistent rates on identical inputs without /mnt/team or GBD access.

usage: python notebooks/refit_consistent_rates.py <src.hdf> <out.hdf>

e.g.  python notebooks/refit_consistent_rates.py \\
        src/vivarium_csu_alzheimers/artifacts/united_states_of_america.hdf \\
        artifact_comparisons/united_states_of_america.new_sampler.hdf
"""
import shutil
import sys
import time

from loguru import logger
from vivarium.artifact import Artifact

from vivarium_csu_alzheimers.data import consistent_rates


def main(src: str, out: str) -> None:
    shutil.copy(src, out)
    art = Artifact(out)
    start = time.time()
    consistent_rates.generate_consistent_rates(art)
    consistent_rates.generate_consistent_susceptible_to_bbbm_transition_count(art)
    consistent_rates.generate_consistent_dementia_conditional_prevalence(art)
    consistent_rates.generate_consistent_csmr(art)
    logger.info(f"refit written to {out} in {(time.time() - start) / 60:.1f} min")
    diagnostics = art.load(consistent_rates.MCMC_DIAGNOSTICS_KEY)
    print(
        diagnostics.groupby(["sex", "parameter"])[["n_eff", "r_hat"]]
        .agg(["min", "median", "max"])
        .to_string()
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
