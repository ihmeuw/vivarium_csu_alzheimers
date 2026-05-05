"""Tests that dead simulants do not undergo disease state transitions.

Regression tests for a bug where BBBMTransitionRate.compute_transition_rate
did not filter out dead simulants, causing ~24% of dead simulants to
transition from BBBM to MCI after death.
"""

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.slow
class TestNoPostDeathTransitions:
    """Run a full simulation and verify no disease transitions occur after death."""

    @pytest.fixture(scope="class")
    def line_list(self):
        """Run a short simulation and return the line list DataFrame."""
        from loguru import logger
        from vivarium import InteractiveContext

        from vivarium_csu_alzheimers.components.observers import SimulantLineListObserver

        logger.disable("vivarium")

        spec_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "vivarium_csu_alzheimers"
            / "model_specifications"
            / "model_spec.yaml"
        )
        artifact_path = Path(__file__).resolve().parents[1] / "united_states_of_america.hdf"
        if not artifact_path.exists():
            pytest.skip("Artifact file not found")

        sim = InteractiveContext(
            str(spec_path),
            configuration={
                "input_data": {"artifact_path": str(artifact_path)},
                "population": {"population_size": 5000},
                "intervention": {"scenario": "baseline"},
            },
        )

        # Step to ~2060 (roughly 77 steps) to give time for deaths + post-death transitions
        for _ in range(77):
            sim.step()

        observer = [
            c
            for c in sim._component_manager._components
            if isinstance(c, SimulantLineListObserver)
        ][0]
        return observer._build_line_list()

    def test_no_mci_after_death(self, line_list):
        """No simulant should transition to MCI after their date of death."""
        dead_with_mci = line_list[
            line_list["date_of_death"].notna() & line_list["date_of_mci_incidence"].notna()
        ]
        mci_after_death = dead_with_mci[
            dead_with_mci["date_of_mci_incidence"] > dead_with_mci["date_of_death"]
        ]
        assert len(mci_after_death) == 0, (
            f"{len(mci_after_death)} simulants transitioned to MCI after death "
            f"(out of {len(dead_with_mci)} dead simulants with MCI)"
        )

    def test_no_dementia_after_death(self, line_list):
        """No simulant should transition to dementia after their date of death."""
        dead_with_dementia = line_list[
            line_list["date_of_death"].notna()
            & line_list["date_of_dementia_incidence"].notna()
        ]
        dementia_after_death = dead_with_dementia[
            dead_with_dementia["date_of_dementia_incidence"]
            > dead_with_dementia["date_of_death"]
        ]
        assert len(dementia_after_death) == 0, (
            f"{len(dementia_after_death)} simulants transitioned to dementia after death "
            f"(out of {len(dead_with_dementia)} dead simulants with dementia)"
        )
