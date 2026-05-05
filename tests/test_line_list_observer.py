"""Tests for the SimulantLineListObserver."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from vivarium_csu_alzheimers.components.observers import SimulantLineListObserver
from vivarium_csu_alzheimers.constants.data_values import COLUMNS
from vivarium_csu_alzheimers.constants.models import (
    ALZHEIMERS_DISEASE_MODEL,
    TREATMENT_DISEASE_MODEL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockPopulationView:
    """Thin mock backed by a DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get(self, index, query=""):
        return self.df.loc[index].copy()

    def subview(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        view = MagicMock()
        view.get = lambda idx: self.df.loc[idx, columns].copy()
        return view

    def update(self, update_df: pd.DataFrame):
        for col in update_df.columns:
            if col not in self.df.columns:
                self.df[col] = pd.NaT
            self.df.loc[update_df.index, col] = update_df[col]


def _make_observer(pop_df, clock_time=pd.Timestamp("2030-01-01"), step_days=182):
    """Create a SimulantLineListObserver wired to *pop_df*."""
    obs = SimulantLineListObserver()
    obs.clock = lambda: clock_time
    obs.step_size = lambda: pd.Timedelta(days=step_days)
    obs.draw = 0
    obs.random_seed = 42
    obs.location = "united_states_of_america"
    obs._all_simulant_ids = pop_df.index.copy()
    obs.line_list = pd.DataFrame()
    obs.results_dir = None
    obs._population_view = MockPopulationView(pop_df)
    return obs


def _base_pop(n=5):
    """Return a population DataFrame with all required columns set to defaults."""
    idx = pd.RangeIndex(n)
    return pd.DataFrame(
        {
            "alive": "alive",
            "tracked": True,
            "age": [65.0 + i * 5 for i in range(n)],
            "entrance_time": pd.Timestamp("2022-01-01"),
            "exit_time": pd.NaT,
            COLUMNS.DISEASE_STATE: ALZHEIMERS_DISEASE_MODEL.BBBM_STATE,
            COLUMNS.PREVIOUS_DISEASE_STATE: ALZHEIMERS_DISEASE_MODEL.BBBM_STATE,
            COLUMNS.BBBM_ENTRANCE_TIME: pd.Timestamp("2020-06-01"),
            COLUMNS.TREATMENT_STATE: TREATMENT_DISEASE_MODEL.SUSCEPTIBLE_STATE,
            "date_of_birth": pd.NaT,
            "mci_event_time": pd.NaT,
            "dementia_event_time": pd.NaT,
            "treatment_start_time": pd.NaT,
            "treatment_end_time": pd.NaT,
        },
        index=idx,
    )


def _collect_event(index):
    """Build a minimal Event-like object for on_collect_metrics."""
    event = MagicMock()
    event.index = index
    return event


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestOnInitializeSimulants:
    def test_date_of_birth_computed(self):
        pop = _base_pop(3)
        pop["age"] = [30.0, 50.0, 70.0]
        pop["entrance_time"] = pd.Timestamp("2022-01-01")
        obs = _make_observer(pop)

        pop_data = MagicMock()
        pop_data.index = pop.index
        obs.on_initialize_simulants(pop_data)

        result = obs.population_view.df
        for i, age in enumerate([30.0, 50.0, 70.0]):
            expected = pd.Timestamp("2022-01-01") - pd.Timedelta(days=age * 365.25)
            assert result.loc[i, "date_of_birth"] == expected

    def test_tracking_columns_initialized_to_nat(self):
        pop = _base_pop(2)
        obs = _make_observer(pop)

        pop_data = MagicMock()
        pop_data.index = pop.index
        obs.on_initialize_simulants(pop_data)

        result = obs.population_view.df
        for col in ["mci_event_time", "dementia_event_time", "treatment_start_time", "treatment_end_time"]:
            assert result[col].isna().all()

    def test_all_simulant_ids_accumulated(self):
        pop = _base_pop(3)
        obs = _make_observer(pop)
        obs._all_simulant_ids = pd.Index([])

        pop_data = MagicMock()
        pop_data.index = pd.RangeIndex(3)
        obs.on_initialize_simulants(pop_data)

        assert len(obs._all_simulant_ids) == 3

        # Simulate a second batch
        pop2 = _base_pop(2)
        pop2.index = pd.RangeIndex(3, 5)
        # Extend backing df
        obs.population_view.df = pd.concat([obs.population_view.df, pop2])
        pop_data2 = MagicMock()
        pop_data2.index = pop2.index
        obs.on_initialize_simulants(pop_data2)

        assert len(obs._all_simulant_ids) == 5


# ---------------------------------------------------------------------------
# Transition detection tests
# ---------------------------------------------------------------------------


class TestMCITransitionDetection:
    def test_detects_bbbm_to_mci(self):
        pop = _base_pop(3)
        # Simulant 1 just transitioned BBBM -> MCI
        pop.loc[1, COLUMNS.DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.MCI_STATE
        pop.loc[1, COLUMNS.PREVIOUS_DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.BBBM_STATE

        clock = pd.Timestamp("2030-01-01")
        obs = _make_observer(pop, clock_time=clock)
        obs.on_collect_metrics(_collect_event(pop.index))

        expected_time = clock + pd.Timedelta(days=182)
        assert obs.population_view.df.loc[1, "mci_event_time"] == expected_time
        # Others unchanged
        assert pd.isna(obs.population_view.df.loc[0, "mci_event_time"])
        assert pd.isna(obs.population_view.df.loc[2, "mci_event_time"])

    def test_does_not_overwrite_existing_mci_date(self):
        pop = _base_pop(1)
        pop.loc[0, COLUMNS.DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.MCI_STATE
        pop.loc[0, COLUMNS.PREVIOUS_DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.BBBM_STATE
        original_time = pd.Timestamp("2028-07-01")
        pop.loc[0, "mci_event_time"] = original_time

        obs = _make_observer(pop, clock_time=pd.Timestamp("2030-01-01"))
        obs.on_collect_metrics(_collect_event(pop.index))

        assert obs.population_view.df.loc[0, "mci_event_time"] == original_time


class TestDementiaTransitionDetection:
    def test_detects_mci_to_dementia(self):
        pop = _base_pop(2)
        pop.loc[0, COLUMNS.DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_DISEASE_STATE
        pop.loc[0, COLUMNS.PREVIOUS_DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.MCI_STATE

        clock = pd.Timestamp("2035-01-01")
        obs = _make_observer(pop, clock_time=clock)
        obs.on_collect_metrics(_collect_event(pop.index))

        expected_time = clock + pd.Timedelta(days=182)
        assert obs.population_view.df.loc[0, "dementia_event_time"] == expected_time
        assert pd.isna(obs.population_view.df.loc[1, "dementia_event_time"])

    def test_does_not_overwrite_existing_dementia_date(self):
        pop = _base_pop(1)
        pop.loc[0, COLUMNS.DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_DISEASE_STATE
        pop.loc[0, COLUMNS.PREVIOUS_DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.MCI_STATE
        original_time = pd.Timestamp("2033-01-01")
        pop.loc[0, "dementia_event_time"] = original_time

        obs = _make_observer(pop, clock_time=pd.Timestamp("2035-01-01"))
        obs.on_collect_metrics(_collect_event(pop.index))

        assert obs.population_view.df.loc[0, "dementia_event_time"] == original_time


class TestTreatmentStartDetection:
    def test_detects_treatment_start(self):
        pop = _base_pop(2)
        pop.loc[0, COLUMNS.TREATMENT_STATE] = TREATMENT_DISEASE_MODEL.TREATMENT_EFFECT

        clock = pd.Timestamp("2035-01-01")
        obs = _make_observer(pop, clock_time=clock)
        obs.on_collect_metrics(_collect_event(pop.index))

        expected_time = clock + pd.Timedelta(days=182)
        assert obs.population_view.df.loc[0, "treatment_start_time"] == expected_time
        assert pd.isna(obs.population_view.df.loc[1, "treatment_start_time"])

    def test_does_not_overwrite_existing_treatment_start(self):
        pop = _base_pop(1)
        pop.loc[0, COLUMNS.TREATMENT_STATE] = TREATMENT_DISEASE_MODEL.TREATMENT_EFFECT
        original_time = pd.Timestamp("2034-01-01")
        pop.loc[0, "treatment_start_time"] = original_time

        obs = _make_observer(pop, clock_time=pd.Timestamp("2035-01-01"))
        obs.on_collect_metrics(_collect_event(pop.index))

        assert obs.population_view.df.loc[0, "treatment_start_time"] == original_time


class TestTreatmentEndDetection:
    def test_detects_treatment_end(self):
        pop = _base_pop(1)
        pop.loc[0, COLUMNS.TREATMENT_STATE] = TREATMENT_DISEASE_MODEL.NO_EFFECT_AFTER_TREATMENT
        pop.loc[0, "treatment_start_time"] = pd.Timestamp("2034-01-01")

        clock = pd.Timestamp("2045-01-01")
        obs = _make_observer(pop, clock_time=clock)
        obs.on_collect_metrics(_collect_event(pop.index))

        expected_time = clock + pd.Timedelta(days=182)
        assert obs.population_view.df.loc[0, "treatment_end_time"] == expected_time

    def test_no_end_without_start(self):
        """treatment_end_time should stay NaT if treatment_start_time was never set."""
        pop = _base_pop(1)
        pop.loc[0, COLUMNS.TREATMENT_STATE] = TREATMENT_DISEASE_MODEL.NO_EFFECT_AFTER_TREATMENT
        # treatment_start_time is NaT

        obs = _make_observer(pop, clock_time=pd.Timestamp("2045-01-01"))
        obs.on_collect_metrics(_collect_event(pop.index))

        assert pd.isna(obs.population_view.df.loc[0, "treatment_end_time"])

    def test_does_not_overwrite_existing_treatment_end(self):
        pop = _base_pop(1)
        pop.loc[0, COLUMNS.TREATMENT_STATE] = TREATMENT_DISEASE_MODEL.NO_EFFECT_AFTER_TREATMENT
        pop.loc[0, "treatment_start_time"] = pd.Timestamp("2034-01-01")
        original_time = pd.Timestamp("2044-01-01")
        pop.loc[0, "treatment_end_time"] = original_time

        obs = _make_observer(pop, clock_time=pd.Timestamp("2045-01-01"))
        obs.on_collect_metrics(_collect_event(pop.index))

        assert obs.population_view.df.loc[0, "treatment_end_time"] == original_time


# ---------------------------------------------------------------------------
# Multiple simultaneous transitions
# ---------------------------------------------------------------------------


class TestMultipleTransitions:
    def test_different_simulants_different_transitions(self):
        pop = _base_pop(4)
        # Simulant 0: BBBM -> MCI
        pop.loc[0, COLUMNS.DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.MCI_STATE
        pop.loc[0, COLUMNS.PREVIOUS_DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.BBBM_STATE
        # Simulant 1: MCI -> Dementia
        pop.loc[1, COLUMNS.DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.ALZHEIMERS_DISEASE_STATE
        pop.loc[1, COLUMNS.PREVIOUS_DISEASE_STATE] = ALZHEIMERS_DISEASE_MODEL.MCI_STATE
        # Simulant 2: enters treatment
        pop.loc[2, COLUMNS.TREATMENT_STATE] = TREATMENT_DISEASE_MODEL.TREATMENT_EFFECT
        # Simulant 3: no transition

        clock = pd.Timestamp("2035-01-01")
        obs = _make_observer(pop, clock_time=clock)
        obs.on_collect_metrics(_collect_event(pop.index))

        expected = clock + pd.Timedelta(days=182)
        df = obs.population_view.df
        assert df.loc[0, "mci_event_time"] == expected
        assert df.loc[1, "dementia_event_time"] == expected
        assert df.loc[2, "treatment_start_time"] == expected
        # Simulant 3 unchanged
        assert pd.isna(df.loc[3, "mci_event_time"])
        assert pd.isna(df.loc[3, "dementia_event_time"])
        assert pd.isna(df.loc[3, "treatment_start_time"])


# ---------------------------------------------------------------------------
# Line list output
# ---------------------------------------------------------------------------


class TestBuildLineList:
    def test_output_columns(self):
        pop = _base_pop(2)
        obs = _make_observer(pop)
        result = obs._build_line_list()

        expected_cols = {
            "simulant_id",
            "draw",
            "random_seed",
            "location",
            "date_of_birth",
            "date_of_death",
            "date_of_bbbm_incidence",
            "date_of_treatment_initiation",
            "date_of_treatment_cessation",
            "date_of_mci_incidence",
            "date_of_dementia_incidence",
        }
        assert set(result.columns) == expected_cols

    def test_scalar_columns(self):
        pop = _base_pop(3)
        obs = _make_observer(pop)
        result = obs._build_line_list()

        assert (result["draw"] == 0).all()
        assert (result["random_seed"] == 42).all()
        assert (result["location"] == "united_states_of_america").all()

    def test_dates_mapped_correctly(self):
        pop = _base_pop(1)
        pop.loc[0, "date_of_birth"] = pd.Timestamp("1960-01-01")
        pop.loc[0, "exit_time"] = pd.Timestamp("2040-06-01")
        pop.loc[0, COLUMNS.BBBM_ENTRANCE_TIME] = pd.Timestamp("2020-06-01")
        pop.loc[0, "mci_event_time"] = pd.Timestamp("2030-07-02")
        pop.loc[0, "dementia_event_time"] = pd.Timestamp("2035-01-01")
        pop.loc[0, "treatment_start_time"] = pd.Timestamp("2031-01-01")
        pop.loc[0, "treatment_end_time"] = pd.Timestamp("2042-01-01")

        obs = _make_observer(pop)
        result = obs._build_line_list()

        row = result.iloc[0]
        assert row["date_of_birth"] == pd.Timestamp("1960-01-01")
        assert row["date_of_death"] == pd.Timestamp("2040-06-01")
        assert row["date_of_bbbm_incidence"] == pd.Timestamp("2020-06-01")
        assert row["date_of_mci_incidence"] == pd.Timestamp("2030-07-02")
        assert row["date_of_dementia_incidence"] == pd.Timestamp("2035-01-01")
        assert row["date_of_treatment_initiation"] == pd.Timestamp("2031-01-01")
        assert row["date_of_treatment_cessation"] == pd.Timestamp("2042-01-01")

    def test_alive_simulant_has_nat_death_date(self):
        pop = _base_pop(1)
        pop.loc[0, "exit_time"] = pd.NaT

        obs = _make_observer(pop)
        result = obs._build_line_list()

        assert pd.isna(result.iloc[0]["date_of_death"])


class TestToCsv:
    def test_writes_csv(self, tmp_path):
        pop = _base_pop(3)
        obs = _make_observer(pop)
        obs.line_list = obs._build_line_list()

        path = tmp_path / "line_list.csv"
        obs.to_csv(str(path))

        loaded = pd.read_csv(path)
        assert len(loaded) == 3
        assert "simulant_id" in loaded.columns

    def test_builds_if_empty(self, tmp_path):
        pop = _base_pop(2)
        obs = _make_observer(pop)
        # line_list is empty DataFrame
        assert obs.line_list.empty

        path = tmp_path / "line_list.csv"
        obs.to_csv(str(path))

        loaded = pd.read_csv(path)
        assert len(loaded) == 2


class TestOnSimulationEnd:
    def test_auto_saves_csv_when_results_dir_set(self, tmp_path):
        pop = _base_pop(3)
        obs = _make_observer(pop)
        obs.results_dir = tmp_path

        obs.on_simulation_end(_collect_event(pop.index))

        output = tmp_path / "simulant_line_list.csv"
        assert output.exists()
        loaded = pd.read_csv(output)
        assert len(loaded) == 3
        assert "simulant_id" in loaded.columns

    def test_no_save_when_results_dir_none(self, tmp_path):
        pop = _base_pop(2)
        obs = _make_observer(pop)
        obs.results_dir = None

        obs.on_simulation_end(_collect_event(pop.index))

        # line_list built in memory but no file written
        assert len(obs.line_list) == 2
        assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Slow integration test (requires artifact + full sim)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntegration:
    def test_line_list_with_simulation(self):
        """Run the full simulation for a few steps and verify the line list."""
        from pathlib import Path

        from loguru import logger
        from vivarium import InteractiveContext

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
                "population": {"population_size": 1000},
                "intervention": {"scenario": "bbbm_testing_and_treatment"},
            },
        )

        # Step to ~2035 (roughly 27 steps of 182 days each)
        for _ in range(27):
            sim.step()

        observer = [
            c
            for c in sim._component_manager._components
            if isinstance(c, SimulantLineListObserver)
        ][0]
        line_list = observer._build_line_list()

        # Basic structure checks
        assert len(line_list) > 0
        assert "simulant_id" in line_list.columns
        assert "date_of_birth" in line_list.columns
        assert "date_of_mci_incidence" in line_list.columns
        assert "date_of_dementia_incidence" in line_list.columns
        assert "date_of_treatment_initiation" in line_list.columns
        assert "date_of_treatment_cessation" in line_list.columns

        # Every simulant should have a date of birth
        assert line_list["date_of_birth"].notna().all()

        # Some simulants should have transitioned to MCI by 2035
        assert line_list["date_of_mci_incidence"].notna().any(), (
            "Expected some MCI transitions by 2035"
        )

        # Draw and location should be constant
        assert (line_list["draw"] == 0).all()
        assert (line_list["location"] == "united_states_of_america").all()

        # BBBM incidence should be set for all simulants
        assert line_list["date_of_bbbm_incidence"].notna().all()
