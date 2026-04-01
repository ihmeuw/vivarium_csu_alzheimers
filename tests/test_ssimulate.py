"""Tests for the ssimulate serial simulation runner."""

from pathlib import Path

import pytest

from vivarium_csu_alzheimers.tools.ssimulate import (
    _flatten_dict,
    _unflatten_dict,
    calculate_input_draws,
    calculate_random_seeds,
    expand_branch_templates,
    parse_branches,
)


# ---------------------------------------------------------------------------
# Branch parsing unit tests
# ---------------------------------------------------------------------------


class TestCalculateInputDraws:
    def test_explicit_draws(self):
        config = {"input_draws": [0, 5, 10]}
        assert calculate_input_draws(config) == [0, 5, 10]

    def test_draw_count(self):
        config = {"input_draw_count": 3}
        draws = calculate_input_draws(config)
        assert len(draws) == 3
        assert all(0 <= d < 500 for d in draws)
        assert len(set(draws)) == 3  # unique

    def test_default_is_one(self):
        assert len(calculate_input_draws({})) == 1

    def test_reproducible(self):
        config = {"input_draw_count": 5}
        a = calculate_input_draws(config)
        b = calculate_input_draws(config)
        assert a == b


class TestCalculateRandomSeeds:
    def test_explicit_seeds(self):
        config = {"random_seeds": [42, 123]}
        assert calculate_random_seeds(config) == [42, 123]

    def test_seed_count(self):
        config = {"random_seed_count": 4}
        seeds = calculate_random_seeds(config)
        assert len(seeds) == 4
        assert all(0 <= s < 10_000 for s in seeds)

    def test_default_is_one(self):
        assert len(calculate_random_seeds({})) == 1

    def test_reproducible(self):
        config = {"random_seed_count": 5}
        a = calculate_random_seeds(config)
        b = calculate_random_seeds(config)
        assert a == b


class TestFlattenUnflatten:
    def test_roundtrip(self):
        original = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        flat = _flatten_dict(original)
        assert flat == {"a.b": 1, "a.c.d": 2, "e": 3}
        assert _unflatten_dict(flat) == original

    def test_empty(self):
        assert _flatten_dict({}) == {}
        assert _unflatten_dict({}) == {}


class TestExpandBranchTemplates:
    def test_no_lists(self):
        templates = [{"a": 1, "b": 2}]
        assert expand_branch_templates(templates) == templates

    def test_single_list(self):
        templates = [{"intervention": {"scenario": ["baseline", "treatment"]}}]
        result = expand_branch_templates(templates)
        assert len(result) == 2
        scenarios = {r["intervention"]["scenario"] for r in result}
        assert scenarios == {"baseline", "treatment"}

    def test_cartesian_product(self):
        templates = [{"a": [1, 2], "b": [3, 4, 5]}]
        result = expand_branch_templates(templates)
        assert len(result) == 6

    def test_multiple_templates(self):
        templates = [
            {"x": [1, 2]},
            {"y": [3, 4, 5]},
        ]
        result = expand_branch_templates(templates)
        assert len(result) == 5  # 2 + 3

    def test_nested_lists(self):
        templates = [
            {
                "intervention": {"scenario": ["a", "b"]},
                "input_data": {"artifact_path": ["/p1", "/p2"]},
            }
        ]
        result = expand_branch_templates(templates)
        assert len(result) == 4


class TestParseBranches:
    def test_parse_test_branch_file(self):
        branch_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "vivarium_csu_alzheimers"
            / "model_specifications"
            / "branches"
            / "ssimulate_test.yaml"
        )
        draws, seeds, branches = parse_branches(branch_path)
        assert len(draws) == 1
        assert len(seeds) == 1
        assert len(branches) == 2
        scenarios = {b["intervention"]["scenario"] for b in branches}
        assert scenarios == {"baseline", "bbbm_testing_and_treatment"}

    def test_total_jobs(self):
        """The scenarios.yaml should expand to 25 * 100 * 30 = 75000 jobs."""
        branch_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "vivarium_csu_alzheimers"
            / "model_specifications"
            / "branches"
            / "scenarios.yaml"
        )
        draws, seeds, branches = parse_branches(branch_path)
        assert len(draws) == 25
        assert len(seeds) == 100
        assert len(branches) == 30  # 3 scenarios * 10 artifacts
        assert len(draws) * len(seeds) * len(branches) == 75_000


# ---------------------------------------------------------------------------
# Slow integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSSimulateIntegration:
    def test_run_two_scenarios(self, tmp_path):
        """Run ssimulate with the test branch file (2 jobs) and verify output."""
        from loguru import logger

        from vivarium_csu_alzheimers.tools.ssimulate import run_single_simulation

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

        branch_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "vivarium_csu_alzheimers"
            / "model_specifications"
            / "branches"
            / "ssimulate_test.yaml"
        )

        draws, seeds, branches = parse_branches(branch_path)
        assert len(branches) == 2

        # Override artifact path and shrink population for speed
        for branch in branches:
            branch.setdefault("input_data", {})["artifact_path"] = str(artifact_path)
            branch.setdefault("population", {})["population_size"] = 100
            branch.setdefault("time", {})["end"] = {"year": 2035, "month": 1, "day": 1}

        all_measures = set()
        for branch in branches:
            metadata, results = run_single_simulation(
                model_specification=str(spec_path),
                branch_config=branch,
                input_draw=draws[0],
                random_seed=seeds[0],
            )

            assert metadata["simulation_run_time"] > 0
            assert len(results) > 0
            all_measures.update(results.keys())

            # The treatment scenario should produce a line list
            scenario = branch["intervention"]["scenario"]
            if scenario == "bbbm_testing_and_treatment":
                assert "simulant_line_list" in results
                ll = results["simulant_line_list"]
                assert "simulant_id" in ll.columns
                assert "date_of_birth" in ll.columns
                assert len(ll) > 0

        # Both runs should produce standard disease observer results
        assert "transition_count_alzheimers_disease_and_other_dementias" in all_measures
