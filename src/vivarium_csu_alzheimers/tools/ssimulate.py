"""
====================================
Serial Simulate (ssimulate)
====================================

A command line tool for running multiple vivarium simulations serially on a
single machine.  Analogous to ``psimulate`` from ``vivarium_cluster_tools``
but without cluster infrastructure (Redis, RQ workers, etc.).

Usage::

    ssimulate run MODEL_SPECIFICATION BRANCH_CONFIGURATION [OPTIONS]

The tool reads a branch configuration file (same format as psimulate),
expands the parameter space (input draws x random seeds x branches), and
runs each combination one at a time.  Results are aggregated across all
runs and written as parquet files, matching psimulate's output layout.
"""

import itertools
import os
from datetime import datetime as dt
from pathlib import Path
from time import time

import click
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from vivarium.framework.engine import SimulationContext
from vivarium.framework.logging import (
    configure_logging_to_file,
    configure_logging_to_terminal,
)
from vivarium.framework.utilities import handle_exceptions
from vivarium.interface.utilities import get_output_model_name_string


# ---------------------------------------------------------------------------
# Branch configuration parsing
# ---------------------------------------------------------------------------

INPUT_DRAW_SEED = 123456
INPUT_DRAW_MAX = 500
RANDOM_SEED_SEED = 654321
RANDOM_SEED_MAX = 10_000


def calculate_input_draws(
    branch_config: dict,
    existing_draws: list[int] | None = None,
) -> list[int]:
    """Select input draws from the branch configuration.

    Uses the same reproducible seeding as ``psimulate`` so that draws are
    identical for a given ``input_draw_count``.
    """
    explicit = branch_config.get("input_draws")
    if explicit is not None:
        return sorted(explicit)

    count = branch_config.get("input_draw_count", 1)
    existing = set(existing_draws or [])
    pool = [d for d in range(INPUT_DRAW_MAX) if d not in existing]
    np.random.seed(INPUT_DRAW_SEED)
    draws = np.random.choice(pool, size=count, replace=False)
    return sorted(int(d) for d in draws)


def calculate_random_seeds(
    branch_config: dict,
    existing_seeds: list[int] | None = None,
) -> list[int]:
    """Select random seeds from the branch configuration.

    Uses the same reproducible seeding as ``psimulate``.
    """
    explicit = branch_config.get("random_seeds")
    if explicit is not None:
        return sorted(explicit)

    count = branch_config.get("random_seed_count", 1)
    existing = set(existing_seeds or [])
    pool = [s for s in range(RANDOM_SEED_MAX) if s not in existing]
    np.random.seed(RANDOM_SEED_SEED)
    seeds = np.random.choice(pool, size=count, replace=False)
    return sorted(int(s) for s in seeds)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict to dot-separated keys."""
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d: dict, sep: str = ".") -> dict:
    """Restore a dot-separated dict to nested form."""
    result: dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def expand_branch_templates(branch_templates: list[dict]) -> list[dict]:
    """Expand branch templates that contain list values into the Cartesian
    product of all list-valued parameters.

    Matches the expansion logic of ``vivarium_cluster_tools``.
    """
    expanded: list[dict] = []
    for template in branch_templates:
        flat = _flatten_dict(template)
        list_keys = [k for k, v in flat.items() if isinstance(v, list)]
        if not list_keys:
            expanded.append(template)
            continue
        list_values = [flat[k] for k in list_keys]
        scalar = {k: v for k, v in flat.items() if not isinstance(v, list)}
        for combo in itertools.product(*list_values):
            branch_flat = dict(scalar)
            for k, v in zip(list_keys, combo):
                branch_flat[k] = v
            expanded.append(_unflatten_dict(branch_flat))
    return expanded


def parse_branches(
    branch_config_path: str | Path,
) -> tuple[list[int], list[int], list[dict]]:
    """Parse a branch configuration YAML file.

    Returns
    -------
    input_draws
        The list of input draw numbers.
    random_seeds
        The list of random seed values.
    branches
        The fully-expanded list of branch configuration dicts, each suitable
        for merging into a simulation's configuration.
    """
    with open(branch_config_path) as f:
        config = yaml.safe_load(f)
    input_draws = calculate_input_draws(config)
    random_seeds = calculate_random_seeds(config)
    branches = expand_branch_templates(config.get("branches", [{}]))
    return input_draws, random_seeds, branches


# ---------------------------------------------------------------------------
# Simulation execution helpers
# ---------------------------------------------------------------------------


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively merge *update* into *base* (mutates *base*)."""
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _build_sim_config(
    branch_config: dict,
    input_draw: int,
    random_seed: int,
) -> dict:
    """Build the configuration override dict for a single simulation run."""
    sim_config: dict = {
        "randomness": {"random_seed": random_seed},
        "input_data": {"input_draw_number": input_draw},
    }
    _deep_update(sim_config, branch_config)
    return sim_config


def _collect_line_list(sim: SimulationContext) -> pd.DataFrame | None:
    """Attempt to retrieve a line list from any component that exposes one."""
    for component in sim._component_manager._components:
        line_list = getattr(component, "line_list", None)
        if isinstance(line_list, pd.DataFrame) and not line_list.empty:
            return line_list
    return None


def run_single_simulation(
    model_specification: str | Path,
    branch_config: dict,
    input_draw: int,
    random_seed: int,
    with_debugger: bool = False,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Run one simulation and return (metadata, results).

    The simulation does **not** write results to disk; that is the caller's
    responsibility.

    Returns
    -------
    metadata
        Dict with ``random_seed``, ``input_draw``, ``simulation_run_time``.
    results
        Dict mapping measure names to DataFrames (same as
        ``SimulationContext.get_results()``).  If a ``SimulantLineListObserver``
        is present, its line list is included under the key
        ``"simulant_line_list"``.
    """
    sim_config = _build_sim_config(branch_config, input_draw, random_seed)

    sim = SimulationContext(
        model_specification=str(model_specification),
        configuration=sim_config,
    )

    start = time()
    if with_debugger:
        runner = handle_exceptions(sim.run_simulation, logger, with_debugger)
        runner()
    else:
        sim.setup()
        sim.initialize_simulants()
        sim.run()
        sim.finalize()
        sim.report(print_results=False)
    run_time = time() - start

    results = sim.get_results()

    # Collect line list if present
    line_list = _collect_line_list(sim)
    if line_list is not None:
        results["simulant_line_list"] = line_list

    metadata = {
        "random_seed": random_seed,
        "input_draw": input_draw,
        "simulation_run_time": run_time,
    }
    return metadata, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def ssimulate() -> None:
    """A command line utility for running simulations serially.

    Reads a model specification and a branch configuration file (same format
    as psimulate) and runs every (draw, seed, branch) combination one at a
    time on the local machine.
    """


@ssimulate.command()
@click.argument(
    "model_specification",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.argument(
    "branch_configuration",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--artifact_path",
    "-i",
    type=click.Path(resolve_path=True),
    default=None,
    help="Override the artifact path for all runs.",
)
@click.option(
    "--results_directory",
    "-o",
    type=click.Path(resolve_path=True),
    default=Path("~/vivarium_results/").expanduser(),
    help="Top-level directory for results (default: ~/vivarium_results/).",
)
@click.option("--verbose", "-v", is_flag=True, help="Log verbosely.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-warning logs.")
@click.option(
    "--pdb",
    "with_debugger",
    is_flag=True,
    help="Drop into the Python debugger on error.",
)
def run(
    model_specification: str,
    branch_configuration: str,
    artifact_path: str | None,
    results_directory: str,
    verbose: bool,
    quiet: bool,
    with_debugger: bool,
) -> None:
    """Run simulations serially from a model spec and branch configuration.

    MODEL_SPECIFICATION is the path to a vivarium model specification YAML.
    BRANCH_CONFIGURATION is the path to a branch YAML (same format as
    psimulate) that defines the parameter space.
    """
    if verbose and quiet:
        raise click.UsageError("Cannot be both verbose and quiet.")
    verbosity = 1 + int(verbose) - int(quiet)
    configure_logging_to_terminal(verbosity=verbosity, long_format=False)

    # ---- parse branches ----
    input_draws, random_seeds, branches = parse_branches(branch_configuration)

    # ---- output directory setup (mirrors simulate / psimulate layout) ----
    model_name = get_output_model_name_string(artifact_path, model_specification)
    launch_time = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_root = Path(results_directory) / model_name / launch_time
    _ = os.umask(0o002)
    results_root.mkdir(parents=True, exist_ok=False)
    output_data_root = results_root / "results"
    output_data_root.mkdir(parents=True, exist_ok=False)

    configure_logging_to_file(output_directory=results_root)

    # ---- apply artifact path override ----
    if artifact_path:
        for branch in branches:
            branch.setdefault("input_data", {})["artifact_path"] = str(artifact_path)

    # ---- persist run configuration ----
    with open(results_root / "keyspace.yaml", "w") as f:
        yaml.dump(
            {
                "input_draws": input_draws,
                "random_seeds": random_seeds,
                "branches": branches,
            },
            f,
            default_flow_style=False,
        )

    # ---- run every combination ----
    total_jobs = len(input_draws) * len(random_seeds) * len(branches)
    all_results: dict[str, list[pd.DataFrame]] = {}
    all_metadata: list[dict] = []

    job_num = 0
    overall_start = time()

    for input_draw in input_draws:
        for random_seed in random_seeds:
            for branch in branches:
                job_num += 1
                flat = _flatten_dict(branch)
                desc_parts = [
                    f"{k}={v}"
                    for k, v in flat.items()
                    if "artifact_path" not in k
                ]
                logger.info(
                    f"[{job_num}/{total_jobs}] draw={input_draw}, "
                    f"seed={random_seed}, {', '.join(desc_parts)}"
                )

                metadata, results = run_single_simulation(
                    model_specification=model_specification,
                    branch_config=branch,
                    input_draw=input_draw,
                    random_seed=random_seed,
                    with_debugger=with_debugger,
                )

                # Tag every result DataFrame with job parameters
                for measure, df in results.items():
                    df = df.copy()
                    df["input_draw_number"] = input_draw
                    df["random_seed"] = random_seed
                    for k, v in flat.items():
                        if "artifact_path" not in k:
                            df[k.replace(".", "_")] = v
                    all_results.setdefault(measure, []).append(df)

                metadata.update(
                    {k: v for k, v in flat.items() if "artifact_path" not in k}
                )
                all_metadata.append(metadata)

                logger.info(
                    f"[{job_num}/{total_jobs}] done "
                    f"({metadata['simulation_run_time']:.1f}s)"
                )

    # ---- write aggregated results ----
    for measure, dfs in all_results.items():
        combined = pd.concat(dfs, ignore_index=True)
        output_file = output_data_root / f"{measure}.parquet"
        combined.to_parquet(output_file, index=False)

    pd.DataFrame(all_metadata).to_csv(
        results_root / "finished_sim_metadata.csv", index=False
    )

    total_time = time() - overall_start
    logger.info(
        f"All {total_jobs} simulations complete in {total_time:.1f}s.\n"
        f"Results written to {output_data_root}"
    )
