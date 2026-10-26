"""Compare the result files of two single simulations, for a smoke test of an artifact change.

usage: python notebooks/compare_smoke_sims.py <results_dir_a> <results_dir_b> [label_a label_b]

Each directory is the ``results/`` folder that ``simulate run`` writes. For every result
file the values are summed over age and year, by sex and by the file's own category
column (disease state, transition, cause of death, ...), and shown side by side with the
ratio b / a. Totals over the whole run are the headline; a ratio far from 1 on
person-time in a state or on deaths points at a change in the rates the artifact carries.
"""
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

FILES = {
    "person_time_alzheimers_disease_and_other_dementias": ["sub_entity"],
    "transition_count_alzheimers_disease_and_other_dementias": ["sub_entity"],
    "deaths": ["entity"],
    "ylls": ["entity"],
    "ylds": ["entity"],
    "counts_new_simulants": [],
    "counts_baseline_tests_among_eligible": ["testing_state"],
    "person_time_eligible_for_bbbm_testing": [],
}


def load(results_dir: Path, name: str, by: list[str]) -> pd.Series:
    df = pd.read_parquet(results_dir / f"{name}.parquet")
    keys = ["sex", *by]
    return df.groupby(keys, observed=True)["value"].sum()


def main(dir_a: str, dir_b: str, label_a: str = "a", label_b: str = "b") -> None:
    a_dir, b_dir = Path(dir_a), Path(dir_b)
    for name, by in FILES.items():
        if not (a_dir / f"{name}.parquet").exists():
            continue
        a, b = load(a_dir, name, by), load(b_dir, name, by)
        table = pd.DataFrame({label_a: a, label_b: b})
        table["ratio"] = table[label_b] / table[label_a]
        print(f"\n== {name} (summed over age and year)")
        print(table.round(3).to_string())
    # Year trend of the headline quantities, both sexes together
    print("\n== by year: person-time in any AD state, new simulants, deaths")
    rows = {}
    for label, d in [(label_a, a_dir), (label_b, b_dir)]:
        pt = pd.read_parquet(d / "person_time_alzheimers_disease_and_other_dementias.parquet")
        ns = pd.read_parquet(d / "counts_new_simulants.parquet")
        de = pd.read_parquet(d / "deaths.parquet")
        rows[(label, "person_time")] = pt.groupby("event_year", observed=True)["value"].sum()
        rows[(label, "new_simulants")] = ns.groupby("event_year", observed=True)[
            "value"
        ].sum()
        rows[(label, "deaths")] = de.groupby("event_year", observed=True)["value"].sum()
    print(pd.DataFrame(rows).round(1).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
