# Shared-function migration (MIC-7490) — investigation notebooks

Not for merge. These exist so the investigation behind MIC-7490 can be reviewed and
replicated from the PR.

## The finding

For `modelable_entity_id=24351` (Unadjusted dementia, post-mortality; DisMod-MR) at
`release_id=16`, `year_id=2023`, the DisMod draws behind the January 2026 artifact
differ from what the same call returns today:

| measure | today vs January artifact |
|---|---|
| excess mortality rate | **unchanged** — bit-reproduces, ratio exactly 1.00000 |
| prevalence | **0.48–0.63×**, rising monotonically with age |
| incidence | same ratio as prevalence |

The code path is demonstrably unchanged. Replaying the exact January call — same
kwargs, same `get_draws` line (5.1.7) — reproduces the artifact's EMR to every digit.
And `get_draws` and `get_model_estimates` return identical values today, with and
without the production downsample arguments, so the migration is not the cause.

That is what makes the prevalence result conclusive: the same deterministic pipeline
that bit-reproduces EMR returns prevalence and incidence at roughly half their January
values. Same `release_id`, same kwargs, different data.

## Notebooks

| notebook | what it shows |
|---|---|
| `notebooks/mcve_get_draws_vs_get_model_estimates.ipynb` | The MCVE. Start here. |
| `notebooks/mic_7489_vv_model12_2_comparison.ipynb` | Earlier V&V of the framework migration (MIC-7489). |

Two standalone probes sit alongside them:

| script | what it shows |
|---|---|
| `probe_exact_january_call.py` | That the exact January call, replayed today, **bit-reproduces** the January artifact's EMR. |
| `probe_sex_age_filters.py` | That the `sex_id` / `age_group_id` arguments our migrated code passes are no-ops. |

`probe_exact_january_call.py` is the tightest control available. The notebook's `old`
side omits `downsample=True, n_draws=500`, so it is not byte-identical to the call that
built the artifact; this probe replays the January kwargs verbatim (from commit
`4f75828`, with `gbd_constants.SOURCES.EPI == "epi"` confirmed at v4.2.6) on the same
`get_draws` line, and reproduces the artifact's EMR to every digit — ratio exactly
1.00000 at all 12 ages. That proves the call and the code path are deterministic and
unchanged, which is what makes the prevalence result conclusive: the same pipeline that
bit-reproduces EMR returns prevalence and incidence at 0.48–0.63×. Run it with an env
that has `get_draws` (`gbd_midnight`).

`probe_sex_age_filters.py` covers a question that doesn't arise in the notebook — its
`old` side omits those arguments, matching the pre-migration call — but it was a real
suspect during the investigation, so the evidence is kept here. Run it with the artifact
env. Neither probe writes any files.

The MCVE notebook runs four checks:

1. **Do the two APIs agree today?** Same kwargs to `get_draws` and
   `get_model_estimates`. Identical on every shared row and draw.
2. **Does the production downsample path agree?** `downsample=True, n_draws=500` vs
   `n_draws=500`. Also identical, within 1e-9.
3. **EMR today vs the January artifact.** `load_emr` applies no post-processing, so the
   artifact value *is* the raw draw value — a direct comparison with no
   simulation-science code in the path.
4. **Prevalence/incidence, new artifact vs January artifact.** Both artifacts multiply
   the raw draws by the same AD-proportion frame, so the proportions **cancel** in the
   ratio. This measures the raw data ratio without depending on the proportions being
   correct — only unchanged. (Verified separately: the CSV is untouched since
   2025-09-22 and `load_dementia_proportions` is byte-identical to its pre-migration
   version.)

## Running it

`get_draws` and `ihme_cc_get_estimates` pull incompatible folio/grpc versions and
cannot be installed together. Rather than requiring two kernels, the notebook runs in an
environment needing only **pandas** and shells out to each GBD interpreter via
`subprocess`. The code dispatched to each is a visible string literal in the notebook,
not a hidden import.

Any kernel with pandas works; this one was run with
`/ihme/homes/sbachmei/miniconda3/envs/vivarium_csu_alzheimers_simulation`. The two
interpreter paths are constants in the first code cell:

- `OLD_PY` — an env with `get_draws` (`/ihme/code/central_comp/miniconda/envs/gbd_midnight`)
- `NEW_PY` — an env with `ihme_cc_get_estimates` (the alz artifact env)

Fetches are cached as parquet under `/mnt/share/homes/sbachmei/scratch/alz/mcve_notebook`,
so a re-run is instant. Pass `force=True` to `fetch()` to re-pull from GBD.

## Which `get_draws` built the January artifact

**5.1.7 or 5.1.8**, derived rather than assumed:

- alz (commit `4f75828`, 2025-11-12) required `vivarium_inputs>=6.0.1`
- `vivarium_gbd_access` must have been 4.x — v5.0.0 moved `get_draws` out of
  `utilities`, and the January code calls `vi_utils.get_draws`
- v4.2.6 requires `vivarium_dependencies[...,gbd]`
- `vivarium-dependencies` 1.0.2 (2025-12-10, the last release before the build) pins
  `get_draws>=5.1.4,<6.0.0`
- Central Comp's date-stamped environments carry 5.1.7 from 2025-08-07 and 5.1.9 by
  2026-02-19

The notebook's `old` side runs 5.1.7, so it *is* January's `get_draws` — a behavioural
change in `get_draws` itself is excluded.

## Caveats for reviewers

- Everything here uses **Israel** (`location_id=85`), which is most-detailed — no
  location aggregation anywhere in the comparison. The USA shows the same ratio but
  adds aggregation over 51 states.
- `get_draws` returns more rows than `get_model_estimates` (56 vs 24 for prevalence):
  it zero-fills age groups below 40 and includes aggregates such as 27, 33 and 164.
  Comparisons use only the rows the two share.
- EMR has wider age coverage than prevalence within the same modelable entity
  (50 rows vs 24 from `get_model_estimates`). All shared rows are identical.
- The notebook filters to age groups present in `AGE_MAP`. An unmapped `age_group_id`
  survives the rename as a bare integer and can collide with an artifact `age_start` —
  `age_group_id=10` is ages 25–29, while `age_start=10.0` is ages 10–15. That collision
  silently produced a `0/0` and a wrong verdict before it was caught.
