# Integration Testing Notes for vivarium_csu_alzheimers

## 1. Codebase Understanding

### Simulation Structure
- **Sim period**: 2022-01-01 to 2100-12-31
- **Step size**: 182 days (~6 months, guarantees 2 steps per year)
- **Population**: 20,000 simulants (ages 25-100)
- **Artifact**: `united_states_of_america.hdf`

### Disease Progression (3 states)
1. **BBBM** (preclinical) → 2. **MCI** (mild cognitive impairment) → 3. **Dementia**
- BBBM→MCI transition uses a Weibull hazard (shape=1.22, scale=6.76, mean ~8 years)
- MCI→Dementia uses a data-loaded transition rate

### Three Scenarios
| Scenario | BBBM Testing | Treatment |
|---|---|---|
| `baseline` | No | No |
| `bbbm_testing` | Yes | No |
| `bbbm_testing_and_treatment` | Yes | Yes |

### BBBM Testing Component (`components/testing.py`)
- **Eligibility**: BBBM state, age 65-80, no positive test, not tested in last 3-5 years
- **Testing rate** (piecewise-linear, propensity threshold):

| Code knots | Year | Rate |
|---|---|---|
| 1 | 2027-01-01 | 0% |
| 2 | 2030-07-01 | 10% |
| 3 | 2045-01-01 | 50% |
| 4 | 2055-01-01 | 60% (plateau) |

- **Positive diagnosis probability**: 50% (independent each test)
- **Retesting interval**: uniform 3-5 years (drawn from `[6,7,8,9,10]` time steps)
- On initialization: assigns historical BBBM test dates to avoid an unrealistic surge

### Treatment Component (`components/treatment.py`)
- **Treatment probability ramp** (piecewise-linear, propensity threshold):

| Code knots | Year | Probability |
|---|---|---|
| 1 | 2027-01-01 | 0% |
| 2 | 2035-07-01 | 30% |
| 3 | 2100-01-01 | 80% |

- **Treatment state machine**:
  `susceptible → positive_test (transient) → waiting_for_treatment → treatment_effect → waning_effect → no_effect_after_treatment`
  or `→ no_effect_never_treated` (if declining)
- **Treatment effect on BBBM→MCI**: reduces transition rate via relative risk
  - Full effect: RR loaded from data (~0.4-0.6)
  - Waning: linearly interpolates from RR_min back to 1.0
- **Dwell times**: waiting ~6mo, effect ~6yr, waning ~11yr
- **Duration**: 90% get 9 months; 10% get 1-8 months random

## 2. Spec Discrepancy (Fixed)

**PR 1888 spec** says 10% testing coverage at mid-2030 (`(2030.5, 10%)`).
**Code** originally had 10% at mid-2035 (`(2035-07-01, 10%)`). User confirmed spec is
correct; fixed `data_values.py` to `(2030-07-01, 10%)`. There is also a PR on GitHub
fixing this.

## 3. Expected Values at Key Time Points

### Testing rates (from spec, via piecewise linear interpolation)

**At 2030-07-01** (mid-2030):
- Exactly at knot: **10.0%** propensity threshold

**At 2035-07-01** (mid-2035):
- Between knots (2030-07-01, 10%) and (2045-01-01, 50%)
- Interpolated: 10% + 40% * (5.0/14.5) ≈ **23.8%** propensity threshold

### Treatment probabilities (from code)

**At 2030-07-01**:
- Between (2027-01-01, 0%) and (2035-07-01, 30%)
- Interpolated: ~0.3 * (3.5/8.5) ≈ **10.6%** of BBBM-positive simulants

**At 2035-07-01**:
- Exactly at knot: **30.0%** of BBBM-positive simulants

### What fraction of all eligible simulants are "tested" at a given time step?
The testing rate is a propensity threshold. A simulant is tested at time t if:
1. They are eligible (BBBM state, age 65-80, no positive test, test interval elapsed)
2. Their `testing_propensity < testing_rate(t)`

Since propensity is uniform[0,1], the fraction of newly-eligible simulants who get tested
equals the testing rate. But the **cumulative** fraction tested is complicated by:
- Testing history initialization
- Age-in/age-out dynamics
- Retesting intervals (3-5 years)
- Positive tests removing simulants from the testing pool

## 4. Proposed Testing Approaches

### Approach A: Causal Chain Integration Tests (Recommended)

Tests the full pipeline using InteractiveContext at key time points. Validates that
testing drives treatment and treatment drives slower progression.

**Strengths**:
- Tests the actual integrated behavior end-to-end
- Directly validates the causal logic the user cares about
- Catches interaction bugs between components

**Weaknesses**:
- Slow (~60-90s per sim setup)
- Stochastic variation requires tolerance bands or large populations
- Needs the HDF artifact file present

**Key tests**:
1. At 2030/2035, fraction of eligible tested matches propensity threshold
2. BBBM-positive simulants are entering treatment (in treatment scenario)
3. No treatment occurs in baseline/testing-only scenarios
4. Treatment reduces BBBM→MCI transition count vs baseline

### Approach B: Scenario Comparison Tests

Runs two identical-seed simulations with different scenarios and compares outcomes.

**Strengths**:
- Directly tests the "what changed" question
- More robust than absolute checks (relative comparisons absorb noise)
- Clearly demonstrates the intervention effect

**Weaknesses**:
- 2x the run time
- CRN (common random numbers) may not perfectly eliminate stochastic variation
- Hard to pin down exact expected magnitudes

**Key tests**:
1. `bbbm_testing` vs `baseline`: testing scenario should have BBBM tests, baseline should not
2. `bbbm_testing_and_treatment` vs `bbbm_testing`: treatment scenario should have
   simulants in treatment states, testing-only should not
3. `bbbm_testing_and_treatment` vs `baseline`: treatment scenario should have fewer
   MCI transitions (due to treatment effect slowing BBBM→MCI)

### Approach C: Parameter Verification + Focused Component Tests

Unit-test the piecewise-linear functions and propensity logic, combined with lightweight
integration tests that run just a few steps.

**Strengths**:
- Fast (seconds, not minutes)
- Precise — tests exact numerical values without stochastic noise
- Good for catching regressions in the rate functions

**Weaknesses**:
- Doesn't test real integration between components
- May miss bugs in how components interact via population columns

**Key tests**:
1. `_get_bbbm_testing_rate()` returns correct values at knot points and between them
2. `start_treatment_probs()` returns correct propensity threshold at key dates
3. Eligibility masks compute correctly for known population states

### Approach D: Hybrid (Recommended Final Approach)

Combine the approaches:
1. **Fast unit tests** (Approach C) for rate functions and eligibility logic
2. **Integration tests** (Approach A) for the causal chain at 2030/2035
3. **Scenario comparisons** (Approach B) for treatment-reduces-progression

This gives fast feedback on simple logic, while catching integration bugs with slower tests.

Structure:
```
tests/
├── conftest.py                    # Shared fixtures (sim setup, helpers)
├── test_sample.py                 # Existing placeholder
├── test_testing_rates.py          # Approach C: unit tests for rate functions
├── test_integration.py            # Approach A+B: InteractiveContext tests
```

Mark the InteractiveContext tests as `@pytest.mark.slow` so they can be skipped for
fast iteration but run in CI.

## 5. Key Design Decisions

### Population size for tests
- Full sim uses 20,000 — too slow for tests
- Recommend 5,000 for integration tests (enough for stable fractions among eligible)
- Only ~15-20% of population is in the eligible age range (65-80), so a 5,000 pop
  gives ~750-1000 eligible simulants, which is enough for ~5% tolerance bands

### How far to step
- 2022 → 2030: 16 time steps (8 years × 2 steps/year)
- 2022 → 2035: 26 time steps
- Stepping is fast once the sim is initialized; the bottleneck is setup

### Tolerance for stochastic checks
- Use statistical tests (e.g., binomial confidence intervals) rather than fixed tolerances
- For N=1000 eligible and expected rate p=0.10, 95% CI is roughly [0.08, 0.12]
- Alternative: use a fixed random seed and test exact values (more brittle but deterministic)

### Fixture reuse
- Setting up InteractiveContext is expensive; use `scope="module"` fixtures
- Step the sim forward in-order (don't reset between tests)
- This means tests must be ordered: 2030 checks first, then step to 2035 and check

## 6. Smoke Test Findings

InteractiveContext works with the model spec. Key observations:

- Constructor: `InteractiveContext(spec_path, configuration={...})` — do NOT call
  `setup()` separately; the constructor handles it
- At init (2022), all simulants have `bbbm_test_result='not_tested'` and
  `treatment='susceptible_to_treatment'` — correct since testing starts in 2027
- Pop grows over time due to AlzheimersIncidence adding new simulants
- The sim starts at 2022-01-01, each step adds 182 days
- ~42 columns available per simulant
- Stepping to 2030 requires ~17 steps; to 2035 requires ~27 steps

### Key column names (for test assertions)
- Disease state: `alzheimers_disease_and_other_dementias`
  - Values: `alzheimers_blood_based_biomarker_state`, `alzheimers_mild_cognitive_impairment_state`, `alzheimers_disease_state`
- Testing: `testing_state`, `bbbm_test_result`, `bbbm_test_date`, `next_bbbm_test_date`, `testing_propensity`
- Treatment: `treatment`, `treatment_propensity`, `treatment_duration`
- Demographics: `age`, `sex`, `alive`, `entrance_time`

## 7. Small Population Size Issue for Treatment→Progression Tests

With 5,000 simulants, only ~15-20% are in the eligible age range (65-80) and in BBBM
state. After testing rates (~4-10%) and positive diagnosis (50%) and treatment acceptance
(~10-30%), the treated subpopulation is extremely small (~10-20 simulants). This makes
statistical testing of treatment's effect on BBBM→MCI progression unreliable.

**Mitigation options**:
1. Use larger populations (20,000+) but accept slower tests
2. Run to later years (2045+) when testing/treatment rates are higher
3. Test the mechanism indirectly: verify the value pipeline applies the correct RR
   to treated simulants (deterministic check, no stochastic noise)
4. Compare scenarios over enough time steps for the effect to compound

## 8. Integration Test Results (10,000 simulants, with code fix)

Notebook: `tests/integration_tests.ipynb`

### Test 1: Testing Coverage

**At ~end of 2030** (sim time 2030-12-21):
- 15,568 alive; 3,978 eligible for BBBM test
- 177 eligible simulants tested (4.4% of eligible)
- Propensity check (threshold ~10%):
  - Low-propensity (<0.10): 294 eligible, **156 tested (53.1%)**
  - High-propensity (>=0.10): 3,684 eligible, **21 tested (0.6%)**
- The 53% tested-rate among low-propensity is expected: new simulants entering via
  AlzheimersIncidence are assigned a future first-test date (0-4.5 years out) to avoid
  testing surges. Investigation confirms 100% of untested low-propensity simulants are
  recent entrants (post-2030) with future `next_bbbm_test_date`
- The 0.6% in high-propensity is from the rate increasing past 10% by end-2030

**At ~end of 2035** (sim time 2035-12-15):
- 18,120 alive; 3,892 eligible
- 335 eligible tested (8.6%)
- Propensity check (threshold ~23.8%):
  - Low-propensity (<0.238): 643 eligible, **305 tested (47.4%)**
  - High-propensity (>=0.238): 3,249 eligible, **30 tested (0.9%)**

### Test 2: Testing Drives Treatment

At end of 2035 (bbbm_testing_and_treatment scenario):
- 760 alive BBBM-positive simulants
- 154 (20.3%) in treatment pipeline; 606 (79.7%) declined
- **0 simulants in treatment without a positive test** (correct)
- Treatment propensity check (threshold ~30%):
  - Low-propensity (<0.30): 224 positive, **154 in treatment (68.8%)**
  - High-propensity (>=0.30): 536 positive, **0 in treatment (0.0%)**

### Test 3: Treatment Reduces Progression

At 2045, scenario comparison:
| State | Testing Only | Testing+Treatment | Difference |
|---|---|---|---|
| BBBM | 9,529 | 9,625 | **+96** (retained) |
| MCI | 4,405 | 4,359 | **-46** (reduced) |
| Dementia | 6,403 | 6,363 | **-40** (reduced) |

Treatment scenario retains more in BBBM and has fewer MCI/Dementia — **confirms
treatment effect on BBBM→MCI progression**.

### Test 4: Baseline Verification

At 2035 (baseline scenario):
- BBBM tests: **0** (correct)
- Non-susceptible treatment: **0** (correct)
- CSF/PET testing: 1,580 PET + 730 CSF (working correctly)

## 9. Scenario Comparison Visualization

Notebook: `tests/scenario_comparison.ipynb`

Runs all 3 scenarios (baseline, bbbm_testing, bbbm_testing_and_treatment) from 2022
to 2060 and collects per-step snapshots. Produces:

1. **Disease state trajectories** — BBBM, MCI, Dementia counts over time per scenario
2. **Differences from baseline** — isolates the intervention effect
3. **Testing coverage** — BBBM-positive tests and eligible population over time
4. **Treatment activity** — simulants in treatment pipeline and active treatment effect
5. **Stacked area charts** — disease state proportions per scenario
6. **Animated bar chart** — HTML5 video of disease state counts across scenarios
7. **Summary table** — counts at key years (2030, 2035, 2040, 2045, 2050, 2055)

Key observations from the visualization:
- Testing and treatment effects become visible after ~2030 (when testing starts ramping)
- Treatment effect compounds over time: by 2060, the gap in MCI/Dementia counts widens
- The treatment scenario retains more simulants in BBBM state (slower progression)
- Baseline has zero BBBM testing and zero treatment activity throughout

## 10. Deep Dive: Why Testing/Treatment Numbers Seem Modest at 2055

At 2055, the summary table shows (for `bbbm_testing_and_treatment`, 10k pop):

| Metric | Value |
|---|---|
| Alive | ~20,100 |
| BBBM state | ~9,500 (47%) |
| BBBM+ (tested positive) | ~2,900 (14% of alive) |
| In treatment pipeline | ~1,200 (41% of BBBM+) |

This may look low given a 60% testing rate and ~46% treatment probability, but
the numbers are correct. Here is why.

### Rate functions verified

| Year | Testing rate | Treatment prob |
|---|---|---|
| 2027 | 1.4% | 1.7% |
| 2030 | 10.0% | 12.3% |
| 2035 | 23.8% | 30.0% |
| 2040 | 37.6% | 33.9% |
| 2045 | 50.5% | 37.8% |
| 2050 | 55.5% | 41.6% |
| 2055 | 60.0% | 45.5% |

Both functions match the spec knots and interpolate correctly.

### Why only ~2,900 BBBM+ out of ~9,500 BBBM?

Three compounding filters limit who gets tested:

1. **Narrow age window**: Only BBBM simulants aged 65-80 are eligible. At 2055,
   only 43% of BBBM simulants (4,142 of 9,536) fall in this range. Half (50%)
   are over 80 and have aged out.

2. **Propensity ceiling**: The testing rate plateaus at 60%, so 40% of eligible
   simulants will *never* be tested regardless of time.

3. **Staggered first-test dates**: New entrants via `AlzheimersIncidence` get a
   random future `next_bbbm_test_date` (0-4.5 years out). At any snapshot, recent
   entrants haven't been tested yet. At 2055, ~900 of ~1,500 below-threshold
   eligible simulants are recent entrants awaiting their first test.

4. **50% positive diagnosis rate**: Each test only has a 50% chance of returning
   positive, so negative-tested simulants re-enter the queue for 3-5 years.

### Propensity distribution is correct (not a bug)

Initial concern: 52% of remaining eligible had propensity >= 0.60, suggesting
a non-uniform distribution. Investigation showed this is **selection bias**, not
a bug:

- Among **all** BBBM 65-80 (n=4,142): propensity >= 0.60 = **40.3%** (correct)
- Among the **not-yet-positive** subset (n=3,210): propensity >= 0.60 = **52.0%**
- The 932 simulants who tested positive are **100% low-propensity** (< 0.60)

Removing positive-tested simulants from the denominator depletes the low-propensity
group and inflates the high-propensity fraction. The underlying distribution is
perfectly uniform across the full population.

### Why only ~1,200 in treatment out of ~2,900 BBBM+?

The treatment decision is **one-time and permanent**. When a simulant tests positive
for BBBM, they immediately either enter treatment (propensity < treatment_prob) or
permanently decline (`no_effect_never_treated` has no exit transitions).

At 2055 (10k pop, `bbbm_testing_and_treatment`):
- 2,946 BBBM+ alive
- 1,215 (41.2%) entered the treatment pipeline
- 1,731 (58.8%) permanently declined
- 0 stuck in `susceptible_to_treatment` (all processed correctly)

The 58.8% decline rate reflects the **historical average** of treatment probability
across all years when positive tests occurred, not the current-year rate. Breakdown
of when declined simulants tested positive:

| Year | Declined | Treatment prob at time |
|---|---|---|
| 2029-2034 | ~27 | 9-26% |
| 2035-2039 | ~100 | 30-33% |
| 2040-2044 | ~291 | 34-37% |
| 2045-2049 | ~470 | 38-41% |
| 2050-2055 | ~843 | 42-46% |

Simulants who tested positive in early years faced very low treatment probabilities,
permanently locking them out even as the probability later rose. A simulant with
`treatment_propensity = 0.35` who tested positive in 2040 (treatment prob ~34%)
was permanently declined, even though by 2055 the threshold reached 46%.

Treatment propensity of declined simulants confirms this:
- 0% have propensity < 0.20 (very-low-propensity always got treated)
- 2.3% have propensity < 0.30 (almost all early-low-propensity got treated)
- 32.6% have propensity < 0.60 (many would qualify NOW but were locked out earlier)

### Why baseline and testing-only have identical disease counts

The disease state numbers (BBBM, MCI, Dementia) are identical between baseline and
testing-only scenarios at every time point. This is correct:

- Testing is pure observation — it does not affect disease progression
- Vivarium uses Common Random Numbers (CRN) keyed by simulant attributes, so
  disease model random draws are identical across scenarios
- Only the `bbbm_testing_and_treatment` scenario differs because the treatment
  RiskEffect modifies the BBBM→MCI transition rate

### Consistency checks passed

- Propensity distribution is uniform across the full alive population
- 0 positive simulants stuck in `susceptible_to_treatment`
- 0 BBBM+ simulants without a test date
- All rate functions return correct values at all time points

### Design question for the research team

The permanent-decline behavior is the dominant factor limiting treatment coverage.
Whether this is the desired model design depends on the research question — should
simulants who declined treatment early be re-evaluated as treatment availability
increases? Currently `no_effect_never_treated` is an absorbing state.

## 11. Treatment Effect Quantification and the 71.8% Zero-Benefit Question

### Notebooks
- `tests/treatment_effect_analysis.ipynb` — CRN-matched comparison of per-simulant outcomes
- `tests/treatment_timing_investigation.ipynb` — deep dive into treatment state at exit

### Method

Ran both `bbbm_testing` and `bbbm_testing_and_treatment` scenarios to 2070 with
10,000 simulants. Using CRN, tracked each simulant's BBBM entry/exit times in both
scenarios. For treated simulants, computed:

- **Additional preclinical years** = exit_time(treatment) - exit_time(testing_only)
- **DALYs averted per treatment** = additional_years × (DW_MCI - DW_BBBM) where
  DW_BBBM=0, DW_MCI=0.031, DW_Dementia=0.31

### Top-line results

| Metric | Value |
|---|---|
| Treated simulants (entered pipeline) | 3,969 |
| Uncensored (exited BBBM in both scenarios) | 3,032 |
| Zero additional preclinical years | 2,177 (71.8%) |
| Has additional preclinical years | 855 (28.2%) |
| Mean additional years (all uncensored) | 1.32 |
| Mean additional years (benefit group only) | 4.68 |
| Mean DALYs averted per treatment | 0.18 |

### Why 71.8% show zero benefit — RESOLVED (no bug)

Initial expectation: with RR=0.528, the probability of "rescue" (transition avoided
under treatment) should be ~47.2% per step, so more than half of treated simulants
should see benefit. But 71.8% showed identical exit times.

Investigation confirmed the model is working correctly. The 71.8% is fully explained
by four compounding factors:

#### Factor 1: Deaths in BBBM (25.7% of zero-benefit)

Treatment RR only modifies the BBBM→MCI transition rate, not mortality. Simulants
who die while in BBBM exit at the same time in both scenarios regardless of treatment.
Of 2,177 zero-benefit simulants, **559 (25.7%) exited via death**. Of 855
has-benefit simulants, **0 (0%) exited via death** — logically consistent since
death timing is identical across scenarios.

#### Factor 2: `waiting_for_treatment` period (14.3% of zero-benefit)

The treatment state machine has a 1-step waiting period (`waiting_for_treatment`)
with RR=1.0. Simulants evaluated during this step get no protection. Additionally,
10% of simulants fail to advance from waiting to treatment_effect (decline
probability). Due to component priority ordering (disease model runs first at
default priority, treatment model runs at priority 7), a simulant's first step
with the treatment RR applied is actually 2 steps after entering the pipeline.

#### Factor 3: CRN mechanism (~52.8% theoretical rescue rate)

With Common Random Numbers, both scenarios use identical random draws for the
BBBM→MCI transition. The draw `d` is compared against:
- `p_base` (base transition probability) in the testing-only scenario
- `p_treat = p_base × RR` in the treatment scenario

A simulant transitions in testing-only when `d < p_base`. Treatment "rescues" them
only when `p_treat ≤ d < p_base`. The rescue probability given transition is:
`(p_base - p_treat) / p_base = 1 - RR = 1 - 0.528 = 0.472`

So even with perfect, instant treatment, **at most 47.2% of transitions can be
rescued per step**. The remaining 52.8% have draws below p_treat and transition
regardless.

#### Factor 4: Treatment dropout and short duration (~6pp residual)

- 10% of treated simulants receive a short treatment duration (1-8 months instead
  of 9 months), which scales down both treatment_effect and waning_effect dwell times
- Simulants in `waning_effect` have an interpolated RR between 0.528 and 1.0, with
  rescue probability decreasing toward zero
- Simulants who reach `no_effect_after_treatment` have RR=1.0 (no protection)

#### Accounting

After removing deaths: 1,618 alive zero-benefit / 2,473 alive uncensored = **65.4%**
Per-step alive-only zero-benefit fractions:

| Steps after pipeline | Zero-benefit % | Expected (CRN) |
|---|---|---|
| 2 | 66.5% | ~52.8% |
| 3 | 54.9% | ~52.8% |
| 4 | 56.1% | ~52.8% |
| 5 | 50.5% | ~52.8% |
| 6 | 52.5% | ~52.8% |
| 7-15 | ~55-68% | ~52.8% |

The ~6pp excess above 52.8% is consistent with the combined effect of the
waiting_for_treatment step (RR=1.0), treatment dropout, and waning_effect periods.

### RR verification (confirmed correct)

Directly queried the vivarium pipeline for transition rates:

| Group | Mean rate | Ratio |
|---|---|---|
| `treatment_effect` simulants | pipeline rate | pipeline/base = **0.5275** |
| `susceptible_to_treatment` | pipeline rate | identical across scenarios (diff=0.00) |

The per-simulant ratio was exactly 0.5275 for all 20 sampled simulants. The rate
conversion is linear (`probability = rate × step_size/365`), so
`p_treat / p_base = RR` exactly — no non-linearity distortion.

### CRN verification (confirmed working)

For susceptible simulants present in both scenarios at the same time:
- Transition rates are **exactly identical** (diff = 0.00e+00)
- Disease states diverge only for treated simulants (as expected)

### Conclusion

The treatment effect is implemented correctly. The seemingly high zero-benefit rate
(71.8%) is an expected consequence of the CRN design combined with deaths, treatment
timing, and dropout — not a bug. The treatment does provide meaningful benefit when
it works: 4.68 additional preclinical years for the 28.2% who benefit, averaging
1.32 years across all treated simulants.

## 12. Hazard Ratio Degradation Analysis

### Notebook
- `tests/hr_degradation_analysis.ipynb` — standalone, well-commented notebook for sharing

### Method

Using Kaplan-Meier survival curves and Cox Proportional Hazards regression, measured
the hazard ratio (HR) for MCI progression (treated vs control) under five increasingly
inclusive analysis strategies. The dialed-in relative risk is RR=0.528.

### Five Analysis Strategies

| # | Strategy | Time zero | Censoring | Control group | Expected HR |
|---|---|---|---|---|---|
| 1 | As-treated (effect only) | treatment_effect start | treatment_effect end | Refused (same scenario) | ~0.53 (matches RR) |
| 2 | As-treated (effect + waning) | treatment_effect start | waning_effect end | Refused | ~0.55-0.56 |
| 3 | Per-protocol (full course) | treatment_effect start | None | Refused | ~0.56-0.57 |
| 4 | From treatment offer | Pipeline entry | None | Refused | ~0.58-0.60 |
| 5 | Intention to treat (CRN) | BBBM entry | None | CRN-matched untreated | ~0.61-0.62 |

### Why HR degrades from RR

The observed HR attenuates toward 1.0 as more "diluted" (unprotected) person-time is
included in the analysis. Each strategy adds a source of dilution:

- **Strategy 1→2**: Waning period has RR interpolating from 0.528 back to 1.0
- **Strategy 2→3**: Post-treatment period has RR=1.0 (no protection)
- **Strategy 3→4**: Waiting-for-treatment period has RR=1.0 (1 step delay)
- **Strategy 4→5**: ITT includes simulants who never entered treatment, plus uses
  BBBM entry (earlier) as time zero, adding unprotected person-time at the start

### Significance

The close match between Strategy 1 HR (~0.531) and the dialed-in RR (0.528) confirms
the simulation correctly implements the treatment effect. The progressive degradation
demonstrates how real-world clinical trial designs (which can't perfectly isolate
treatment-effect windows) will observe attenuated effect sizes — a useful validation
that the model produces epidemiologically realistic patterns.

## 13. Vivarium Enhancement Ideas

Based on experience building analysis notebooks for this project, the following
enhancements to the vivarium framework would improve the researcher experience:

### 13.1 Configurable Logging Verbosity

**Problem**: Vivarium uses **loguru** (not Python's standard `logging` module) for
all its log output. This means `logging.getLogger().setLevel(logging.WARNING)` has
no effect on vivarium's INFO spam during `InteractiveContext` construction and
`sim.step()` calls. The correct workaround is:
```python
from loguru import logger
logger.disable("vivarium")
```
But this is non-obvious and catches researchers off guard.

**Suggestion**: Add a `log_level` parameter to `InteractiveContext` (and/or a
`vivarium.quiet_mode` configuration key) that configures loguru's level for vivarium
loggers without requiring users to know about the loguru dependency:
```python
sim = InteractiveContext(spec_path, configuration={'vivarium': {'log_level': 'WARNING'}})
```

### 13.2 Built-in State Transition Event Tracking

**Problem**: Tracking when simulants enter/exit specific states requires manual
per-step diffing of the population table — comparing column values before and after
each `sim.step()`. This is error-prone (component priority ordering means columns
may reflect post-transition states) and requires significant boilerplate code.

**Suggestion**: Add an optional event log that records state transitions as they
happen, accessible via something like `sim.get_event_log()`. Each entry would include:
- Simulant index
- Column name (e.g., `treatment`, `alzheimers_disease_and_other_dementias`)
- Old value, new value
- Simulation time

This would eliminate the need for manual "snapshot before step, diff after step"
patterns that are currently necessary for survival analysis.

### 13.3 Per-Simulant Event History / Audit Trail

**Problem**: To do survival analysis (KM curves, Cox regression), researchers need
per-simulant timelines: when did each simulant enter BBBM? When did they get tested?
When did treatment start/end? Currently this must be reconstructed by tracking state
changes at every step and building DataFrames manually.

**Suggestion**: An opt-in per-simulant event history that records key lifecycle events:
```python
sim = InteractiveContext(spec_path, configuration={'vivarium': {'track_history': True}})
# ... run simulation ...
history = sim.get_simulant_history(simulant_id=42)
# Returns: [(time, 'disease_state', 'susceptible', 'bbbm'), (time, 'treatment', 'susceptible', 'waiting'), ...]
```

### 13.4 Snapshot / Checkpoint Support for InteractiveContext

**Problem**: Running a simulation to year 2070 takes ~25 minutes. If you want to
analyze different metrics at different time points, you must either: (a) collect
everything in one run, or (b) re-run from scratch. Adding a new analysis often means
re-running the entire simulation.

**Suggestion**: Allow saving and restoring simulation state:
```python
sim.step_n(50)  # Run to ~2047
sim.save_checkpoint('checkpoint_2047.pkl')
# ... later, in a new session ...
sim = InteractiveContext.from_checkpoint('checkpoint_2047.pkl')
sim.step_n(10)  # Continue from 2047
```

### 13.5 Built-in CRN-Matched Scenario Comparison

**Problem**: Comparing outcomes across scenarios (e.g., treatment vs no-treatment)
with Common Random Numbers requires running two separate simulations and manually
joining results by simulant index. This is a very common analysis pattern but has
no framework support.

**Suggestion**: A utility that runs multiple scenarios in parallel and returns a
matched comparison DataFrame:
```python
from vivarium.tools import compare_scenarios
results = compare_scenarios(
    spec_path,
    scenarios={'control': {...}, 'treatment': {...}},
    track_columns=['disease_state', 'treatment'],
    duration_steps=50,
)
# Returns per-simulant, per-step data for both scenarios, pre-joined
```

### 13.6 Progress Bar for sim.step() and sim.step_n()

**Problem**: Long simulation runs (~50+ steps) provide no progress feedback beyond
log messages. In notebooks, this makes it hard to estimate completion time.

**Suggestion**: Optional `tqdm` integration:
```python
sim.step_n(50, progress=True)  # Shows a progress bar
```
