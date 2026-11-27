# Balanced covariance-model experiment

27 November 2026. The task was to improve Euclidean, Frobenius, QLIKE, and long-only portfolio risk together. The model-selection protocol in [design.json](design.json) was fixed before evaluating the new candidate. All dates are retrospective and have been used in earlier project research.

## Outcome

The learned joint selector improves the three mean covariance losses but worsens later-period portfolio risk, so it fails the primary joint criterion. The predefined equal-weight covariance pool is a better balanced experimental option: all four point estimates improve in both evaluation periods. Its QLIKE and risk confidence intervals still include zero change. There is no joint statistical success or fresh-confirmation claim.

Highlighting the fixed pool after inspecting these comparisons is exploratory model selection. Its weights were not retuned, and the failed learned candidate remains in every result table. The fixed pool is an ensemble extension, not a numerical reproduction of the paper's original GHAR model.

## Implemented fixed ensemble

For each forecast date, form five positive-definite covariance forecasts using only earlier observations:

1. HAR-DRD.
2. GHAR(GL,L) in variance/correlation levels.
3. GHAR(GL,L) in log-variance/Fisher-correlation coordinates, inverted with exp/tanh.
4. HAR-Cholesky.
5. The arithmetic average of yesterday's covariance, the previous five-session mean, and the previous 22-session mean. These overlapping averages are an additional expert, not the paper's disjoint HAR regression predictors.

The fixed forecast is the equal-weight matrix average:

\[
\widehat H_t^{\mathrm{pool}}=\frac15\sum_{j=1}^{5}\widehat H_{j,t}.
\]

All constituent matrices are positive definite, so their positive weighted average is positive definite. This is a matrix-validity property, not a guarantee of lower future error or portfolio variance.

The graph models use the independently verified denser 12-alpha CV grid. Their rolling regressions still refit monthly. Complete preceding HAR windows and complete targets are required against the screened SPY calendar; that reference is not a complete exchange calendar. The log/Fisher transformation follows the previously documented direct-inverse interpretation, with no unreported retransformation correction.

A direct API is supplied in `joint_pool.py`:

```python
from joint_pool import fixed_covariance_pool

# Shape: dates × 5 experts × assets × assets, in the documented expert order.
improved_covariance = fixed_covariance_pool(expert_bank)
```

It requires no learned selector or future outcomes. `prepare_joint_experts.py` builds the causal expert bank. The API was checked against all 642 evaluated ensemble forecasts and matches them exactly.

## Dates and comparability

The input files are the same pinned public trade-bar dataset, with stored file hashes. The expert bank has 1,020 eligible origins from 1 July 2022 through 19 October 2026. The learned model requires 378 earlier expert labels. All controls are evaluated on its common eligible dates:

| Period | Origins | First date | Last date |
| --- | ---: | --- | --- |
| Development | 245 | 2024-01-08 | 2024-12-31 |
| Reused later benchmark | 397 | 2025-01-02 | 2026-10-19 |

The later sample is smaller than the earlier project's 443-origin benchmark because complete-lag calendar screening excludes additional dates. The new results must not be compared directly to old absolute risk figures on different dates. All new model/control comparisons use identical dates and targets.

## Fixed-ensemble results

Ratios below use HAR-DRD on the same dates, with the same calendar screening. Lower is better.

| Period | Euclidean ratio | Frobenius ratio | QLIKE ratio | HAR-DRD annual risk (%) | Fixed-pool annual risk (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development 2024 | 0.952003 | 0.949705 | 0.995885 | 7.419627 | 7.410661 |
| Reused 2025–2026 | 0.987767 | 0.987351 | 0.996494 | 9.721778 | 9.673684 |

On the later benchmark this corresponds to 1.22% lower Euclidean loss, 1.26% lower Frobenius loss, 0.35% lower QLIKE, and 0.99% lower portfolio **variance**. Annualized standard deviation falls by about 0.0481 percentage points, or 0.49% relatively. Risk is measured from open-to-close returns with no overnight exposure or transaction costs, using exact long-only optimization.

Exploratory paired stationary-bootstrap intervals use 10,000 draws, mean block length 10, and the fixed seed in the design:

| Later benchmark endpoint | Change (%) | Paired 95% interval |
| --- | ---: | --- |
| Euclidean mean loss | −1.2233 | [−2.1665, −0.5534] |
| Frobenius mean loss | −1.2649 | [−2.2917, −0.5256] |
| QLIKE mean loss | −0.3506 | [−0.8326, +0.0177] |
| Long-only portfolio variance | −0.9870 | [−2.2218, +0.6633] |

The distance-loss intervals exclude zero; QLIKE and portfolio-risk intervals do not. Because this is an exploratory comparison on reused data, these intervals do not establish a new independently confirmed advantage. Development also has all-four favorable point estimates, while its QLIKE and risk intervals include zero.

## Learned candidate and strong controls

The more ambitious candidate uses the same expert bank. It searches a fixed simplex grid with weights in thirds plus the uniform vector, diagonal shrinkage 0 or 0.1, and four scalar-calibration recipes. Positive scale is bounded to [0.5, 2].

At each quarterly refit, it uses at most 504 earlier labels, reserving the last 126 for validation. Scale recipes are fitted only on the earlier fitting block. The chosen setting minimizes the largest validation ratio across Euclidean loss, Frobenius loss, QLIKE divergence, and realized long-only portfolio variance, with HAR-DRD retained as an available fallback. Calibration is then refitted on the full past window; no same-date or future target enters selection.

QLIKE divergence subtracts the realized-matrix-dependent constant from ordinary QLIKE. This leaves paired model-loss differences unchanged and provides a nonnegative training normalization. The published evaluation tables retain ordinary QLIKE.

The construction is motivated by combining covariance forecasts and shrinkage toward a structured target; it is not an implementation of the Ledoit–Wolf shrinkage-intensity estimator. See the [primary covariance-shrinkage paper](https://ledoit.net/honey_abstract.htm) for the broader principle.

Later-period controls show the tradeoff:

| Model | Euclidean ratio | Frobenius ratio | QLIKE ratio | Long-only annual risk (%) |
| --- | ---: | ---: | ---: | ---: |
| HAR-DRD | 1.000000 | 1.000000 | 1.000000 | 9.721778 |
| Levels GHAR(GL,L), dense grid | 1.013334 | 1.012807 | 1.192064 | 9.984072 |
| Log/Fisher GHAR(GL,L), dense grid | 0.973542 | 0.968018 | 0.998614 | 9.660256 |
| HAR-Cholesky | 1.010665 | 1.003375 | 1.075982 | 9.678457 |
| Arithmetic HAR covariance | 1.089300 | 1.098111 | 1.016754 | 9.477321 |
| Fixed uniform covariance pool | 0.987767 | 0.987351 | 0.996494 | 9.673684 |
| Learned joint calibrated pool | 0.975033 | 0.970653 | 0.993031 | 9.819951 |

The learned pool reduces mean losses by 2.50%, 2.93%, and 0.70%, but raises realized portfolio variance by 2.03%. Its primary joint rule fails. The arithmetic expert has lower risk but worse distance losses. The log/Fisher expert has favorable all-four point estimates, but its Euclidean, QLIKE, and risk intervals include zero; its separate control intervals are saved. No model is claimed to dominate every strong control.

The fixed pool's maximum condition number is about 238 in the later period, versus about 131,829 for the levels graph expert. This is a descriptive stability improvement. It does not establish that conditioning caused its realized risk difference.

## Validation and reproducibility

All 44 relevant tests passed, including six new analytic/causal tests. They check known scalar calibration, positive covariance construction, solver objectives against exact constrained optimization, a perfect-baseline fallback, and invariance of earlier choices to future realized outcomes.

Validation portfolio selection uses accelerated simplex optimization with a convex duality-gap bound, increasing numerical iterations as needed. The largest observed relative objective-gap bound is 9.69e-5; no exact-solver fallbacks were required. Final portfolio evaluation uses the exact CVXPY/OSQP procedure. Fit, validation, and refit labels are all earlier than their forecast origins.

`design.json` predates candidate evaluation. The run manifest preserves hashes of the evaluated sources and expert archive. The fixed-average API was factored out after evaluation without changing its arithmetic; all evaluated fixed forecasts were rechecked for exact equality. Additional confidence intervals for predefined controls were calculated as exploratory summaries without refitting or retuning them.

Run from `covariance_replication` with its numerical requirements:

```sh
python prepare_joint_experts.py \
  --data-dir /absolute/path/to/cached/syncster \
  --output-dir /absolute/path/to/joint-experts

python run_joint_pool.py \
  --experts /absolute/path/to/joint-experts/experts.npz \
  --config docs/joint-pool/design.json \
  --output-dir /absolute/path/to/joint-results
```

Compact tables, quarter-by-quarter choices, solver diagnostics, expert lineage, and uncertainty summaries are saved with this report. Raw prices and large forecast archives remain outside the repository. The original replication notebook and website are unchanged; the ensemble is an experimental extension requiring new data for confirmation.
