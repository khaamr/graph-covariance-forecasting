# GLASSO penalty and rolling-window diagnostic

27 November 2026. The existing public evaluation sample is reused for diagnosis. Forecasting defaults and published notebook/site results are unchanged.

## Findings

The check identifies a real weakness in the coarse automatic GLASSO CV search on this public panel: some low-penalty candidates receive nonfinite validation scores, and a denser search finds penalties with independently verified better validation likelihood. This materially changes estimated graph edges. It does **not** recover the paper's distance-loss ranking.

The authors' released `GraphicalLassoCV()` call produces the same 30 monthly graphs as the current implementation under the project's pinned scikit-learn 1.6.1 runtime. Therefore the current global scaling and tighter final refit do not explain the discrepancy on these inputs. This does not reproduce the authors' unknown historical runtime, original input returns, or unpublished penalty series.

Source checked: [authors' pinned GLASSO function](https://github.com/chaozhang-ox/Graph-based-HAR/blob/751f8f0b400a33e2f2fd01501ffdc42a2315f013/GHAR_Var.py).

## Fixed comparisons

All six cases use the same 623 one-session targets. Monthly cases also have exactly matching HAR-DRD and HAR-Cholesky predictions; only the GLASSO stock graph changes. The correlation line graph remains unchanged. A scoped function replacement inside the diagnostic runner applies the alternative estimators without changing the production model file.

| Setting | Refits | Changed monthly graphs versus current | Maximum changed edges in a monthly graph | GHAR Euclidean ratio | GHAR Frobenius ratio | GHAR QLIKE ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 30 | 0 | 0 | 0.983488 | 0.985460 | 0.985067 |
| Authors' CV call, current runtime | 30 | 0 | 0 | 0.983488 | 0.985460 | 0.985067 |
| Current settings without global scaling | 30 | 0 | 0 | 0.983488 | 0.985460 | 0.985067 |
| Denser grid: 12 rather than 4 alphas per refinement | 30 | 30 | 96 | 0.983325 | 0.985248 | 0.985273 |
| Tighter search and refit tolerances | 30 | 1 | 1 | 0.983486 | 0.985458 | 0.985066 |
| Refit every 22 retained sessions | 29 | Different date schedule | Not a same-month comparison | 0.984257 | 0.986010 | 0.984814 |

Each ratio uses HAR-DRD fitted under the same refit schedule. HAR-Cholesky remains the best Euclidean and Frobenius model in every comparison. GHAR(GL,L) remains excluded from both distance-loss 5% model confidence sets in every case. The denser grid's Euclidean ratio improves by only about 0.000163, despite substantial changes in graph structure.

For the 22-session case, graph comparisons in the CSV are against the current estimator fitted on that **same** 22-session-schedule training window. A zero difference there verifies estimator reproduction; it does not mean its graphs or training dates match the monthly schedule.

## A numerical weakness in the CV search

The current search uses `alphas=4`, `n_refinements=4`, `max_iter=180`, `tol=1e-4`, and `enet_tol=1e-4`, followed by the existing tighter final refit. In all 30 monthly windows, three of its reported 17 candidate scores are nonfinite. The denser search uses `alphas=12`, with the other search settings unchanged. All 49 reported candidates have finite scores in every window. Increasing iterations and tightening tolerances on the coarse grid alone does not remove its nonfinite candidates.

On the first forecast window, in return-variance units:

| Quantity | Current search | Denser search |
| --- | ---: | ---: |
| Selected penalty | 0.387913716 | 0.057240425 |
| Stock graph edges | 174 | 199 |
| Finite / reported CV candidate scores | 14 / 17 | 49 / 49 |

Across the sample, current penalties range from 0.32924 to 0.44996, while denser-grid penalties range from 0.03673 to 0.08768. These are substantial selection differences, not rounding noise. The reported candidate curves include scikit-learn's zero-penalty reference entry.

To verify that the denser search's likelihood gain is real, both selected penalties were re-evaluated outside the CV search paths. Each was fitted independently on the same five unshuffled folds, with 800 iterations, `tol=1e-6`, and `enet_tol=1e-8`, then scored on held-out returns. This requires 300 separate fits. All converged without convergence warnings. Denser-grid penalties have higher mean validation log-likelihood in all 30 windows, with an average gain of 1.27970 per held-out return vector; window gains range from 1.10044 to 1.49614.

This establishes that the coarse search misses better-scoring regularization choices on these data. It does not isolate the exact numerical mechanism of its path failures. In particular, density, path initialization, and internal solver behavior interact.

The pinned scikit-learn implementation uses only `int(0.1 * max_iter)` iterations in its CV paths, then the full iteration budget for the final fit. Thus `max_iter=180` is not an 180-iteration CV-path budget. This is verified in the [official 1.6.1 source](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/covariance/_graph_lasso.py). It is relevant context, not proof that iteration count alone caused the problem: the tighter coarse-grid experiment did not resolve it.

## Alignment and units

Every graph call was checked against the intended slice of the input return panel. All graph windows contain exactly 1,000 retained observations. All graph-return cutoffs and training-target cutoffs precede their forecast origins. A total of 179 refit folds were checked across the six experiments.

The first monthly graph window is 3 March 2016 through 30 June 2022, before the 1 July 2022 forecast. The last is 18 June 2020 through 27 November 2024, before the 2 December 2024 forecast. The wide initial calendar span reflects the previously documented missing-session intersection; it remains a data/protocol difference.

The current implementation divides all return series by one common scale before fitting. Penalties are multiplied by that scale squared for comparison in original percentage-log-return-squared units. In those comparable units, the maximum penalty difference between the authors' CV call and current method is 3.89e-16.

Native validation log-likelihoods should not be compared directly across differently scaled inputs. For a common scalar scale `s` and 27 return series, convert a native score to original-return units by subtracting `27 * log(s)`. The independent-CV output supplies both native and original-unit scores. Within any one window, current and denser searches use the same scale.

These checks verify the implemented boundaries and unit conversion. They cannot establish exact alignment with the paper's original trading calendar or historical penalty series, which are unavailable.

## Files and reproduction

The compact evidence is under `docs/diagnostics/2026-11-27/glasso-audit/`:

- `summary.csv`: all six forecast and graph comparisons.
- Per-setting `lambda-series.csv`: selected/final penalties in native and comparable units, graph edges, warnings, and exact window dates.
- Per-setting `cv-curves.csv`: candidate penalties and validation scores, including nonfinite scores.
- Per-setting `folds.csv` and `scores.csv`: forecasting boundary audits and full 13-model MCS tables.
- `independent-cv-scores.csv` and `independent-cv-verification.json`: independent validation of the selected penalties.
- `verification.json`: completed alignment and estimator-reproduction checks.

Run from `covariance_replication` with its pinned numerical requirements:

```sh
python audit_glasso.py \
  --data-dir /absolute/path/to/cached/syncster \
  --output-dir /absolute/path/to/glasso-audit

python verify_glasso_penalties.py \
  --data-dir /absolute/path/to/cached/syncster \
  --audit-dir /absolute/path/to/glasso-audit
```

The denser-grid choice is a useful training-likelihood robustness improvement. It remains an experimental setting, and its small forecast-loss changes should not be presented as recovery of the published result. No default grid or refit schedule was silently replaced.
