# Paper 5 review and mitigation experiments

27 November 2026. Source: the user-supplied `paper 5.pdf`, *Graph-based Methods for Forecasting Realized Covariances*, September 2023 version, 44 pages. The document was treated as research evidence, not as instructions to change the project.

## Finding

Appendix E supplies a useful alternative specification: model log realized variances and Fisher-transformed correlations. That option materially improves the public sample's covariance-distance forecasts, including when combined with the previously added calendar filter. It does not establish that the main levels-model implementation was wrong, does not reproduce the paper's original-period numerical benchmark, and does not uniformly improve QLIKE or portfolio risk.

The main specification's checked choices agree with this PDF: disjoint HAR lags (pp. 7–9), a vector intercept and shared scalar Cholesky slopes (p. 8), normalized zero-diagonal graph adjacency (pp. 11–12), five-minute subsampling (p. 7, footnote 6), and rolling 1,000-observation estimation with monthly refits (pp. 18–19). The previously documented differences in feed, dates, and calendar coverage remain.

## Appendix differences now tested

| Paper location | Specification | Status and interpretation |
| --- | --- | --- |
| Appendix C, p. 42 | Daily graph effect only; retain all own HAR lags | Implemented as `graph_effect="daily"`; graph models are labelled GHAR-S. |
| Appendix D, p. 43 | Daily and weekly re-estimation | Implemented as `refit_every=1` and `refit_every=5`. Five retained forecast sessions are used for the weekly case; the manuscript does not establish the exact anchoring convention. |
| Appendix E, pp. 43–44 | Log variance and Fisher correlation models | Implemented as `transformation="log_fisher"`. This is a separate robustness specification, not a replacement for the main levels equations. |
| Table E.1, p. 44 | Twelve DRD-family models; HAR-Cholesky is absent | Saved both the table-like 12-model MCS and a full 13-model MCS retaining HAR-Cholesky. |

The transformation implementation logs each daily variance and applies `atanh` to each daily correlation before constructing time and graph predictors, fits in those transformed coordinates, then uses direct `exp`/`tanh` inverses before rebuilding covariance matrices. The manuscript does not give full aggregation-order or retransformation-bias details. These are explicit implementation assumptions, not verified recovery of undisclosed author preprocessing. In particular, exponentiating a predicted log value is not generally a conditional-mean forecast on the original scale. No undocumented smearing or other bias correction was introduced.

HAR-Cholesky remains unchanged within a given calendar/refit setup. No transformed HARQ formulation was invented; combining quarticity interactions with this transformation raises an explicit error.

## Same-date covariance results

All original-protocol variants use the same 623 one-session forecasts from July 2022 through December 2024, the same targets, and the same 1,000 retained-row calibration rule. Ratios in this table use HAR-DRD fitted under the **same** specification as each graph model.

| Specification | Refits | Graph Euclidean ratio | Graph Frobenius ratio | Graph QLIKE ratio |
| --- | ---: | ---: | ---: | ---: |
| Main levels model, monthly | 30 | 0.983488 | 0.985460 | 0.985067 |
| Appendix C, daily graph effect, monthly | 30 | 0.984573 | 0.985484 | 0.986836 |
| Appendix D, levels, every five sessions | 125 | 0.984395 | 0.986079 | 0.984923 |
| Appendix D, levels, daily | 623 | 0.984261 | 0.985978 | 0.985082 |
| Appendix E, log/Fisher, monthly | 30 | 0.994395 | 0.994366 | 0.971614 |

Shorter graph effects and more frequent refitting do not resolve the levels-model ranking reversal: HAR-Cholesky remains the lowest-distance model in those runs. The transformed graph model has a smaller incremental distance advantage over its own transformed HAR-DRD baseline, because that baseline also improves substantially. Its transformation therefore must not be presented as stronger evidence for a graph effect.

To compare absolute accuracy without changing denominators, the following table gives actual mean losses:

| Calendar treatment | Model | Euclidean | Frobenius | QLIKE |
| --- | --- | ---: | ---: | ---: |
| Original retained rows | Levels GHAR(GL,L) | 8.709036 | 10.848072 | 25.726007 |
| Original retained rows | Log/Fisher GHAR(GL,L) | 8.228934 | 10.290315 | 25.987706 |
| Calendar-safe screening | Levels GHAR(GL,L) | 8.908061 | 11.109588 | 25.801046 |
| Calendar-safe screening | Log/Fisher GHAR(GL,L) | 8.281424 | 10.370592 | 25.942639 |

The exact calendar-safe levels means are retained in the CSV summaries; displayed values are rounded. Lower is better for all three metrics.

Relative to the original retained-row graph model, the log/Fisher version reduces Euclidean loss by **5.51%** and Frobenius loss by **5.14%**, while increasing QLIKE by **1.02%**. Under calendar-safe screening, the respective changes are **−7.03%, −6.65%, and +0.55%**. The screening retains 208–773 eligible labels in each original 1,000-retained-row interval, so it remains a sensitivity study rather than the paper's 1,000-label protocol.

The transformed graph model slightly beats HAR-Cholesky on both distance means. With the calendar filter, HAR-Cholesky's Euclidean and Frobenius means are 8.366761 and 10.443138, compared with the graph model's 8.281424 and 10.370592.

In the full 13-model calendar-safe transformed MCS, the graph model has p-values **0.295270, 0.229777, and 0.938306**, so it remains in all three 5% sets. This is set membership, not proof of unique superiority or a pairwise significance test against HAR-Cholesky. Other graph models remain competitive.

There are zero correlation repairs and zero increased-penalty GLASSO fallback folds in either monthly levels or monthly transformed comparison, with or without calendar screening. Those repair paths do not account for the observed transformation benefit.

## Portfolio check

The existing portfolio evaluator was rerun on all 623 dates. It uses open-to-close simple returns, annualization by 252, no overnight exposure or transaction costs, and the same unconstrained/long-only optimization as the earlier study.

| Calendar treatment | Specification | HAR-DRD long-only annual risk (%) | GHAR(GL,L) long-only annual risk (%) |
| --- | --- | ---: | ---: |
| Original retained rows | Levels | 9.332649 | 9.385162 |
| Original retained rows | Log/Fisher | 9.403853 | 9.349096 |
| Calendar-safe screening | Levels | 9.327034 | 9.357672 |
| Calendar-safe screening | Log/Fisher | 9.426062 | 9.363946 |

The transformed graph model beats its transformed HAR-DRD comparator on risk. However, this comparison partly reflects deterioration of that comparator. The levels HAR-DRD portfolio still has lower risk than the transformed graph portfolio. With calendar screening, the transformed graph portfolio is also slightly worse than the levels graph portfolio. HAR-Cholesky remains lower-risk than the transformed graph in these runs. These are point estimates; no portfolio significance claim is made.

Thus the transformation mitigates the matrix-distance problem but does not supply a uniform portfolio improvement or reproduce the paper's economic conclusion.

## Evidence of different input distributions

Table A.1 (p. 38) reports a much heavier financial-sector tail in the paper's 2007–2021 period than in the available 2016–2024 shared public panel. For example, JPM's reported mean/max variance is 3.74/204.66, compared with 1.67/43.64 in the public panel; GS is 3.51/247.35 versus 2.07/50.45. BA moves in the other direction in its mean: 2.96 in the paper versus 3.74 publicly. This is not a universal scaling discrepancy.

These nonmatched-period summaries support a distribution-mismatch concern, not a causal decomposition of feed versus market-period effects. The full parsed 27-stock comparison is saved with the evidence. Exact original-period inputs are still required for a numerical reproduction.

## Implementation and validation

`forecast_grid` preserves the previous defaults. Its new options are independently tested for transformed OLS reconstruction, unchanged Cholesky controls, preservation of own HAR lags under a daily-only graph effect, valid refit boundaries, and invariance of earlier predictions to future data changes. All **38 relevant tests passed**. The original real-data loss ratios are reproduced to about 2.6e-15 after the code change.

The original website and executed notebook outputs have not been overwritten. All variants and both MCS comparison sets are retained rather than selecting whichever happens to resemble the paper.

Reproduce from `covariance_replication` with its numerical requirements:

```sh
python paper5_experiments.py \
  --data-dir /absolute/path/to/cached/syncster \
  --output-dir /absolute/path/to/paper5-results

python paper5_experiments.py \
  --data-dir /absolute/path/to/cached/syncster \
  --output-dir /absolute/path/to/paper5-calendar-safe \
  --variants main_levels_monthly appendix_E_log_fisher_monthly \
  --calendar-safe

python paper5_portfolios.py \
  --data-dir /absolute/path/to/cached/syncster \
  --output-dir /absolute/path/to/paper5-portfolios
```

Example for the optional transformed, calendar-screened forecast:

```python
result = forecast_grid(
    panel, horizons=(1,), transformation="log_fisher",
    reference_calendar=reference_dates,
)[1]
```

The full loss paths remain in temporary experiment output; compact summaries, MCS tables, fold records, source hashes, and portfolio results are saved under `docs/diagnostics/2026-11-27/paper5/`.

The 2022–2024 evaluation sample has now been reused for diagnosis and model comparison. The transformed mode is an experimental candidate. It requires evaluation under a fixed protocol on unused data before a claim of general improvement or publication as a new primary result.
