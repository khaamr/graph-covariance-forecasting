# Diagnosing and mitigating covariance-forecast failures

27 November 2026. The [fixed protocol](design.json) defines seven diagnostic challengers, evaluated alongside the 17 earlier models on exactly the same 642 dates. The later comparison uses the same 397 dates from 2 January 2025 to 19 October 2026. The targets, returns, controls and portfolio rules remain unchanged.

## What the diagnostics found

**Underprediction after log transformations.** The original market-factor LASSO forecasts have a mean trace equal to 0.6848 of the realized covariance trace: the average sum of individual predicted variances is about 31.5% too low. Directly exponentiating a conditional log forecast generally does not produce a conditional mean on the original scale. The earlier Log/Fisher graph forecast also has a mean trace ratio of 0.8221. These are observed calibration diagnostics; they do not prove that retransformation alone causes all forecast errors.

**Cleaning suppresses a relevant stress episode.** The factor paper's four-standard-deviation filter replaces five eligible dates, 4–10 April 2025, with earlier nonflagged covariance averages. On the later benchmark, this changes mean input trace to 0.9014 of its raw value. The original rule is intentional in the published specification, but it can slow adaptation during a sustained volatility increase on this sample. We evaluate raw inputs as a separate extension, preserving the original default.

**The market-only factor leaves important residual relationships.** Using each date's realized factor decomposition, then zeroing cross-sector residual entries, produces an average Frobenius discrepancy equal to 30.63% of average realized matrix magnitude. This calculation uses same-date outcomes **only for structural diagnosis**. It is not a forecast, a usable oracle input, or a lower bound on all possible forecast losses. It shows that the authorized one-factor sector restriction approximates this universe poorly; the unavailable characteristic-factor portfolios could produce a different residual structure.

**Daily-return information is weaker than the available realized data.** The original DCC-NL has higher variance QLIKE divergence and correlation error than HAR-DRD. Its model uses one daily return vector per observation, while the target aggregates intraday returns. Both targets and portfolio returns cover open-to-close, so overnight mismatch is not offered as the explanation here. The realized-input HEAVY extension directly addresses the information gap.

Complete [component diagnostics](tables/component-diagnostics.csv) and [structural diagnostics](tables/structural-diagnostics.json) reproduce these quantities. These findings identify mechanisms worth testing; they do not attribute the full ranking to one isolated cause.

## Research and implemented changes

[Duan (1983), Smearing Estimate](https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017) motivates estimating original-scale conditional means through the empirical residual distribution. The scalar principle does not by itself establish an exact matrix correction. Our experimental matrix extension averages exp(L_hat + E_j), with paired symmetric training-residual matrices E_j. This retains noncommuting matrix structure and positive definiteness. It assumes past residual distributions remain relevant at the forecast date. We also add the positive-semidefinite second-moment contribution from training loading errors; independence approximations between loading errors and factor forecasts remain explicit limitations.

[Bauwens and Xu (2023), DCC- and DECO-HEAVY](https://orca.cardiff.ac.uk/id/eprint/150752/7/1-s2.0-S0169207022000450-main.pdf), equations 16–18 and 43–44, supplies the realized-covariance process used here. Each realized variance follows a free-intercept positive HEAVY recursion. Correlations follow the published recursion driven by the previous realized correlation matrix, with the empirical mean correlation as target. Estimation uses separate variance QLIKE fits followed by the full-matrix correlation quasi-likelihood. This is the M process for realized covariance; the separate H process for daily-return covariance is not fitted or claimed.

Across missing reference sessions, the code propagates linear variance and correlation conditional means over the elapsed session count. These component forecasts have closed-form recursions under the model; the reconstructed multi-step covariance remains a component plug-in, rather than a claimed exact expectation of their nonlinear product. All reported origins have complete preceding HAR windows. The screened SPY reference remains an imperfect exchange calendar.

The frozen challengers are raw-input and raw-input/smearing factor LASSO/adaLASSO; realized DCC-HEAVY M; and two component controls exchanging HEAVY and existing graph/ridge forecasts. Every coefficient and residual correction uses strictly earlier observations. No architecture or tuning parameter was changed after viewing the new scores.

## Before and after

Ratios below are relative to HAR-DRD on the same 397 dates. Lower is better. Each row is a complete model; metrics from different variants are not combined into one result.

| Model | Euclidean ratio | Frobenius ratio | QLIKE ratio | Annual long-only risk (%) |
| --- | ---: | ---: | ---: | ---: |
| HAR-DRD | 1.000000 | 1.000000 | 1.000000 | 9.721778 |
| Original DCC-NL | 1.231951 | 1.232161 | 1.085950 | 11.062301 |
| Realized DCC-HEAVY M | 1.006759 | 1.002041 | 0.995455 | 10.347812 |
| Original market-factor LASSO | 1.070834 | 1.073100 | 1.095369 | 10.335602 |
| Raw-input market-factor LASSO | 1.027745 | 1.027575 | 1.077849 | 10.005169 |
| Raw/smeared market-factor LASSO | 1.032493 | 1.041830 | 1.028901 | 9.861754 |

The realized-input change reduces the DCC Frobenius gap from **23.22% above HAR-DRD to 0.20% above**. QLIKE becomes 0.45% below HAR-DRD, but portfolio risk remains higher. Relative to the earlier DCC version itself, Frobenius loss falls 18.68%, QLIKE falls 8.33%, and realized portfolio variance falls 12.50%.

For the corrected market-factor LASSO, the trace ratio rises from **0.6848 to 0.8953**. Its QLIKE gap falls from **9.54% to 2.89% above HAR-DRD**, while annualized risk falls from **10.3356% to 9.8618%**. Relative to its original version, QLIKE falls 6.07% and portfolio variance falls 8.96%. Its distance-loss before/after confidence intervals still include zero.

These changes mitigate the failing methods substantially. None of the seven new models improves all four later-period endpoints versus HAR-DRD. The existing Log/Fisher graph model still has the lowest distance losses among the 24 models, and the earlier ridge/graph model still has the lowest later QLIKE. Correcting scale and incorporating realized information does not establish a portfolio-risk advantage. The simplified factor structure and covariance-to-portfolio objective difference remain limitations.

Complete [later scores](tables/reused_2025_2026/scores.csv), [development scores](tables/development_2024/scores.csv), [baseline comparisons](tables/reused_2025_2026/paired-intervals.csv), and [before/after intervals](tables/before-after-intervals.csv) retain every failure. Intervals use 10,000 paired stationary-bootstrap draws, mean block length 10 and seed 20261127. Before/after intervals were added as exploratory reporting after evaluation, without parameter changes. All dates have been reused in earlier research; this is not independent confirmation or reproduction of either original paper's numerical results.

## Code and reproduction

`realized_heavy.py` implements realized-covariance HEAVY fitting and forecasting. `factor_vhar.py` adds optional `clean=False` and `bias_correction=True`; original defaults stay unchanged. Both default factor estimators exactly match their earlier saved forecasts at three sampled origins. The earlier published-method manifest preserves its original source hash; the new manifest records the extended module.

Seven new tests verify analytic gradients across missing sessions, closed-form gap propagation, scalar and noncommuting matrix smearing, positive-definite forecasts and invariance to future observations. The full existing tests also run against the extended factor module.

From `covariance_replication`, with `requirements-published.txt` installed and the earlier panel and published-method artifacts prepared:

```sh
python run_diagnosis_repairs.py \
  --panel /absolute/path/to/panel.npz \
  --controls /absolute/path/to/published-results/forecasts.npz \
  --control-weights /absolute/path/to/published-results/portfolio-weights.npz \
  --config docs/model-diagnosis/design.json \
  --output /absolute/path/to/repair-results

python audit_model_diagnosis.py \
  --panel /absolute/path/to/panel.npz \
  --repaired-folder /absolute/path/to/repair-results \
  --output /absolute/path/to/diagnostics
```

Original portfolio allocations are reused unchanged; new portfolios use the existing exact CVXPY/OSQP solver. Compact fit audits, diagnostics, scores and paired intervals accompany this report. Raw inputs, daily losses and large forecast arrays remain outside git.
