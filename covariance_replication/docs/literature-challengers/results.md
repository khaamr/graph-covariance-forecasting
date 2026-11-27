# Literature-guided covariance challengers

27 November 2026. The [protocol](design.json) was saved before fitting or evaluating these six challengers. The experiment remains exploratory because the benchmark dates have been inspected in earlier project work.

## Result

None of the six new challengers improves all four later-period endpoints relative to HAR-DRD. The primary factor-regularized ridge log-HAR reduces mean Euclidean, Frobenius and QLIKE losses by 0.90%, 0.92% and 0.79%, but increases realized portfolio variance by 1.25%. It fails the joint objective.

Ridge log-HAR with the original graph correlation forecast achieves the lowest later-period QLIKE among the 14 models in this common-date comparison: **1.5854% below HAR-DRD**, versus 0.3506% for the fixed ensemble. Its exploratory paired 95% interval is [−2.2365%, −1.0484%]. Annualized long-only risk increases from 9.7218% to 9.7353%; this is a covariance-likelihood challenger, not an overall replacement for the fixed ensemble.

This QLIKE result was highlighted after inspecting the six fixed challengers. The interval is not adjusted for that selection or for the wider history of experiments. No parameters were retuned after seeing these results. This is evidence on reused dates, not independent confirmation.

## Research used

[Engle, Ledoit and Wolf (2019), Large Dynamic Covariance Matrices](https://www.ledoit.net/Engle_GARCH_2019.pdf) combines dynamic conditional correlation with nonlinear shrinkage of its long-run correlation target. The method separates volatility dynamics from cross-sectional estimation error. The authors provide [Matlab reference code](https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html). [Ledoit's shrinkage package](https://github.com/oledoit/covShrinkage) supplies reference estimators, including quadratic-inverse shrinkage.

[Alves, de Brito, Medeiros and Ribeiro, Forecasting Large Realized Covariance Matrices: The Benefits of Factor Models and Shrinkage](https://arxiv.org/abs/2303.16151) uses factor structure and regularized forecasting to address high-dimensional covariance estimation.

These papers motivate the separation and regularization used here. The implemented challengers are adaptations: they do not reproduce DCC-NL, estimate Ledoit–Wolf nonlinear shrinkage intensity, or reproduce the fundamental-factor VHAR model. A negative result for this three-factor projection does not reject those published methods.

## Implemented algorithm

1. Form disjoint daily, weekly and monthly log-variance lags for each stock. Add leave-one-out market and sector signals; use the market signal for a sector with no peer. The multivariate control instead uses every stock's three lags.
2. Fit separate ridge coefficients per stock on at most 1,000 earlier retained rows, excluding incomplete 22-session feature windows. Choose a penalty from the fixed five-point grid using the last 126 earlier labels. Normalization, coefficients and smearing for validation are fitted only on the preceding block. Minimize average univariate QLIKE divergence, then refit on all eligible earlier labels.
3. Convert log predictions to positive variance forecasts using per-stock exponential smearing estimated from training residuals. This is an explicit additional retransformation assumption.
4. Estimate three correlation eigenvectors from the preceding 252 retained realized correlation matrices at the start of each forecast month. Project an EWMA of observed correlations onto those factors; use half-life 20 reference sessions and a 126-reference-session window. Add a positive residual diagonal and renormalize. The long-run estimate uses retained rows; the SPY reference is a screened calendar, not a full exchange calendar.
5. For the primary model, average this correlation target with the previously fitted Log/Fisher GHAR correlation matrix, then apply the new variance forecasts. Evaluate five fixed controls that remove components or use multivariate predictors.

All inputs are earlier than the forecast. Regression and factor loadings refit monthly; the fast correlation statistic updates daily. There are 33 monthly refits with 586–1,000 eligible training labels. Every predicted covariance matrix is positive definite.

## Common-date comparison

All models use the same 245 development dates in 2024 and 397 reused dates from 2 January 2025 to 19 October 2026 as the previous fixed-ensemble experiment. Existing controls are loaded without refitting. Realized targets and returns stay identical; rebuilding the pinned panel is checked within 2e-12 absolute numerical tolerance. Lower loss ratios and risk are better.

| Model, reused 2025–2026 | Euclidean ratio | Frobenius ratio | QLIKE ratio | Annual long-only risk (%) |
| --- | ---: | ---: | ---: | ---: |
| HAR-DRD | 1.000000 | 1.000000 | 1.000000 | 9.721778 |
| Fixed uniform covariance pool | 0.987767 | 0.987351 | 0.996494 | 9.673684 |
| Log/Fisher GHAR(GL,L) | 0.973542 | 0.968018 | 0.998614 | 9.660256 |
| Primary factor-regularized ridge log-HAR | 0.990965 | 0.990809 | 0.992120 | 9.782298 |
| Ridge log-HAR + graph correlation | 0.987367 | 0.986161 | 0.984146 | 9.735293 |
| Ridge log-HAR + factor correlation | 1.034511 | 1.043879 | 1.034573 | 10.183605 |
| Graph log-variance + factor blend | 0.978772 | 0.974645 | 1.005840 | 9.832328 |
| Graph log-variance + factor correlation | 1.018204 | 1.022770 | 1.056185 | 10.315702 |
| Multivariate ridge log-HAR + factor blend | 1.005058 | 1.000881 | 1.000644 | 10.071593 |

The existing Log/Fisher graph model remains the lowest-distance-loss model in this comparison. The arithmetic covariance expert, retained in the full table, has the lowest observed portfolio risk but worse forecast losses. Complete [later-period scores](tables/reused_2025_2026/scores.csv), [development scores](tables/development_2024/scores.csv), and [paired intervals](tables/reused_2025_2026/paired-intervals.csv) retain all successes and failures.

Risk is annualized standard deviation of open-to-close long-only minimum-variance portfolio returns, before transaction costs. Optimization uses the existing CVXPY/OSQP solver. Paired stationary-bootstrap intervals use 10,000 draws, mean block length 10 and seed 20261127. Forecast loss definitions remain those of the replication.

## Reproduction

From `covariance_replication`, after preparing the existing joint-pool experts and controls:

```sh
python prepare_literature_panel.py --data-dir /absolute/path/to/cached/syncster --output /absolute/path/to/panel.npz
python run_literature.py \
  --panel /absolute/path/to/panel.npz \
  --experts /absolute/path/to/joint-experts/experts.npz \
  --controls /absolute/path/to/joint-results/forecasts.npz \
  --config docs/literature-challengers/design.json \
  --output /absolute/path/to/literature-results
```

Four new tests check factor-matrix validity, stock-specific coefficient recovery and smearing, unchanged earlier forecasts when future realized values change, and rejection of incomplete calendar windows. The manifest records the evaluated source hashes and input hashes. Raw data, covariance archives and daily portfolio returns remain outside git. The original replication and fixed-ensemble results are unchanged.
