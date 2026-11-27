# DCC-NL and factor-VHAR implementations

27 November 2026. These implementations replace the earlier simplified challengers with the complete estimation components. The frozen [design](design.json) records the benchmark choices. They are method implementations on our public-data sample, with the empirical differences below; they do not reproduce the original papers' numerical experiments.

## Implemented methods

**DCC-NL.** Fit Gaussian GARCH(1,1) to each demeaned return series. Apply Ledoit-Wolf analytical nonlinear shrinkage to the covariance of the standardized residuals and normalize to a correlation target. Estimate the two DCC parameters by the authors' contiguous-pair composite likelihood, with nonnegative parameters and sum at most 0.99998. Update the full DCC state and GARCH variances to forecast the next observation. Each forecast is formed before consuming that date's return.

The original 2019 study used QuEST. The author package documentation endorses direct analytical shrinkage as a faster alternative. This implementation uses that published alternative, not fixed eigenvalue clipping or a factor projection. Its analytical estimator supports both fewer and more assets than observations.

**Factor-VHAR.** Given dated factor-portfolio weights W, calculate factor covariances W' Sigma W, daily loadings (W' Sigma W)^(-1) W' Sigma, and residual covariances. Forecast matrix-log factor covariances with equationwise LASSO or adaptive LASSO. Select penalties by BIC over the full LASSO path knots. Forecast each daily loading with OLS HAR using overlapping daily, weekly and monthly predictors. Forecast sector residual blocks separately from their lagged diagonal entries. Exponentiate forecast log blocks and recombine factor and residual forecasts. Adaptive LASSO follows the paper's two-stage active-set restriction and coefficient weighting.

The general API accepts any independent factor portfolios, including 1/3/5/7 factors. Historical weights must include timestamps proving availability before their corresponding trading date. The benchmark uses the **equal-weight market factor authorized by the user**. It does not substitute principal components for the unavailable characteristic-based portfolios.

## Source specifications and checks

- [Engle, Ledoit and Wolf (2019)](https://www.econ.uzh.ch/dam/jcr:28fa9939-753e-4f5c-932d-945872f30cfd/jbes_2019.pdf), equations 2.1–2.4 and Section 2.3: GARCH, DCC recursion and contiguous-pair composite likelihood.
- [Authors' DCC-NL software](https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html): target construction, backcast, starting-value grid and parameter bound. The code returns an in-sample last-date estimate; our forecasting function explicitly advances it by one observation.
- [Authors' analytical nonlinear-shrinkage estimator](https://www.econ.uzh.ch/dam/jcr:11d24ab0-7ec2-4b3f-8ef4-7affaa727d25/analytical_shrinkage.m.zip): kernel density and Hilbert-transform formulas, degrees of freedom and high-dimensional branch.
- [Alves et al. (2023/2024)](https://arxiv.org/html/2303.16151v1), equations 6–17 and Appendix A: factor decomposition, VHAR, BIC, adaptive LASSO, loading HAR and restricted sector residual dynamics.

The numerical DCC likelihood matches the **unchanged author C numerical core on 12 cases**, including 27 assets. Maximum absolute difference in total negative log likelihood is 4.33e-12. The verification replaces only MATLAB allocation wrappers. This validates the likelihood core, not end-to-end equality of MATLAB and Python optimizers. Author files and compiled binaries remain outside the repository.

Twelve new tests check nonlinear-shrinkage formulas, rotation and scale equivariance, high-dimensional positive definiteness, analytic DCC gradients, next-observation alignment, future-data invariance, factor decomposition against intraday OLS, overlapping HAR lags, sparse coefficient recovery, timestamp restrictions, causal cleaning, and 1/3/5/7-factor execution.

## Empirical choices and remaining differences

| Item | This implementation and benchmark |
| --- | --- |
| DCC target | Author-supported analytical nonlinear shrinkage; original paper's QuEST output is not reproduced |
| DCC sample | 1,250 earlier retained return observations; monthly estimation, daily state updates |
| Return definition | Existing public open-to-close returns, rather than the DCC paper's CRSP close-to-close data |
| Factor inputs | Authorized equal-weight market portfolio; characteristic-based 3/5/7-factor empirical runs await historical portfolio weights |
| Factor sample | Daily estimation; up to 1,000 earlier retained rows, excluding incomplete 22-session feature windows |
| HAR lags | Overlapping lags 1, 1–5, 1–22, following this factor paper; these differ from our graph-paper disjoint lag convention |
| Sector groups | Existing 27-stock sector labels, rather than the factor paper's SIC groups |
| Residual matrix logarithms | Apply the Section 5.2 log-matrix specification to each residual sector block; Section 3.3 writes its equations in levels, so this interpretation is explicit |
| Penalized regression | Training-only column standardization, unpenalized intercept, BIC over LARS knots; numerical grid/scaling details are not fully specified by the paper |
| Outlier cleaning | Prior-only four-standard-deviation screen after 22 observations; flag if more than 25% of unique entries are extreme, replace with the preceding ten nonflagged matrices. Cleaning affects historical fitting inputs; evaluation targets stay raw |
| Calendar | Inherited screened SPY calendar and sparse retained panel; this is not a complete exchange-day sample |

GARCH fitting uses unit-variance numerical scaling and converts estimates back to original units. There are zero failed final fits. Three optimizer solutions exceeded the GARCH sum constraint by less than 1e-6; feasibility was restored and variance paths recomputed under those parameters. No failed forecast was substituted with another model. No residual or factor log eigenvalues required repair in the benchmark.

## Results on identical dates

The comparison includes all 14 previous controls plus these three new models. The 642 dates are exactly the previous experiment's 245 development dates and 397 reused later dates. Actual covariance targets and portfolio returns remain identical. No settings were changed in response to test-period scores.

Later dates span 2 January 2025 through 19 October 2026. Ratios below use HAR-DRD; lower is better.

| Model | Euclidean ratio | Frobenius ratio | QLIKE ratio | Annual long-only risk (%) |
| --- | ---: | ---: | ---: | ---: |
| HAR-DRD | 1.000000 | 1.000000 | 1.000000 | 9.721778 |
| Fixed uniform covariance pool | 0.987767 | 0.987351 | 0.996494 | 9.673684 |
| Log/Fisher GHAR(GL,L) | 0.973542 | 0.968018 | 0.998614 | 9.660256 |
| Ridge log-HAR + graph correlation | 0.987367 | 0.986161 | 0.984146 | 9.735293 |
| DCC-NL, analytical target | 1.231951 | 1.232161 | 1.085950 | 11.062301 |
| Factor-VHAR, market LASSO | 1.070834 | 1.073100 | 1.095369 | 10.335602 |
| Factor-VHAR, market adaptive LASSO | 1.068466 | 1.071669 | 1.096052 | 10.334489 |

All three new methods worsen all four later-period endpoints versus HAR-DRD. Their paired 95% intervals exclude zero in the adverse direction on these reused dates. These are exploratory intervals, without adjustment for the broader history of model searches. The market-only factor model's result cannot establish the performance of the unavailable characteristic-factor specifications.

Development 2024 also fails the joint objective: the factor estimators reduce distance losses but worsen QLIKE and risk; DCC-NL worsens all endpoints. Complete [later scores](tables/reused_2025_2026/scores.csv), [development scores](tables/development_2024/scores.csv), and [paired intervals](tables/reused_2025_2026/paired-intervals.csv) retain every model.

Risk is annualized standard deviation of the existing open-to-close long-only minimum-variance strategy, before transaction costs. Final allocation uses CVXPY/OSQP. Paired stationary bootstrap uses 10,000 draws, mean block length 10 and seed 20261127. The factor estimator made 642 daily fits; DCC made 33 monthly fits. All covariance predictions are positive definite. All LASSO paths passed KKT checks without warnings; maximum KKT error is below 8.4e-16.

## Reproduce and supply additional factors

From `covariance_replication`, install `requirements-published.txt`. Prepare the pinned panel using `prepare_literature_panel.py`; build existing comparison forecasts using the documented joint-pool and literature-challenger commands. Then run:

```sh
python run_published_models.py \
  --panel /absolute/path/to/panel.npz \
  --controls /absolute/path/to/literature-results/forecasts.npz \
  --config docs/published-methods/design.json \
  --output /absolute/path/to/published-results
```

For characteristic-factor portfolios add `--factor-weights /absolute/path/to/weights.npz`. Required arrays are `dates` (T), `symbols` (N), `weights` (T,N,K), and `known_at` (T). Dates and symbol order must match the panel exactly; each weight timestamp must precede its trading date. The API accepts a static N-by-K matrix for portfolios defined in advance. Supplied-factor results receive a separate model label.

For the independent author-core check, download the university DCC_NL06 archive and run:

```sh
python verify_dcc_reference.py \
  --author-source /absolute/path/to/DCC_NL06/dcc_likestream.c \
  --output /absolute/path/to/reference-check.json
```

The manifest and source records preserve evaluated code and input hashes. Compact tables include daily/monthly fit audits, GARCH parameters, cleaning flags and equation summaries. Full equation paths, raw data and forecast archives stay outside git. Existing model defaults and earlier results remain unchanged.

The factor module subsequently gained optional raw-input and matrix-smearing paths in the [diagnostic study](../model-diagnosis/results.md). Its original defaults exactly match both earlier saved estimators at three sampled origins. This report retains the source hashes from its initial evaluation; the diagnostic study records the extended source.
