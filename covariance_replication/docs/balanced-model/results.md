# Balanced covariance model

27 November 2026. The selected model improves all four observed point comparisons against HAR-DRD in both evaluation periods. On the 397 reused later dates, Euclidean, Frobenius and QLIKE losses fall **2.68%, 2.94% and 1.19%**, while annualized long-only risk falls **9.7218% to 9.5113%**. Portfolio variance falls 4.28%.

The portfolio-risk interval still includes zero. This candidate was highlighted after inspecting a broad retrospective search. Its intervals are exploratory and are not adjusted for that selection. Joint statistical superiority and independent confirmation remain unestablished. This is an experimental extension of the public-data project, not a numerical reproduction of the paper's original GHAR model.

## Results on unchanged dates

All comparisons use the same raw covariance targets, open-to-close returns and exact CVXPY/OSQP long-only optimization. Lower is better.

| Model, reused 2025–2026 | Euclidean ratio | Frobenius ratio | QLIKE ratio | Annual risk (%) |
| --- | ---: | ---: | ---: | ---: |
| HAR-DRD | 1.000000 | 1.000000 | 1.000000 | 9.721778 |
| Earlier five-expert uniform pool | 0.987767 | 0.987351 | 0.996494 | 9.673684 |
| Log/Fisher GHAR | 0.973542 | 0.968018 | 0.998614 | 9.660256 |
| Ridge log-HAR with graph correlation | 0.987367 | 0.986161 | 0.984146 | 9.735293 |
| Equal log/ridge pool | 0.974604 | 0.972241 | 0.988069 | 9.670007 |
| Selected balanced model | **0.973199** | **0.970572** | **0.988102** | **9.511277** |

The selected model improves all four later-period point estimates versus the earlier five-expert pool. The Log/Fisher model retains lower Frobenius loss, and the ridge model retains lower QLIKE. There is no claim of dominance over every control.

| Period | Dates | Euclidean ratio | Frobenius ratio | QLIKE ratio | HAR-DRD risk (%) | Selected risk (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development 2024 | 245 | 0.939030 | 0.938890 | 0.984660 | 7.419627 | 7.411811 |
| Reused 2025–2026 | 397 | 0.973199 | 0.970572 | 0.988102 | 9.721778 | 9.511277 |

The development-period risk improvement is small. The later sample runs from 2 January 2025 through 19 October 2026; the development sample runs from 8 January through 31 December 2024. These are the existing 642 common dates, rather than a new sample chosen for this candidate.

Paired stationary-bootstrap intervals use 10,000 draws, mean block length 10 and seed 20261127:

| Later endpoint versus HAR-DRD | Change (%) | Exploratory paired 95% interval |
| --- | ---: | --- |
| Euclidean loss | −2.6801 | [−5.4413, −0.4346] |
| Frobenius loss | −2.9428 | [−5.7821, −0.6910] |
| QLIKE loss | −1.1898 | [−1.8217, −0.6801] |
| Long-only portfolio variance | −4.2836 | [−7.5395, +0.7367] |

The three covariance-loss intervals exclude zero; the risk interval does not. Annual standard deviation decreases by 0.2105 percentage points, or 2.17% relatively. Later-period turnover is 0.3443 versus HAR-DRD's 0.3289. Performance is before transaction costs.

## Difference from the first version

The first balanced version is the [five-expert uniform pool](../joint-pool/results.md) committed in [`b425def`](https://github.com/khaamr/graph-covariance-forecasting/commit/b425defbcda3b6f09290b5ed60190e5a20601731). This comparison holds the 397 later dates, raw targets, returns and long-only portfolio constraints fixed. The loss ratios above are relative to HAR-DRD; the percentage changes in the last column below are relative to the first ensemble itself.

| Endpoint, reused 2025–2026 | First ensemble | Current model | Change versus first |
| --- | ---: | ---: | ---: |
| Euclidean loss / HAR-DRD | 0.987767 | 0.973199 | 1.47% lower loss |
| Frobenius loss / HAR-DRD | 0.987351 | 0.970572 | 1.70% lower loss |
| QLIKE / HAR-DRD | 0.996494 | 0.988102 | 0.84% lower loss |
| Annualized long-only risk (%) | 9.673684 | 9.511277 | 0.1624 percentage points lower |
| Mean turnover | 0.332044 | 0.344328 | 3.70% higher |

The risk change corresponds to 3.33% lower portfolio variance. Among these four forecast-loss and variance comparisons against the first ensemble, only QLIKE has a paired 95% interval excluding zero. The saved [paired intervals](tables/reused_2025_2026/paired-intervals.csv) report all four; they remain exploratory after model search. Turnover is higher and results exclude transaction costs.

The first version averages five covariance forecasts equally. The current model averages two components, Log/Fisher GHAR and ridge log-HAR, then uses earlier realized covariance to adjust its desired portfolio during volatility stress. A positive-definite covariance construction makes the standard long-only optimizer recover that allocation. The graph correlation topology and portfolio constraints are preserved; the forecast architecture changes.

In development 2024, the current model also improves all three losses versus the first ensemble, but annualized risk increases slightly from **7.410661% to 7.411811%**. Both remain below HAR-DRD's 7.419627%. Thus the current model improves all four point estimates versus HAR-DRD in both periods, and versus the first ensemble in the later period only.

The project's original paper replication is a separate comparison: its [GHAR results](../result-comparison.md) use July 2022–December 2024 and a different specification. Its absolute risks cannot be compared directly with this 2025–2026 table. That original replication remains available, and the current model is an extension rather than a reproduction of the paper's architecture or original data.

## Algorithm

1. Average the causal Log/Fisher graph covariance forecast and ridge log-HAR covariance forecast to obtain F.
2. Compute a fast covariance estimate from earlier realized matrices, using exponential smoothing with a five-observation half-life. Obtain its long-only allocation and the allocation implied by F.
3. Measure the previous realized covariance trace relative to its preceding 22-observation mean. The allocation moves gradually toward the fast estimate as that ratio rises from 1 to 3; it stays with the base allocation below 1.
4. Reconcile F with that desired allocation using a positive-definite matrix construction and its Gaussian-KL scalar optimum. Standard long-only optimization of the resulting covariance recovers the desired allocation.

The model uses five observed sessions for smoothing; the inherited panel has missing historical reference sessions. Eligible forecast origins retain complete preceding HAR windows. The reference is screened SPY data, not a complete exchange calendar.

The matrix construction is developed in this project and proved in [mathematics.md](mathematics.md). It is not the published Dynamic Conditional Weights algorithm. The [direct-weight forecasting literature](https://arxiv.org/abs/2004.12400) motivates considering allocations separately, while the [covariance shrinkage literature](https://www.econ.uzh.ch/dam/jcr:68117157-a339-4600-8d9c-71151e69c177/jmva_2021.pdf) explains why the portfolio objective matters when estimating a matrix. Neither source is claimed to establish this new model's performance.

The ridge component uses own-stock, leave-one-out market and sector log-variance HAR signals. It chooses a ridge penalty using only the last 126 earlier labels, with fitting on the preceding block, then refits on at most 1,000 earlier retained rows. Training-residual smearing is estimated before the forecast. Graph components remain the previously documented 12-alpha Log/Fisher specification.

## Search record and why this candidate was highlighted

The investigation first tested return-risk calibration, then covariance reconciliation with smoothed weight forecasts, then faster covariance targets with lagged volatility gates. Each family's settings were fixed before evaluating that family; earlier outcomes informed later research directions. The studies contain **204, 148 and 166 configurations**, including repeated baseline and scalar variants. These are not independent statistical trials. Earlier project studies also inspected this benchmark.

- [Risk calibration](studies/risk-calibration/design.json) uses genuine prequential candidate forecasts and tests integrated-risk and return-risk selectors. Its combined-risk primary still worsens later return risk. The integrated-risk control improves all four later point estimates but slightly worsens development risk. All [candidate scores](studies/risk-calibration/tables/all-candidate-scores.csv) remain available.
- [Initial reconciliation](studies/reconciliation/design.json) gives good matrix losses but its fixed slow-weight primary worsens risk. All [scores](studies/reconciliation/tables/all-candidate-scores.csv) and KKT checks are retained.
- [Stress-sensitive reconciliation](studies/stress-reconciliation/design.json) contains 166 configurations. Of these, 157 improve all four later point estimates and 32 improve all four in both periods; these counts include closely related scalar variants. The predeclared yesterday-covariance primary improves later risk but worsens development risk. The selected five-session EWMA configuration is **candidate 159**, a predefined variant highlighted after inspecting both periods. All [scores](studies/stress-reconciliation/tables/all-candidate-scores.csv), fixed-primary failures and selector outcomes are retained.

The [selection record](selection.json) identifies the chosen variant explicitly. This is retrospective model development with causal rolling fits, not a fresh holdout or a causal choice of architecture from an unseen test period. No confidence interval corrects for the whole search. New uninspected data are needed to establish a reliable future risk advantage.

## Reproduction and API

Install `requirements-published.txt` from `covariance_replication`. The chosen model can be run without evaluating the failed challengers:

```sh
python prepare_literature_panel.py --data-dir /path/to/cached/syncster --output /path/to/panel.npz
python prepare_joint_experts.py --data-dir /path/to/cached/syncster --output-dir /path/to/experts
python run_balanced_model.py --panel /path/to/panel.npz --experts /path/to/experts/experts.npz --output /path/to/balanced-results
```

The standalone pipeline refits its ridge component and reproduces all **642** selected forecasts exactly. It does not load the earlier losing-model outcomes. The earlier comparison archive and the independently rebuilt pipeline produce identical targets, component forecasts and final predictions.

For a single forecast, supply the two causal component forecasts and strictly earlier realized observations:

```python
from balanced_covariance import balanced_covariance_forecast

prediction = balanced_covariance_forecast(
    log_graph_forecast, ridge_graph_forecast,
    observed_covariance, history_dates, forecast_date,
)
```

The API rejects same-day or future history. Mathematical and numerical tests check positive definiteness, the prescribed portfolio's KKT conditions against independent CVXPY optimization, scalar-distance optima, unit scaling, missing future influence, and full-pipeline chronology. All 80 relevant tests passed, including 13 new tests for these studies. Exact source and input hashes, solver certificates and [verification](tables/verification.json) are retained.

Compact [later results](tables/reused_2025_2026/scores.csv), [development results](tables/development_2024/scores.csv), and [paired comparisons](tables/reused_2025_2026/paired-intervals.csv) accompany this report. Raw data and large forecast archives remain outside git. Original replication results and website layouts are unchanged.
