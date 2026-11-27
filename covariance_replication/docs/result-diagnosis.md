# Diagnosis of the paper-result discrepancy

Audit date: 26 November 2026. This audit leaves the forecasting models, notebook outputs, and website unchanged. It adds a reproducible diagnostic script.

## Findings

### 1. The retained calendar changes the meaning of the HAR lags

`market_data.load_market` intersects the valid dates of all 27 stocks and SPY. `models.har_lags` then shifts and averages rows of that intersection. A missing trading session disappears from the time axis rather than remaining an explicitly missing observation.

The first forecasting fold begins on 1 July 2022. Its 1,000 training rows begin on 3 March 2016. Measured against the screened SPY input calendar:

| Retained-row lag | Intended span in consecutive sessions | Median observed span | Maximum observed span | Training rows spanning missing reference sessions |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 14 | 293 / 1,000 |
| 5 | 5 | 7 | 26 | 657 / 1,000 |
| 22 | 22 | 34 | 67 | 792 / 1,000 |

Only 208 of the first 1,000 training rows have an uninterrupted preceding 22-session history. These counts are lower bounds: the reference is the screened SPY calendar, not a complete exchange calendar. Holidays and weekends are not being counted as gaps.

The largest individual contributors to missing screened SPY sessions before 2022 are HON (185), MMM (171), and AXP (147). Counts overlap across stocks and must not be added. The existing notebook already reports the aggregate missing-date problem; this audit measures its effect on the actual lag inputs.

This is a verified protocol mismatch for replication of daily, weekly, and monthly market-time dynamics. It does not prove how much of the loss-ranking difference it causes. The one-session test period is much more complete, so the training and test lag inputs also differ in their market-time interpretation.

A valid correction requires more complete input data and explicit alignment to an exchange calendar. Simply forward-filling entire missing daily covariance matrices, shortening the training window, or dropping difficult stocks would change the experiment. Preserving missing dates and dropping incomplete feature windows alone would leave only 208 eligible rows in the first original training interval, far short of 1,000.

### 2. An exchange-malfunction day passes the input screens

On 24 January 2023 the cached minute bars include:

| Stock | 09:30 open | 09:30 close | First later bar shown | Later close |
| --- | ---: | ---: | --- | ---: |
| VZ | 38.70 | 42.49 | 09:38 | 38.57 |
| MCD | 270.43 | 248.24 | 09:40 | 267.25 |
| NKE | 123.58 | 117.98 | 09:39 | 127.73 |

The within-session fill rule propagates these early prices through missing minute bars, and the realized-covariance estimator incorporates the subsequent reversals. The realized variances for VZ, MCD, and NKE on that date are approximately 110, 108, and 50 times their respective median test-period variances, excluding that date.

The SEC documents that the NYSE failed to conduct opening auctions on this date, producing trading pauses and thousands of subsequently cancelled trades. This is external evidence to investigate the bars; it is not proof that each particular cached price corresponds to an individually cancelled execution. OHLC bars do not contain the execution identifiers or correction flags needed to establish that correspondence. [SEC administrative proceeding](https://www.sec.gov/enforcement-litigation/administrative-proceedings/34-104934-s).

The current completeness, positivity, and same-session fill screens do not establish that cancelled trades have been removed. Verify against corrected source records or use a vetted quote-based input. Do not apply a blanket large-return filter: large genuine market moves also occur in these files.

A diagnostic that omits 24 and 25 January 2023 from scoring, while retaining the original model fits and predictors, gives Euclidean ratios of 0.9444 for HAR-Cholesky and 0.9853 for GHAR(GL,L), relative to HAR-DRD. Original ratios are 0.9376 and 0.9835. This is NOT a corrected-data backtest: contamination remains in features and later training windows. It shows that those two scoring dates alone do not explain the ranking reversal.

## Checks that did not explain the discrepancy

- Re-running the current implementation on the cached inputs reproduces the notebook's one-session ratios: GHAR(GL,L) 0.9835 / 0.9855 / 0.9851, and HAR-Cholesky 0.9376 / 0.9388 / 1.0790, for Euclidean / Frobenius / QLIKE.
- Cholesky forecasts depend on stock ordering. Sector, alphabetical, and reverse order give Euclidean ratios 0.9376, 0.9357, and 0.9443, respectively. All retain the same distance-loss ranking. Ordering is therefore a sensitivity to document, not the explanation found here. No authors' Cholesky ordering was established.
- The upstream scripts use overlapping 1/5/22 lag averages, while this implementation follows the paper's disjoint 1, 2–5, 6–22 form. These span the same regressors in unpenalized OLS. A regression check confirms equal predictions under both intercept conventions. Changing this basis cannot legitimately recover the paper's result.
- The released correlation script includes the target pair in its neighbor average. Since the model also includes the target pair's own lags and the graph degree is constant, this is another linear change of regressor basis.
- There were zero correlation repairs and zero increased-penalty GLASSO fallback months in the reproduced one-session main-model run. Those repair paths cannot explain these one-session results.
- The common-intercept versus series-intercept discrepancy was already tested in the notebook; it does not restore the published conclusion.

No verified error in the one-session forecast/target indexing, DRD reconstruction, loss definitions, or graph normalization was identified in this audit. That is a bounded finding, not a claim that every part of the project is error-free.

## What is needed for a comparable reproduction

The paper uses LOBSTER intraday observations from July 2007 through July 2021 and forecasts out of sample from July 2011. This run uses a different feed, history, and July 2022–December 2024 test period. These are different experiments. See the [published paper, Sections 3 and 3.3](https://academic.oup.com/jfec/article/23/2/nbae026/7889003) and the [pinned authors' implementation](https://github.com/chaozhang-ox/Graph-based-HAR/tree/751f8f0b400a33e2f2fd01501ffdc42a2315f013).

Use the original daily realized covariances/returns or the corresponding licensed intraday data, align dates and ticker order explicitly, verify correction handling, and then run the fixed protocol. If those inputs are unavailable, retain this as a public-data robustness study and report the different outcome. Similar numbers or a desired ranking must not be a tuning target.

## Reproduce this diagnosis

From `covariance_replication`, in an environment with its numerical requirements:

```sh
python audit_replication.py \
  --data-dir /absolute/path/to/cached/syncster \
  --output-dir /absolute/path/to/audit-output
python -m unittest discover -s tests -p 'test_audit.py'
```

The script writes loss ratios, daily loss paths, lag spans, fold diagnostics, and a JSON summary with implementation hashes. It downloads missing ticker files through the existing loader if the supplied cache is incomplete. It does not overwrite the notebooks, saved results, or website.

## Candidate data repositories checked

| Repository | Contents found | Supplies the required 27-stock history? |
| --- | --- | --- |
| [lobsterdata/py4lobster](https://github.com/lobsterdata/py4lobster) | A reader for local orderbook/message CSV files; repository tree contains five files and no sample-data directory. The test references local files that are not bundled. | No. Useful as a reader reference only. |
| [GGLAB-KU/LOBSTER](https://github.com/GGLAB-KU/LOBSTER) | Language-Of-study Bias in ScienTific pEer Review: annotations and code for NLP review analysis. | No. Unrelated use of the same acronym. |
| [financialcomputingucl/lobframe](https://github.com/financialcomputingucl/lobframe) | A limit-order-book processing, prediction, and simulation framework. Its README requires users to place their own raw LOBSTER files in the data directory. No raw market-data files appear in the repository tree. | No. Processing software, not the missing data source. |

The [official sample page](https://php.lobsterdata.com/info/DataSamples.php) offers a small selection of sample stocks from the NASDAQ historical sample. Those can validate an importer but cannot provide this study's multi-year 27-stock panel. Full historical access is described on the [official access page](https://php.lobsterdata.com/info/AccessOptions.php). An existing university subscription or the authors' daily realized-covariance inputs would be the most direct route to comparable data. No contact, purchase, or subscription action was taken.

## Verification

The complete diagnostic was repeated in an isolated temporary Python environment using the project's pinned NumPy, pandas, SciPy, scikit-learn, PyArrow, statsmodels, CVXPY, OSQP, and threadpoolctl versions. The one-session ratios and calendar counts match the initial audit. The 10 existing replication tests and 4 new diagnostic tests all pass. Audit tables and JSON are stored beside this report under `diagnostics/2026-11-26/`; larger temporary forecast arrays remain outside the repository.
