# LOBSTER sample validation and calendar diagnosis

Follow-up completed 27 November 2026. This investigation uses the real stock samples recorded by `lob-forecasting-lab`, then tests calendar handling on the longer public covariance panel. It does not substitute a one-day dataset for a multi-year forecasting experiment.

## Outcome

The LOBSTER samples do not reveal a basic price-unit, timestamp-conversion, minute-loader, or realized-covariance formula error in the checked path. A separate, verified issue remains: deleting missing sessions before constructing HAR lags changes their time span. An optional reference-calendar filter now excludes incomplete feature and target windows. That correction does **not** recover the paper's model ranking on the available public data.

## Measurement test using the lab's data

The AAPL, INTC, and MSFT archives for 21 June 2012 were read from the original local cache. All three SHA-256 hashes match the lab's archived manifest. These are the three sample stocks that overlap with the covariance study's universe.

The validation reads matching orderbook and message rows, parses decimal event timestamps to integer nanoseconds, divides LOBSTER fixed-point prices by 10,000, and forms the best bid/ask midpoint. It samples quotes at minute boundaries over 09:30–16:00 New York time and writes start-labelled minute CSVs for the existing `market_data.minute_panel` adapter. It deliberately does not apply the LOB forecasting lab's 09:40–15:50 training-session crop.

The files begin 4.24 ms, 5.74 ms, and 13.99 ms after the exact opening time. The first observed quote is explicitly used as an opening proxy; the validator rejects delays above one second. Every subsequent grid point uses only quotes observed at or before its sampling time. The maximum nonopening quote age is 5.39 seconds. This opening convention is a documented measurement approximation, not a claim that an exact opening quote was recovered.

The independent covariance calculation enumerates all 386 rolling five-minute returns and accumulates their outer products divided by five. It does not call the production estimator to construct its reference. A separate quarticity calculation also agrees.

| Check | Observed result |
| --- | ---: |
| Maximum price difference through the existing CSV adapter | 3.55e-15 dollars |
| Maximum covariance difference against the independent calculation | 3.13e-14 percentage-points squared |
| Maximum quarticity difference | 1.93e-13 in the corresponding fourth-power units |
| Minimum covariance eigenvalue | 0.410851 |
| Cholesky factorization of the measured matrix | Passed |

The measured realized covariance matrix, in percentage-points squared, is:

| | AAPL | INTC | MSFT |
| --- | ---: | ---: | ---: |
| AAPL | 0.688085 | 0.500461 | 0.605284 |
| INTC | 0.500461 | 1.595826 | 0.989232 |
| MSFT | 0.605284 | 0.989232 | 1.770353 |

The values, archive hashes, quote timing, and numerical checks are saved in [LOBSTER evidence](diagnostics/2026-11-27/lobster/summary.json). The one-day sample cannot establish forecasting accuracy or validate every source-feed convention in the longer trade-bar dataset.

## What calendar compression can change

A controlled experiment uses the 2022–2024 public covariance panel, which contains all dates in the screened SPY reference for that interval. This reference itself is not a complete exchange calendar.

For seeds 11, 27, and 41, 40% of the pre-2024 observations are randomly hidden from the lag history. The two variants use identical training-target dates and values, identical monthly 400-reference-session calibration intervals, identical GLASSO graphs, and the same 249 forecast dates in 2024. One variant uses the complete observed historical lag path; the other constructs lags after compressing the retained rows. No 2024 forecast target is deleted. Historical gaps are artificial and independently randomized; they do not reproduce the original missingness pattern.

The distortion can change relative performance. For seed 27, the HAR-Cholesky Frobenius ratio against HAR-DRD moves from **1.0068** with complete history to **0.9972** with compressed history, reversing that pair's ordering. The graph model's ratios also change, but the three seeds do not show a consistent improvement or deterioration in its distance-loss advantage. This is a mechanism check, not an estimate of the causal contribution of missing data to the original paper discrepancy.

[Controlled-experiment ratios](diagnostics/2026-11-27/artificial-gaps/ratios.csv) and [configuration](diagnostics/2026-11-27/artificial-gaps/config.json) retain all seeds and variants; no seed was selected to match the paper.

## Calendar-safe test on the actual history

`forecast_grid` now accepts `reference_calendar`. When supplied, it excludes a training row unless all preceding 22 reference sessions occur consecutively in the retained panel. It also excludes multi-session target windows that bridge missing reference dates. The same checks apply to forecast origins and their targets. An omitted reference preserves the previous default behavior.

This option does not impute prices or replace the original data. It preserves each original rolling 1,000-retained-row calibration interval, so the number of eligible training labels decreases. A true 1,000-calendar-complete-label replication still needs more complete historical inputs.

The actual one-session experiment uses the observed SPY reference. Both runs have identical 623 forecast dates, realized targets, and GLASSO graphs. The original run uses 1,000 labels per fold; the screened run uses 208–773. That reduction is a limitation of this available-data correction.

| Model and training rule | Euclidean ratio | Frobenius ratio | QLIKE ratio |
| --- | ---: | ---: | ---: |
| GHAR(GL,L), original | 0.983488 | 0.985460 | 0.985067 |
| GHAR(GL,L), calendar-safe | 0.983146 | 0.982367 | 0.984148 |
| HAR-Cholesky, original | 0.937597 | 0.938758 | 1.078985 |
| HAR-Cholesky, calendar-safe | 0.923405 | 0.923436 | 1.059045 |

Each ratio uses HAR-DRD from the same training rule as its denominator. The screened graph model's Euclidean and Frobenius MCS p-values are 0.0004 and 0.0003: it remains excluded from those 5% model confidence sets. It remains in the QLIKE set, with p = 0.840916. MCS calculations use the same 13 main models, 10,000 stationary-bootstrap draws and block length two as the original run. These p-values are not tests against the paper's estimates.

The calendar-safe run therefore retains the original qualitative disagreement: HAR-Cholesky wins the distance losses, whereas the graph model improves on HAR-DRD without becoming the best distance-loss model. The calendar issue is real, but this test does not support claiming that it explains or resolves the published ranking difference. [Scores and fold records](diagnostics/2026-11-27/calendar-safe/).

## Reproduction

From `covariance_replication`, using its numerical dependency versions:

```sh
python validate_lobster.py \
  --data-dir /absolute/path/to/lob-lab/data/raw \
  --lab-manifest /absolute/path/to/lob-forecasting-lab/data/manifests/manifest.json \
  --output-dir /absolute/path/to/lobster-validation

python calendar_sensitivity.py \
  --data-dir /absolute/path/to/cached/syncster \
  --output-dir /absolute/path/to/artificial-gap-results

python calendar_safe_replication.py \
  --data-dir /absolute/path/to/cached/syncster \
  --output-dir /absolute/path/to/calendar-safe-results
```

The quote validator reads existing archives without downloading data. The covariance experiments use the existing loader, which downloads missing public ticker files if the supplied cache is incomplete. Large raw inputs, sampled minute grids, daily loss paths, and forecast arrays remain in temporary or ignored locations; compact validation evidence is stored with this report.

The new filter was checked on complete and gapped synthetic reference calendars, including multi-session targets. Complete-calendar predictions match the default predictions, and no future observations affect current features in the compression experiment. In total, 31 relevant tests passed across the replication, audit, quote-sampling, calendar-screening, and adaptive-extension checks. The original real-data loss ratios were also reproduced after the code change.

The website and executed notebook results have not been replaced with these sensitivity results. The original [diagnosis](result-diagnosis.md) remains the record of the initial audit. Comparable original-period data are still required to distinguish feed and market-period effects from implementation differences.
