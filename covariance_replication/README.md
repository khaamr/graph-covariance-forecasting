# forecasting realized covariances

a replication of the graph-based har covariance study. the target is a full covariance matrix, including the co-movement of every pair of stocks.

[live website](https://khaamr.github.io/graph-covariance-forecasting/) · [executed notebook](Covariance_Replication.ipynb) · [protocol audit](docs/protocol-audit.md) · [journal article](https://doi.org/10.1093/jjfinec/nbae026) · [authors' code](https://github.com/chaozhang-ox/Graph-based-HAR/tree/751f8f0b400a33e2f2fd01501ffdc42a2315f013)

![architecture showing intraday prices, the stock and pair graphs, covariance forecasts, and evaluation](docs/architecture.png)

## data

the paper uses licensed lobster quotes for 27 dow stocks during 2007–2021. this run uses public one-minute syncster trade bars for the **same 27 tickers** and spy. daily variances alone would not capture how assets move together, so the pipeline synchronizes minute prices and computes the realized covariance from five-minute return vectors. there were 798 sessions aligned across the whole panel through june 2021. extending the public data through december 2024 gives 1,647 sessions and supports the paper's 1,000-session rolling calibration.

[pinned public data](https://huggingface.co/datasets/syncster-data/historical-market-data/tree/eff0dac55d843b3b83caec819efa213021a3e3e5) · [capire academic alternative](https://capire.stat.unipd.it/download/)

## what’s covered

- har-cholesky, har-drd, and eleven graph combinations at 1, 5, and 22 sessions.
- matrix euclidean, frobenius, and qlike losses, with a stationary-bootstrap model confidence set.
- equal weights, unrestricted gmvp, long-only gmvp+, and turnover.
- spy stress regimes, raw- and square-root-quarticity harq/gharq checks, and three interactive plotly charts.

published ratios are included as reference points from the original licensed study. portfolio returns here are open-to-close and exclude transaction costs. the saved run tests the method with a different feed and forecast period. the graph model is excluded from the one-session euclidean and frobenius confidence sets, where har-cholesky performs best. [comparison with the paper](docs/result-comparison.md).

## public backtest

the saved run contains 623 held-out one-session forecasts from july 2022 through december 2024. relative to har-drd, ghar with the glasso and line graphs lowered euclidean loss by **1.65% at one aligned session** and **2.60% at five aligned sessions**. its 22-session frobenius loss was **0.45% higher**. one-session long-only portfolio risk was 9.39% for the graph model and 9.33% for har-drd, compared with 11.17% for equal weights.

![relative forecast losses on the public 27-stock panel](docs/forecast-loss.png)

## run

from this folder, use python 3.10 or 3.11:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name covariance-har --display-name "python (covariance har)"
```

open the notebook with the new kernel and run all cells from this folder. the first run caches the public parquet files locally (approximately 0.5 gb). to run on licensed data, use the documented `SOURCE = "lobster"` input mode, with aligned minute-price csvs for the 27 tickers and spy. the raw data stays in the ignored data folder. run `python -m unittest discover -s tests -v` to repeat the protocol regression checks.

## data validation and calendar sensitivity

The [LOBSTER validation and follow-up diagnosis](docs/lobster-validation.md) checks real quote samples from `lob-forecasting-lab` against the covariance input path and reports controlled calendar-gap experiments. The measured covariance agrees with an independent calculation. Screening incomplete lag histories does not recover the paper's distance-loss ranking on the public panel.

`forecast_grid(..., reference_calendar=reference_dates)` optionally excludes features and targets that bridge missing reference sessions. This preserves the original retained-row calibration intervals while reducing eligible training-label counts; it does not recreate the paper's 1,000-label experiment. Without a reference calendar, the default forecasts are unchanged. `validate_lobster.py`, `calendar_sensitivity.py`, and `calendar_safe_replication.py` reproduce the additional checks.

## paper 5 appendix checks

The [paper 5 review](docs/paper5-review.md) compares the uploaded September 2023 manuscript with the implementation and tests Appendices C–E. Optional log/Fisher models reduce distance losses on the reused public sample, including with calendar screening, but do not uniformly improve QLIKE or portfolio risk. The review retains all variants, explicit inverse-transform assumptions, and both the appendix's model set and a full comparison including HAR-Cholesky.

Use `transformation="log_fisher"`, `graph_effect="daily"`, or `refit_every=1` / `5` for separate experiments. Defaults remain the original levels model with all graph lags and monthly refits. `paper5_experiments.py` and `paper5_portfolios.py` reproduce the comparisons; the existing notebook and website results are unchanged.

## extended research

[executed extension](Extended_Covariance.ipynb) · [proofs](docs/theory.md) · [results and use cases](docs/extension-results.md) · [saved tables](docs/tables)

the extension learns an adaptive mixture of covariance shapes from har-drd, ghar(gl,l) and har-cholesky. it keeps the har-drd total variance and updates mixture weights only after targets are observed. proofs establish the line-graph spectrum and contraction, a loss–risk counterexample, a conditional portfolio bound, positive mixing and shape-loss regret bounds.

on 443 confirmation origins in 2025–october 2026, its one-session loss ratios against har-drd are **0.9977, 0.9979 and 0.9918**. the primary shape-loss comparison against a uniform mixture has a 95% interval crossing zero, and the adaptive model does not improve long-only portfolio risk. the extension reports this mixed result directly.

![adaptive covariance architecture](docs/adaptive-architecture.png)

![proved statements and their limits](docs/guarantee-map.png)

## geometric and spectral architecture

[executed notebook](Spectral_Covariance.ipynb) · [literature and architecture](docs/spectral/research.md) · [results with uncertainty](docs/spectral/results.md)

the newer candidate combines geometric covariance experts, a nonlinear regime gate, learned eigenvalue cleaning and a condition-number guard. its equal training objective targets frobenius distance and long-only portfolio risk. on the reused 2025–2026 benchmark, frobenius distance is 1.79% lower than har-drd and annualized long-only risk is 9.406%, versus 9.435%. the risk interval crosses zero, so the joint improvement criterion is not passed. the static control has stronger matrix-distance point results. fresh confirmation remains necessary.

![geometric spectral architecture and equal objectives](docs/spectral-architecture.png)
