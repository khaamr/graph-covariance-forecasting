"""Diagnose calendar and benchmark sensitivity without changing saved results.

Run from any directory:
    python audit_replication.py --data-dir /path/to/cached/syncster --output-dir /tmp/audit

The reference calendar is the screened SPY input, so measured gaps are lower bounds
relative to a complete exchange calendar. No prices are imputed across sessions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from market_data import load_market, minute_panel
from models import covariance_components, forecast_grid, har_lags, panel_ols


def lag_session_spans(dates, reference, lags=(1, 5, 22)):
    """Count reference sessions between a row and its lagged retained observation."""
    dates, reference = pd.DatetimeIndex(dates), pd.DatetimeIndex(reference)
    if (not dates.is_unique or not dates.is_monotonic_increasing
            or not reference.is_unique or not reference.is_monotonic_increasing):
        raise ValueError("Dates and reference dates must be unique and increasing.")
    positions = reference.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("Every retained date must occur in the reference calendar.")
    result = pd.DataFrame(index=dates)
    for lag in lags:
        if not isinstance(lag, int) or lag < 1:
            raise ValueError("Lags must be positive integers.")
        result[f"lag_{lag}"] = pd.Series(positions, index=dates).diff(lag)
    return result


def daily_losses(actual, predicted):
    lower = np.tril_indices(actual.shape[-1])
    error = predicted - actual
    np.linalg.cholesky(predicted)
    return np.column_stack([
        np.linalg.norm(error[:, lower[0], lower[1]], axis=1),
        np.linalg.norm(error, axis=(1, 2)),
        np.linalg.slogdet(predicted)[1]
        + np.trace(np.linalg.solve(predicted, actual), axis1=-2, axis2=-1),
    ])


def ordering_forecast(panel, forecast, order, lookback):
    """Change only the Cholesky ordering; map predictions back before scoring."""
    covariance = panel.covariance[:, order][:, :, order]
    target = covariance_components(covariance)[2]
    features = har_lags(target)
    months = panel.dates.to_period("M")
    lower = np.tril_indices(len(order))
    inverse = np.argsort(order)
    result = []
    for month in months[forecast.positions].unique():
        future = forecast.positions[months[forecast.positions] == month]
        origin = future[0]
        train = np.arange(origin - lookback, origin)
        values = panel_ols(features, target, train, future)
        triangle = np.zeros((len(future), len(order), len(order)))
        triangle[:, lower[0], lower[1]] = values
        matrices = triangle @ triangle.swapaxes(-1, -2)
        result.append(matrices[:, inverse][:, :, inverse])
    return np.concatenate(result)


def run(data_dir, output_dir):
    start, end, lookback = "2016-01-04", "2024-12-31", 1000
    panel = load_market("syncster", data_dir, start, end)
    reference = minute_panel("SPY", "syncster", data_dir, start, end).index
    forecast = forecast_grid(panel, horizons=(1,), lookback=lookback)[1]
    origin = forecast.positions[0]
    train = np.arange(origin - lookback, origin)
    spans = lag_session_spans(panel.dates, reference)
    selected = spans.iloc[train]
    span_summary = {
        str(lag): {
            "median_reference_sessions": float(selected[f"lag_{lag}"].median()),
            "max_reference_sessions": int(selected[f"lag_{lag}"].max()),
            "affected_training_rows": int(selected[f"lag_{lag}"].gt(lag).sum()),
        } for lag in (1, 5, 22)
    }
    metric_names = ["Euclidean", "Frobenius", "QLIKE"]
    loss = {k: daily_losses(forecast.actual, v) for k, v in forecast.predictions.items()}
    baseline = loss["HAR-DRD"].mean(axis=0)
    ratios = pd.DataFrame({k: v.mean(axis=0) / baseline for k, v in loss.items()}, index=metric_names).T
    orders = {"sector": np.arange(len(panel.symbols)),
              "alphabetical": np.argsort(panel.symbols),
              "reversed": np.arange(len(panel.symbols))[::-1]}
    order_ratios = {}
    for name, order in orders.items():
        prediction = ordering_forecast(panel, forecast, order, lookback)
        order_ratios[name] = dict(zip(metric_names, (daily_losses(forecast.actual, prediction).mean(axis=0) / baseline).tolist()))
    # Diagnostic scoring exclusion only: original training/features remain unchanged.
    # This is not a corrected-data backtest or a reason to discard adverse outcomes.
    exclude = pd.to_datetime(["2023-01-24", "2023-01-25"])
    keep = ~forecast.dates.isin(exclude)
    sensitivity = {k: dict(zip(metric_names, (v[keep].mean(axis=0) / loss["HAR-DRD"][keep].mean(axis=0)).tolist())) for k, v in loss.items()}
    pre2022 = reference[reference < "2022-01-01"]
    missing = {ticker: len(pre2022.difference(minute_panel(ticker, "syncster", data_dir, start, end).index)) for ticker in panel.symbols}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "reference_calendar": "screened SPY sessions; gaps are lower bounds",
        "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("models.py", "market_data.py")},
        "first_forecast": str(forecast.dates[0].date()),
        "first_training_feature": str(panel.dates[train[0]].date()),
        "origins": len(forecast.dates), "first_fold_lag_spans": span_summary,
        "first_fold_complete_22_session_training_rows": int(selected.lag_22.eq(22).sum()),
        "missing_spy_sessions_before_2022_by_stock": dict(sorted(missing.items(), key=lambda t: -t[1])),
        "ordering_ratios": order_ratios,
        "scoring_only_exclusion_2023_01_24_and_25": sensitivity,
        "scoring_exclusion_warning": "Training and predictors are unchanged; not a corrected backtest.",
        "glasso_penalty_increased_months": int(forecast.audit.penalty_increased.sum()),
        "correlation_repairs": int(forecast.audit.correlation_repairs.sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ratios.to_csv(output_dir / "ratios.csv")
    spans.to_csv(output_dir / "lag-session-spans.csv", index_label="date")
    forecast.audit.to_csv(output_dir / "folds.csv", index=False)
    rows = [pd.DataFrame(v, columns=metric_names).assign(date=forecast.dates, model=k) for k, v in loss.items()]
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "daily-losses.csv", index=False)
    print(ratios.loc[["HAR-Cholesky", "HAR-DRD", "GHAR(GL,L)"]].round(6).to_string())
    print(json.dumps(span_summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Cached Syncster Parquet directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir, args.output_dir)
