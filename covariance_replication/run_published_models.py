"""Run DCC-NL and full factor-VHAR estimators against the frozen controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from dcc_nl import MODEL, dcc_forecasts
from evaluation import covariance_losses, long_only_weights, point_ratios, portfolio_stats
from factor_vhar import factor_forecasts
from market_data import SECTORS
from models import Forecasts
from run_joint_pool import paired_intervals


def run(panel_path, controls_path, config_path, output, factor_weights_path=None):
    output.mkdir(parents=True, exist_ok=True)
    panel, controls = np.load(panel_path), np.load(controls_path)
    config = json.loads(config_path.read_text())
    inputs = [panel_path, controls_path, config_path]
    if factor_weights_path:
        inputs.append(factor_weights_path)
    manifest = {"config": config,
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("dcc_nl.py", "factor_vhar.py", "run_published_models.py")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    dates = pd.DatetimeIndex(controls["dates"])
    positions = controls["positions"]
    np.testing.assert_array_equal(panel["dates"][positions], controls["dates"])
    np.testing.assert_allclose(panel["covariance"][positions], controls["actual"], rtol=1e-12, atol=2e-12)
    returns = panel["returns"]
    predictions = {key: controls[key] for key in controls.files
                   if key not in ("dates", "positions", "actual")}
    log = lambda message: print(message, flush=True)
    cache = output / "dcc-forecasts.npz"
    if cache.exists():
        cached = np.load(cache)
        if str(cached["source_sha256"]) != manifest["source_sha256"]["dcc_nl.py"]:
            raise ValueError("DCC resume cache does not match evaluated source.")
        if str(cached["panel_sha256"]) != manifest["input_sha256"][str(panel_path)]:
            raise ValueError("DCC resume cache does not match input panel.")
        if int(cached["lookback"]) != config["dcc"]["lookback"]:
            raise ValueError("DCC resume cache uses a different estimation window.")
        np.testing.assert_array_equal(cached["positions"], positions)
        predictions[MODEL] = cached["forecast"]
    else:
        predictions[MODEL], audit, parameters = dcc_forecasts(
            panel["dates"], 100 * returns, positions, lookback=config["dcc"]["lookback"], progress=log)
        audit.to_csv(output / "dcc-folds.csv", index=False)
        parameters.to_csv(output / "garch-parameters.csv", index=False)
        np.savez_compressed(cache, forecast=predictions[MODEL], positions=positions,
                            lookback=config["dcc"]["lookback"],
                            source_sha256=manifest["source_sha256"]["dcc_nl.py"],
                            panel_sha256=manifest["input_sha256"][str(panel_path)])
    known_at = None
    if factor_weights_path:
        archive = np.load(factor_weights_path)
        np.testing.assert_array_equal(archive["dates"], panel["dates"])
        np.testing.assert_array_equal(archive["symbols"], panel["symbols"])
        weights, known_at = archive["weights"], archive["known_at"]
    else:
        weights = np.ones((len(panel["symbols"]), 1)) / len(panel["symbols"])
    factor = factor_forecasts(panel["dates"], panel["covariance"], positions, weights,
                              [SECTORS[symbol] for symbol in panel["symbols"]], panel["reference"],
                              known_at=known_at, lookback=config["factor"]["lookback"], progress=log)
    # Name the empirical factors explicitly; the estimator API stays general.
    identity = "supplied factors" if factor_weights_path else "market"
    factor_predictions = {name.replace("Factor-VHAR ", f"Factor-VHAR {identity} "): value
                          for name, value in factor.predictions.items()}
    predictions.update(factor_predictions)
    factor.audit.to_csv(output / "factor-folds.csv", index=False)
    factor.equation_audit.to_csv(output / "factor-equations.csv", index=False)
    factor.cleaning.to_csv(output / "factor-cleaning.csv", index=False)
    (output / "factor-numerics.json").write_text(json.dumps(factor.cleaning.attrs, indent=2) + "\n")
    forecast = Forecasts(dates, positions, controls["actual"], predictions, pd.DataFrame())
    np.savez_compressed(output / "forecasts.npz", dates=dates.to_numpy(), positions=positions,
                        actual=forecast.actual, **predictions)
    losses = covariance_losses(forecast)
    losses.to_csv(output / "daily-losses.csv", index=False)
    stock_weights, realized = {}, {}
    for name, matrices in predictions.items():
        log("Exact portfolio: " + name)
        stock_weights[name] = long_only_weights(matrices)
        realized[name] = np.sum(stock_weights[name] * returns[positions], axis=1)
    realized = pd.DataFrame(realized, index=dates)
    realized.to_csv(output / "portfolio-returns.csv", index_label="date")
    np.savez_compressed(output / "portfolio-weights.npz", **stock_weights)
    for period, mask in [("development_2024", dates < "2025-01-01"),
                         ("reused_2025_2026", dates >= "2025-01-01")]:
        folder = output / period
        folder.mkdir(exist_ok=True)
        selected = np.flatnonzero(mask)
        subset = losses[losses.date.isin(dates[selected])]
        scores = point_ratios(subset)
        risk = pd.DataFrame([{"model": name,
                              "annual_risk_percent": portfolio_stats(w[selected], returns[positions[selected]])[0],
                              "turnover": portfolio_stats(w[selected], returns[positions[selected]])[1]}
                             for name, w in stock_weights.items()]).set_index("model")
        scores = scores.join(risk)
        scores.to_csv(folder / "scores.csv")
        intervals = []
        for name in (MODEL, *factor_predictions):
            frame = paired_intervals(subset, realized.iloc[selected], "HAR-DRD", name,
                                     config["bootstrap"]["draws"], config["bootstrap"]["mean_block"],
                                     config["bootstrap"]["seed"])
            frame.insert(0, "model", name)
            intervals.append(frame)
        pd.concat(intervals).to_csv(folder / "paired-intervals.csv", index=False)
        log(f"{period}: {len(selected)} origins\n" + scores.round(6).to_string())
    log("Finished full-method comparison.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "controls", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--factor-weights", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.panel, args.controls, args.config, args.output, args.factor_weights)
