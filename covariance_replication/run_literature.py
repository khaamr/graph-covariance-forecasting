"""Evaluate the frozen literature-guided challenger study on common dates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluation import covariance_losses, point_ratios, long_only_weights, portfolio_stats
from literature_models import challenger_forecasts, PRIMARY
from models import Forecasts
from run_joint_pool import paired_intervals


def run(panel_path, expert_path, controls_path, config_path, output):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    manifest = {'config': config, 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (panel_path, expert_path, controls_path)},
                'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ('literature_models.py', 'run_literature.py')}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    panel = np.load(panel_path)
    experts = np.load(expert_path)
    controls = np.load(controls_path)
    dates = pd.DatetimeIndex(controls['dates'])
    positions = controls['positions']
    aligned = pd.DatetimeIndex(experts['dates']).get_indexer(dates)
    assert (aligned >= 0).all()
    np.testing.assert_array_equal(panel['dates'][positions], controls['dates'])
    # BLAS reductions may differ at about 1e-12 when rebuilding the pinned panel.
    np.testing.assert_allclose(panel['covariance'][positions], controls['actual'], rtol=1e-12, atol=2e-12)
    np.testing.assert_array_equal(panel['returns'][positions], experts['returns'][aligned])
    np.testing.assert_array_equal(experts['actual'][aligned], controls['actual'])
    result = challenger_forecasts(panel['dates'], panel['covariance'], tuple(panel['symbols']),
                                 panel['reference'], dates, positions, experts['bank'][aligned, 2],
                                 config, progress=lambda message: print(message, flush=True))
    result.audit.to_csv(output / 'folds.csv', index=False)
    result.validation.to_csv(output / 'ridge-validation.csv', index=False)
    predictions = {key: controls[key] for key in controls.files
                   if key not in ('dates', 'positions', 'actual')}
    predictions.update(result.predictions)
    forecast = Forecasts(dates, positions, controls['actual'], predictions, result.audit)
    np.savez_compressed(output / 'forecasts.npz', dates=dates.to_numpy(), positions=positions,
                        actual=forecast.actual, **predictions)
    returns = panel['returns'][positions]
    all_losses = covariance_losses(forecast)
    all_losses.to_csv(output / 'daily-losses.csv', index=False)
    weights, portfolio_returns = {}, {}
    for name, matrices in predictions.items():
        print('Exact portfolio:', name, flush=True)
        weights[name] = long_only_weights(matrices)
        portfolio_returns[name] = np.sum(weights[name] * returns, axis=1)
    portfolio_returns = pd.DataFrame(portfolio_returns, index=dates)
    portfolio_returns.to_csv(output / 'portfolio-returns.csv', index_label='date')
    np.savez_compressed(output / 'portfolio-weights.npz', **weights)
    for period, mask in [('development_2024', dates < '2025-01-01'),
                         ('reused_2025_2026', dates >= '2025-01-01')]:
        folder = output / period
        folder.mkdir(exist_ok=True)
        selected = np.flatnonzero(mask)
        losses = all_losses[all_losses.date.isin(dates[selected])]
        ratios = point_ratios(losses)
        risk = pd.DataFrame([
            {'model': name, 'annual_risk_percent': portfolio_stats(w[selected], returns[selected])[0],
             'turnover': portfolio_stats(w[selected], returns[selected])[1]} for name, w in weights.items()
        ]).set_index('model')
        scores = ratios.join(risk)
        scores.to_csv(folder / 'scores.csv')
        interval_frames = []
        for name in result.predictions:
            intervals = paired_intervals(losses, portfolio_returns.iloc[selected], 'HAR-DRD', name,
                                         10000, 10, 20261127)
            intervals.insert(0, 'model', name)
            interval_frames.append(intervals)
        pd.concat(interval_frames).to_csv(folder / 'paired-intervals.csv', index=False)
        print(period, len(selected), '\n' + scores.round(6).to_string(), flush=True)
        print('Primary intervals:\n' + interval_frames[0].to_string(index=False), flush=True)
    print('Finished: primary model', PRIMARY, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('panel', 'experts', 'controls', 'config', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.panel, args.experts, args.controls, args.config, args.output)
