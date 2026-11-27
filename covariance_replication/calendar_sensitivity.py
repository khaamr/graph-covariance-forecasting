"""Controlled calendar-gap experiment on the dense 2022-2024 public panel.

Within each paired run, targets, training-label rows, forecast dates, graphs,
fit windows and estimators match. Only lag-history availability differs.
Synthetic removal is diagnostic and does not repair the original experiment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from market_data import load_market
from models import (covariance_components, correlation_graph_features, graph_augmented_features,
                    har_lags, normalized_graph, panel_ols, precision_graph, rebuild_drd)
from audit_replication import daily_losses


def mapped_lags(values, retained):
    """Recreate row-compressed history and map it back onto the full time axis."""
    retained = np.asarray(retained, dtype=bool)
    if retained.shape != (len(values),):
        raise ValueError("Retained mask must have one entry per observation.")
    result = np.full((*values.shape, 3), np.nan)
    result[retained] = har_lags(values[retained])
    return result


def run(data_dir, output_dir, seeds=(11, 27, 41), removal_fraction=.4, lookback=400):
    if not 0 <= removal_fraction < 1:
        raise ValueError("Removal fraction must lie in [0, 1).")
    # load_market's existing shared-session screening is unchanged.
    panel = load_market('syncster', data_dir, '2022-01-01', '2024-12-31')
    dates, covariance = panel.dates, panel.covariance
    # Verify density against the source's screened SPY calendar, not weekdays.
    from market_data import minute_panel
    reference = minute_panel('SPY', 'syncster', data_dir, '2022-01-01', '2024-12-31').index
    if not dates.equals(reference):
        raise ValueError("The experiment requires a complete shared screened-SPY panel.")
    n = len(panel.symbols)
    pairs, lower = np.tril_indices(n, -1), np.tril_indices(n)
    variance, correlation, cholesky = covariance_components(covariance)
    targets = {'v': variance, 'r': correlation, 'c': cholesky}
    full = {k: har_lags(v) for k, v in targets.items()}
    month_index = dates.to_period('M')
    test = np.flatnonzero((dates >= '2024-01-01') & (np.arange(len(dates)) >= lookback + 22))
    records, folds = [], []
    for seed in seeds:
        retained = np.ones(len(dates), dtype=bool)
        history = dates < '2024-01-01'
        retained[history] = np.random.default_rng(seed).random(history.sum()) >= removal_fraction
        compressed = {k: mapped_lags(v, retained) for k, v in targets.items()}
        inputs = {'complete_history': full, 'compressed_history': compressed}
        for clock in inputs:
            inputs[clock] = dict(inputs[clock])
            inputs[clock]['corr'] = correlation_graph_features(inputs[clock]['r'], pairs, n)
        for month in month_index[test].unique():
            future = test[month_index[test] == month]
            origin = future[0]
            candidates = np.arange(origin - lookback, origin)
            usable = retained[candidates] & np.isfinite(compressed['v'][candidates]).all(axis=(1, 2))
            train = candidates[usable]
            # The graph and all training target rows match between both clocks.
            edges, _, _, _ = precision_graph(panel.graph_return[train])
            weight = normalized_graph(edges)
            for clock, x in inputs.items():
                v = panel_ols(x['v'], variance, train, future)
                v = np.where(v > 0, v, variance[train].min(axis=0))
                gv = panel_ols(graph_augmented_features(x['v'], weight), variance, train, future)
                gv = np.where(gv > 0, gv, variance[train].min(axis=0))
                r = panel_ols(x['corr']['-'], correlation, train, future)
                gr = panel_ols(x['corr']['L'], correlation, train, future)
                c = panel_ols(x['c'], cholesky, train, future)
                triangle = np.zeros((len(future), n, n))
                triangle[:, lower[0], lower[1]] = c
                predictions = {'HAR-DRD': rebuild_drd(v, r)[0],
                               'HAR-Cholesky': triangle @ triangle.swapaxes(-1, -2),
                               'GHAR(GL,L)': rebuild_drd(gv, gr)[0]}
                for model, matrices in predictions.items():
                    rows = pd.DataFrame(daily_losses(covariance[future], matrices), columns=['Euclidean', 'Frobenius', 'QLIKE'])
                    records.append(rows.assign(date=dates[future], seed=seed, clock=clock, model=model))
            folds.append({'seed': seed, 'month': str(month), 'training_labels_both_clocks': len(train),
                          'reference_window': lookback, 'graph_edges_both_clocks': int(edges.sum()//2)})
    losses = pd.concat(records, ignore_index=True)
    means = losses.groupby(['seed', 'clock', 'model'])[['Euclidean', 'Frobenius', 'QLIKE']].mean()
    # Explicit merge avoids relying on broadcasting across a missing model level.
    ratios = means.reset_index().merge(means.xs('HAR-DRD', level='model').reset_index(), on=['seed', 'clock'], suffixes=('', '_baseline'))
    for metric in ['Euclidean', 'Frobenius', 'QLIKE']:
        ratios[metric] /= ratios.pop(metric + '_baseline')
    output_dir.mkdir(parents=True, exist_ok=True)
    losses.to_csv(output_dir / 'daily-losses.csv', index=False)
    means.to_csv(output_dir / 'mean-losses.csv')
    ratios.to_csv(output_dir / 'ratios.csv', index=False)
    pd.DataFrame(folds).to_csv(output_dir / 'folds.csv', index=False)
    config = {'seeds': seeds, 'historical_removal_probability': removal_fraction,
              'reference_window_sessions': lookback, 'input_period': ['2022-01-01','2024-12-31'],
              'test_start': str(dates[test[0]].date()), 'test_end': str(dates[test[-1]].date()), 'test_origins': len(test),
              'density_reference': 'screened SPY input calendar, not a complete exchange calendar',
              'paired_controls': 'identical training targets/rows, graphs, dates, model formulas and fit windows',
              'difference': 'complete past covariance observations versus row-compressed lag histories',
              'scope': 'Artificial deletion sensitivity; cannot estimate the causal effect of original missing data or reproduce the paper.'}
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(ratios.round(6).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir, args.output_dir)
