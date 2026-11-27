"""Fixed, paper-motivated Appendix C-E checks; no tuning to published numbers.

Every variant uses the same public sample and 623 one-session targets. Inverse
log/Fisher transformations are exp/tanh without an unspecified bias correction.
Appendix-specific MCS sets and the full set including Cholesky are both retained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from market_data import load_market, minute_panel
from models import forecast_grid
from evaluation import METRICS, covariance_losses, forecast_table

SPECIFICATIONS = {
    'main_levels_monthly': {},
    'appendix_E_log_fisher_monthly': {'transformation': 'log_fisher'},
    'appendix_C_daily_graph_monthly': {'graph_effect': 'daily'},
    'appendix_D_levels_every_5_sessions': {'refit_every': 5},
    'appendix_D_levels_daily': {'refit_every': 1},
}


def run(data_dir, output_dir, variants=None, calendar_safe=False):
    selected_specs = SPECIFICATIONS if variants is None else {name: SPECIFICATIONS[name] for name in variants}
    if next(iter(selected_specs)) != 'main_levels_monthly':
        raise ValueError('Begin with main_levels_monthly so absolute ratios have a fixed reference.')
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {'variants': selected_specs, 'horizons': [1], 'lookback_retained_sessions': 1000,
                'source_period': ['2016-01-04', '2024-12-31'], 'intercept': 'series',
                'inverse_transforms': 'exp/tanh, no bias correction; transform before temporal/graph aggregation',
                'weekly_definition': 'five retained forecast sessions anchored to the common first date',
                'calendar_mode': 'complete lag/target windows on screened SPY calendar' if calendar_safe else 'original retained-row protocol for controlled comparison to saved results',
                'MCS': {'draws': 10000, 'block': 2, 'seed': 4274989},
                'source_sha256': {f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ['models.py','market_data.py']},
                'scope': 'Existing public evaluation sample reused for diagnosis; not fresh confirmation.'}
    (output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    panel = load_market('syncster', data_dir, '2016-01-04', '2024-12-31')
    reference = minute_panel('SPY', 'syncster', data_dir, '2016-01-04', '2024-12-31').index if calendar_safe else None
    baseline_dates = None
    baseline_means = None
    records = []
    for variant, options in selected_specs.items():
        print('RUNNING', variant, flush=True)
        forecast = forecast_grid(panel, horizons=(1,), reference_calendar=reference, **options)[1]
        if baseline_dates is None:
            baseline_dates = forecast.dates
        if not baseline_dates.equals(forecast.dates):
            raise AssertionError('Forecast dates changed across paper variants.')
        losses = covariance_losses(forecast)
        means = losses.groupby('model')[list(METRICS)].mean()
        if baseline_means is None:
            baseline_means = means.loc['HAR-DRD']
        folder = output_dir / variant
        folder.mkdir(exist_ok=True)
        losses.to_csv(folder / 'daily-losses.csv', index=False)
        forecast.audit.to_csv(folder / 'folds.csv', index=False)
        full_scores = forecast_table(losses)
        full_scores.to_csv(folder / 'full-scores.csv')
        # The paper's C.1/E.1 tables omit Cholesky; keep that effect explicit.
        if options.get('graph_effect') == 'daily' or options.get('transformation') == 'log_fisher':
            appendix_losses = losses[losses.model != 'HAR-Cholesky']
            forecast_table(appendix_losses).to_csv(folder / 'appendix-scores.csv')
        for model, row in means.iterrows():
            record = {'variant': variant, 'model': model, 'origins': len(forecast.dates), 'refits':len(forecast.audit)}
            for metric in METRICS:
                record[metric + '_mean'] = row[metric]
                record[metric + '_within_variant_ratio'] = row[metric] / means.loc['HAR-DRD',metric]
                record[metric + '_relative_to_original_DRD'] = row[metric] / baseline_means[metric]
            records.append(record)
        pd.DataFrame(records).to_csv(output_dir / 'summary.csv', index=False)
        selected = ['HAR-Cholesky','HAR-DRD', 'GHAR-S(GL,L)' if options.get('graph_effect') == 'daily' else 'GHAR(GL,L)']
        print(full_scores.loc[selected].round(6).to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=list(SPECIFICATIONS))
    parser.add_argument('--calendar-safe', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir, args.output_dir, args.variants, args.calendar_safe)
