"""Compare the saved protocol with calendar-complete training-window screening.

The screen uses the observed SPY session calendar and retains the same 1000-row
calibration intervals and GLASSO graphs. It reduces training-label counts; this
is a sensitivity experiment, not the paper's original 1000-label protocol.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from market_data import load_market, minute_panel
from models import forecast_grid
from evaluation import covariance_losses, forecast_table


def run(data_dir, output_dir, draws=10_000):
    panel = load_market('syncster', data_dir, '2016-01-04', '2024-12-31')
    reference = minute_panel('SPY', 'syncster', data_dir, '2016-01-04', '2024-12-31').index
    original = forecast_grid(panel, horizons=(1,))[1]
    screened = forecast_grid(panel, horizons=(1,), reference_calendar=reference)[1]
    if not original.dates.equals(screened.dates):
        raise ValueError('The reference screen changed forecast dates. Align evaluation dates explicitly before comparing.')
    np.testing.assert_array_equal(original.actual, screened.actual)
    np.testing.assert_array_equal(original.audit.glasso_edges, screened.audit.glasso_edges)
    np.testing.assert_allclose(original.audit.glasso_alpha, screened.audit.glasso_alpha)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in [('original', original), ('calendar_safe', screened)]:
        losses = covariance_losses(result)
        table = forecast_table(losses, draws=draws)
        table.to_csv(output_dir / f'{name}-scores.csv')
        losses.to_csv(output_dir / f'{name}-daily-losses.csv', index=False)
        result.audit.to_csv(output_dir / f'{name}-folds.csv', index=False)
        print(name)
        print(table.loc[['HAR-DRD','HAR-Cholesky','GHAR(GL,L)']].round(6).to_string())
    summary = {'origins': len(original.dates), 'forecast_dates_and_targets_match': True,
               'glasso_graphs_match': True, 'reference': 'screened SPY sessions; not a complete exchange calendar',
               'original_training_labels': [int(original.audit.training_labels.min()), int(original.audit.training_labels.max())],
               'screened_training_labels': [int(screened.audit.training_labels.min()), int(screened.audit.training_labels.max())],
               'calibration_window': 'same rolling 1000 retained rows; screening reduces usable training labels',
               'scope': 'Calendar sensitivity only, not a reproduction of the paper data/protocol.'}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir, args.output_dir)
