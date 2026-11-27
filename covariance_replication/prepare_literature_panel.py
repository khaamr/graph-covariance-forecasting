"""Cache the pinned public realized panel for the literature-guided study."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from market_data import load_market, minute_panel


def run(data_dir, output):
    panel = load_market('syncster', data_dir, '2016-01-04', '2026-10-23')
    reference = minute_panel('SPY', 'syncster', data_dir, '2016-01-04', '2026-10-23').index
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, dates=panel.dates.to_numpy(), symbols=panel.symbols,
                        covariance=panel.covariance, returns=panel.portfolio_return,
                        reference=reference.to_numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir, args.output)
