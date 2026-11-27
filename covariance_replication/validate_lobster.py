"""Validate the covariance input path against the LOB lab's real quote samples.

No downloads or forecasting: uses three cached archives verified against the lab
manifest, preserves the full session, and compares independent calculations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from market_data import minute_panel, realized_measures

OPEN_NS = 34200 * 10**9
CLOSE_NS = 57600 * 10**9
SYMBOLS = ("AAPL", "INTC", "MSFT")


def previous_tick_grid(time_ns, midpoint, max_open_delay_ns=10**9):
    """Sample past quotes at minute boundaries, recording the opening proxy.

    The samples start milliseconds after 09:30. Only the first grid point uses
    the first observed quote, and only within a one-second tolerance. Its true
    timestamp is recorded; all other points strictly use observed past quotes.
    """
    time_ns, midpoint = np.asarray(time_ns), np.asarray(midpoint)
    if len(time_ns) == 0 or len(time_ns) != len(midpoint):
        raise ValueError("Supply nonempty paired times and midpoints.")
    if np.any(np.diff(time_ns) < 0):
        raise ValueError("Quote timestamps must be nondecreasing.")
    if not (np.isfinite(midpoint).all() and (midpoint > 0).all()):
        raise ValueError("Midpoints must be finite and positive.")
    if time_ns[0] < OPEN_NS or time_ns[-1] > CLOSE_NS:
        raise ValueError("Supply regular-session quotes only.")
    grid = OPEN_NS + np.arange(391, dtype=np.int64) * 60 * 10**9
    positions = np.searchsorted(time_ns, grid, side="right") - 1
    opening_proxy = positions[0] == -1
    if opening_proxy:
        if time_ns[0] - OPEN_NS > max_open_delay_ns:
            raise ValueError("The first quote is too late to proxy the opening point.")
        positions[0] = 0
    if (positions < 0).any():
        raise ValueError("A grid point has no observed price.")
    source_ns = time_ns[positions]
    age_seconds = (grid - source_ns) / 1e9
    return midpoint[positions], source_ns, {
        "opening_proxy": bool(opening_proxy),
        "opening_delay_seconds": max(0., float(-age_seconds[0])),
        "max_nonopening_quote_age_seconds": float(age_seconds[1:].max()),
    }


def read_sample(archive, ticker):
    with zipfile.ZipFile(archive) as bundle:
        def name(part):
            matches = [n for n in bundle.namelist() if n.endswith('.csv') and f'_{part}_' in n and Path(n).name.startswith(ticker + '_')]
            if len(matches) != 1:
                raise ValueError(f"Expected one {ticker} {part} file.")
            return matches[0]
        book = pd.read_csv(bundle.open(name('orderbook')), header=None, usecols=[0, 1, 2, 3])
        messages = pd.read_csv(bundle.open(name('message')), header=None, usecols=[0], dtype=str)
    if len(book) != len(messages):
        raise ValueError("Orderbook and message rows do not align.")
    # Decimal parsing avoids rounding event timestamps through binary floats.
    parts = messages[0].str.split('.', expand=True)
    fraction = parts[1] if len(parts.columns) > 1 else pd.Series('0', index=parts.index)
    time_ns = (parts[0].astype('int64') * 10**9
               + fraction.fillna('0').str.pad(9, side='right', fillchar='0').str[:9].astype('int64'))
    ask, bid = book[0].to_numpy() / 10000., book[2].to_numpy() / 10000.
    valid = np.isfinite(ask) & np.isfinite(bid) & (ask > bid) & (bid > 0) & (ask < 1e8)
    valid &= (book[1].to_numpy() > 0) & (book[3].to_numpy() > 0)
    valid &= time_ns.between(OPEN_NS, CLOSE_NS).to_numpy()
    return time_ns.to_numpy()[valid], ((ask + bid) / 2)[valid], {"raw_rows": len(book), "usable_top_quote_rows": int(valid.sum())}


def run(data_dir, manifest_path, output_dir):
    manifest = json.loads(manifest_path.read_text())
    records = {r['asset'].upper(): r for r in manifest if r['kind'] == 'lobster'}
    dates = {records[s]['date'] for s in SYMBOLS}
    if len(dates) != 1:
        raise ValueError("The stock samples must share a date.")
    date = next(iter(dates))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / 'minute_csv'
    csv_dir.mkdir(exist_ok=True)
    grids, through_loader, provenance = [], [], {}
    for ticker in SYMBOLS:
        record = records[ticker]
        archive = data_dir / Path(record['path']).name
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise ValueError(f"Archive checksum mismatch for {ticker}.")
        times, mid, counts = read_sample(archive, ticker)
        grid, observed_ns, timing = previous_tick_grid(times, mid)
        grids.append(grid)
        stamp = pd.date_range(f'{date} 09:30', periods=390, freq='min', tz='America/New_York').tz_convert('UTC')
        # The loader's convention is a start-labelled minute bar plus its endpoint.
        pd.DataFrame({'ts': stamp, 'open': grid[:-1], 'close': grid[1:]}).to_csv(csv_dir / f'{ticker}.csv', index=False)
        via = minute_panel(ticker, 'lobster', csv_dir, date, date)
        if len(via) != 1:
            raise AssertionError("The minute adapter lost the sample session.")
        through_loader.append(via.to_numpy()[0])
        pd.DataFrame({'minute_offset': np.arange(391), 'midpoint': grid, 'quote_seconds_since_midnight': observed_ns / 1e9}).to_csv(output_dir / f'{ticker}-grid.csv', index=False)
        provenance[ticker] = {'sha256': digest, **counts, **timing}
    direct = np.stack(grids, axis=-1)
    loaded = np.stack(through_loader, axis=-1)
    np.testing.assert_allclose(loaded, direct, rtol=1e-14, atol=1e-12)
    log_prices = np.log(direct) * 100
    estimated, quarticity = realized_measures(np.log(loaded)[None] * 100)
    # Independent implementation: enumerate every rolling 5-minute window.
    rolling = log_prices[5:] - log_prices[:-5]
    reference = np.zeros((3, 3))
    for increment in rolling:
        reference += np.outer(increment, increment) / 5
    np.testing.assert_allclose(estimated[0], reference, rtol=1e-12, atol=1e-12)
    # Quarticity has a grid-specific number-of-increments multiplier.
    independent_q = np.zeros(3)
    for offset in range(5):
        grid_values = log_prices[offset::5]
        increments = grid_values[1:] - grid_values[:-1]
        independent_q += len(increments) * np.power(increments, 4).sum(axis=0) / 15
    np.testing.assert_allclose(quarticity[0], independent_q, rtol=1e-12, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(estimated[0])
    np.linalg.cholesky(estimated[0])
    summary = {'date': date, 'symbols': SYMBOLS, 'session': '09:30-16:00 America/New_York',
               'opening_policy': 'first observed quote within one second, recorded explicitly; no exact 09:30 quote is available',
               'archives': provenance, 'minute_adapter_max_price_error': float(abs(loaded-direct).max()),
               'covariance_max_absolute_error': float(abs(estimated[0]-reference).max()),
               'quarticity_max_absolute_error': float(abs(quarticity[0]-independent_q).max()),
               'minimum_covariance_eigenvalue': float(eigenvalues.min()),
               'covariance_percent_squared': estimated[0].tolist(),
               'scope': 'One-day measurement validation only; not a forecasting replication.'}
    pd.DataFrame(estimated[0], index=SYMBOLS, columns=SYMBOLS).to_csv(output_dir / 'covariance.csv')
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--lab-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_dir, args.lab_manifest, args.output_dir)
