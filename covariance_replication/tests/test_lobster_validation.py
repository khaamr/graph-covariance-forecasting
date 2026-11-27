"""Checks for real-quote sampling and the controlled calendar experiment."""
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calendar_sensitivity import mapped_lags
from models import har_lags
from validate_lobster import OPEN_NS, CLOSE_NS, previous_tick_grid, read_sample


class QuoteValidationChecks(unittest.TestCase):
    def test_quotes_are_previous_tick_and_duplicate_times_use_last_update(self):
        times = np.array([OPEN_NS + 100, OPEN_NS + 60 * 10**9, OPEN_NS + 60 * 10**9, CLOSE_NS])
        grid, source, audit = previous_tick_grid(times, np.array([100., 101., 102., 103.]))
        self.assertEqual(grid[0], 100.)
        self.assertEqual(grid[1], 102.)
        self.assertEqual(grid[2], 102.)
        self.assertEqual(grid[-1], 103.)
        self.assertTrue(audit['opening_proxy'])
        self.assertTrue(np.all(source[1:] <= OPEN_NS + np.arange(1, 391) * 60 * 10**9))

    def test_late_opening_proxy_is_rejected(self):
        with self.assertRaises(ValueError):
            previous_tick_grid(np.array([OPEN_NS + 2 * 10**9, CLOSE_NS]), np.array([100., 101.]))

    def test_exact_opening_needs_no_proxy(self):
        _, _, audit = previous_tick_grid(np.array([OPEN_NS, CLOSE_NS]), np.array([100., 101.]))
        self.assertFalse(audit['opening_proxy'])
        self.assertEqual(audit['opening_delay_seconds'], 0.)

    def test_lobster_decimal_timestamps_and_fixed_point_prices(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'sample.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('AAPL_2012-06-21_orderbook_10.csv', '1000100,5,999900,8\n1000200,9,1000000,7\n')
                bundle.writestr('AAPL_2012-06-21_message_10.csv', '34200.000000001,1,1,1,1,1\n34200.1,1,1,1,1,1\n')
            times, mid, _ = read_sample(archive, 'AAPL')
            np.testing.assert_array_equal(times, [OPEN_NS + 1, OPEN_NS + 100000000])
            np.testing.assert_allclose(mid, [100., 100.01])


class CalendarExperimentChecks(unittest.TestCase):
    def test_no_deletion_recovers_original_features(self):
        values = np.arange(120).reshape(40, 3)
        np.testing.assert_allclose(mapped_lags(values, np.ones(40, dtype=bool)), har_lags(values), equal_nan=True)

    def test_removed_session_changes_lag_one_without_changing_target_rows(self):
        values = np.arange(80, dtype=float).reshape(40, 2)
        retained = np.ones(40, dtype=bool)
        retained[30] = False
        compressed = mapped_lags(values, retained)
        np.testing.assert_array_equal(compressed[31, :, 0], values[29])
        np.testing.assert_array_equal(har_lags(values)[31, :, 0], values[30])
        self.assertTrue(np.isnan(compressed[30]).all())

    def test_future_values_cannot_change_current_features(self):
        rng = np.random.default_rng(12)
        values = rng.normal(size=(80, 3))
        retained = rng.random(80) > .2
        retained[50] = True
        before = mapped_lags(values, retained)[50].copy()
        values[50:] += 10000
        np.testing.assert_allclose(mapped_lags(values, retained)[50], before)


if __name__ == '__main__':
    unittest.main()
