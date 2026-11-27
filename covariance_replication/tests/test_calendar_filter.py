"""The optional calendar filter must not train or score across missing sessions."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from market_data import PAPER_SYMBOLS, RealizedPanel
from models import forecast_grid


class CalendarFilterChecks(unittest.TestCase):
    def make_panel(self, dates):
        rng = np.random.default_rng(31)
        n = len(dates)
        factors = rng.normal(scale=.1, size=(n, 3, 3))
        covariance = factors @ factors.swapaxes(-1, -2) + np.eye(3)
        return RealizedPanel(dates, PAPER_SYMBOLS[:3], covariance, np.ones((n,3)),
                             rng.normal(size=(n,3)), np.zeros((n,3)), np.ones(n), pd.Series(dtype=int))

    def test_complete_calendar_preserves_default_forecasts(self):
        dates = pd.bdate_range('2020-01-02', periods=100)
        panel = self.make_panel(dates)
        original = forecast_grid(panel, horizons=(1,5), lookback=40)
        filtered = forecast_grid(panel, horizons=(1,5), lookback=40, reference_calendar=dates)
        for h in original:
            np.testing.assert_array_equal(original[h].positions, filtered[h].positions)
            self.assertTrue(filtered[h].audit.calendar_excluded_labels.eq(0).all())
            for model in original[h].predictions:
                np.testing.assert_allclose(original[h].predictions[model], filtered[h].predictions[model], atol=1e-12)

    def test_gap_filter_checks_training_histories_and_future_targets(self):
        reference = pd.bdate_range('2020-01-02', periods=120)
        dates = reference.delete([50,91])
        panel = self.make_panel(dates)
        positions = reference.get_indexer(dates)
        results = forecast_grid(panel, horizons=(1,5), lookback=40, reference_calendar=reference)
        for h,f in results.items():
            for t in f.positions:
                self.assertEqual(positions[t] - positions[t-22], 22)
                self.assertEqual(positions[t+h-1] - positions[t], h-1)
            for row in f.audit.itertuples():
                origin = dates.get_loc(row.first_forecast)
                candidates = np.arange(origin-40, origin-h+1)
                valid = ((positions[candidates] - positions[candidates-22] == 22)
                         & (positions[candidates+h-1] - positions[candidates] == h-1))
                self.assertEqual(row.training_labels, int(valid.sum()))
                self.assertEqual(row.calendar_excluded_labels, len(candidates)-int(valid.sum()))
        self.assertTrue(results[1].audit.calendar_excluded_labels.gt(0).any())

    def test_reference_calendar_cannot_omit_panel_dates(self):
        dates = pd.bdate_range('2020-01-02', periods=100)
        with self.assertRaisesRegex(ValueError,'Every panel date'):
            forecast_grid(self.make_panel(dates), horizons=(1,), lookback=40, reference_calendar=dates[:-1])

    def test_no_complete_history_fails_instead_of_using_compressed_rows(self):
        reference = pd.bdate_range('2020-01-02', periods=200)
        with self.assertRaisesRegex(ValueError,'No complete held-out targets'):
            forecast_grid(self.make_panel(reference[::2]), horizons=(1,), lookback=40, reference_calendar=reference)


if __name__ == '__main__':
    unittest.main()
