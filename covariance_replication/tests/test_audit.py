"""Checks for reference-calendar diagnostics and equivalent HAR feature bases."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from audit_replication import lag_session_spans
from models import har_lags, panel_ols


class AuditChecks(unittest.TestCase):
    def test_gaps_are_counted_against_reference_sessions(self):
        reference = pd.bdate_range("2024-01-01", periods=10)
        dates = reference[[0, 1, 4, 5, 9]]
        actual = lag_session_spans(dates, reference, lags=(1, 2))
        np.testing.assert_allclose(actual.lag_1, [np.nan, 1, 3, 1, 4], equal_nan=True)
        np.testing.assert_allclose(actual.lag_2, [np.nan, np.nan, 4, 4, 5], equal_nan=True)

    def test_complete_reference_history_has_expected_spans(self):
        reference = pd.bdate_range("2024-01-01", periods=40)
        actual = lag_session_spans(reference, reference)
        for lag in (1, 5, 22):
            self.assertTrue(actual[f"lag_{lag}"].iloc[:lag].isna().all())
            self.assertTrue(actual[f"lag_{lag}"].iloc[lag:].eq(lag).all())

    def test_rejects_missing_and_unsorted_reference_dates(self):
        reference = pd.bdate_range("2024-01-01", periods=5)
        with self.assertRaises(ValueError):
            lag_session_spans(reference, reference[:-1])
        with self.assertRaises(ValueError):
            lag_session_spans(reference[::-1], reference)

    def test_overlapping_and_disjoint_lags_give_same_ols_predictions(self):
        rng = np.random.default_rng(18)
        values = rng.lognormal(size=(100, 4))
        frame = pd.DataFrame(values)
        overlapping = np.stack([frame.shift(1).rolling(lag).mean().to_numpy() for lag in (1, 5, 22)], axis=-1)
        disjoint = har_lags(values)
        train, test = np.arange(22, 80), np.arange(80, 100)
        target = rng.normal(size=(100, 4))
        for mode in ("series", "shared"):
            np.testing.assert_allclose(
                panel_ols(overlapping, target, train, test, mode),
                panel_ols(disjoint, target, train, test, mode), atol=1e-11,
            )


if __name__ == "__main__":
    unittest.main()
