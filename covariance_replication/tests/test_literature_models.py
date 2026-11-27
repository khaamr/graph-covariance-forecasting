import unittest
import numpy as np
import pandas as pd

from literature_models import as_correlation, factor_correlation, ridge_fit, ridge_predict, challenger_forecasts


class LiteratureModelTests(unittest.TestCase):
    def test_factor_target_preserves_correlation_and_positive_definiteness(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(6, 6))
        correlation = as_correlation(a @ a.T + np.eye(6))
        _, vectors = np.linalg.eigh(correlation)
        result = factor_correlation(correlation, vectors[:, -3:])
        np.testing.assert_allclose(result, result.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(result), 1, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(result).min(), 0)

    def test_ridge_recovers_distinct_asset_relations_and_smearing(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(400, 3, 2))
        coefficients = np.array([[1., -.5], [-.2, .8], [.3, .4]])
        target = np.einsum('tnf,nf->tn', features, coefficients) + np.array([1., 3., -1.])
        fit = ridge_fit(features, target, 1e-9)
        np.testing.assert_allclose(ridge_predict(features, fit), target, atol=1e-7)
        noisy = target + rng.normal(0, .4, target.shape)
        fitted = ridge_fit(features, noisy, .001)
        self.assertTrue((fitted[-1] > 0).all())

    def test_challengers_are_causal_with_future_realized_values_changed(self):
        rng = np.random.default_rng(18)
        dates = pd.bdate_range('2021-01-01', periods=400)
        raw = rng.normal(size=(400, 3, 3))
        covariance = raw @ raw.swapaxes(-1, -2) + np.eye(3)
        positions = np.arange(350, 370)
        graph = np.broadcast_to(np.eye(3), (len(positions), 3, 3)).copy()
        config = {'history': 300, 'ridge_grid': [.01, .1]}
        args = [dates, covariance, ['AAPL', 'MSFT', 'JPM'], dates, dates[positions], positions, graph, config]
        first = challenger_forecasts(*args)
        changed = covariance.copy(); changed[360:] *= 100
        args[1] = changed
        second = challenger_forecasts(*args)
        for name in first.predictions:
            np.testing.assert_array_equal(first.predictions[name][:10], second.predictions[name][:10])
            np.linalg.cholesky(first.predictions[name])
        self.assertTrue((first.audit.last_training_label < first.audit.first_forecast).all())
        self.assertTrue((first.audit.last_fitting_label < first.audit.first_validation_label).all())

    def test_rejects_forecast_with_missing_lag_session(self):
        dates = pd.bdate_range('2021-01-01', periods=400)
        panel_dates = dates.delete(349)
        matrices = np.broadcast_to(np.eye(3), (399, 3, 3)).copy()
        with self.assertRaisesRegex(ValueError, 'complete preceding HAR window'):
            challenger_forecasts(panel_dates, matrices, ['AAPL', 'MSFT', 'JPM'], dates,
                                 panel_dates[[350]], np.array([350]), matrices[:1],
                                 {'history': 300, 'ridge_grid': [.1]})


if __name__ == '__main__':
    unittest.main()
