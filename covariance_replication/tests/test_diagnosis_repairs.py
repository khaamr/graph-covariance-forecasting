import unittest

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize._numdiff import approx_derivative

from factor_vhar import factor_forecasts, matrix_smearing
from realized_heavy import (HeavyFit, advance_heavy, as_correlation,
                            correlation_objective, heavy_forecasts, variance_objective)


class RepairTests(unittest.TestCase):
    def test_variance_score_gradient_across_missing_sessions(self):
        rng=np.random.default_rng(11);values=np.exp(rng.normal(0,.3,100))
        gaps=np.ones(100,dtype=np.int64);gaps[30]=5;gaps[70]=3
        p=np.array([.1,.15,.7]);value,gradient,path=variance_objective(p,values,gaps)
        numeric=approx_derivative(lambda q:variance_objective(q,values,gaps)[0],p).ravel()
        np.testing.assert_allclose(gradient,numeric,rtol=1e-5,atol=1e-6)
        self.assertTrue(np.isfinite(value));self.assertTrue((path>0).all())

    def test_correlation_score_gradient_across_missing_sessions(self):
        rng=np.random.default_rng(12);raw=rng.normal(size=(70,4,4));cov=raw@raw.swapaxes(-1,-2)+np.eye(4)
        realized=as_correlation(cov);target=realized.mean(axis=0)
        gaps=np.ones(70,dtype=np.int64);gaps[25]=4
        p=np.array([.15,.7]);_,gradient=correlation_objective(p,realized,cov/5,target,gaps)
        numeric=approx_derivative(lambda q:correlation_objective(q,realized,cov/5,target,gaps)[0],p).ravel()
        np.testing.assert_allclose(gradient,numeric,rtol=1e-5,atol=1e-6)

    def test_gap_propagation_matches_closed_form_and_is_causal(self):
        realized=np.array([[1.,.3],[.3,1.]])
        fit=HeavyFit(np.array([[.2,.2,.7],[.2,.2,.7]]),np.ones(2),np.full(2,3.),
                     np.array([.1,.8]),np.eye(2),np.eye(2),realized,{})
        matrices=np.broadcast_to(np.eye(2),(2,2,2)).copy()
        first=advance_heavy(fit,matrices,np.array([3,1]))
        np.testing.assert_allclose(np.diag(first[0]),1.595)
        self.assertAlmostEqual(as_correlation(first[0])[0,1],.0243)
        matrices[0]*=100
        other=advance_heavy(fit,matrices,np.array([3,1]))
        np.testing.assert_array_equal(first[0],other[0]);self.assertFalse(np.allclose(first[1],other[1]))

    def test_scalar_smearing_matches_exact_expectation(self):
        result=matrix_smearing(np.array([[.7]]),np.array([[-.5],[.5]]))
        np.testing.assert_allclose(result,[[np.exp(.7)*np.cosh(.5)]],atol=1e-14)

    def test_matrix_smearing_uses_full_noncommuting_residuals(self):
        mean=np.array([[.3,.1],[.1,-.1]]);error=np.array([[.2,.4],[.4,-.1]])
        vectors=np.array([[.2,.4,-.1],[-.2,-.4,.1]])
        expected=(expm(mean+error)+expm(mean-error))/2
        result=matrix_smearing(mean,vectors)
        np.testing.assert_allclose(result,expected,rtol=1e-13,atol=1e-13)
        np.linalg.cholesky(result);self.assertGreater(np.trace(result),np.trace(expm(mean)))

    def test_repaired_factor_ignores_future_targets(self):
        rng=np.random.default_rng(14);dates=pd.bdate_range('2021-01-01',periods=320)
        raw=rng.normal(size=(320,6,6));cov=raw@raw.swapaxes(-1,-2)+np.eye(6)
        positions=np.array([280,281,282]);weights=np.ones((6,1))/6
        args=[dates,cov,positions,weights,['a','a','b','b','c','c'],dates]
        first=factor_forecasts(*args,lookback=240,clean=False,bias_correction=True)
        args[1]=cov.copy();args[1][282:]*=10
        other=factor_forecasts(*args,lookback=240,clean=False,bias_correction=True)
        for name in first.predictions:
            np.testing.assert_array_equal(first.predictions[name],other.predictions[name])
            np.linalg.cholesky(first.predictions[name])
        self.assertFalse(first.cleaning.flagged.any())

    def test_heavy_pipeline_causality_and_positive_matrices(self):
        rng=np.random.default_rng(15);dates=pd.bdate_range('2021-01-01',periods=180)
        raw=rng.normal(size=(180,3,3));cov=raw@raw.swapaxes(-1,-2)+np.eye(3)
        positions=np.arange(140,145)
        first,audit,_=heavy_forecasts(dates,cov,positions,dates,lookback=120)
        changed=cov.copy();changed[143:]*=100
        other,_,_=heavy_forecasts(dates,changed,positions,dates,lookback=120)
        np.testing.assert_array_equal(first[:4],other[:4]);np.linalg.cholesky(first)
        self.assertTrue((audit.last_training_label<audit.first_forecast).all())


if __name__=='__main__':
    unittest.main()
