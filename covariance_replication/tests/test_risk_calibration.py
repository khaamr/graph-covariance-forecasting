import unittest

import numpy as np
import pandas as pd

from evaluation import long_only_weights
from risk_calibration import build_candidates,calibrate,portfolio_weights,select_paths


class RiskCalibrationTests(unittest.TestCase):
    def test_active_set_objectives_match_independent_cvxpy(self):
        rng=np.random.default_rng(205)
        covariance=[]
        for n in [3,8,27]:
            raw=rng.normal(size=(5,n,n));matrices=raw@raw.swapaxes(-1,-2)+.01*np.eye(n)
            covariance.extend([])
            weights,audit=portfolio_weights(matrices)
            exact=long_only_weights(matrices)
            value=np.einsum('ti,tij,tj->t',weights,matrices,weights)
            expected=np.einsum('ti,tij,tj->t',exact,matrices,exact)
            np.testing.assert_allclose(value,expected,rtol=1e-6,atol=1e-8)
            self.assertLess(audit['max_relative_gap'],1e-7)

    def test_simplex_corner_and_interior_known_solutions(self):
        matrix=np.array([np.diag([1.,2.,4.]),[[1.,1.1,1.2],[1.1,2.,1.5],[1.2,1.5,3.]]])
        weights,_=portfolio_weights(matrix)
        np.testing.assert_allclose(weights[0],np.array([1,.5,.25])/1.75,atol=1e-12)
        np.testing.assert_allclose(weights[1],[1,0,0],atol=1e-12)

    def test_both_calibration_maps_preserve_positive_definiteness(self):
        base=np.array([[[2.,.3],[.3,1.]]]);target=np.array([[1.4,.2],[.2,.7]])
        for kind in ['diagonal','full']:
            for strength in [0,.1,1.]:
                result=calibrate(base,target,strength,kind);np.linalg.cholesky(result)
                np.testing.assert_allclose(result,result.swapaxes(-1,-2),atol=1e-14)
        np.testing.assert_array_equal(calibrate(base,target,0,'none'),base)

    def test_no_future_labels_enter_candidates_or_selector(self):
        rng=np.random.default_rng(206);dates=pd.bdate_range('2022-01-01',periods=430)
        raw=rng.normal(size=(430,3,3));actual=raw@raw.swapaxes(-1,-2)+np.eye(3)
        bases=np.broadcast_to(np.eye(3),(430,1,3,3)).copy();returns=rng.normal(0,.01,size=(430,3))
        config={'minimum_calibration_history':252,'validation':126}
        first=build_candidates(dates,bases,actual,returns,config)
        changed_actual=actual.copy();changed_actual[400:]*=10
        changed_returns=returns.copy();changed_returns[400:]*=10
        second=build_candidates(dates,bases,changed_actual,changed_returns,config)
        a,audit=select_paths(first,dates[378:400],config);b,_=select_paths(second,dates[378:400],config)
        for name in a:np.testing.assert_array_equal(a[name],b[name]);np.linalg.cholesky(a[name])
        self.assertTrue((first.folds.last_calibration_label<first.folds.first_forecast).all())
        self.assertTrue((audit.last_validation_label<audit.first_forecast).all())


if __name__=='__main__':unittest.main()
