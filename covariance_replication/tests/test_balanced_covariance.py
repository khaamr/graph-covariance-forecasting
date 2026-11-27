import unittest
import numpy as np
import pandas as pd

from balanced_covariance import balanced_covariance_forecast
from covariance_reconciliation import reconcile
from risk_calibration import portfolio_weights
from stress_covariance_reconciliation import ewma_covariance_weights,stress_strength


class BalancedCovarianceTests(unittest.TestCase):
    def test_direct_api_matches_batch_construction(self):
        rng=np.random.default_rng(408);raw=rng.normal(size=(80,5,5));history=raw@raw.swapaxes(-1,-2)+np.eye(5)
        history[-1]*=3
        dates=pd.bdate_range('2022-01-01',periods=81);first=np.diag([1.,2.,3.,4.,5.]);second=np.diag([1.2,1.8,2.7,4.5,4.])
        result=balanced_covariance_forecast(first,second,history,dates[:80],dates[80])
        base=(first+second)/2;original,_=portfolio_weights(base[None]);extended=np.concatenate([history,np.eye(5)[None]])
        driver,_=ewma_covariance_weights(extended,np.array([80]),5)
        beta=stress_strength(extended,np.array([80]),'Volatility ratio1to3')
        expected=reconcile(base[None],(1-beta[:,None])*original+beta[:,None]*driver,'Gaussian KL')[0]
        np.testing.assert_allclose(result,expected,rtol=1e-13,atol=1e-13);np.linalg.cholesky(result)

    def test_api_rejects_same_day_or_future_observations(self):
        dates=pd.bdate_range('2022-01-01',periods=30);history=np.broadcast_to(np.eye(3),(30,3,3)).copy()
        with self.assertRaisesRegex(ValueError,'strictly before'):
            balanced_covariance_forecast(np.eye(3),np.eye(3),history,dates,dates[-1])


if __name__=='__main__':unittest.main()
