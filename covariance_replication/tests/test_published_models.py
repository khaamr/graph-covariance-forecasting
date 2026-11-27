import unittest

import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

from dcc_nl import (DCCFit, advance_dcc, composite_objective, correlation, dcc_backcast,
                    dcc_forecasts, filter_q, nonlinear_covariance)
from factor_vhar import (causal_clean, equation_forecasts, factor_decomposition,
                         factor_forecasts, overlap_lags)


class DCCNLTests(unittest.TestCase):
    def test_analytical_shrinkage_rotation_scale_and_high_dimension(self):
        rng = np.random.default_rng(49)
        x = rng.normal(size=(180, 8)) * np.arange(1, 9)
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        estimate = nonlinear_covariance(x)
        np.testing.assert_allclose(nonlinear_covariance(x @ q), q.T @ estimate @ q, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(nonlinear_covariance(3*x), 9*estimate, rtol=1e-10, atol=1e-10)
        high = nonlinear_covariance(rng.normal(size=(20, 35)))
        self.assertGreater(np.linalg.eigvalsh(high).min(), 0)

    def test_nonlinear_shrinkage_matches_scalar_kernel_formula(self):
        rng = np.random.default_rng(34)
        x = rng.normal(size=(120, 4))
        values, vectors = np.linalg.eigh(x.T @ x / len(x))
        n, p = x.shape; h = n**(-1/3); root = np.sqrt(5.)
        oracle = []
        for value in values:
            fs, hs = [], []
            for other in values:
                ratio = (value - other)/(h*other)
                fs.append(3/(4*root)*max(1-ratio*ratio/5, 0)/(h*other))
                kernel = -3*ratio/(10*np.pi) + 3/(4*root*np.pi)*(1-ratio*ratio/5)*np.log(abs((root-ratio)/(root+ratio)))
                hs.append(kernel/(h*other))
            oracle.append(value/((np.pi*p/n*value*np.mean(fs))**2+(1-p/n-np.pi*p/n*value*np.mean(hs))**2))
        np.testing.assert_allclose(nonlinear_covariance(x), (vectors*np.array(oracle))@vectors.T, atol=1e-12)

    def test_composite_likelihood_and_gradient_match_full_matrix_oracle(self):
        rng = np.random.default_rng(16)
        z = rng.normal(size=(90, 5))
        target = correlation(nonlinear_covariance(z)); back = dcc_backcast(z)
        parameter = np.array([.06, .89])
        value, gradient = composite_objective(parameter, z, target, back)
        numerical = approx_derivative(lambda p: composite_objective(p, z, target, back)[0], parameter).ravel()
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-7)
        q = (1-parameter.sum())*target + parameter.sum()*back
        total = 0.
        for t in range(len(z)):
            if t: q=(1-parameter.sum())*target+parameter[0]*np.outer(z[t-1], z[t-1])+parameter[1]*q
            r = correlation(q)
            for i in range(z.shape[1]-1):
                pair = r[i:i+2, i:i+2]; current=z[t,i:i+2]
                total += .5*(2*np.log(2*np.pi)+np.linalg.slogdet(pair)[1]+current@np.linalg.solve(pair,current))
        self.assertAlmostEqual(value, total/(len(z)*(z.shape[1]-1)), places=12)
        np.testing.assert_allclose(filter_q(z,target,parameter,back), q, atol=1e-14)

    def test_forecast_is_next_day_and_does_not_consume_current_return(self):
        fit = DCCFit(np.zeros(2),np.array([[.1,.1,.8],[.1,.1,.8]]),np.array([2.,2.]),
                     np.array([1.,2.]),np.eye(2),np.array([.05,.9]),np.eye(2),{})
        first = advance_dcc(fit, np.array([[100.,-100.],[2.,3.]]))
        other = advance_dcc(fit, np.zeros((2,2)))
        np.testing.assert_array_equal(first[0],other[0])
        np.testing.assert_allclose(np.diag(first[0]),[1.8,2.1])
        self.assertFalse(np.allclose(first[1],other[1]))

    def test_fitted_dcc_causality_and_validity(self):
        rng = np.random.default_rng(310)
        returns = rng.standard_t(9,size=(180,3))
        dates = pd.bdate_range('2020-01-01',periods=180)
        positions = np.arange(140,145)
        first,audit,_ = dcc_forecasts(dates,returns,positions,lookback=120)
        changed=returns.copy();changed[143:]*=10
        second,_,_=dcc_forecasts(dates,changed,positions,lookback=120)
        np.testing.assert_array_equal(first[:4],second[:4])
        np.linalg.cholesky(first)
        self.assertTrue((audit.last_training_return<audit.first_forecast).all())


class FactorVHARTests(unittest.TestCase):
    def test_decomposition_matches_intraday_ols_and_reconstruction(self):
        rng=np.random.default_rng(21)
        returns=rng.normal(size=(100,6)); covariance=returns.T@returns/100
        weights=rng.normal(size=(6,3))
        factor,beta,residual=factor_decomposition(covariance[None],weights)
        expected=np.linalg.lstsq(returns@weights,returns,rcond=None)[0]
        np.testing.assert_allclose(beta[0],expected,atol=1e-12)
        np.testing.assert_allclose(beta[0].T@factor[0]@beta[0]+residual[0],covariance,atol=1e-12)
        np.testing.assert_allclose(weights.T@residual[0],0,atol=1e-12)

    def test_har_uses_overlapping_paper_lags(self):
        values=np.arange(30.)[:,None]
        features=overlap_lags(values)
        np.testing.assert_allclose(features[25,0],[24,22,13.5])

    def test_lasso_and_adaptive_recover_sparse_equation(self):
        rng=np.random.default_rng(24)
        x=rng.normal(size=(350,4)); y=2+1.4*x[:,0]-.8*x[:,2]+rng.normal(0,.03,350)
        a,b,first,second=equation_forecasts(x,y,np.arange(300),np.arange(300,350))
        self.assertLess(np.mean((a-y[300:])**2),.003)
        self.assertLess(np.mean((b-y[300:])**2),.003)
        self.assertLess(first['kkt_error'],1e-6)
        self.assertLessEqual(second['active'],first['active'])

    def test_cleaning_is_causal_and_uses_ten_prior_nonflagged_days(self):
        covariance=np.broadcast_to(np.eye(3),(50,3,3)).copy();covariance[30]*=100
        clean,flag=causal_clean(covariance)
        self.assertTrue(flag[30]);np.testing.assert_array_equal(clean[30],np.eye(3))
        changed=covariance.copy();changed[35:]*=1000
        other,other_flag=causal_clean(changed)
        np.testing.assert_array_equal(clean[:35],other[:35]);np.testing.assert_array_equal(flag[:35],other_flag[:35])

    def test_historical_weights_require_prior_availability(self):
        dates=pd.bdate_range('2020-01-01',periods=30)
        cov=np.broadcast_to(np.eye(4),(30,4,4)).copy()
        weights=np.broadcast_to(np.ones((4,1))/4,(30,4,1)).copy()
        with self.assertRaisesRegex(ValueError,'known before'):
            factor_decomposition(cov,weights,dates,dates)
        factor_decomposition(cov,weights,dates,dates-pd.Timedelta(days=1))

    def test_complete_factor_pipeline_multifactor_causality(self):
        rng=np.random.default_rng(154)
        dates=pd.bdate_range('2020-01-01',periods=340)
        raw=rng.normal(size=(340,6,6));cov=raw@raw.swapaxes(-1,-2)+np.eye(6)
        positions=np.array([310,311,312]);weights=rng.normal(size=(6,2))
        sectors=['a','a','b','b','c','c']
        first=factor_forecasts(dates,cov,positions,weights,sectors,dates,lookback=250)
        changed=cov.copy();changed[312:]*=20
        second=factor_forecasts(dates,changed,positions,weights,sectors,dates,lookback=250)
        for name in first.predictions:
            np.testing.assert_array_equal(first.predictions[name],second.predictions[name])
            np.linalg.cholesky(first.predictions[name])
        self.assertTrue((first.audit.last_training_label<first.audit.date).all())

    def test_factor_counts_one_three_five_seven(self):
        rng=np.random.default_rng(155)
        dates=pd.bdate_range('2020-01-01',periods=290)
        raw=rng.normal(size=(290,27,27));cov=raw@raw.swapaxes(-1,-2)+np.eye(27)
        weights,_=np.linalg.qr(rng.normal(size=(27,7)))
        for k in (1,3,5,7):
            result=factor_forecasts(dates,cov,np.array([280]),weights[:,:k],
                                   [str(i//3) for i in range(27)],dates,lookback=250)
            self.assertEqual(result.audit.factor_count.iloc[0],k)
            for name,value in result.predictions.items():
                self.assertNotIn('market',name)
                self.assertEqual(value.shape,(1,27,27));np.linalg.cholesky(value)


if __name__=='__main__':
    unittest.main()
