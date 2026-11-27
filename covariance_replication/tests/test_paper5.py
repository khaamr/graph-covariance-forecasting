"""Causal and independent checks for the manuscript's optional appendix modes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from market_data import PAPER_SYMBOLS, RealizedPanel
from models import (correlation_graph_features, covariance_components, forecast_grid,
                    graph_augmented_features, har_lags, inverse_variance, panel_ols,
                    rebuild_drd, transform_drd)


class Paper5Checks(unittest.TestCase):
    def panel(self):
        rng = np.random.default_rng(83)
        dates = pd.bdate_range('2020-01-02', periods=100)
        variance = rng.lognormal(size=(100,3))
        corr = np.full((3,3), .2) + np.eye(3)*.8
        covariance = corr[None] * np.sqrt(variance)[:,:,None] * np.sqrt(variance)[:,None,:]
        return RealizedPanel(dates,PAPER_SYMBOLS[:3],covariance,np.ones((100,3)),
                             rng.normal(size=(100,3)),np.zeros((100,3)),np.ones(100),pd.Series(dtype=int))

    def test_transform_round_trip_and_endpoint_guard(self):
        variance=np.array([[.2,2.,20.]])
        corr=np.array([[-.9,0.,.9]])
        v,r=transform_drd(variance,corr,'log_fisher')
        np.testing.assert_allclose(inverse_variance(v,'log_fisher'),variance)
        np.testing.assert_allclose(np.tanh(r),corr)
        _, endpoint=transform_drd(variance,np.array([[-1.,0.,1.]]),'log_fisher')
        self.assertTrue(np.isfinite(endpoint).all())

    def test_short_term_graph_preserves_own_lags(self):
        x=np.arange(90.).reshape(10,3,3)
        weight=(np.ones((3,3))-np.eye(3))/2
        actual=graph_augmented_features(x,weight,graph_effect='daily')
        self.assertEqual(actual.shape[-1],4)
        np.testing.assert_array_equal(actual[...,:3],x)
        np.testing.assert_allclose(actual[...,3:],(weight@x)[...,:1])
        pairs=np.tril_indices(3,-1)
        all_graphs=correlation_graph_features(x,pairs,3)
        short=correlation_graph_features(x,pairs,3,graph_effect='daily')
        for k in ['K','L']:
            np.testing.assert_allclose(short[k],all_graphs[k][...,[0,1,2,3]])

    def test_transformed_baseline_matches_independent_ols_reconstruction(self):
        panel=self.panel()
        result=forecast_grid(panel,horizons=(1,),lookback=40,transformation='log_fisher')[1]
        origin=result.positions[0]
        future=result.positions[result.dates.to_period('M')==result.dates[0].to_period('M')]
        train=np.arange(origin-40,origin)
        v,r,_=covariance_components(panel.covariance)
        vp=panel_ols(har_lags(np.log(v)),np.log(v),train,future)
        rp=panel_ols(har_lags(np.arctanh(r)),np.arctanh(r),train,future)
        expected=rebuild_drd(np.exp(vp),np.tanh(rp))[0]
        np.testing.assert_allclose(result.predictions['HAR-DRD'][:len(future)],expected,rtol=1e-12,atol=1e-12)
        for matrices in result.predictions.values():
            np.linalg.cholesky(matrices)

    def test_transforms_and_graph_lag_change_leave_cholesky_control_unchanged(self):
        panel=self.panel()
        raw=forecast_grid(panel,horizons=(1,),lookback=40)[1]
        for options in [{'transformation':'log_fisher'},{'graph_effect':'daily'}]:
            result=forecast_grid(panel,horizons=(1,),lookback=40,**options)[1]
            np.testing.assert_array_equal(result.positions,raw.positions)
            np.testing.assert_allclose(result.predictions['HAR-Cholesky'],raw.predictions['HAR-Cholesky'],atol=1e-12)
            if options.get('graph_effect')=='daily':
                np.testing.assert_allclose(result.predictions['HAR-DRD'],raw.predictions['HAR-DRD'],atol=1e-12)
                self.assertIn('GHAR-S(GL,L)',result.predictions)

    def test_refit_intervals_are_past_only_for_multiple_horizons(self):
        for interval in [1,5]:
            results=forecast_grid(self.panel(),horizons=(1,5),lookback=40,refit_every=interval)
            for result in results.values():
                self.assertTrue(result.audit.test_origins.le(interval).all())
                self.assertTrue((result.audit.last_train_target<result.audit.first_forecast).all())
                self.assertTrue((result.audit.graph_return_through<result.audit.first_forecast).all())

    def test_future_perturbation_cannot_change_earlier_daily_refits(self):
        panel=self.panel()
        original=forecast_grid(panel,horizons=(1,),lookback=40,refit_every=1,transformation='log_fisher')[1]
        covariance=panel.covariance.copy();covariance[85:]*=9
        returns=panel.graph_return.copy();returns[85:]+=100
        changed=RealizedPanel(panel.dates,panel.symbols,covariance,panel.quarticity,returns,panel.portfolio_return,panel.spy_variance,panel.session_counts)
        result=forecast_grid(changed,horizons=(1,),lookback=40,refit_every=1,transformation='log_fisher')[1]
        early=original.positions<85
        for model in original.predictions:
            np.testing.assert_allclose(result.predictions[model][early],original.predictions[model][early],atol=1e-12)

    def test_undefined_combinations_and_bad_options_fail(self):
        for options in [{'refit_every':0},{'refit_every':True},{'graph_effect':'bad'},
                        {'transformation':'bad'},{'transformation':'log_fisher','quarticity':True}]:
            with self.assertRaises(ValueError):
                forecast_grid(self.panel(),horizons=(1,),lookback=40,**options)


if __name__=='__main__':
    unittest.main()
