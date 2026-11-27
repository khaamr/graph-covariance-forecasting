"""Analytic, solver, fallback, and temporal checks for joint covariance pooling."""
from pathlib import Path
import sys,json,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from evaluation import long_only_weights
from joint_pool import (MODEL,fit_pool,loss_statistics,scaled_losses,scale_from_statistics,
                        select_candidate,shrink_diagonal,simplex_grid,simplex_weights)


class JointPoolChecks(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1]/'docs/joint-pool/design.json').read_text())

    def test_simplex_grid_and_shrinkage_preserve_positive_covariance(self):
        grid=simplex_grid(5)
        self.assertEqual(len(grid),36)
        np.testing.assert_allclose(grid.sum(axis=1),1)
        self.assertTrue((grid>=0).all())
        a=np.array([[[2.,.5],[.5,1.]]])
        cleaned=shrink_diagonal(a,.1)
        np.testing.assert_allclose(cleaned[:,range(2),range(2)],a[:,range(2),range(2)])
        np.linalg.cholesky(cleaned)

    def test_portfolio_solver_matches_exact_objectives(self):
        rng=np.random.default_rng(94);a=rng.normal(size=(5,4,4));cov=a@a.swapaxes(-1,-2)+np.eye(4)
        w,audit=simplex_weights(cov)
        exact=long_only_weights(cov)
        q=lambda z:np.einsum('ti,tij,tj->t',z,cov,z)
        self.assertLess(float(np.max((q(w)-q(exact))/q(exact))),1e-4)
        self.assertLessEqual(audit['max_relative_objective_gap_bound'],5e-4)
        self.assertTrue((w>=0).all());np.testing.assert_allclose(w.sum(axis=1),1)
        scaled,_=simplex_weights(3*cov)
        np.testing.assert_allclose(w,scaled,atol=1e-12)

    def test_scaled_loss_matches_direct_matrix_definitions(self):
        p=np.array([[[2.,.4],[.4,1.]]]);h=np.array([[[3.,.2],[.2,2.]]]);c=1.3
        value=scaled_losses(loss_statistics(p,h),c)[0]
        delta=c*p[0]-h[0]
        expected=[np.linalg.norm(delta[np.tril_indices(2)]),np.linalg.norm(delta),
                  np.linalg.slogdet(c*p[0])[1]-np.linalg.slogdet(h[0])[1]+np.trace(np.linalg.solve(c*p[0],h[0]))-2]
        np.testing.assert_allclose(value,expected,atol=1e-12)

    def test_scalar_calibration_recovers_known_scale(self):
        p=np.repeat(np.diag([1.,2.,4.])[None],12,axis=0);h=1.3*p
        stats=loss_statistics(p,h)
        for recipe in ['Euclidean','Frobenius','QLIKE divergence']:
            self.assertAlmostEqual(scale_from_statistics(stats,recipe),1.3,places=4)

    def test_perfect_baseline_remains_a_valid_fallback(self):
        n=400;h=np.repeat(np.diag([1.,2.,4.])[None],n,axis=0)
        bank=np.stack([h,10*h,10*h,10*h,10*h],axis=1)
        returns=np.random.default_rng(42).normal(scale=.01,size=(n,3))
        best,_,_,_=select_candidate(bank,h,returns,np.arange(252),np.arange(252,378),self.config())
        np.testing.assert_allclose(best[1],[1,0,0,0,0])
        self.assertEqual(best[2],0.)
        self.assertEqual(best[3],'identity')
        self.assertLessEqual(best[4]['worst_ratio'],1+1e-8)

    def test_future_outcomes_cannot_change_first_quarter_choices(self):
        rng=np.random.default_rng(95);n=450;dates=pd.bdate_range('2022-07-01',periods=n)
        a=rng.normal(scale=.15,size=(n,3,3));actual=a@a.swapaxes(-1,-2)+np.eye(3)
        bank=np.stack([actual*.9,actual*1.1,actual*.8,actual*1.2,actual],axis=1)
        returns=rng.normal(scale=.01,size=(n,3));config=self.config()
        first=fit_pool(dates,np.arange(n),bank,actual,returns,config)
        cutoff=dates.get_loc(first.forecasts.dates[0]);changed_actual=actual.copy();changed_actual[cutoff:]*=5
        changed_returns=returns.copy();changed_returns[cutoff:]*=7
        second=fit_pool(dates,np.arange(n),bank,changed_actual,changed_returns,config)
        first_quarter=first.forecasts.dates.to_period('Q')==first.forecasts.dates[0].to_period('Q')
        np.testing.assert_allclose(first.forecasts.predictions[MODEL][first_quarter],second.forecasts.predictions[MODEL][first_quarter],atol=1e-12)
        self.assertTrue((first.audit.last_refit_label<first.audit.first_forecast).all())
        np.linalg.cholesky(first.forecasts.predictions[MODEL])


if __name__=='__main__':unittest.main()
