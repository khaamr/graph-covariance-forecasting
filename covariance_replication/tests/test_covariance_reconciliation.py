import unittest
import numpy as np

from covariance_reconciliation import reconcile,smoothed_realized_weights
from evaluation import long_only_weights
from risk_calibration import portfolio_weights


class ReconciliationTests(unittest.TestCase):
    def test_requested_weights_are_actual_long_only_optima(self):
        rng=np.random.default_rng(403);raw=rng.normal(size=(8,6,6));cov=raw@raw.swapaxes(-1,-2)+np.eye(6)
        weights=rng.random((8,6));weights[weights<.5]=0;weights/=weights.sum(axis=1,keepdims=True)
        for distance in ['Gaussian KL','Reverse Gaussian KL','Frobenius']:
            adjusted=reconcile(cov,weights,distance)
            exact=long_only_weights(adjusted)
            np.testing.assert_allclose(exact,weights,atol=3e-6,rtol=1e-5)
            np.linalg.cholesky(adjusted)

    def test_original_optimum_leaves_covariance_unchanged(self):
        rng=np.random.default_rng(404);raw=rng.normal(size=(6,5,5));cov=raw@raw.swapaxes(-1,-2)+np.eye(5)
        weights,_=portfolio_weights(cov)
        for distance in ['Gaussian KL','Reverse Gaussian KL','Frobenius']:
            np.testing.assert_allclose(reconcile(cov,weights,distance),cov,atol=1e-9,rtol=1e-9)

    def test_scalar_distance_optimum_matches_neighbor_comparisons(self):
        rng=np.random.default_rng(405);raw=rng.normal(size=(5,5));cov=raw@raw.T+np.eye(5)
        w=rng.random(5);w/=w.sum();h=np.ones(5)
        def objective(s,distance):
            if distance=='Gaussian KL':return np.trace(np.linalg.solve(cov,s))-np.linalg.slogdet(s)[1]+np.linalg.slogdet(cov)[1]-5
            if distance=='Reverse Gaussian KL':return np.trace(np.linalg.solve(s,cov))+np.linalg.slogdet(s)[1]-np.linalg.slogdet(cov)[1]-5
            return np.sum((s-cov)**2)
        for distance in ['Gaussian KL','Reverse Gaussian KL','Frobenius']:
            s=reconcile(cov[None],w[None],distance)[0];c=(s@w)[0]
            for shift in [-.1,.1]:
                other=s+shift*c*np.outer(h,h)
                self.assertLess(objective(s,distance),objective(other,distance))

    def test_scale_equivariance(self):
        covariance=np.array([[[2.,.3],[.3,1.]]]);weights=np.array([[.3,.7]])
        for distance in ['Gaussian KL','Reverse Gaussian KL','Frobenius']:
            np.testing.assert_allclose(reconcile(4*covariance,weights,distance),4*reconcile(covariance,weights,distance),atol=1e-13)

    def test_smoothed_driver_is_strictly_past(self):
        rng=np.random.default_rng(406);raw=rng.normal(size=(50,4,4));cov=raw@raw.swapaxes(-1,-2)+np.eye(4)
        first,_=smoothed_realized_weights(cov,20);changed=cov.copy()
        scale=np.array([10.,1.,1.,1.]);changed[30:]*=scale[None,:,None]*scale[None,None,:]
        second,_=smoothed_realized_weights(changed,20)
        np.testing.assert_array_equal(first[:31],second[:31])
        self.assertFalse(np.allclose(first[32:],second[32:]))
        np.testing.assert_allclose(first.sum(axis=1),1)


if __name__=='__main__':unittest.main()
