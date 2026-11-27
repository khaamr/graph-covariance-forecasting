import unittest
import numpy as np

from stress_covariance_reconciliation import stress_strength,ewma_covariance_weights


class StressReconciliationTests(unittest.TestCase):
    def test_gate_uses_only_lagged_covariances(self):
        covariance=np.broadcast_to(np.eye(3),(80,3,3)).copy();covariance[29]*=5
        positions=np.arange(22,60);first=stress_strength(covariance,positions,'Volatility ratio1to3')
        self.assertEqual(first[positions==29][0],0)
        self.assertGreater(first[positions==30][0],0)
        changed=covariance.copy();changed[40:]*=100
        second=stress_strength(changed,positions,'Volatility ratio1to3')
        np.testing.assert_array_equal(first[positions<=40],second[positions<=40])
        self.assertTrue(((first>=0)&(first<=1)).all())

    def test_fast_covariance_weights_are_strictly_past(self):
        rng=np.random.default_rng(407);raw=rng.normal(size=(80,4,4));covariance=raw@raw.swapaxes(-1,-2)+np.eye(4)
        positions=np.arange(22,60);first,_=ewma_covariance_weights(covariance,positions,2)
        changed=covariance.copy();scale=np.array([10.,1.,1.,1.]);changed[40:]*=scale[None,:,None]*scale[None,None,:]
        second,_=ewma_covariance_weights(changed,positions,2)
        np.testing.assert_array_equal(first[positions<=40],second[positions<=40])
        self.assertFalse(np.allclose(first[positions>40],second[positions>40]))


if __name__=='__main__':unittest.main()
