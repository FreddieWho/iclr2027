import itertools
import unittest
import numpy as np
import torch
from interaction_witness import (checkerboard,crossing,group8,AdditivePair,
                                 invariant_witness_score,features8)

class WitnessTests(unittest.TestCase):
    def test_exact_checkerboard(self):
        x,y=checkerboard()
        np.testing.assert_array_equal(y,[[1,0],[0,1]])
    def test_g8_preserves_oracle(self):
        x,y=checkerboard()
        for p in group8():
            np.testing.assert_array_equal([crossing(s[list(p)]) for s in x],y.ravel())
    def test_additive_identity(self):
        x,_=checkerboard()
        for seed in range(10):
            torch.manual_seed(seed)
            with torch.no_grad():
                s=AdditivePair().double()(torch.tensor(x)).numpy().reshape(2,2)
            self.assertAlmostEqual(float(s[0,0]+s[1,1]-s[0,1]-s[1,0]),0.,places=12)
    def test_additive_grid_cannot_fit_checkerboard(self):
        # Exhaustive finite grid is a guard; the README gives the general proof.
        y=np.array([[1,0],[0,1]])
        for a,b,c,d,t in itertools.product((-2.,0.,2.),repeat=5):
            z=np.array([[a+c,a+d],[b+c,b+d]])
            self.assertFalse(np.array_equal(z>t,y))
    def test_nonlinear_witness_solves_and_is_invariant(self):
        x,y=checkerboard()
        s=invariant_witness_score(x)
        np.testing.assert_array_equal(s>0,y.ravel())
        for p in group8():
            np.testing.assert_allclose(invariant_witness_score(x[:,list(p)]),s,atol=1e-12)
    def test_all_point_permutations_not_allowed(self):
        x=np.array([[-.6,0],[.6,0],[0,-.6],[0,.6]])
        self.assertEqual(crossing(x),1)
        self.assertEqual(crossing(x[[0,2,1,3]]),0)
    def test_features_eight_and_invariant(self):
        x,_=checkerboard(); f=features8(x)
        self.assertEqual(f.shape,(4,8))
        for p in group8():
            np.testing.assert_allclose(features8(x[:,list(p)]),f,atol=1e-12)
    def test_same_target_label_does_not_define_same_features(self):
        # A sanity guard against accidentally using the oracle label as features.
        rng=np.random.default_rng(42)
        x=rng.uniform(-.8,.8,size=(20,4,2))
        f=features8(x)
        self.assertGreater(np.unique(np.round(f,8),axis=0).shape[0],2)

if __name__=='__main__':
    unittest.main(verbosity=2)
