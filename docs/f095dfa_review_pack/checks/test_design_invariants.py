"""Design-level tests only; not a reproduction of iclr2027 empirical findings."""
from __future__ import annotations
import itertools
import unittest
import numpy as np
from threshold_certificate import threshold_certificate


def group8():
    out=[]
    for reverse_a,reverse_b,swap in itertools.product([False,True], repeat=3):
        a=[1,0] if reverse_a else [0,1]
        b=[3,2] if reverse_b else [2,3]
        out.append(tuple(b+a if swap else a+b))
    return tuple(out)


def crosses(x):
    a,b,c,d = np.asarray(x,dtype=float)
    def orient(a,b,c):
        u,v=b-a,c-a
        return u[0]*v[1]-u[1]*v[0]
    return orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0


class DesignTests(unittest.TestCase):
    def test_three_way_identity(self):
        s=np.array([[0,2],[3,4],[2,1],[1,2]],float)
        y=np.tile([0,1],(4,1))
        c=threshold_certificate(s,y,0)
        self.assertAlmostEqual(c.S_local_separable,.75)
        self.assertAlmostEqual(c.J_star_global_oracle,.5)
        self.assertAlmostEqual(c.J_at_threshold,.25)
        self.assertAlmostEqual(c.ordering_failure+c.global_incompatibility+c.operating_point_gap,.75)

    def test_strict_monotone_invariance(self):
        s=np.array([[-2,1],[2,4],[3,2],[0,0]],float)
        y=np.tile([0,1],(4,1))
        a=threshold_certificate(s,y,.5)
        b=threshold_certificate(np.exp(s),y,np.exp(.5))
        for key in ['S_local_separable','J_star_global_oracle','J_at_threshold']:
            self.assertAlmostEqual(getattr(a,key),getattr(b,key))

    def test_ties_and_half_open(self):
        y=np.tile([0,1],(2,1))
        c=threshold_certificate([[0,1],[1,2]],y,1)
        self.assertAlmostEqual(c.J_star_global_oracle,.5)
        self.assertAlmostEqual(c.J_at_threshold,.5)
        self.assertEqual(threshold_certificate([[1,1]],[[0,1]]).S_local_separable,0)

    def test_bruteforce_matches_sweep(self):
        r=np.random.default_rng(13)
        s=r.normal(size=(200,3));y=r.integers(0,2,size=s.shape)
        y[:,0]=0;y[:,-1]=1
        w=r.uniform(size=200);w=w/w.sum()
        c=threshold_certificate(s,y,0,w)
        brute=max(np.sum(w*np.all((s>t)==y,axis=1)) for t in np.unique(s))
        self.assertAlmostEqual(c.J_star_global_oracle,brute)

    def test_input_validation(self):
        with self.assertRaises(ValueError): threshold_certificate([[1,2]],[[1,1]])
        with self.assertRaises(ValueError): threshold_certificate([[np.nan,2]],[[0,1]])
        with self.assertRaises(ValueError): threshold_certificate([[1,2]],[[0,1]],weights=[-1])

    def test_group8_closed(self):
        G=group8(); self.assertEqual(len(set(G)),8)
        for p in G:
            self.assertIn(tuple(np.argsort(p)),G)
            for q in G: self.assertIn(tuple(p[i] for i in q),G)

    def test_group8_oracle(self):
        r=np.random.default_rng(32)
        for _ in range(100):
            x=r.normal(size=(4,2));v=crosses(x)
            for g in group8(): self.assertEqual(v,crosses(x[list(g)]))

    def test_untyped_set_is_insufficient(self):
        x=np.array([[-1,-1],[1,1],[-1,1],[1,-1]],float)
        z=x[[0,2,1,3]]
        self.assertEqual(set(map(tuple,x)),set(map(tuple,z)))
        self.assertTrue(crosses(x)); self.assertFalse(crosses(z))
        self.assertNotIn((0,2,1,3),group8())

    def test_group_average_invariant(self):
        x=np.random.default_rng(2).normal(size=(4,2))
        def f(a): return np.sin(a[0,0])+a[2,1]**2-2*a[3,0]
        def avg(a): return np.mean([f(a[list(g)]) for g in group8()])
        for g in group8(): self.assertAlmostEqual(avg(x),avg(x[list(g)]))

if __name__ == '__main__':
    unittest.main(verbosity=2)
