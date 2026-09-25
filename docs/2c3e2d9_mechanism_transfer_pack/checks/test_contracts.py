import unittest
import numpy as np
import torch
import torch.nn.functional as F
from contracts import (t1_collision_pair,t1_legacy_rich,t2_legacy_rich,
                       t2_role_distances,point_in_triangle,crossing,
                       orbit_distance_representative,T1_PERMS,T2_PERMS,
                       SOURCE_PERMS,joint_metrics)

class AuditContracts(unittest.TestCase):
    def test_t1_witness_has_opposite_labels(self):
        a,b=t1_collision_pair()
        self.assertTrue(point_in_triangle(a));self.assertFalse(point_in_triangle(b))
    def test_t1_legacy_sorted_aliases_labels(self):
        a,b=t1_collision_pair()
        np.testing.assert_allclose(t1_legacy_rich(a),t1_legacy_rich(b),atol=1e-12,rtol=0)
    def test_t1_unsorted_preserves_this_distinction(self):
        a,b=t1_collision_pair()
        self.assertGreater(np.max(np.abs(t1_legacy_rich(a,False)-t1_legacy_rich(b,False))),.1)
    def test_t1_whole_orbit_separates_witness(self):
        a,b=t1_collision_pair()
        self.assertGreater(np.max(np.abs(orbit_distance_representative(a,T1_PERMS)-
                                        orbit_distance_representative(b,T1_PERMS))),.1)
    def test_t1_whole_orbit_is_group_invariant(self):
        a,_=t1_collision_pair(); ref=orbit_distance_representative(a,T1_PERMS)
        for g in T1_PERMS:
            np.testing.assert_allclose(orbit_distance_representative(a[list(g)],T1_PERMS),ref,atol=1e-12)
    def test_source_all_point_sum_loses_pair_roles(self):
        x=torch.tensor([[-.6,-.6],[.6,.6],[-.6,.6],[.6,-.6]],dtype=torch.float64)
        xp=x[[0,2,1,3]]
        self.assertTrue(crossing(x.numpy()));self.assertFalse(crossing(xp.numpy()))
        torch.manual_seed(11)
        phi=torch.nn.Sequential(torch.nn.Linear(2,48),torch.nn.ReLU(),torch.nn.Linear(48,48),torch.nn.ReLU()).double()
        rho=torch.nn.Sequential(torch.nn.Linear(48,16),torch.nn.GELU(),torch.nn.Linear(16,1)).double()
        def legacy(p):
            h=phi(p);return rho(h[:2].sum(0)+h[2:].sum(0))
        torch.testing.assert_close(legacy(x),legacy(xp),atol=1e-12,rtol=0)
    def test_source_legal_orbit_does_not_merge_regrouping(self):
        x=np.array([[-.6,-.6],[.6,.6],[-.6,.6],[.6,-.6]])
        self.assertGreater(np.max(np.abs(orbit_distance_representative(x,SOURCE_PERMS)-
                   orbit_distance_representative(x[[0,2,1,3]],SOURCE_PERMS))),.1)
    def test_t2_wrong_axis_breaks_rotation(self):
        x=np.array([[-.6,.1],[.5,.2],[.1,.7]])
        rotation=np.array([[0.,-1.],[1.,0.]])
        xp=x@rotation.T
        self.assertGreater(np.max(np.abs(t2_legacy_rich(x,True)-t2_legacy_rich(xp,True))),1e-3)
    def test_t2_reference_is_role_and_rotation_invariant(self):
        x=np.array([[-.6,.1],[.5,.2],[.1,.7]])
        rotation=np.array([[0.,-1.],[1.,0.]])
        for xp in (x[[1,0,2]],x@rotation.T):
            np.testing.assert_allclose(t2_role_distances(x),t2_role_distances(xp),atol=1e-12)
    def test_nearest_pool_can_erase_a_visible_segment(self):
        m=torch.zeros(1,1,64,64);m[:,:,4:9,8:57]=1
        self.assertEqual(m.sum().item(),245.)
        self.assertEqual(F.interpolate(m,size=(7,7),mode='nearest').sum().item(),0.)
        self.assertGreater(F.interpolate(m,size=(7,7),mode='area').sum().item(),0.)
    def test_j4_must_include_start(self):
        scores=np.array([[-1.,-1.,-1.,1.]])
        labels=np.array([[1.,0.,0.,1.]])
        metrics=joint_metrics(scores,labels)
        self.assertEqual(metrics['J3'],1.);self.assertEqual(metrics['J4'],0.)
    def test_correct_metrics_handle_all_correct(self):
        self.assertEqual(joint_metrics(np.array([[1.,-1.,-1.,1.]]),np.array([[1.,0.,0.,1.]])),{'J3':1.,'J4':1.})

if __name__=='__main__':unittest.main(verbosity=2)
