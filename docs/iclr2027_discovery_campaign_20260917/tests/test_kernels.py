from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
from core.perturbations import *
from core.metric_edit import fit_metric_initializer, squared_distance
from core.relations import segment_relation, make_relational_scenes


class KernelTests(unittest.TestCase):
    def test_bank_balance_and_uniqueness(self):
        s = balanced_sign_patterns([4,6],64,7)
        self.assertTrue(np.all(s[:,:4].sum(axis=1)==0))
        self.assertTrue(np.all(s[:,4:].sum(axis=1)==0))
        self.assertTrue(np.all(s[:,0]==1))
        self.assertEqual(len(s),len(set(map(tuple,s))))

    def test_small_bank_complete(self):
        self.assertEqual(balanced_sign_patterns([4],100).shape,(3,4))

    def test_seed_reproducibility(self):
        np.testing.assert_array_equal(balanced_sign_patterns([10,10],64,5),balanced_sign_patterns([10,10],64,5))

    def test_invalid_bank_input(self):
        for g in [[],[3],[0],[2.5]]:
            with self.assertRaises(ValueError): balanced_sign_patterns(g)

    def test_equal_marginals_different_weights(self):
        s = balanced_sign_patterns([8],12)
        one = weighted_sign_law(s,np.ones(len(s)))
        other = weighted_sign_law(s,np.arange(1,len(s)+1)**2)
        np.testing.assert_allclose(one['plus_probability'],.5)
        np.testing.assert_allclose(other['plus_probability'],.5)
        np.testing.assert_allclose(other['mean'],0)
        np.testing.assert_allclose(np.diag(other['covariance']),1)
        self.assertGreater(np.linalg.norm(one['covariance']-other['covariance']),0)

    def test_energy_and_antithetic(self):
        s=balanced_sign_patterns([4,4],12)
        f=antithetic_fields(s,.3,np.array([2.,1.]))
        np.testing.assert_allclose(f[:,0],-f[:,1])
        np.testing.assert_allclose(np.sum(f*f,axis=(-1,-2)),.09)
        np.testing.assert_allclose(f[:,:,:4].sum(axis=2),0,atol=1e-15)

    def test_zero_energy(self):
        f=antithetic_fields(np.array([[1,-1]]),0,np.array([1,0]))
        self.assertEqual(float(abs(f).max()),0)

    def test_bad_direction_and_pattern(self):
        with self.assertRaises(ValueError): antithetic_fields(np.array([[1,-1]]),1,np.zeros(2))
        with self.assertRaises(ValueError): validate_patterns(np.array([[1,0]]))
        with self.assertRaises(ValueError): normalize_weights(np.zeros(2),2)

    def test_whole_pair_filtering(self):
        f=antithetic_fields(balanced_sign_patterns([4],3),.1,np.array([1,0]))
        kept,mask=keep_complete_pairs(f,np.array([[True,True],[True,False],[False,True]]))
        self.assertEqual(len(kept),1)
        np.testing.assert_array_equal(mask,[True,False,False])

    def test_quadratic_identity(self):
        rng=np.random.default_rng(4)
        s=balanced_sign_patterns([6],10)
        j=rng.normal(size=(3,6,2)); v=np.array([1.,3.]); eps=.2
        f=antithetic_fields(s,eps,v); law=weighted_sign_law(s,np.arange(1,len(s)+1))
        response=np.einsum('lnd,bqnd->bql',j,f)
        direct=law['weights']@(.5*np.sum(response**2,axis=-1).mean(axis=1))
        q=expected_quadratic(directional_gram(j,v),law['covariance'],eps/np.sqrt(6))
        self.assertAlmostEqual(q,float(direct),places=12)

    def test_equal_spectral_power(self):
        rng=np.random.default_rng(5)
        u,_=np.linalg.qr(rng.normal(size=(6,6)))
        r=np.array([0,.1,.2,.3,.4,.5])
        s=np.where(rng.random((10,6))>.5,1,-1)
        f=spectral_sign_fields(u,r,s,np.array([1.,0.]))
        coeff=np.einsum('in,bid->bnd',u,f)
        np.testing.assert_allclose(np.sum(coeff**2,axis=-1),np.broadcast_to(r*r,(10,6)),atol=1e-12)

    def test_bad_spectral_basis(self):
        with self.assertRaises(ValueError): spectral_sign_fields(np.ones((2,2)),np.ones(2),np.array([[1,-1]]),np.array([1,0]))

    def test_dro_weights(self):
        w=worst_case_weights(np.array([1.,2.,3.]),.5)
        self.assertAlmostEqual(w.sum(),1)
        self.assertTrue(w[2]>w[1]>w[0])
        np.testing.assert_allclose(worst_case_weights(np.ones(3)),np.ones(3)/3)
        self.assertEqual(worst_case_weights(np.arange(3.),1.,np.array([.5,.5,0]))[2],0)

    def test_dro_keeps_marginals(self):
        s=balanced_sign_patterns([6],10)
        w=worst_case_weights(np.arange(len(s),dtype=float),.2)
        np.testing.assert_allclose(weighted_sign_law(s,w)['plus_probability'],.5)

    def test_greedy_cover(self):
        selected,covered=greedy_mode_cover(np.array([[1,1,0],[0,1,1],[0,0,0]],dtype=bool),2)
        self.assertEqual(selected,[0,1]); self.assertTrue(covered.all())
        self.assertEqual(greedy_mode_cover(np.zeros((2,3),dtype=bool),2)[0],[])

    def test_metric_psd_bounded(self):
        rng=np.random.default_rng(3)
        fit=fit_metric_initializer(rng.normal(size=(30,8))*3,rng.normal(size=(30,8)),3)
        ev=np.linalg.eigvalsh(fit['metric'])
        self.assertGreaterEqual(ev.min(),1-1e-10)
        self.assertLessEqual(ev.max(),2+1e-10)
        self.assertLessEqual(np.linalg.matrix_rank(fit['metric']-np.eye(8),tol=1e-8),3)

    def test_zero_signal_identity_and_collision(self):
        fit=fit_metric_initializer(np.zeros((3,4)),np.ones((3,4)),2)
        np.testing.assert_allclose(fit['metric'],np.eye(4))
        self.assertEqual(squared_distance(np.ones(4),np.ones(4),fit['metric']),0)

    def test_metric_rejects_bad_inputs(self):
        with self.assertRaises(ValueError): fit_metric_initializer(np.ones((3,4)),np.ones((4,5)))
        with self.assertRaises(ValueError): squared_distance(np.ones(2),np.zeros(2),-np.eye(2))

    def test_oracle_crossing_and_noncrossing(self):
        self.assertEqual(segment_relation(np.array([[-1,0],[1,0],[0,-1],[0,1]]))['label'],1)
        self.assertEqual(segment_relation(np.array([[-1,0],[1,0],[0,1],[.5,2]]))['label'],0)
        self.assertTrue(segment_relation(np.array([[-1,0],[1,0],[0,0],[0,1]]))['ambiguous'])

    def test_scenes_reproducible_balanced(self):
        x,y,m=make_relational_scenes(32,5)
        a,b,c=make_relational_scenes(32,5)
        np.testing.assert_array_equal(x,a); np.testing.assert_array_equal(y,b)
        np.testing.assert_array_equal(np.bincount(y),[16,16])
        self.assertTrue((m>=.02).all())
        self.assertTrue(all(segment_relation(xx)['label']==yy for xx,yy in zip(x,y)))

    def test_scenes_invalid_input(self):
        with self.assertRaises(ValueError): make_relational_scenes(0)

    def test_task_dag_and_referenced_routes(self):
        tasks=json.loads((PACK/'configs/tasks.json').read_text())['tasks']
        ids={t['id'] for t in tasks}; self.assertEqual(len(ids),len(tasks))
        graph={t['id']:t['depends_on'] for t in tasks}
        def visit(k,trail):
            self.assertNotIn(k,trail)
            for dep in graph[k]:
                self.assertIn(dep,ids); visit(dep,trail+[k])
        for k in ids: visit(k,[])
        for i in range(1,7): self.assertEqual(len(list((PACK/'routes').glob(f'R{i:02d}_*.md'))),1)

    def test_example_is_not_measured(self):
        obj=json.loads((PACK/'examples/route_result.example.json').read_text())
        self.assertEqual(obj['evidence_level'],'not_measured')
        self.assertIsNone(obj['primary_result']['value'])

    def test_inventory_is_read_only(self):
        spec=importlib.util.spec_from_file_location('inventory_test',PACK/'scripts/bootstrap_run.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'scripts').mkdir(); f=root/'scripts/run_t5r3_sanity.py'; f.write_text('original')
            obj=mod.inventory(root)
            self.assertEqual(f.read_text(),'original')
            self.assertIsNone(obj['current_commit'])
            self.assertEqual(obj['known_entries'][0]['exists'],True)

if __name__=='__main__':
    unittest.main()
