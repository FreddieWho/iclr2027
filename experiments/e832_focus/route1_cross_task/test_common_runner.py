import json, os, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common_runner as r

class Route1Tests(unittest.TestCase):
 def test_group_actions_preserve_roles(self):
  x=np.arange(8,dtype=np.float32).reshape(1,4,2)
  for p in r.perms('T1'):
   z=r.apply_perm('T1',x,p); self.assertTrue(np.array_equal(z[0,3],x[0,3])); self.assertEqual(set(map(tuple,z[0,:3])),set(map(tuple,x[0,:3])))
  for p in r.perms('T2'):
   z=r.apply_perm('T2',x[:,:3],p); self.assertEqual(set(map(tuple,z[0,:2])),set(map(tuple,x[0,:2])))
 def test_sorted_rich_is_group_invariant(self):
  rng=np.random.default_rng(2)
  for task,n in [('source',4),('T1',4),('T2',3)]:
   x=rng.normal(size=(5,n,2)).astype(np.float32);f=r.sort_features(task,r.rich(task,x))
   for p in r.perms(task):
    self.assertTrue(np.allclose(f,r.sort_features(task,r.rich(task,r.apply_perm(task,x,p))),atol=1e-6))
 def test_additive_witness_and_repair(self):
  sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'f095_campaign'))
  from u01_models import TypedPairMLP
  add=TypedPairMLP(64); rep=r.Typed('source',True)
  # Add cannot satisfy the established crossing-difference identity; repaired
  # rho has no algebraic requirement to retain it.
  # Four independent combinations S1/S2 and T1/T2; no point is shared across roles.
  ss=torch.randn(2,2,2);tt=torch.randn(2,2,2)
  combos=torch.stack([torch.cat([ss[i],tt[j]]) for i in range(2) for j in range(2)])
  a=add(combos).detach();d=(a[0]+a[3]-a[1]-a[2]).abs().item();self.assertLess(d,1e-5)
  y=rep(combos);self.assertEqual(tuple(y.shape),(4,))
 def test_official_E_contract(self):
  import inspect
  s=inspect.getsource(r)
  self.assertIn('both preserve, sum flips',s)
  self.assertNotIn('/dev512',s)
 def test_explicit_cli_and_environment_settings(self):
  argv=['--tasks','T1','--test-parents','8192','--edit-cap','8',
        '--candidate-draws','4','--epochs','300','--test-seed','12345']
  a=r.parse_args(argv)
  self.assertEqual((a.train_parents,a.dev_parents,a.test_parents,a.edit_cap,
                    a.candidate_draws,a.epochs,a.test_seed),
                   (256,64,8192,8,4,300,12345))
  env={'ROUTE1_TRAIN_PARENTS':'11','ROUTE1_DEV_PARENTS':'12','ROUTE1_TEST_PARENTS':'13',
       'ROUTE1_EDIT_CAP':'14','ROUTE1_CANDIDATE_DRAWS':'15','ROUTE1_EPOCHS':'16'}
  with mock.patch.dict(os.environ,env,clear=False):
   e=r.parse_args(['--tasks','T2'])
  self.assertEqual((e.train_parents,e.dev_parents,e.test_parents,e.edit_cap,
                    e.candidate_draws,e.epochs),(11,12,13,14,15,16))
 def test_single_task_results_merge_preserves_other_tasks(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'results.json'
   r.merge_results(p,{'source':{'round':1},'T1':{'round':1}})
   r.merge_results(p,{'T1':{'round':2}})
   self.assertEqual(json.loads(p.read_text()),
                    {'source':{'round':1},'T1':{'round':2}})
 def test_parent_bootstrap_uses_equal_parent_weight(self):
  # Pair-weighted mean is 0.50, but equal-parent mean is 1/3. The point
  # estimate must agree with the parent-cluster resampling estimand.
  a=np.array([1,1,0,0],bool);b=np.zeros(4,bool);pa=np.array([0,0,1,2])
  z=r.boot_delta(a,b,pa,7)
  self.assertAlmostEqual(z['estimate'],1/3)
  self.assertEqual(z['parents'],3)
  self.assertEqual(z['unit'],'test parent (equal weight)')
if __name__=='__main__':unittest.main()
