import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));import strong_baselines as b
class Tests(unittest.TestCase):
 def test_parser_crossing(self):
  x=np.array([[-.5,0],[.5,0],[0,-.5],[0,.5]],float);self.assertEqual(b.intersects(x),1)
  y=x.copy();y[2:]+=2;self.assertEqual(b.intersects(y),0)
 def test_parser_respects_declared_endpoint_swap_group(self):
  crossing=np.array([[-.5,0],[.5,0],[0,-.5],[0,.5]],float)
  separated=crossing.copy();separated[2:]+=2
  for x in (crossing,separated):
   expected=b.intersects(x)
   for p in b.cr.perms('source'):
    self.assertEqual(b.intersects(x[p]),expected)
 def test_legal_relational_model_shape(self):
  import torch
  m=b.SegmentSetRel();self.assertEqual(tuple(m(torch.randn(3,4,2)).shape),(3,))
class RoundTests(unittest.TestCase):
 def test_orbit_canonical_invariant(self):
  import route3_rounds as r
  x=np.array([[[-.6,-.6],[.6,.6],[-.6,.6],[.6,-.6]]],float)
  ref=r.orbit_canonical(x)
  for perm in ([1,0,2,3],[2,3,0,1],[0,1,3,2]):
   np.testing.assert_allclose(r.orbit_canonical(x[:,perm,:]),ref,atol=1e-12)
 def test_orbit_canonical_separates_regroup(self):
  import route3_rounds as r
  x=np.array([[[-.6,-.6],[.6,.6],[-.6,.6],[.6,-.6]]],float)
  self.assertGreater(np.abs(r.orbit_canonical(x)-r.orbit_canonical(x[:,[0,2,1,3],:])).max(),.1)
if __name__=='__main__':unittest.main()
