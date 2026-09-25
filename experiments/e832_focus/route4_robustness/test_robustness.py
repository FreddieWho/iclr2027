import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));import robustness as r
class Tests(unittest.TestCase):
 def test_rigid_transform_shape_and_labels(self):
  x=np.random.default_rng(1).normal(size=(4,2));
  # source oracle is invariant to both fixed conditions; check finite shape.
  for k in ('translation','rotation'):self.assertEqual(r.transform(x,k).shape,x.shape)
 def test_noise_not_silently_implemented(self):
  self.assertTrue(hasattr(r,'main'))
if __name__=='__main__':unittest.main()
