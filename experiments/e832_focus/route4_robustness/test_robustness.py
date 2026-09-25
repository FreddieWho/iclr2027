import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));import robustness as r
class Tests(unittest.TestCase):
 def test_rigid_transform_shape_and_labels(self):
  x=np.array([[-.5,0],[.5,0],[0,-.5],[0,.5]],float)
  label=r.c.oracle('source',x)[0]
  for k in ('translation','rotation'):
   y=r.transform(x,k)
   self.assertEqual(y.shape,x.shape)
   self.assertEqual(r.c.oracle('source',y)[0],label)
 def test_noise_not_silently_implemented(self):
  self.assertTrue(hasattr(r,'main'))
  with self.assertRaisesRegex(ValueError,'new label-preserving E bank'):
   r.transform(np.zeros((4,2)), 'noise')
if __name__=='__main__':unittest.main()
