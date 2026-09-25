import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));import strong_baselines as b
class Tests(unittest.TestCase):
 def test_parser_crossing(self):
  x=np.array([[-.5,0],[.5,0],[0,-.5],[0,.5]],float);self.assertEqual(b.intersects(x),1)
  y=x.copy();y[2:]+=2;self.assertEqual(b.intersects(y),0)
 def test_legal_relational_model_shape(self):
  import torch
  m=b.SegmentSetRel();self.assertEqual(tuple(m(torch.randn(3,4,2)).shape),(3,))
if __name__=='__main__':unittest.main()
