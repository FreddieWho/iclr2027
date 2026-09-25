#!/usr/bin/env python3
"""Route 3 source-task strong baselines. Uses route1's fresh source bank logic.
The analytic parser is a diagnostic upper bound, not a learned relational model.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/e832_focus/route1_cross_task'))
import common_runner as cr

torch.set_num_threads(4);ART=ROOT/'artifacts/e832_focus/route3';SEEDS=(11,23,47);EPOCHS=100;LR=1e-3
def intersects(x):
 def cr(a,b):return float(np.cross(b-a,a))
 d1=cr(x[1]-x[0],x[2]-x[0]);d2=cr(x[1]-x[0],x[3]-x[0]);d3=cr(x[3]-x[2],x[0]-x[2]);d4=cr(x[3]-x[2],x[1]-x[2]);return int(((d1>0)!=(d2>0)) and ((d3>0)!=(d4>0)))

class SegmentSetRel(nn.Module):
 def __init__(self,hidden=64):
  super().__init__();self.phi=nn.Sequential(nn.Linear(2,48),nn.ReLU(),nn.Linear(48,48),nn.ReLU());self.head=nn.Sequential(nn.Linear(96,hidden),nn.ReLU(),nn.Linear(hidden,1))
 def forward(self,x):
  p=x.reshape(-1,4,2);h=self.phi(p);z=torch.cat([h[:,:2].sum(1),h[:,2:].sum(1)],1);return self.head(z).reshape(-1)
class CapacityRaw(nn.Module):
 def __init__(self,hidden=96):super().__init__();self.net=nn.Sequential(nn.Linear(8,hidden),nn.ReLU(),nn.Linear(hidden,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).reshape(-1)
class GroupRaw(nn.Module):
 def __init__(self):super().__init__();self.net=CapacityRaw(64)
 def forward(self,x):return self.net(x).reshape(-1)

def fit_predict(arm,seed,X,Y,F,bank,stats,sup='flip'):
 mu,sd=stats
 if arm=='capacity':m=CapacityRaw()
 elif arm=='relational':m=SegmentSetRel()
 else:m=GroupRaw()
 def prep(x):return torch.from_numpy(((x.reshape(len(x),-1)-mu)/sd).astype(np.float32))
 xc=prep(X);yc=torch.from_numpy(Y.astype(np.float32));xf=prep(F[0]);yf=torch.from_numpy(F[1].astype(np.float32));torch.manual_seed(seed);opt=torch.optim.Adam(m.parameters(),lr=LR);lf=nn.BCEWithLogitsLoss();t=time.time()
 for _ in range(EPOCHS):
  opt.zero_grad();loss=lf(m(xc),yc)+lf(m(xf),yf);loss.backward();opt.step()
 m.eval();pred=[]
 with torch.no_grad():
  for i in range(4):
   xx=bank['states'][i]
   if arm=='group_average_raw':
    pred.append(np.mean([m(prep(cr.apply_perm('source',xx,q))).numpy() for q in cr.perms('source')],axis=0))
   else: pred.append(m(prep(xx)).numpy())
 z=np.stack(pred);lab=np.stack([bank['y0'],bank['y0'],bank['y0'],bank['yab']]);ok=(z>0)==(lab==1)
 j=ok[1]&ok[2]&ok[3];atomic=ok[1]&ok[2]
 return {'arm':arm,'seed':seed,'J3':float(j.mean()),'J4':None,'A':float(ok[1].mean()),'B':float(ok[2].mean()),'AB':float(ok[3].mean()),'atomic_joint':float(atomic.mean()),'CCM_miss_given_atomic':float(1-j[atomic].mean()) if atomic.any() else None,'n_E':len(j),'n_parents':len(np.unique(bank['parent'])),'parameters':sum(p.numel() for p in m.parameters()),'inference_macs':sum(q.in_features*q.out_features for q in m.modules() if isinstance(q,nn.Linear)),'train_state_forwards':EPOCHS*(len(X)+len(F[1])),'train_macs':sum(q.in_features*q.out_features for q in m.modules() if isinstance(q,nn.Linear))*EPOCHS*(len(X)+len(F[1])),'seconds':time.time()-t,'threads':4}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ART);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 X,Y,M=cr.gen('source',256,832501);F=cr.mine_flips('source',X,Y,832502,4);T,TY,TM=cr.gen('source',256,832503);B=cr.mine_E('source',T,TY,832504);assert B is not None
 # Train-statistics only; no test labels.
 mu=X.reshape(-1,2).mean(0);sd=X.reshape(-1,2).std(0)+1e-8;mu=np.tile(mu,4);sd=np.tile(sd,4);stats=(mu,sd)
 analytic=(cr.oracle('source',B['states'][2])[0] if False else None)
 # Analytic segment-intersection parser on each quartet's AB state.
 pa=[]
 for arm in ('capacity','relational','group_average_raw'):
  for s in SEEDS:pa.append(fit_predict(arm,s,X,Y,F,B,stats))
 # group_average_raw is explicitly applied as legal G8 orbit score average.
 analytic_pred=np.stack([np.array([intersects(s) for s in B['states'][i]]) for i in range(4)]);lab=np.stack([B['y0'],B['y0'],B['y0'],B['yab']]);aok=analytic_pred==lab
 payload={'status':'COMPLETE','task':'source','fresh_parent_seeds':{'train':832501,'test':832503,'edit':832504},'E':len(B['parent']),'parents':len(np.unique(B['parent'])),'denominator':'official label-preserving atomic edit pairs, unordered; no metric substitution','analytic_parser':{'J3':float((aok[1]&aok[2]&aok[3]).mean()),'A':float(aok[1].mean()),'B':float(aok[2].mean()),'AB':float(aok[3].mean()),'n_E':len(B['parent']),'note':'coordinate parser diagnostic; source task, not visual image parser'},'learned':pa,'classical_visual_parser':'NOT_APPLICABLE: source task is coordinate-only; route2 has no visual bank','invalid_S4_all_point_sum':'not run; not a valid relational baseline','threads':4}
 (a.out/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'E':len(B['parent']),'analytic_J3':payload['analytic_parser']['J3'],'learned_rows':len(pa)},indent=2))
if __name__=='__main__':main()
