#!/usr/bin/env python3
"""Independent source contrast under fresh parents and fixed observation shifts."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments/e832_focus/route1_cross_task'));import common_runner as c
ART=ROOT/'artifacts/e832_focus/route4';THREADS=4
def transform(x,kind):
 x=np.asarray(x,np.float64).copy()
 if kind=='canonical':return x
 if kind=='translation':return x+np.array([.071,-.043])
 if kind=='rotation':
  t=np.deg2rad(7.0);R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]);return x@R.T
 if kind=='noise':
  raise ValueError('noise is not implemented; freeze and audit a new label-preserving E bank first')
 raise KeyError(kind)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ART);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);c.THREADS=THREADS
 # New parent and edit seeds, disjoint from Route 1; same task generator/oracle.
 X,Y,M=c.gen('source',256,832701);DX,DY,DM=c.gen('source',64,832702);DS=c.mine_flips('source',DX,DY,832703,3);F=c.mine_flips('source',X,Y,832704,4);T,TY,TM=c.gen('source',256,832705);B=c.mine_E('source',T,TY,832706);assert B is not None
 dev=(DX,DY,DS[0],DS[1]);conditions=['canonical','translation','rotation'];rows=[]
 for arm in ('raw','additive','repaired','rich_unsorted','rich_sorted'):
  for seed in (11,23,47):
   m,mu,sd,rec=c.train_one('source',arm,seed,'flip',X,Y,F,dev,60)
   for cond in conditions:
    st=transform(B['states'],cond)
    zz=np.stack([c.predict('source',arm,m,mu,sd,st[i]) for i in range(4)]);lab=np.stack([B['y0'],B['y0'],B['y0'],B['yab']]);ok=(zz>0)==(lab==1);j=ok[1]&ok[2]&ok[3];atomic=ok[1]&ok[2]
    rows.append({'arm':arm,'seed':seed,'condition':cond,'J3':float(j.mean()),'J4':None,'A':float(ok[1].mean()),'B':float(ok[2].mean()),'AB':float(ok[3].mean()),'atomic_joint':float(atomic.mean()),'CCM_miss_given_atomic':float(1-j[atomic].mean()) if atomic.any() else None,'n_E':len(j),'n_parents':len(np.unique(B['parent'])),'parameters':rec['parameters'],'inference_macs':rec['inference_macs'],'train_state_forwards':rec['train_state_forwards'],'threads':THREADS,'oracle_consistency':'not recomputed: rigid transforms preserve source label by construction'})
 # A fixed non-rigid noise diagnostic is not silently folded into denominator.
 noise_rows=[]
 for arm in ('rich_sorted','rich_unsorted'):
  # use first trained setting's saved logits is not retained, so a small explicit
  # diagnostic is reported as not run rather than fabricated.
  noise_rows.append({'arm':arm,'condition':'noise_sigma_0.003','status':'NOT_RUN','reason':'would require a separately frozen new E label audit; no silent denominator change'})
 payload={'status':'COMPLETE','fresh_parent_seeds':{'train':832701,'dev':832702,'test':832705,'edit':832706},'E':len(B['parent']),'parents':len(np.unique(B['parent'])),'denominator':'same fresh E denominator for canonical/translation/rotation; no post-hoc filtering','conditions':conditions,'rows':rows,'noise_diagnostic':noise_rows,'visual_renderer_contract':'SEPARATE_NOT_RUN: coordinate-only source task; see Route 2 BLOCKED_GPU','main_contrast':'rich_sorted minus rich_unsorted','threads':THREADS}
 (a.out/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'E':len(B['parent']),'rows':len(rows),'status':'COMPLETE'},indent=2))
if __name__=='__main__':main()
