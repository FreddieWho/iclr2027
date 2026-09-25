"""Standalone algebraic audit, NOT a reproduction of repository training.

Inspected commit: e832887c938948b23e8783719d493f2b9b8c3a17.
The AdditivePair model reproduces the FUNCTIONAL FORM of TypedPairMLP
(experiments/f095_campaign/u01_models.py) and G8SetMLP
(experiments/e1a933_review/leads_l014_l015_l006.py).
No repository checkpoints, evaluation banks, or archived numerical results
are loaded here. Fixed witnesses test expressivity and input invariances.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def crossing(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (4, 2) or not np.isfinite(x).all():
        raise ValueError('Expected finite [4,2] coordinates')
    def orient(a, b, c):
        u, v = b-a, c-a
        return u[0]*v[1]-u[1]*v[0]
    a,b,c,d=x
    return int(orient(a,b,c)*orient(a,b,d)<0 and
               orient(c,d,a)*orient(c,d,b)<0)


def checkerboard() -> tuple[np.ndarray,np.ndarray]:
    # Two horizontal alternatives and two short vertical alternatives.
    # Proper crossings on the diagonal; no touching/collinearity.
    s=[np.array([[-.7,.4],[.7,.4]]),np.array([[-.7,-.4],[.7,-.4]])]
    t=[np.array([[.1,.2],[.1,.6]]),np.array([[.1,-.6],[.1,-.2]])]
    xs=np.stack([np.concatenate([a,b]) for a in s for b in t])
    ys=np.array([crossing(x) for x in xs]).reshape(2,2)
    return xs,ys


def group8():
    out=[]
    for a,b,c in itertools.product((0,1), repeat=3):
        p=[0,1,2,3]
        if a: p[0],p[1]=p[1],p[0]
        if b: p[2],p[3]=p[3],p[2]
        if c: p=p[2:]+p[:2]
        out.append(tuple(p))
    return out


class AdditivePair(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        self.phi=nn.Sequential(nn.Linear(2,width),nn.ReLU(),
                               nn.Linear(width,width),nn.ReLU())
        self.psi=nn.Sequential(nn.Linear(width,width),nn.ReLU(),
                               nn.Linear(width,width),nn.ReLU())
        self.rho=nn.Linear(width,1)
    def forward(self,x):
        h=self.phi(x.reshape(-1,4,2))
        return self.rho(self.psi(h[:,0]+h[:,1])+
                        self.psi(h[:,2]+h[:,3])).reshape(-1)


def invariant_witness_score(x: np.ndarray) -> np.ndarray:
    """An after-pooling nonlinearity solves ONLY this 4-case witness.

    This is deliberately not claimed to solve segment intersection generally.
    It shows the additive impossibility is not a property of G8 invariance.
    """
    x=np.asarray(x)
    centers=x.reshape(-1,2,2,2).mean(axis=2)
    summed_y=centers[:,:,1].sum(axis=1)
    return np.abs(summed_y)-.4


def features8(x: np.ndarray) -> np.ndarray:
    """Functional copy of the inspected 8-d handcrafted feature recipe."""
    x=np.asarray(x,dtype=np.float64).reshape(-1,4,2)
    u=x[:,1]-x[:,0]; v=x[:,3]-x[:,2]
    lengths=np.sort(np.stack([np.linalg.norm(u,axis=1),
                              np.linalg.norm(v,axis=1)],axis=1),axis=1)
    cross=np.sort(np.stack([np.linalg.norm(x[:,i]-x[:,j],axis=1)
                            for i,j in [(0,2),(0,3),(1,2),(1,3)]],axis=1),axis=1)
    mid=np.linalg.norm((x[:,0]+x[:,1]-x[:,2]-x[:,3])/2,axis=1)
    angle=np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-8)
    return np.concatenate([lengths,cross,mid[:,None],angle[:,None]],axis=1)


def run_audit() -> dict:
    torch.set_num_threads(1)
    xs,ys=checkerboard()
    residuals=[]
    accuracies=[]
    invariance=[]
    for seed in range(10):
        torch.manual_seed(seed)
        model=AdditivePair().double().eval()
        with torch.no_grad():
            z=model(torch.tensor(xs)).numpy().reshape(2,2)
            residuals.append(float(abs(z[0,0]+z[1,1]-z[0,1]-z[1,0])))
            accuracies.append(float(((z>0)==ys).mean()))
            zg=model(torch.tensor(xs[:,list(group8()[5]),:])).numpy()
            invariance.append(float(np.max(abs(z.ravel()-zg))))
    rich=features8(xs)
    feature_error=max(float(np.max(abs(features8(xs[:,list(p),:])-rich))) for p in group8())
    znew=invariant_witness_score(xs).reshape(2,2)
    return {
        'audit_type':'standalone algebraic witness; no repository model reproduction',
        'source_commit':'e832887c938948b23e8783719d493f2b9b8c3a17',
        'source_files':['experiments/f095_campaign/u01_models.py',
                        'experiments/e1a933_review/leads_l014_l015_l006.py'],
        'oracle_matrix':ys.tolist(),
        'additive_mixed_residual_max':max(residuals),
        'additive_group_error_max':max(invariance),
        'random_weight_witness_accuracies':accuracies,
        'nonlinear_after_sum_witness_logits':znew.tolist(),
        'nonlinear_after_sum_witness_accuracy':float(((znew>0)==ys).mean()),
        'handcrafted_feature_dimension':rich.shape[1],
        'handcrafted_feature_g8_error':feature_error,
        'limitations':['No inference about corrected-model performance on the project bank.',
                       'No inference about target-task TypedTriMLP/TypedDiskMLP; their fusion differs.',
                       'The nonlinear witness is not a general segment-intersection solver.'],
    }

if __name__=='__main__':
    r=run_audit()
    out=Path(__file__).resolve().parents[1]/'results'/'ALGEBRAIC_AUDIT.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))
