#!/usr/bin/env python3
"""Run selected real repository definitions without importing training entrypoints.

Use on a trusted checkout. This executes named function/class definitions from
that checkout in an isolated namespace. It neither trains nor modifies it.
A retained legacy negative-control representation may intentionally still fail
one semantic contract; interpret results against the declared claim, not a
blanket 'all pass' gate.
"""
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from contracts import t1_collision_pair, point_in_triangle


def definitions(path: Path, names: list[str], extra=None):
    text=path.read_text(encoding='utf-8')
    module=ast.parse(text,filename=str(path))
    nodes=[n for n in module.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    missing=set(names)-{n.name for n in nodes}
    if missing:raise RuntimeError(f'Missing definitions: {sorted(missing)}')
    ns={'np':np,'torch':torch,'nn':nn,'F':F}
    if extra:ns.update(extra)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns,hashlib.sha256(text.encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();rows=[];sources={}
    def record(name,passed,detail):rows.append({'contract':name,'passed':bool(passed),'detail':detail})
    path=args.repo/'experiments/e832_focus/route1_cross_task/common_runner.py'
    try:
        ns,sha=definitions(path,['rich','sort_features','Typed']);sources[str(path)]=sha
        a,b=t1_collision_pair()
        fa=ns['sort_features']('T1',ns['rich']('T1',a[None].astype(np.float32)))
        fb=ns['sort_features']('T1',ns['rich']('T1',b[None].astype(np.float32)))
        gap=float(np.max(np.abs(fa-fb)))
        record('T1_representation_distinguishes_opposite_labels',not np.allclose(fa,fb,atol=1e-7,rtol=0),
               {'max_feature_gap':gap,'labels':[point_in_triangle(a),point_in_triangle(b)],'bitwise_equal':bool(np.array_equal(fa,fb))})
        x=np.array([[[-.6,.1],[.5,.2],[.1,.7]]],np.float32)
        expected=np.linalg.norm(x[:,2,None]-x[:,:2],axis=-1)
        actual=ns['rich']('T2',x)
        record('T2_center_endpoint_distances',np.allclose(actual[:,1:3],expected,atol=1e-6),
               {'actual':actual[:,1:3].tolist(),'expected':expected.tolist()})
        r=np.array([[0.,-1.],[1.,0.]],np.float32)
        fr=ns['sort_features']('T2',ns['rich']('T2',x@r.T))
        f=ns['sort_features']('T2',actual)
        record('T2_rigid_invariance',np.allclose(f,fr,atol=1e-6),{'max_gap':float(np.abs(f-fr).max())})
        torch.manual_seed(11);model=ns['Typed']('source',True).double().eval()
        x=torch.tensor([[-.6,-.6],[.6,.6],[-.6,.6],[.6,-.6]],dtype=torch.float64)
        with torch.no_grad():
            pa=model(x.reshape(1,8));pb=model(x[[0,2,1,3]].reshape(1,8))
        record('source_not_forced_S4_invariant',not torch.allclose(pa,pb,atol=1e-12,rtol=0),
               {'counterexample_logit_gap':float((pa-pb).abs().max()),'interpretation':'A random witness; equal outputs plus all-point-sum source establishes the legacy defect. Non-equality alone does not prove full expressive adequacy.'})
    except Exception as exc:
        rows.append({'contract':'route1_definition_execution','passed':False,'error':repr(exc)})
    path=args.repo/'experiments/e832_focus/route2_visual/gpu_run.py'
    try:
        ns,sha=definitions(path,['metrics']);sources[str(path)]=sha
        m=ns['metrics'](np.array([[-1.,-1.,-1.,1.]]),np.array([[1.,0.,0.,1.]]))
        record('J4_includes_P',m.get('J3')==1 and m.get('J4')==0,m)
    except Exception as exc:
        rows.append({'contract':'metric_definition_execution','passed':False,'error':repr(exc)})
    path=args.repo/'experiments/e832_focus/route2_visual/visual_mechanism.py'
    try:
        ns,sha=definitions(path,['_image_nchw','prep','visible_segment_features'],
               {'MEAN':torch.tensor([.485,.456,.406]).view(1,3,1,1),'STD':torch.tensor([.229,.224,.225]).view(1,3,1,1)})
        sources[str(path)]=sha
        class Dummy(nn.Module):
            def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.tensor(1.))
            def backbone(self,x):return torch.ones(len(x),2,7,7,device=x.device)
        image=torch.full((1,3,64,64),.5)
        image[:,0,4:9,8:57]=.9;image[:,1:,4:9,8:57]=.1
        pooled=ns['visible_segment_features'](Dummy(),image)['red']
        record('visible_mask_not_erased_at_pooling',bool(pooled.abs().sum()>0),{'red_pool':pooled.detach().tolist(),'native_red_pixels':245})
    except Exception as exc:
        rows.append({'contract':'pool_definition_execution','passed':False,'error':repr(exc)})
    output={'scope':'Definition-level contract audit on trusted checkout; not model performance replication',
            'sources':sources,'checks':rows,'n_checks':len(rows),'n_failed':sum(not x['passed'] for x in rows)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
