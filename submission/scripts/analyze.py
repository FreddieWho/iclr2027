"""Paired history-cluster bootstrap. Exploratory unless independently confirmed."""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
import numpy as np
from core import read_jsonl,write_json,digest

def bootstrap(values, family_size=1, n_boot=5000, seed=127):
    x=np.asarray(values,dtype=float)
    if len(x)<2 or not np.isfinite(x).all(): raise ValueError('Need at least 2 finite independent history contrasts')
    if family_size<1 or n_boot<100:raise ValueError('Invalid family size / bootstrap count')
    rng=np.random.default_rng(seed)
    # Chunked resampling bounds memory for larger analyses.
    out=[]
    for start in range(0,n_boot,500):
        n=min(500,n_boot-start)
        out.extend(x[rng.integers(0,len(x),size=(n,len(x)))].mean(axis=1).tolist())
    alpha=.05/family_size
    return {'estimate':float(x.mean()),'ci_low':float(np.quantile(out,alpha/2)), 'ci_high':float(np.quantile(out,1-alpha/2)), 'ci_level':1-alpha, 'n_histories':len(x),'bootstrap_replicates':n_boot,'bootstrap_seed':seed,'family_size':family_size}

def paired_values(rows, arm_a, arm_b, budget, qtype=None):
    relevant=[r for r in rows if r['arm'] in (arm_a,arm_b) and (r.get('budget')==budget or r['arm'] in ('raw','no_memory','oracle')) and (qtype is None or r['question_type']==qtype)]
    sides={arm_a:{},arm_b:{}};universe=set()
    for r in relevant:
        key=(r['group_id'],r['question_id'],r['replicate'],r.get('reader',''))
        universe.add(key)
        if key in sides[r['arm']]:raise ValueError('Duplicate score row; do not concatenate repeated runs blindly')
        sides[r['arm']][key]=r
    groups=defaultdict(list);nvalid=0
    for key in universe:
        a=sides[arm_a].get(key);b=sides[arm_b].get(key)
        if not a or not b or a.get('status')!='valid' or b.get('status')!='valid':continue
        if a.get('score') is None or b.get('score') is None:continue
        groups[key[0]].append(float(a['score'])-float(b['score']));nvalid+=1
    vals={g:float(np.mean(v)) for g,v in groups.items()}
    return vals,{'paired_question_replicates':nvalid,'planned_union_question_replicates':len(universe),'valid_pair_fraction':nvalid/len(universe) if universe else 0,'contains_mock':any(r.get('mock',False) for r in relevant)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True);p.add_argument('--contrast',required=True,help='arm_A,arm_B; estimate=A-B');p.add_argument('--budget',type=int,required=True);p.add_argument('--type');p.add_argument('--family-size',type=int,default=1);p.add_argument('--n-boot',type=int,default=5000);p.add_argument('--out',required=True)
    a=p.parse_args();arms=a.contrast.split(',')
    if len(arms)!=2 or arms[0]==arms[1]:raise SystemExit('Need two distinct arm IDs')
    rows=read_jsonl(a.scores);vals,qc=paired_values(rows,*arms,a.budget,a.type)
    if len(vals)<2:raise SystemExit('Insufficient valid paired histories; not a negative scientific result')
    obj={**bootstrap(list(vals.values()),a.family_size,a.n_boot),**qc,'contrast':arms,'budget':a.budget,'question_type':a.type,'history_contrasts':vals,'scores_hash':digest(rows),'analysis_status':'MOCK_ONLY' if qc['contains_mock'] else 'REQUIRES_CONFIRMATION_AND_AUDITS'}
    write_json(a.out,obj);print(f"{arms[0]} - {arms[1]} = {obj['estimate']:.4f}, CI [{obj['ci_low']:.4f}, {obj['ci_high']:.4f}], n={len(vals)}, mock={obj['contains_mock']}")
if __name__=='__main__':main()
