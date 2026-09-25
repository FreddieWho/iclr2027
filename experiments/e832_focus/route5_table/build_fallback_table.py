#!/usr/bin/env python3
"""Build an explicit fallback measurement table from archived U1 per-quartet rows.

This is not a new experiment. It exposes missing fields and keeps flow
calculations tied to same-seed raw_clean baselines. It must be replaced or
amended only after route1/route2 evidence is accepted.
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PQ=ROOT/'artifacts/next_novelty/u1_factorial/U1_PERQUARTET.csv'
SUM=ROOT/'artifacts/next_novelty/u1_factorial/U1_SUMMARY.json'
OUT=ROOT/'reports/e832_focus/MAIN_TABLE_FALLBACK.csv'
TEX=ROOT/'reports/e832_focus/MAIN_TABLE_FALLBACK.tex'

ARMS=('raw_clean','raw_flipmine','relfeat','relflip')
SEEDS=(11,23,47)

def bits(r): return (int(r['correct_A']),int(r['correct_B']),int(r['correct_AB']))
def main():
 rows=list(csv.DictReader(PQ.open())); summary=json.loads(SUM.read_text())
 idx={(int(r['seed']),r['arm'],int(r['qid'])):r for r in rows}
 qids=sorted({int(r['qid']) for r in rows})
 parents={int(r['qid']):int(r['parent']) for r in rows}
 out=[]
 for seed in SEEDS:
  base={q:idx[(seed,'raw_clean',q)] for q in qids}
  h=[q for q in qids if bits(base[q])==(1,1,0)]
  h111=[q for q in qids if bits(base[q])==(1,1,1)]
  for arm in ARMS:
   rs=[idx[(seed,arm,q)] for q in qids]
   A=np.mean([int(r['correct_A']) for r in rs]); B=np.mean([int(r['correct_B']) for r in rs]); AB=np.mean([int(r['correct_AB']) for r in rs])
   atomic=np.mean([bits(r)[0] and bits(r)[1] for r in rs]); J=np.mean([r['J']=='1' for r in rs])
   def flow(q): return bits(idx[(seed,arm,q)])
   rf=np.mean([flow(q)==(1,1,1) for q in h]) if h else None
   re=np.mean([flow(q)[2]==1 for q in h]) if h else None
   mg=np.mean([flow(q)[2]==1 and not (flow(q)[0] and flow(q)[1]) for q in h]) if h else None
   reg=np.mean([flow(q)!=(1,1,1) for q in h111]) if h111 else None
   out.append({'seed':seed,'arm':arm,'n_E':len(qids),'n_parent':len(set(parents.values())),
     'base':'raw_clean same seed' if arm!='raw_clean' else 'self',
     'A':A,'B':B,'AB':AB,'atomic_joint':atomic,'J3':J,'J4':'missing',
     'CCM_miss_given_atomic':(1-J/atomic if atomic else None),
     'baseline_n110':len(h) if arm!='raw_clean' else 'missing',
     'baseline_n111':len(h111) if arm!='raw_clean' else 'missing',
     'R_endpoint':re if arm!='raw_clean' else 'missing','R_full':rf if arm!='raw_clean' else 'missing',
     'M_migration':mg if arm!='raw_clean' else 'missing','regression111':reg if arm!='raw_clean' else 'missing',
     'parent_bootstrap':'U1 summary; see source','parameters':'missing','inference_macs':'missing',
     'train_state_forwards':'missing','source':str(PQ.relative_to(ROOT))})
 fields=list(out[0]);
 with OUT.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
 # compact latex fallback, one row per arm with seed mean J3 and raw-clean flow range
 lines=['\\begin{tabular}{@{}lrrrrrrrr@{}}','\\toprule','arm & $J_3$ & atomic & A & B & AB & $R_{full}$ & M & 111 regress\\\\','\\midrule']
 for arm in ARMS:
  rr=[r for r in out if r['arm']==arm]
  def avg(k):
   v=[r[k] for r in rr if isinstance(r[k],(int,float))]
   return sum(v)/len(v) if v else None
  def rng(k):
   v=[r[k] for r in rr if isinstance(r[k],(int,float))]
   return f'{min(v):.3f}--{max(v):.3f}' if v else 'missing'
  lines.append(f"{arm} & {avg('J3'):.3f} & {avg('atomic_joint'):.3f} & {avg('A'):.3f} & {avg('B'):.3f} & {avg('AB'):.3f} & {rng('R_full')} & {rng('M_migration')} & {rng('regression111')} \\\\")
 lines += ['\\bottomrule','\\end{tabular}']
 TEX.write_text('\n'.join(lines)+'\n')
 print(json.dumps({'rows':len(out),'n_E':len(qids),'n_parent':len(set(parents.values())),'csv':str(OUT),'tex':str(TEX)},indent=2))
if __name__=='__main__':main()
