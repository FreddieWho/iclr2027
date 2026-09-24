"""Evaluate frozen D01/U02 models and all O01 recipes on the independent new pool."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/f095_campaign'))
from u01_eval import load_arm,featurize_raw
from common import rel_features
from symmetry import GROUP,apply
from data_stats import parent_bootstrap
torch.set_num_threads(2)
OUT=ROOT/'artifacts/e1a933_review/data_source_eval'


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument("--out-dir",type=Path,default=OUT);p.add_argument("--o01-dir",type=Path,default=ROOT/"artifacts/e1a933_review/data_o01");a=p.parse_args();OUT=a.out_dir
    if (OUT/"FROZEN_AND_ALL_RECIPES.csv").exists():raise FileExistsError("Use a fresh --out-dir")
    OUT.mkdir(parents=True,exist_ok=True)
    ep=a.o01_dir/'eval.npz';d=np.load(ep);x=d['states'];lab=d['labels'];parents=d['parents'];rows=[]
    paths=[]
    for family in ['D01','U02']:
        paths.extend((ROOT/f'artifacts/f095_campaign/{family}').glob('**/model.pt'))
    paths.extend(a.o01_dir.glob('*/model.pt'))
    for mp in paths:
        model,ck=load_arm(mp.parent);flat=x.reshape(-1,4,2)
        fe=ck.get('featurize')
        if fe in ['rel12','sixdist']:
            f=rel_features(flat);f=f[:,:6] if fe=='sixdist' else f
            t=torch.tensor((f-ck['mu'])/ck['sd'],dtype=torch.float32)
        else:t=featurize_raw(flat,ck)
        with torch.no_grad():sc=model(t).numpy().reshape(-1,3)
        ok=(sc>0)==lab;name=hashlib.sha256(str(mp).encode()).hexdigest()[:16]
        np.savez_compressed(OUT/(name+'.npz'),scores=sc,correct=ok,parents=parents,labels=lab)
        ci=parent_bootstrap(ok.all(1),parents)
        rows.append(dict(checkpoint=str(mp.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
            score_file=name+'.npz',J=float(ok.all(1).mean()),J_ci95_low=ci['ci95'][0][0],J_ci95_high=ci['ci95'][0][1],
            atomic_pass=float(ok[:,:2].all(1).mean()),AB_correct=float(ok[:,2].mean()),n=len(lab),n_parents=len(np.unique(parents))))
    with (OUT/'FROZEN_AND_ALL_RECIPES.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Existing source typed uses a true global scalar normalization; audit, no retraining.
    audits={};xx=np.load(ROOT/'artifacts/discovery_campaign/scenes/eval_202/scenes.npz')['positions']
    for mp in (ROOT/'artifacts/f095_campaign/U01').glob('**/model.pt'):
        model,ck=load_arm(mp.parent)
        with torch.no_grad():
            ref=model(featurize_raw(xx,ck));delta=[];changes=[]
            for g in GROUP:
                pred=model(featurize_raw(np.stack([apply(v,g) for v in xx]),ck))
                delta.append(float((pred-ref).abs().max()));changes.append(int(((pred>0)!=(ref>0)).sum()))
        audits[str(mp.relative_to(ROOT))]=dict(scalar_mu=bool(np.ptp(ck['mu'])==0),scalar_sd=bool(np.ptp(ck['sd'])==0),max_logit_difference=max(delta),prediction_difference=max(changes))
    (OUT/'SOURCE_TYPED_PIPELINE.json').write_text(json.dumps(audits,indent=2))
    print('evaluated',len(rows),'checkpoints',flush=True)


if __name__=='__main__':main()
