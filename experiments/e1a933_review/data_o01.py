"""O01 bounded 2x2x3 source factorial, equal optimizer search, independent parent pool.

Selection: 3 fixed learning rates on independent static/single development BCE.
All 108 scratch fits retained. No E labels used for selection. Full-batch 300 steps.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/f095_campaign'))
from d01_capacity import build_model
from d03_build_bank import oracle,candidates_for_scene
from core.relations import make_relational_scenes
from common import rel_features
from data_stats import parent_bootstrap
OUT=ROOT/'artifacts/e1a933_review/data_o01'
torch.set_num_threads(2)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mine(x,y,seed):
    states=[];labels=[];parents=[];calls=0
    rng=np.random.default_rng(seed)
    for i in range(len(x)):
        n=0
        for e,_ in candidates_for_scene(x[i],rng):
            calls+=1;r=oracle(x[i]+e)
            if r is not None and r[0]!=int(y[i]):
                states.append(x[i]+e);labels.append(r[0]);parents.append(i);n+=1
                if n==12:break
    return dict(states=np.asarray(states),labels=np.asarray(labels),parents=np.asarray(parents),oracle_calls=calls)


def generate():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'eval.npz').exists():return
    x,y,m=make_relational_scenes(768,seed=241933,min_margin=.02)
    training=np.load(ROOT/'artifacts/f095_campaign/D01/scenes_16N/scenes.npz')['positions']
    dist,_=cKDTree(training.reshape(-1,8)).query(x.reshape(-1,8),p=np.inf)
    assert dist.min()>1e-4
    np.savez_compressed(OUT/'new_parents.npz',positions=x,labels=y,margin=m,nearest_16N_linf=dist)
    # First 256 are development only, final 512 evaluation only, split before editing.
    dev=mine(x[:256],y[:256],241934)
    np.savez_compressed(OUT/'dev.npz',positions=x[:256],labels=y[:256],**{'single_'+k:v for k,v in dev.items()})
    states=[];labels=[];parents=[];oracle_calls=0;flow=[]
    for i in range(256,768):
        rng=np.random.default_rng(241935+i);keeps=[]
        for e,_ in candidates_for_scene(x[i],rng):
            oracle_calls+=1;r=oracle(x[i]+e)
            if r is not None and r[0]==int(y[i]):keeps.append(e)
        pairs=[(a,b) for a in range(len(keeps)) for b in range(a+1,len(keeps))]
        if len(pairs)>3000:
            pairs=[pairs[j] for j in sorted(rng.choice(len(pairs),3000,replace=False))]
        n=0
        for a,b in pairs:
            ea,eb=keeps[a],keeps[b];oracle_calls+=1;r=oracle(x[i]+ea+eb)
            if r is None or r[0]==int(y[i]):continue
            states.append(np.stack([x[i]+ea,x[i]+eb,x[i]+ea+eb]));labels.append([y[i],y[i],r[0]]);parents.append(i);n+=1
            if n==12:break
        flow.append(dict(parent=i,keeps=len(keeps),quartets=n))
        if i%64==0:print('new bank parent',i,'quartets',len(states),flush=True)
    np.savez_compressed(OUT/'eval.npz',states=np.asarray(states),labels=np.asarray(labels),parents=np.asarray(parents))
    (OUT/'bank_manifest.json').write_text(json.dumps(dict(seed=241933,development_parents=256,evaluation_parents=512,
        cap=12,pair_cap=3000,oracle_calls=oracle_calls,flow=flow,nearest_16N_min_linf=float(dist.min()),
        generation='new independent parent pool; no target scores consulted'),indent=2))


def features(x,arch):
    return x.reshape(len(x),8) if arch=='raw' else rel_features(x)[:,:6]


def train():
    dev=np.load(OUT/'dev.npz');dxx=np.concatenate([dev['positions'],dev['single_states']]);dyy=torch.tensor(np.concatenate([dev['labels'],dev['single_labels']]),dtype=torch.float32)
    rows=[]
    for size in ['N','4N']:
        tp=ROOT/f'artifacts/f095_campaign/D01/scenes_{size}/scenes.npz';d=np.load(tp);x=d['positions'];y=torch.tensor(d['labels'],dtype=torch.float32)
        fp=OUT/f'{size}_flips.npz'
        if not fp.exists():np.savez_compressed(fp,**mine(x,d['labels'],5))
        flips=np.load(fp);fy=torch.tensor(flips['labels'],dtype=torch.float32)
        subset=np.random.default_rng(934).permutation(len(fy))[:len(fy)//4];subset.sort()
        for arch in ['raw','sixdist']:
            f=features(x,arch);mu=f.mean(0);sd=f.std(0)+1e-8
            xc=torch.tensor((f-mu)/sd,dtype=torch.float32);xf=torch.tensor((features(flips['states'],arch)-mu)/sd,dtype=torch.float32);xd=torch.tensor((features(dxx,arch)-mu)/sd,dtype=torch.float32)
            for seed in [11,23,47]:
                for frac in [0,25,100]:
                    for lr in [.01,.003,.001]:
                        name=f'{size}_{arch}_s{seed}_f{frac}_lr{lr}';dest=OUT/name;dest.mkdir(exist_ok=True)
                        if (dest/'receipt.json').exists():rows.append(json.loads((dest/'receipt.json').read_text()));continue
                        torch.manual_seed(seed);model,extra=build_model(arch,64,32)
                        init=hashlib.sha256(b''.join(t.numpy().tobytes() for t in model.state_dict().values())).hexdigest()
                        opt=torch.optim.Adam(model.parameters(),lr=lr);bce=torch.nn.BCEWithLogitsLoss();curve=[]
                        ids=subset if frac==25 else np.arange(len(fy));xb,yb=xf[ids],fy[ids]
                        for ep in range(300):
                            opt.zero_grad();loss=bce(model(xc),y)
                            if frac:loss=loss+bce(model(xb),yb)
                            loss.backward();opt.step();curve.append(float(loss.detach()))
                        with torch.no_grad():dev_bce=float(bce(model(xd),dyy));train_error=float(((model(xc)>0)!=y.bool()).float().mean())
                        torch.save(dict(state=model.state_dict(),hidden=64,feat=32,mu=mu,sd=sd,seed=seed,**extra),dest/'model.pt')
                        r=dict(name=name,size=size,arch=arch,seed=seed,frac=frac,lr=lr,dev_bce=dev_bce,train_error=train_error,
                            final_loss=curve[-1],steps=300,init_sha256=init,checkpoint_sha256=sha(dest/'model.pt'),train_sha256=sha(tp),
                            flip_sha256=sha(fp),batch_order='full static then full fixed subset each step',subset_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                            oracle_candidate_queries=int(flips['oracle_calls']),retained_flip_labels=0 if frac==0 else len(ids),
                            training_state_forwards=300*(len(x)+(len(ids) if frac else 0)))
                        (dest/'receipt.json').write_text(json.dumps(r,indent=2));(dest/'loss.json').write_text(json.dumps(curve));rows.append(r)
                        print(name,dev_bce,curve[-1],flush=True)
    with (OUT/'O01_ALL_CANDIDATES.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evaluate():
    ev=np.load(OUT/'eval.npz');xx=ev['states'];lab=ev['labels'];par=ev['parents'];rows=[];oks={};selected=[]
    receipts=[json.loads(p.read_text()) for p in OUT.glob('*/receipt.json')]
    for size in ['N','4N']:
        for arch in ['raw','sixdist']:
            for seed in [11,23,47]:
                for frac in [0,25,100]:
                    candidates=[r for r in receipts if (r['size'],r['arch'],r['seed'],r['frac'])==(size,arch,seed,frac)]
                    r=min(candidates,key=lambda r:r['dev_bce']);selected.append(r)
                    ck=torch.load(OUT/r['name']/'model.pt',weights_only=False);model,_=build_model(arch,64,32);model.load_state_dict(ck['state']);model.eval()
                    f=features(xx.reshape(-1,4,2),arch);t=torch.tensor((f-ck['mu'])/ck['sd'],dtype=torch.float32)
                    with torch.no_grad():sc=model(t).numpy().reshape(-1,3)
                    ok=(sc>0)==lab;oks[(size,arch,seed,frac)]=ok
                    np.savez_compressed(OUT/(r['name']+'_scores.npz'),scores=sc,labels=lab,parents=par,correct=ok)
                    rows.append(dict(**r,J=float(ok.all(1).mean()),n=len(lab),n_parents=len(np.unique(par)),atomic_pass=float(ok[:,:2].all(1).mean()),AB_correct=float(ok[:,2].mean())))
    for r in rows:
        base=oks[(r['size'],'raw',r['seed'],0)]
        ok=oks[(r['size'],r['arch'],r['seed'],r['frac'])]
        h=base[:,:2].all(1)&~base[:,2];old111=base.all(1)
        r.update(CCM_denominator=int(h.sum()), R_full=float(ok[h].all(1).mean()) if h.any() else None,
                 M=float((ok[h,2]&~ok[h,:2].all(1)).mean()) if h.any() else None,
                 original111_n=int(old111.sum()), degradation111=float((~ok[old111].all(1)).mean()) if old111.any() else None)
    contrasts={}
    for size in ['N','4N']:
        for seed in [11,23,47]:
            r0,r1,s0,s1=[oks[(size,a,seed,f)].all(1).astype(float) for a,f in [('raw',0),('raw',100),('sixdist',0),('sixdist',100)]]
            contrasts[f'{size}_s{seed}']=dict(I=parent_bootstrap(s1-s0-r1+r0,par),distance_flip_advantage=parent_bootstrap(s1-r1,par),
                 retained25_vs_raw100=parent_bootstrap(oks[(size,'sixdist',seed,25)].all(1).astype(float)-r1,par))
    with (OUT/'O01_FACTORIAL.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'O01_PAIRED.json').write_text(json.dumps(contrasts,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['gen','train','eval'],required=True);p.add_argument('--out-dir',type=Path,default=OUT);a=p.parse_args()
    OUT=a.out_dir
    final={'gen':'eval.npz','train':'O01_ALL_CANDIDATES.csv','eval':'O01_FACTORIAL.csv'}[a.stage]
    if (OUT/final).exists():raise FileExistsError('Choose a fresh --out-dir; completed stage is immutable')
    OUT.mkdir(parents=True,exist_ok=True)
    {'gen':generate,'train':train,'eval':evaluate}[a.stage]()
