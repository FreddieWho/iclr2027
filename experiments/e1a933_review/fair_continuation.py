"""R07: matched clean-start continuation, immutable checkpoints and explicit order."""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/f095_campaign'))
sys.path.insert(0,str(ROOT/'experiments/discovery_campaign'))
sys.path.insert(0,str(ROOT/'docs/iclr2027_discovery_campaign_20260917'))
from d08_fairfight import train_tensors,clean_ckpt,eval_bank
from common import load_model,preprocess
from coord_mlp import CoordMLP
from core.relations import segment_relation
OUT=ROOT/'artifacts/e1a933_review/fair'
torch.set_num_threads(4)

def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def state_metrics(before,after,labels,parents):
    old=before==labels; new=after==labels
    transition=np.zeros((8,8),int)
    bit=np.array([4,2,1]); a=old@bit;b=new@bit
    np.add.at(transition,(a,b),1)
    at=old[:,:2];reg=(at&~new[:,:2]).sum()
    # Parent-cluster bootstrap of the complete old-correct regression ratio.
    unique=np.unique(parents);rng=np.random.default_rng(924)
    numer=np.array([(at&~new[:,:2])[parents==p].sum() for p in unique])
    denom=np.array([at[parents==p].sum() for p in unique])
    draws=rng.integers(0,len(unique),(2000,len(unique)))
    ds=denom[draws].sum(1);rat=numer[draws].sum(1)/np.maximum(ds,1)
    return {'J':float(new.all(1).mean()),'A':float(new[:,0].mean()),'B':float(new[:,1].mean()),'AB':float(new[:,2].mean()),
            'old_correct_atomic_n':int(at.sum()),'old_correct_atomic_regressed':int(reg),
            'old_correct_atomic_regression':float(reg/max(at.sum(),1)),
            'old_correct_atomic_regression_CI_parent_bootstrap':np.quantile(rat,[.025,.975]).tolist(),
            'transition_rows_old_cols_new_000_to_111':transition.tolist()}

def main():
    global OUT
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);args=ap.parse_args()
    OUT=args.out
    if (OUT/'CONTINUATION_RESULTS.json').exists():raise FileExistsError('Immutable completed/partial run exists; use --out NEW_DIRECTORY')
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();epochs=300
    train=np.load(ROOT/'artifacts/discovery_campaign/scenes/train_101/scenes.npz')
    X=train['positions'].astype(np.float32);y=train['labels']
    mine=np.load(ROOT/'artifacts/discovery_campaign/r04b_s11/mined.npz',allow_pickle=True)
    ii=np.array(mine['meta'][:,0],int);K=len(ii)
    rng=np.random.default_rng(260924)
    preserve=[];preserve_labels=[];attempts=0
    # Model-blind perturbations on same mining parents, capped by that flip's norm.
    for k,i in enumerate(ii):
      cap=float(np.linalg.norm(mine[f'edit_{k}']))
      for j in range(1000):
        e=rng.normal(size=(4,2));e*=cap/(np.linalg.norm(e)+1e-12);z=X[i]+e
        lab=segment_relation(z);attempts+=1
        if lab['label']==y[i] and not lab['ambiguous']:
          preserve.append(z);preserve_labels.append(y[i]);break
      else:raise RuntimeError('no preserve within finite search budget')
    preserve=np.asarray(preserve,np.float32);preserve_labels=np.asarray(preserve_labels,np.float32)
    np.savez_compressed(OUT/'preserve_pool.npz',positions=preserve,labels=preserve_labels,parent_indices=ii)
    b=np.load(ROOT/'artifacts/p123_upgrade/bank/bank_dev512.npz',allow_pickle=True)
    meta=json.loads(str(b['Qmeta']));by={}
    for j,m in enumerate(meta):by.setdefault(m['qid'],[]).append((j,m))
    quads=[(q,ps[0],ps[1]) for q,ps in by.items() if len(ps)==2]
    parents=np.array([a[1]['parent'] for q,a,c in quads])
    held=np.load(ROOT/'artifacts/discovery_campaign/scenes/eval_202/scenes.npz')
    train_hash={hashlib.sha256(z.tobytes()).hexdigest() for z in X}
    overlap=sum(hashlib.sha256(z.astype(np.float32).tobytes()).hexdigest() in train_hash for z in held['positions'])
    assert overlap==0
    result={'epochs':epochs,'lr':.01,'base_coefficient':1.,'augmentation_mean_coefficient':1.,
            'added_examples_each_arm':K,'preserve_search_attempts':attempts,
            'bank':'dev512 from eval_202, reused development evaluation, not sealed confirmation',
            'unseen_parent_coordinate_overlap_with_train':overlap,'quartets':len(quads),'parents':len(np.unique(parents)),
            'forward_contract':'base N + augmentation K + identical old-correct replay count every arm; replay coefficient zero for plain/balanced/scratch',
            'arms':{}}
    for enc in ['raw','rel']:
      Xc,yc,Xf,yf,mu,sd,_,_=train_tensors(enc)
      for seed in [11,23,47]:
        clean,stats=load_model(clean_ckpt(enc,seed))
        # Exact loaded checkpoint preprocessing, validated against legacy helper.
        assert torch.allclose(Xc,preprocess(X,stats),atol=1e-5)
        Xp=preprocess(preserve,stats);yp=torch.from_numpy(preserve_labels)
        with torch.no_grad():base_logits=clean(Xc);keep=(base_logits>0)==yc.bool();soft=base_logits.sigmoid()
        before,labels=eval_bank(clean,stats,quads,b['Qx'],b['Qe']);before=before>0
        g=np.random.default_rng(seed+260924);nf=K//2;fi=np.sort(g.choice(K,nf,replace=False));pi=np.sort(g.choice(K,K-nf,replace=False))
        for method in ['plain','protection','hard_replay','preserve_flip_balanced','scratch']:
          torch.manual_seed(seed)
          model=CoordMLP(64,32,in_dim=Xc.shape[1]) if method=='scratch' else copy.deepcopy(clean)
          initial=state_hash(model);opt=torch.optim.Adam(model.parameters(),lr=.01);bce=nn.BCEWithLogitsLoss()
          Xa,ya=(torch.cat([Xf[fi],Xp[pi]]),torch.cat([yf[fi],yp[pi]])) if method=='preserve_flip_balanced' else (Xf,yf)
          # Full batch: this exact array order repeats each epoch; no shuffled minibatch.
          ids=(['flip:'+str(i) for i in fi]+['preserve:'+str(i) for i in pi]) if method=='preserve_flip_balanced' else ['flip:'+str(i) for i in range(K)]
          order={'base':list(range(len(Xc))),'augmentation':ids,'replay':torch.nonzero(keep).reshape(-1).tolist()}
          order_hash=hashlib.sha256(json.dumps(order,sort_keys=True).encode()).hexdigest();hist=[]
          for ep in range(epochs):
            opt.zero_grad();model.train();lb=bce(model(Xc),yc);lf=bce(model(Xa),ya)
            replay=model(Xc[keep]);target=soft[keep] if method=='protection' else yc[keep]
            lp=bce(replay,target);coef=1. if method in ['protection','hard_replay'] else 0.
            loss=lb+lf+coef*lp;loss.backward();opt.step()
            hist.append([float(lb.detach()),float(lf.detach()),float(lp.detach())])
          model.eval();lg,lab=eval_bank(model,stats,quads,b['Qx'],b['Qe'])
          metrics=state_metrics(before,lg>0,lab,parents)
          with torch.no_grad():train_reg=float(((model(Xc[keep])>0)!=yc[keep].bool()).float().mean())
          tag=f'{enc}_{method}_s{seed}';od=OUT/tag;od.mkdir(exist_ok=True)
          torch.save({'state':model.state_dict(),'hidden':64,'feat':32,'in_dim':Xc.shape[1],**stats},od/'model.pt')
          (od/'batch_order.json').write_text(json.dumps({'repeated_for_epochs':epochs,'order':order}))
          np.savez_compressed(od/'evaluation.npz',before=before,after=lg>0,logits=lg,labels=lab,parents=parents)
          row={'initial_state_sha256':initial,'clean_state_sha256':state_hash(clean),'final_state_sha256':state_hash(model),
               'clean_checkpoint_sha256':hashlib.sha256((clean_ckpt(enc,seed)/'model.pt').read_bytes()).hexdigest(),
               'batch_order_sha256':order_hash,'train_old_correct_regression':train_reg,'metrics':metrics,'loss_trajectory':hist}
          (od/'receipt.json').write_text(json.dumps(row,indent=2));result['arms'][tag]=row
          print(tag,'J',round(metrics['J'],4),'unseen atom regression',round(metrics['old_correct_atomic_regression'],4),flush=True)
          result['wall_seconds']=time.monotonic()-started;(OUT/'CONTINUATION_RESULTS.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
