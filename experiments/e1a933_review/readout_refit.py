#!/usr/bin/env python3
"""O04: frozen encoder head reanalysis. Old exposed bank; no fresh confirmation."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"
import argparse, hashlib, json, sys, subprocess, itertools
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/f095_campaign"))
from u04_refit_heads import arm_checkpoint, load_model, preprocess, frozen_features, backbone_sha, threshold_certificate, ARMS, SEEDS

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def objective(theta, z, y, w, lam, rank_pairs=None):
    score = z @ theta[:-1] + theta[-1]
    value = np.dot(w, np.logaddexp(0, score) - y * score) + lam * np.dot(theta[:-1], theta[:-1]) / 2
    err = w * (expit(score) - y)
    grad = np.r_[z.T @ err + lam * theta[:-1], err.sum()]
    if rank_pairs is not None:
        dz = z[rank_pairs[:, 0]] - z[rank_pairs[:, 1]]
        delta = dz @ theta[:-1]
        value += .2 * np.logaddexp(0, -delta).mean()
        grad[:-1] += .2 * (dz.T @ (expit(delta) - 1)) / len(delta)
    return value, grad

def convex(z, y, w, lam, pairs=None):
    trace = []
    def fun(t):
        v,g = objective(t,z,y,w,lam,pairs)
        trace.append(float(v))
        return v,g
    r = minimize(fun, np.zeros(z.shape[1]+1), jac=True, method="L-BFGS-B",
                 options={"maxiter":1000,"gtol":1e-7,"ftol":1e-12,"maxls":40})
    return r.x, dict(success=bool(r.success),message=str(r.message),nit=int(r.nit),
        nfev=int(r.nfev),grad_inf=float(np.max(np.abs(r.jac))),objective=float(r.fun),trace=trace)

def bce(score,y,w):
    return float(np.dot(w,np.logaddexp(0,score)-y*score))

def mlp_fit(z,y,w,zv,yv,wv,width,seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(z.shape[1],width),torch.nn.ReLU(),torch.nn.Linear(width,1))
    opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=1e-4)
    zz,yy,ww=[torch.as_tensor(a,dtype=torch.float32) for a in (z,y,w)]
    zvv=torch.as_tensor(zv,dtype=torch.float32)
    best=float("inf"); state=None; best_ep=0
    trace=[]
    for epoch in range(300):
        opt.zero_grad(); loss=(torch.nn.functional.binary_cross_entropy_with_logits(net(zz).flatten(),yy,reduction="none")*ww).sum()
        loss.backward(); opt.step()
        if (epoch+1)%10==0:
            with torch.no_grad(): v=bce(net(zvv).flatten().numpy(),yv,wv)
            trace.append([epoch+1,float(loss),v])
            if v<best:
                best=v; best_ep=epoch+1; state={k:v.detach().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(state)
    return net,dict(dev_bce=best,best_epoch=best_ep,width=width,trace=trace,convergence="nonconvex; no global convergence claim")

def flags(scores,labels,t=0):
    lo=np.where(labels==0,scores,-np.inf).max(1); hi=np.where(labels==1,scores,np.inf).min(1)
    cert=threshold_certificate(scores,labels,t)
    jt=cert.diagnostic_optimal_threshold
    return dict(S=(lo<hi).astype(float),J=((lo<=t)&(t<hi)).astype(float),
        J_star=((lo<=jt)&(jt<hi)).astype(float) if jt is not None else np.zeros(len(lo))), cert

def paired(a,b,parents):
    delta=np.asarray(b)-np.asarray(a)
    unique=np.unique(parents)
    vals=np.array([delta[parents==p].mean() for p in unique])
    rng=np.random.default_rng(904)
    boots=vals[rng.integers(len(vals),size=(2000,len(vals)))].mean(1)
    return dict(quartet_mean=float(delta.mean()),parent_equal_mean=float(vals.mean()),
        parent_cluster_ci95=np.quantile(boots,[.025,.975]).tolist(),n_parent=len(vals))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS));a=ap.parse_args()
    out=a.output; out.mkdir(parents=True,exist_ok=False)
    tr=ROOT/"artifacts/discovery_campaign/scenes/train_101/scenes.npz"; mine=ROOT/"artifacts/discovery_campaign/r04b_s11/mined.npz"; bank=ROOT/"artifacts/p123_upgrade/bank/bank_dev512.npz"
    d=np.load(tr); X=d["positions"]; y=d["labels"].astype(float)
    m=np.load(mine,allow_pickle=True); ii=m["meta"][:,0].astype(int); yf=m["meta"][:,1].astype(float)
    xf=X[ii]+np.stack([m[f"edit_{k}"] for k in range(len(ii))])
    rng=np.random.default_rng(4404); perm=rng.permutation(len(X)); fit=np.zeros(len(X),bool); fit[perm[:int(.8*len(X))]]=True
    b=np.load(bank,allow_pickle=True); meta=json.loads(str(b["Qmeta"])); group={}
    for i,row in enumerate(meta):group.setdefault(row["qid"],{})[row["ptype"]]=(i,row)
    xe=[]; labels=[]; parents=[]; qids=[]
    for qid,ps in sorted(group.items()):
        if set(ps)!={"A","B"}:raise ValueError("incomplete quartet")
        ia,ma=ps["A"];ib,mb=ps["B"]
        np.testing.assert_allclose(b["Qx"][ia]+b["Qe"][ia], b["Qx"][ib]+b["Qe"][ib],atol=1e-6)
        xe.extend([b["Qx"][ia],b["Qx"][ib],b["Qx"][ia]+b["Qe"][ia]])
        labels.append([ma["yA"],ma["yB"],ma["yAB"]]);parents.append(ma["parent"]);qids.append(qid)
    xe=np.array(xe);labels=np.array(labels);parents=np.array(parents)
    summary=dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        command=sys.argv,script_sha256=sha(__file__),inputs={str(p.relative_to(ROOT)):sha(p) for p in (tr,mine,bank)},
        status="REANALYSIS_EXPOSED_BANK", n_quartets=len(labels),n_parents=len(np.unique(parents)),
        criterion="A/B/AB joint event (same legacy U04 event, not four-state correctness)",
        split=dict(fit_parent_indices=np.flatnonzero(fit).tolist(),head_dev_parent_indices=np.flatnonzero(~fit).tolist(),
        caveat="head-dev parents excluded from head fit but seen by original frozen encoder; bank/encoder lineage is separately audited"),
        recipe=dict(linear_l2_candidates=[.0001,.001,.01],mlp_width_candidates=[16,64],mlp_epochs=300,
        mlp_selection="head-dev BCE every 10 epochs; equal budgets across regimes",normalization="clean fit parents only; shared across static/flip; no eval moments",
        ranking="positive-v-negative different fit parent pairs; 0.2 pair logistic + same BCE; different supervision contract"),rows={},paired={})
    np.savez_compressed(out/"sample_contract.npz",labels=labels,parents=parents,qids=qids,fit=fit)
    allflags={}
    for seed in a.seeds:
      for arm in ARMS:
        tag=f"{arm}_s{seed}";od=out/tag;od.mkdir()
        mp=arm_checkpoint(arm,seed);model,stats=load_model(mp)
        for p in model.parameters():p.requires_grad=False
        bs=backbone_sha(model)
        zc=frozen_features(model,preprocess(X,stats)).numpy().astype(float)
        zf=frozen_features(model,preprocess(xf,stats)).numpy().astype(float)
        ze=frozen_features(model,preprocess(xe,stats)).numpy().astype(float)
        mu=zc[fit].mean(0);sd=zc[fit].std(0);sd=np.maximum(sd,1e-6)
        zc=(zc-mu)/sd;zf=(zf-mu)/sd;ze=(ze-mu)/sd
        np.savez_compressed(od/"features.npz",clean=zc,flip=zf,evaluation=ze,mu=mu,sd=sd)
        rows={};af={};logs={}
        for regime in ("static","flip"):
          z=np.r_[zc[fit],zf[fit[ii]]] if regime=="flip" else zc[fit]
          yy=np.r_[y[fit],yf[fit[ii]]] if regime=="flip" else y[fit]
          par=np.r_[np.flatnonzero(fit),ii[fit[ii]]] if regime=="flip" else np.flatnonzero(fit)
          zv=np.r_[zc[~fit],zf[~fit[ii]]] if regime=="flip" else zc[~fit]
          yv=np.r_[y[~fit],yf[~fit[ii]]] if regime=="flip" else y[~fit]
          w=np.r_[np.ones(fit.sum())/fit.sum()/2,np.ones(fit[ii].sum())/fit[ii].sum()/2] if regime=="flip" else np.ones(len(z))/len(z)
          wv=np.r_[np.ones((~fit).sum())/(~fit).sum()/2,np.ones((~fit[ii]).sum())/(~fit[ii]).sum()/2] if regime=="flip" else np.ones(len(zv))/len(zv)
          pair_rng=np.random.default_rng(seed);pos=np.flatnonzero(yy==1);neg=np.flatnonzero(yy==0)
          pairs=np.c_[pair_rng.choice(pos,4096),pair_rng.choice(neg,4096)];pairs=pairs[par[pairs[:,0]]!=par[pairs[:,1]]]
          for family in ("logistic","ranking","mlp"):
            name=f"{family}_{regime}";candidates=[]
            if family in ("logistic","ranking"):
              for lam in (.0001,.001,.01):
                theta,log=convex(z,yy,w,lam,pairs if family=="ranking" else None)
                log.update(l2=lam,dev_bce=bce(zv@theta[:-1]+theta[-1],yv,wv));candidates.append((log["dev_bce"],theta,log))
              _,theta,chosen=min(candidates,key=lambda v:v[0]);score=(ze@theta[:-1]+theta[-1]).reshape(-1,3)
              dv=zv@theta[:-1]+theta[-1];np.save(od/f"{name}_theta.npy",theta)
            else:
              for width in (16,64):
                net,log=mlp_fit(z,yy,w,zv,yv,wv,width,1000+SEEDS.index(seed));candidates.append((log["dev_bce"],net,log))
              _,net,chosen=min(candidates,key=lambda v:v[0])
              with torch.no_grad():
                score=net(torch.as_tensor(ze,dtype=torch.float32)).flatten().numpy().reshape(-1,3)
                dv=net(torch.as_tensor(zv,dtype=torch.float32)).flatten().numpy()
              torch.save(net.state_dict(),od/f"{name}_head.pt")
            # Threshold selected solely for head-dev single-state weighted accuracy.
            ts=np.unique(np.r_[np.nextafter(dv.min(),-np.inf),dv])
            acc=np.array([np.dot(wv,(dv>t)==yv) for t in ts]);dt=float(ts[int(np.argmax(acc))])
            ff,cert=flags(score,labels);af[name]=ff
            jc=((score>dt)==labels).all(1)
            rows[name]=dict(S=cert.S_local_separable,J_star=cert.J_star_global_oracle,J=cert.J_at_threshold,
                J_head_dev_threshold=float(jc.mean()),head_dev_threshold=dt,oracle_threshold=cert.diagnostic_optimal_threshold,
                ordering_failure=cert.ordering_failure,global_incompatibility=cert.global_incompatibility,operating_point_gap=cert.operating_point_gap,
                selected={k:v for k,v in chosen.items() if k!="trace"})
            np.savez_compressed(od/f"{name}_scores.npz",scores=score,labels=labels,parents=parents,qids=qids,**ff,J_head_dev=jc)
            logs[name]=[c[2] for c in candidates]
            print(tag,name,{k:round(rows[name][k],4) for k in ("S","J_star","J","J_head_dev_threshold")},flush=True)
        assert backbone_sha(model)==bs
        (od/"solver_logs.json").write_text(json.dumps(logs,indent=2))
        summary["rows"][tag]=dict(encoder_checkpoint=str(mp.relative_to(ROOT)),encoder_sha256=sha(mp/"model.pt"),frozen_backbone_sha256=bs,results=rows)
        allflags[tag]=af
        for a1,a2 in itertools.combinations(af,2):
            summary["paired"][f"{tag}:{a2}-{a1}"]={metric:paired(af[a1][metric],af[a2][metric],parents) for metric in ("S","J_star","J")}
        (out/"summary.json").write_text(json.dumps(summary,indent=2))
    for seed in a.seeds:
      for ar1,ar2 in itertools.combinations(ARMS,2):
       for name in allflags[f"{ar1}_s{seed}"]:
        summary["paired"][f"s{seed}/{name}:{ar2}-{ar1}"]={m:paired(allflags[f"{ar1}_s{seed}"][name][m],allflags[f"{ar2}_s{seed}"][name][m],parents) for m in ("S","J_star","J")}
    summary["limitations"]=["Bank reused, not an independent confirmation", "J_star uses evaluation truth and is not deployable accuracy", "paired J_star bootstrap holds sample-selected oracle threshold fixed; diagnostic descriptive interval only", "head-dev split cannot undo frozen-encoder training exposure", "new task/model intervention prediction NOT_RUN", "D10 downstream echo NOT_RUN", "two MLP widths cannot characterize all nonlinear readouts"]
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
if __name__=="__main__":main()
