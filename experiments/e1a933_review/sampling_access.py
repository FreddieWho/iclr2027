#!/usr/bin/env python3
"""N02 frozen-checkpoint diagnostic; old-bank reanalysis, not confirmation."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/last15h'))
sys.path.insert(0, str(ROOT/'experiments/last15h/shared'))
from n08_visual import render

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def metric(z, y):
    ok=(z>0)==y
    atom=ok[:,1]&ok[:,2]; joint=atom&ok[:,3]
    return dict(n=len(y), static=float(ok[:,0].mean()), A=float(ok[:,1].mean()), B=float(ok[:,2].mean()), atomic=float(atom.mean()), AB=float(ok[:,3].mean()), J=float(joint.mean()), CCM_denom=int(atom.sum()), CCM=float(joint.sum()/atom.sum()) if atom.sum() else None)
def paired(a,b,y,groups):
    oa=(a>0)==y; ob=(b>0)==y
    ja=oa[:,1:].all(1); jb=ob[:,1:].all(1)
    d=jb.astype(float)-ja
    u=np.unique(groups); vals=np.array([d[groups==g].mean() for g in u])
    rng=np.random.default_rng(2924)
    boots=vals[rng.integers(len(vals),size=(4000,len(vals)))].mean(1)
    return dict(delta_J=float(d.mean()), parent_mean_delta_J=float(vals.mean()), parent_bootstrap_95pct=np.quantile(boots,[.025,.975]).tolist(), n_unique_parent=len(u), gains=int((~ja&jb).sum()), losses=int((ja&~jb).sum()), baseline_110=int((oa[:,1]&oa[:,2]&~oa[:,3]).sum()), transition_110_to_111=int((oa[:,1]&oa[:,2]&~oa[:,3]&jb).sum()), baseline_111=int(ja.sum()), degradation_111=int((ja&~jb).sum()), changed_state_predictions=int(((a>0)!=(b>0)).sum()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=64);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=803);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    a.out.mkdir(parents=True,exist_ok=False)
    bank=ROOT/'artifacts/next6_ef0f7a3/u01/quartets_eval.npz'; data=np.load(bank)
    eligible=[]; images=[]; qs=[]
    for i in range(len(data['meta'])):
        q=data[f'q{i}'].reshape(3,4,2); x,ea,eb=q
        im=np.stack([render(s,np.random.default_rng(50000+i)) for s in (x,x+ea,x+eb,x+ea+eb)])
        # All phase shifts retain every foreground pixel. Labels do not change.
        fg=np.max(np.abs(im-im[:,:,:1,:1]),axis=1)>1e-6
        if fg[:,-1,:].any() or fg[:,:,-1].any(): continue
        eligible.append(i);images.append(im);qs.append(q)
    pick=np.random.default_rng(2924).permutation(len(eligible))[:a.n]
    ids=np.array(eligible)[pick]; base=np.stack(images)[pick]; coords=np.stack(qs)[pick]; y=data['meta'][ids]
    groups=np.array([ah(x[0]) for x in coords]); n=len(ids)
    np.savez_compressed(a.out/'samples.npz',bank_index=ids,coords=coords,labels=y,parent_hash=groups,base_images=base)
    # AB label contradictions in identical raster are reported, never silently removed.
    contradictions=[int(ids[i]) for i in range(n) if any(np.array_equal(base[i,j],base[i,k]) and y[i,j]!=y[i,k] for j in range(4) for k in range(j))]
    mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1);std=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)
    manifests=[]; scores={}; preds={}; pairs={}; compute={}; hashes={}
    ckpts=[('train64_pretrained',f'resnet_s{a.seed}/pretrained'),('train64_random',f'resnet_s{a.seed}/random'),('train224_pretrained',f'resnet224_s{a.seed}/pretrained'),('train224_random',f'resnet224x_s{a.seed}/random')]
    phases=[(0,0),(1,0),(0,1),(1,1)]
    tensors={}
    for dx,dy in phases:
        shifted=np.broadcast_to(base[:,:,:,:1,:1],base.shape).copy()
        shifted[:,:,:,dy:,dx:]=base[:,:,:,:64-dy,:64-dx]
        for res in (64,224):
            t=torch.from_numpy(shifted.reshape(-1,3,64,64))
            if res==224:t=torch.nn.functional.interpolate(t,(224,224),mode='bilinear',align_corners=False)
            key=f'input{res}_phase{dx}{dy}';hashes[key]=ah(t.numpy());tensors[key]=(t-mean)/std
    for name,rel in ckpts:
        cp=ROOT/'artifacts/f095_campaign/D04'/rel/'model.pt';c=torch.load(cp,map_location='cpu',weights_only=False)
        net=resnet18(weights=None);net.fc=torch.nn.Linear(512,1);net.load_state_dict(c['state']);net.eval()
        manifests.append(dict(name=name,path=str(cp.relative_to(ROOT)),sha256=sha(cp),stored_res=c['res'],stored_mode=c['mode'],stored_seed=c['seed']))
        for key,t in tensors.items():
            start=time.perf_counter();zs=[]
            with torch.inference_mode():
                for b in t.split(32):zs.append(net(b).flatten().numpy())
            sec=time.perf_counter()-start;z=np.concatenate(zs).reshape(n,4);pk=f'{name}__{key}';preds[pk]=z;scores[pk]=metric(z,y);compute[pk]=dict(forward_seconds=sec,n_images=len(t),seconds_per_image=sec/len(t))
            print(pk,scores[pk]['J'],round(sec,2),flush=True)
        pairs[name+'__224_minus_64']=paired(preds[name+'__input64_phase00'],preds[name+'__input224_phase00'],y,groups)
        for res in (64,224):
            for dx,dy in phases[1:]:
                pairs[f'{name}__input{res}_phase{dx}{dy}_minus00']=paired(preds[f'{name}__input{res}_phase00'],preds[f'{name}__input{res}_phase{dx}{dy}'],y,groups)
    np.savez_compressed(a.out/'predictions.npz',bank_index=ids,labels=y,parent_hash=groups,**preds)
    structure={}
    for res in (64,224):
        counts=[];sizes={};hooks=[]
        def conv_hook(m,inp,out):counts.append(int(out.numel()*m.kernel_size[0]*m.kernel_size[1]*m.in_channels/m.groups))
        def linear_hook(m,inp,out):counts.append(int(out.numel()*m.in_features))
        for nm,m in net.named_modules():
            if isinstance(m,torch.nn.Conv2d):hooks.append(m.register_forward_hook(conv_hook))
            if isinstance(m,torch.nn.Linear):hooks.append(m.register_forward_hook(linear_hook))
            if nm in ['conv1','maxpool','layer1','layer2','layer3','layer4']:
                hooks.append(m.register_forward_hook(lambda m,i,o,nm=nm:sizes.update({nm:list(o.shape)})))
        with torch.inference_mode():net(torch.zeros(1,3,res,res))
        for h in hooks:h.remove()
        structure[str(res)]=dict(conv_linear_MACs=sum(counts),feature_maps=sizes,parameters=sum(p.numel() for p in net.parameters()))
    result=dict(status='COMPLETED_BOUNDED_FROZEN_FORWARD_REANALYSIS',full_four_cell_retraining='NOT_RUN',native224_C_and_downsampleD='NOT_RUN_PHYSICAL_LINEWIDTH_CONTRACT_UNVALIDATED',low_stride='NOT_RUN',bank_sha256=sha(bank),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=sha(__file__),renderer_sha256=sha(ROOT/'experiments/last15h/n08_visual.py'),command=' '.join(sys.argv),n_bank=len(data['meta']),n_boundary_eligible=len(eligible),n_selected=n,n_unique_parent=len(np.unique(groups)),identical_raster_label_contradiction_parent=contradictions,selection='rng2924 permutation among no-bottom/right-foreground boundary eligible rows; no model outputs used',phase_contract='translate native64 raster by dx,dy pixels with original background fill, then optionally bilinear upsample; no clipping of any foreground pixel',observation_contract='A native64; B bilinear interpolation of exactly A, align_corners=False. Same color, field, raster information. C/D unrun.',checkpoints=manifests,input_tensor_sha256=hashes,metrics=scores,paired=pairs,forward_budget=compute,structure=structure,samples_sha256=sha(a.out/'samples.npz'),predictions_sha256=sha(a.out/'predictions.npz'),limits=['old bank, exposed analysis; training-parent overlap not resolved here','one seed; correlated rows grouped by base-coordinate hash','static metric is quartet x, not standalone static-only checkpoint','input size switch is distribution shift at fixed checkpoint; cannot identify retraining benefit or aliasing mechanism','no full four-cell retraining or new parent confirmation; no ImageNet prior exclusion'])
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
