"""Corrected image-only visual experiment; fresh parent splits precede edits."""
import hashlib, json, sys, time, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
for directory in ['experiments/f095_campaign', 'experiments/next6_ef0f7a3', 'experiments/last15h']:
    sys.path.insert(0, str(ROOT / directory))
from d04_vision_train import make_net, prep, render as legacy_render
from u06_relation_distill import target_orbit, normalize_orbit, matched_mse, DistillNet
from u01_quartet import mine_quartets
from paths import atomic_edits, oracle_at
from core.relations import make_relational_scenes

def render(x, rng):
    """Canonical endpoint order removes raster roundoff identity leakage.

    Colors and physical line segments stay fixed. The legacy rasterizer is
    direction-sensitive for a small fraction of scenes; all new arms use this
    single corrected observation protocol, including test and dev.
    """
    x = np.asarray(x).reshape(4, 2)
    ordered = np.array(sorted(map(tuple, x[:2])) + sorted(map(tuple, x[2:])))
    return legacy_render(ordered, rng)

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def canonical(x):
    x = np.asarray(x).reshape(4,2)
    red=sorted(map(tuple,x[:2])); blue=sorted(map(tuple,x[2:]))
    return digest(np.array(red+blue,dtype=np.float64))

def generate(out):
    """Only singleton labels are used in train and dev; quartets are test-only."""
    data={}; manifests={}; allparents={}
    for split, n, seed in [('train',512,941001),('dev',128,941002),('test',512,941003)]:
        scenes=make_relational_scenes(n, seed=seed)
        xs=np.asarray(scenes[0])
        allparents[split]=set(map(canonical,xs))
        imgs=[]; coords=[]; labels=[]; parents=[]; clean=[]
        for pi,x in enumerate(xs):
            y0=oracle_at(x)[0]
            # Exactly the same base and noise for static vs singleton-flip arms.
            for view in range(4):
                coords.append(x); labels.append(y0); parents.append(pi); clean.append(True)
                imgs.append(render(x,np.random.default_rng(seed*10+pi*100+view)))
            count=0
            candidates=[]
            for trial in range(16):
                candidates.extend(atomic_edits(x,np.random.default_rng(seed*1000+pi*20+trial),radius=[.1,.2,.35,.5][trial%4]))
            for e,_ in candidates:
                try: y,m,_=oracle_at(x+e)
                except ValueError: continue
                if y==y0 or m<.02: continue
                r0=render(x,np.random.default_rng(seed*10+pi*100+4+count))
                r1=render(x+e,np.random.default_rng(seed*10+pi*100+4+count))
                if int((np.abs(r1-r0).max(0)>.2).sum())<20: continue
                coords.append(x+e);labels.append(y);parents.append(pi);clean.append(False)
                imgs.append(render(x+e,np.random.default_rng(seed*10+pi*100+4+count)))
                count+=1
                if count==2:break
        data[split+'_images']=np.asarray(imgs,dtype=np.float32)
        data[split+'_coords']=np.asarray(coords)
        data[split+'_labels']=np.asarray(labels,dtype=np.float32)
        data[split+'_parents']=np.asarray(parents)
        data[split+'_clean']=np.asarray(clean)
        manifests[split]={'seed':seed,'parent_count':len(xs),'image_count':len(imgs),'images_sha256':digest(data[split+'_images']), 'coords_sha256':digest(data[split+'_coords'])}
        if split=='test':
            quartets=mine_quartets(xs,np.random.default_rng(941004))
            # Retain at most four quartets per parent in stable miner order.
            qimages=[];qlabels=[];qparents=[];counts={};qcoords=[]
            lookup={canonical(x):i for i,x in enumerate(xs)}
            for x,ea,eb,y0,ya,yb,yab in quartets:
                pi=lookup[canonical(x)]; counts[pi]=counts.get(pi,0)+1
                if counts[pi]>4:continue
                states=[x,x+ea,x+eb,x+ea+eb]
                qimages.append([render(s,np.random.default_rng(941005+pi)) for s in states])
                qlabels.append([y0,ya,yb,yab]);qparents.append(pi);qcoords.append(states)
            data['quartet_images']=np.asarray(qimages,dtype=np.float32)
            data['quartet_labels']=np.asarray(qlabels);data['quartet_parents']=np.asarray(qparents)
            data['quartet_coords']=np.asarray(qcoords)
    assert not allparents['train']&allparents['dev']
    assert not allparents['train']&allparents['test']
    assert not allparents['dev']&allparents['test']
    # Audit the whole train/dev/test render bank, not only unedited parents.
    sets={s:set(digest(x) for x in data[s+'_images']) for s in ['train','dev','test']}
    assert not sets['train']&sets['test'] and not sets['train']&sets['dev']
    manifests['training_edit_recipe']='max2 single flips per parent; 16 fixed draws with radii .1/.2/.35/.5; oracle margin>=.02; >=20 changed pixels; no AB train/dev'
    manifests['observation_protocol']='same-color endpoint coordinate sort, legacy native64 render, bilinear224'
    manifests['split_contract']='physical parents independently generated before images/edits; color-preserving endpoint canonical parent hashes disjoint'
    manifests['quartet_n']=len(data['quartet_labels'])
    manifests['quartet_parent_n']=len(set(data['quartet_parents'].tolist()))
    np.savez_compressed(out/'data.npz',**data)
    (out/'data_manifest.json').write_text(json.dumps(manifests,indent=2))
    return data

def logits(net, images, res=224, batch=32):
    net.eval(); outputs=[]
    with torch.no_grad():
        for i in range(0,len(images),batch):
            y=net(prep(images[i:i+batch],res))
            outputs.append((y[0] if isinstance(y,tuple) else y.squeeze(-1)).numpy())
    return np.concatenate(outputs)

def metrics(z,y):
    ok=(z>0)==y; atoms=ok[:,1]&ok[:,2];joint=atoms&ok[:,3]
    return {'n':len(y),'P':float(ok[:,0].mean()),'A':float(ok[:,1].mean()),'B':float(ok[:,2].mean()),'atomic_joint':float(atoms.mean()),'AB':float(ok[:,3].mean()),'J':float(joint.mean()),'CCM_denominator':int(atoms.sum()),'CCM':float(joint.sum()/atoms.sum()) if atoms.sum() else None}

def train_arm(data,out,arm,seed,epochs=20,res=224,batch=32):
    out.mkdir(exist_ok=False)
    auxmode=arm in ['ordered','matched','shuffled_matched']
    torch.manual_seed(seed)
    net=DistillNet(10) if (auxmode or arm=='pretrained_flip') else make_net('random' if arm.startswith('random') else 'pretrained',seed)
    images=data['train_images'];labels=data['train_labels'];coords=data['train_coords']
    indices=np.arange(len(images))
    if arm.endswith('static'):
        ci=np.flatnonzero(data['train_clean']);indices=np.resize(ci,len(images))
    orbit,stats=normalize_orbit(target_orbit(coords))
    target=torch.tensor(orbit,dtype=torch.float32)
    lambda_aux=0.;calibration=None
    if auxmode:
        # All B/C/D arms use the SAME ordered-target, train-only calibration.
        # No dev or quartet outcomes enter this weight; eval mode avoids BN updates.
        with torch.random.fork_rng():
            torch.manual_seed(seed);probe=DistillNet(10);probe.eval()
            ix=torch.randperm(len(images),generator=torch.Generator().manual_seed(seed+100000))[:batch].numpy()
            cz,ca=probe(prep(images[ix],res))
            cb=F.binary_cross_entropy_with_logits(cz,torch.tensor(labels[ix]))
            cl=F.mse_loss(ca,target[ix,0]);pars=list(probe.base.parameters())
            gb=torch.autograd.grad(cb,pars,retain_graph=True,allow_unused=True)
            ga=torch.autograd.grad(cl,pars,allow_unused=True)
            nb=float(sum(g.square().sum() for g in gb if g is not None).sqrt())
            na=float(sum(g.square().sum() for g in ga if g is not None).sqrt())
            lambda_aux=float(np.clip(.25*nb/max(na,1e-12),1e-4,10.))
            calibration={'initial_BCE_gradient':nb,'initial_ordered_aux_gradient':na,
                         'desired_aux_to_BCE_gradient_ratio':.25,'mode':'eval, first train batch only'}
            del probe
    if arm=='shuffled_matched':
        perm=torch.randperm(len(target),generator=torch.Generator().manual_seed(seed+300000));target=target[perm]
    devorbit,_=normalize_orbit(target_orbit(data['dev_coords']),stats)
    devtarget=torch.tensor(devorbit,dtype=torch.float32)
    optimizer=torch.optim.Adam(net.parameters(),lr=3e-4)
    rng=torch.Generator().manual_seed(seed+100000)
    history=[];best=float('inf');beststate=None
    for ep in range(epochs):
        started=time.monotonic(); net.train();sumbce=sumaux=0.;gn=None
        order=torch.randperm(len(indices),generator=rng).numpy()
        for k in range(0,len(order),batch):
            ix=indices[order[k:k+batch]]; y=torch.tensor(labels[ix])
            optimizer.zero_grad();output=net(prep(images[ix],res))
            z,a=output if isinstance(output,tuple) else (output.squeeze(-1),None)
            bce=F.binary_cross_entropy_with_logits(z,y)
            aux=matched_mse(a,target[ix]) if arm in ['matched','shuffled_matched'] else F.mse_loss(a,target[ix,0]) if arm=='ordered' else torch.tensor(0.)
            if auxmode and k==0:
                params=[p for p in net.base.parameters() if p.requires_grad]
                gb=torch.autograd.grad(bce,params,retain_graph=True,allow_unused=True)
                ga=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
                gn={'BCE':float(sum((g.square().sum() for g in gb if g is not None)).sqrt()),'aux_unweighted':float(sum((g.square().sum() for g in ga if g is not None)).sqrt())}
            loss=bce+lambda_aux*aux;loss.backward();optimizer.step()
            sumbce+=float(bce.detach())*len(ix);sumaux+=float(aux.detach())*len(ix)
        dz=logits(net,data['dev_images'],res,batch)
        devbce=float(F.binary_cross_entropy_with_logits(torch.tensor(dz),torch.tensor(data['dev_labels'])))
        auxdiag={}
        if auxmode or arm=='pretrained_flip':
            preds=[]
            with torch.no_grad():
                for k in range(0,len(devtarget),batch):preds.append(net(prep(data['dev_images'][k:k+batch],res))[1])
            ap=torch.cat(preds);auxdiag={'dev_ordered_mse':float(F.mse_loss(ap,devtarget[:,0])),'dev_matched_mse':float(matched_mse(ap,devtarget))}
        row={'epoch':ep+1,'train_BCE':sumbce/len(indices),'train_aux':sumaux/len(indices),'dev_BCE':devbce,'gradient_norms':gn,'seconds':time.monotonic()-started,**auxdiag};history.append(row)
        if devbce<best:best=devbce;beststate={k:v.detach().clone() for k,v in net.state_dict().items()};bestep=ep+1
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps({'arm':arm,'seed':seed,**row}),flush=True)
    net.load_state_dict(beststate);torch.save({'state':beststate,'arm':arm,'seed':seed,'res':res,'best_epoch':bestep,'target_stats':stats,'observation_protocol':'same-color canonical endpoint rendering, native64 bilinear224'},out/'model.pt')
    testz=logits(net,data['test_images'],res,batch)
    np.savez_compressed(out/'test_single_predictions.npz',logits=testz,
                        labels=data['test_labels'],parent=data['test_parents'],clean=data['test_clean'])
    if auxmode or arm=='pretrained_flip':
        preds=[]
        with torch.no_grad():
            for k in range(0,len(devtarget),batch):
                preds.append(net(prep(data['dev_images'][k:k+batch],res))[1])
        ap=torch.cat(preds)
        np.savez_compressed(out/'dev_aux_predictions.npz',prediction=ap.numpy(),
                            target_orbit=devtarget.numpy(),parent=data['dev_parents'],
                            ordered_error=((ap-devtarget[:,0])**2).mean(-1).numpy(),
                            matched_error=matched_mse(ap,devtarget,reduction='none').numpy(),
                            normalization_mean=stats[0],normalization_std=stats[1])
    qi=data['quartet_images'];z=logits(net,qi.reshape(-1,*qi.shape[2:]),res,batch).reshape(-1,4)
    np.savez_compressed(out/'predictions.npz',logits=z,labels=data['quartet_labels'],parent=data['quartet_parents'])
    result=metrics(z,data['quartet_labels']);result.update({'best_epoch':bestep,'selection':'single-dev BCE only','checkpoint_sha256':hashlib.sha256((out/'model.pt').read_bytes()).hexdigest(),'train_exposure_indices_sha256':digest(indices),'train_images_sha256':digest(images),'lambda_aux':lambda_aux,'lambda_status':'same train-only ordered-target initial gradient calibration for B/C/D; training gradient ratios may drift','lambda_calibration':calibration})
    (out/'result.json').write_text(json.dumps(result,indent=2))
    return result

def analyze(out):
    """Paired same-state contrasts; cluster uncertainty is by parent, not image."""
    comparisons={}
    for candidate,baseline in [('pretrained_flip','pretrained_static'),
                                ('random_flip','random_static'),('matched','ordered'),
                                ('matched','shuffled_matched'),('matched','pretrained_flip')]:
        rows=[]
        for seed in [803,805,806]:
            a=np.load(out/f'{baseline}_s{seed}'/'predictions.npz')
            b=np.load(out/f'{candidate}_s{seed}'/'predictions.npz')
            assert np.array_equal(a['labels'],b['labels']) and np.array_equal(a['parent'],b['parent'])
            ao=(a['logits']>0)==a['labels'];bo=(b['logits']>0)==b['labels']
            aj=ao[:,1:].all(1);bj=bo[:,1:].all(1);h=ao[:,1]&ao[:,2]&~ao[:,3]
            parent=a['parent'];unique=np.unique(parent);delta=bj.astype(float)-aj.astype(float)
            rng=np.random.default_rng(951000+seed);boot=[]
            for _ in range(2000):
                sampled=rng.choice(unique,len(unique),replace=True)
                ix=np.concatenate([np.flatnonzero(parent==p) for p in sampled]);boot.append(delta[ix].mean())
            rows.append({'seed':seed,'J_difference':float(delta.mean()),'parent_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
                         'baseline110_n':int(h.sum()),'full_repair_n':int((bj&h).sum()),
                         'migration_n':int((bo[:,3]&~(bo[:,1]&bo[:,2])&h).sum()),
                         'baseline111_n':int(aj.sum()),'111_regression_n':int((aj&~bj).sum()),
                         'A_regression_n':int((ao[:,1]&~bo[:,1]).sum()),'B_regression_n':int((ao[:,2]&~bo[:,2]).sum())})
        comparisons[candidate+' minus '+baseline]={'seeds':rows,'mean_J_difference':float(np.mean([r['J_difference'] for r in rows])),
            'uncertainty_note':'Per-seed intervals resample physical test parents; three seeds remain separate model replicates. No image/labeling pseudo-replication.'}
    (out/'paired_analysis.json').write_text(json.dumps(comparisons,indent=2))
    return comparisons
