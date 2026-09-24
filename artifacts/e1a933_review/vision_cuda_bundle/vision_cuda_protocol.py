"""Independent FP32 CUDA port. CPU running source remains immutable.
Generated from frozen production functions; precomputed data only, no generation.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
DEVICE=torch.device('cpu')  # Runner requires explicit CUDA for actual matrix.
IMNET_MEAN=torch.tensor([.485,.456,.406]).reshape(1,3,1,1)
IMNET_STD=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)
H_ENDPOINTS=np.array([[0,1,2,3],[1,0,2,3],[0,1,3,2],[1,0,3,2]])

def prep(images,res):
    t=torch.from_numpy(np.stack(images).astype(np.float32))
    if res!=64:t=F.interpolate(t,size=(res,res),mode='bilinear',align_corners=False)
    return ((t-IMNET_MEAN)/IMNET_STD).to(DEVICE)

def make_net(mode, seed):
    torch.manual_seed(seed)
    if mode == "pretrained":
        net = resnet18(weights="IMAGENET1K_V1")
    else:
        net = resnet18(weights=None)
    net.fc = nn.Linear(512, 1)
    if mode in {"headonly", "bnadapt"}:
        base = resnet18(weights="IMAGENET1K_V1")
        net = resnet18(weights=None)
        net.load_state_dict(base.state_dict(), strict=False)
        net.fc = nn.Linear(512, 1)
        for n, p in net.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    return net

def rel10(X44):
    X = np.asarray(X44, float).reshape(-1, 4, 2)
    i, j = np.triu_indices(4, k=1)
    pw = np.linalg.norm(X[:, i] - X[:, j], axis=-1)
    cen = X.mean(axis=1, keepdims=True)
    return np.concatenate([pw, np.linalg.norm(X - cen, axis=-1)], axis=-1)

def target_orbit(coords, kind="rel10"):
    """One consistent red/blue endpoint permutation per entire structure."""
    x = np.asarray(coords, float).reshape(-1, 4, 2)
    return np.stack([rel10(x[:, p]) if kind == "rel10" else
                     x[:, p].reshape(len(x), 8) for p in H_ENDPOINTS], axis=1)

def normalize_orbit(orbit, stats=None):
    """Train-only orbit-pooled statistics commute with the endpoint action."""
    if stats is None:
        stats = (orbit.mean(axis=(0, 1)), orbit.std(axis=(0, 1)).clip(1e-8))
    return (orbit - stats[0]) / stats[1], stats

def matched_mse(pred, orbit, reduction="mean"):
    losses = ((pred[:, None, :] - orbit) ** 2).mean(dim=-1).min(dim=1).values
    return losses.mean() if reduction == "mean" else losses

class DistillNet(nn.Module):
    def __init__(self, aux_dim):
        super().__init__()
        self.base = resnet18(weights="IMAGENET1K_V1")
        self.base.fc = nn.Identity()
        self.head = nn.Linear(512, 1)
        # All rel10 arms, including BCE-only, allocate the same auxiliary head.
        with torch.random.fork_rng():
            torch.manual_seed(torch.initial_seed() + 200_000)
            self.aux = nn.Linear(512, aux_dim) if aux_dim else None

    def forward(self, x):
        z = self.base(x)
        out = (self.head(z).squeeze(-1),)
        if self.aux is not None:
            out += (self.aux(z),)
        return out[0] if len(out) == 1 else out

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def logits(net, images, res=224, batch=32):
    net.eval(); outputs=[]
    with torch.no_grad():
        for i in range(0,len(images),batch):
            y=net(prep(images[i:i+batch],res))
            outputs.append((y[0] if isinstance(y,tuple) else y.squeeze(-1)).cpu().numpy())
    return np.concatenate(outputs)

def metrics(z,y):
    ok=(z>0)==y; atoms=ok[:,1]&ok[:,2];joint=atoms&ok[:,3]
    return {'n':len(y),'P':float(ok[:,0].mean()),'A':float(ok[:,1].mean()),'B':float(ok[:,2].mean()),'atomic_joint':float(atoms.mean()),'AB':float(ok[:,3].mean()),'J':float(joint.mean()),'CCM_denominator':int(atoms.sum()),'CCM':float(joint.sum()/atoms.sum()) if atoms.sum() else None}

def train_arm(data,out,arm,seed,epochs=20,res=224,batch=32):
    out.mkdir(exist_ok=False)
    auxmode=arm in ['ordered','matched','shuffled_matched']
    torch.manual_seed(seed)
    net=DistillNet(10) if (auxmode or arm=='pretrained_flip') else make_net('random' if arm.startswith('random') else 'pretrained',seed)
    net=net.to(DEVICE)
    images=data['train_images'];labels=data['train_labels'];coords=data['train_coords']
    indices=np.arange(len(images))
    if arm.endswith('static'):
        ci=np.flatnonzero(data['train_clean']);indices=np.resize(ci,len(images))
    orbit,stats=normalize_orbit(target_orbit(coords))
    target=torch.tensor(orbit,dtype=torch.float32,device=DEVICE)
    lambda_aux=0.;calibration=None
    if auxmode:
        # All B/C/D arms use the SAME ordered-target, train-only calibration.
        # No dev or quartet outcomes enter this weight; eval mode avoids BN updates.
        with torch.random.fork_rng():
            torch.manual_seed(seed);probe=DistillNet(10).to(DEVICE);probe.eval()
            ix=torch.randperm(len(images),generator=torch.Generator().manual_seed(seed+100000))[:batch].cpu().numpy()
            cz,ca=probe(prep(images[ix],res))
            cb=F.binary_cross_entropy_with_logits(cz,torch.tensor(labels[ix],device=DEVICE))
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
    devtarget=torch.tensor(devorbit,dtype=torch.float32,device=DEVICE)
    optimizer=torch.optim.Adam(net.parameters(),lr=3e-4)
    rng=torch.Generator().manual_seed(seed+100000)
    history=[];best=float('inf');beststate=None
    for ep in range(epochs):
        started=time.monotonic(); net.train();sumbce=sumaux=0.;gn=None
        order=torch.randperm(len(indices),generator=rng).cpu().numpy()
        for k in range(0,len(order),batch):
            ix=indices[order[k:k+batch]]; y=torch.tensor(labels[ix],device=DEVICE)
            optimizer.zero_grad();output=net(prep(images[ix],res))
            z,a=output if isinstance(output,tuple) else (output.squeeze(-1),None)
            bce=F.binary_cross_entropy_with_logits(z,y)
            aux=matched_mse(a,target[ix]) if arm in ['matched','shuffled_matched'] else F.mse_loss(a,target[ix,0]) if arm=='ordered' else torch.tensor(0.,device=DEVICE)
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
        if devbce<best:best=devbce;beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};bestep=ep+1
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps({'arm':arm,'seed':seed,**row}),flush=True)
    net.load_state_dict(beststate);torch.save({'state':beststate,'arm':arm,'seed':seed,'res':res,'best_epoch':bestep,'target_stats':stats,'backend':str(DEVICE),'dtype':'FP32','observation_protocol':'same-color canonical endpoint rendering, native64 bilinear224'},out/'model.pt')
    testz=logits(net,data['test_images'],res,batch)
    np.savez_compressed(out/'test_single_predictions.npz',logits=testz,
                        labels=data['test_labels'],parent=data['test_parents'],clean=data['test_clean'])
    if auxmode or arm=='pretrained_flip':
        preds=[]
        with torch.no_grad():
            for k in range(0,len(devtarget),batch):
                preds.append(net(prep(data['dev_images'][k:k+batch],res))[1])
        ap=torch.cat(preds)
        np.savez_compressed(out/'dev_aux_predictions.npz',prediction=ap.cpu().numpy(),
                            target_orbit=devtarget.cpu().numpy(),parent=data['dev_parents'],
                            ordered_error=((ap-devtarget[:,0])**2).mean(-1).cpu().numpy(),
                            matched_error=matched_mse(ap,devtarget,reduction='none').cpu().numpy(),
                            normalization_mean=stats[0],normalization_std=stats[1])
    qi=data['quartet_images'];z=logits(net,qi.reshape(-1,*qi.shape[2:]),res,batch).reshape(-1,4)
    np.savez_compressed(out/'predictions.npz',logits=z,labels=data['quartet_labels'],parent=data['quartet_parents'])
    result=metrics(z,data['quartet_labels']);result.update({'backend':str(DEVICE),'dtype':'FP32','best_epoch':bestep,'selection':'single-dev BCE only','checkpoint_sha256':hashlib.sha256((out/'model.pt').read_bytes()).hexdigest(),'train_exposure_indices_sha256':digest(indices),'train_images_sha256':digest(images),'lambda_aux':lambda_aux,'lambda_status':'same train-only ordered-target initial gradient calibration for B/C/D; training gradient ratios may drift','lambda_calibration':calibration})
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
