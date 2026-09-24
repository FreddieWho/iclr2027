"""N03: matched relation bottleneck versus capacity-identical generic bottleneck.
New runs only. No changes to existing CUDA matrix or CPU frozen source.
"""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import vision_cuda_protocol as shared
from vision_cuda_protocol import target_orbit,normalize_orbit,matched_mse,digest,metrics,logits,H_ENDPOINTS
DEVICE=torch.device('cpu')
def prep(images,res):
    shared.DEVICE=DEVICE
    return shared.prep(images,res)

pairs=list(zip(*np.triu_indices(4,1)))
REL_PERMS=np.array([[pairs.index(tuple(sorted((int(p[i]),int(p[j]))))) for i,j in pairs]+[6+int(p[i]) for i in range(4)] for p in H_ENDPOINTS])
class BottleneckNet(nn.Module):
    def __init__(self):
        super().__init__();self.base=resnet18(weights='IMAGENET1K_V1');self.base.fc=nn.Identity()
        with torch.random.fork_rng():
            torch.manual_seed(torch.initial_seed()+200000);self.aux=nn.Linear(512,10)
        self.relation=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,1))
        self.register_buffer('rel_perms',torch.tensor(REL_PERMS,dtype=torch.long))
    def classify(self,geometry):
        return self.relation(geometry[:,self.rel_perms]).mean(1).squeeze(-1)
    def forward(self,x):
        geometry=self.aux(self.base(x));return self.classify(geometry),geometry

def oracle_diagnostic(net,data,stats,out,res,batch):
    # Diagnostic only: frozen SAME relation head, no test fitting or retraining.
    coords=data['quartet_coords'].reshape(-1,4,2)
    target,_=normalize_orbit(target_orbit(coords),stats)
    net.eval();truth=[]
    with torch.no_grad():
        for k in range(0,len(target),batch):
            truth.append(net.classify(torch.tensor(target[k:k+batch,0],dtype=torch.float32,device=DEVICE)).cpu().numpy())
    truth=np.concatenate(truth).reshape(-1,4)
    np.savez_compressed(out/'oracle_replacement_DIAGNOSTIC.npz',logits=truth,labels=data['quartet_labels'],parent=data['quartet_parents'])
    result=metrics(truth,data['quartet_labels']);result['scope']='True geometry replaces predicted geometry in the frozen same head. Diagnostic, never image-only deployment performance; meaningful semantic replacement only for geometry_bottleneck.'
    (out/'oracle_replacement_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2))

def train_arm(data,out,arm,seed,epochs=20,res=224,batch=32):
    out.mkdir(exist_ok=False)
    auxmode=arm=='geometry_bottleneck'
    torch.manual_seed(seed)
    net=BottleneckNet()
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
            torch.manual_seed(seed);probe=BottleneckNet().to(DEVICE);probe.eval()
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
            aux=matched_mse(a,target[ix]) if arm=='geometry_bottleneck' else F.mse_loss(a,target[ix,0]) if arm=='ordered' else torch.tensor(0.,device=DEVICE)
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
        if True:
            preds=[]
            with torch.no_grad():
                for k in range(0,len(devtarget),batch):preds.append(net(prep(data['dev_images'][k:k+batch],res))[1])
            ap=torch.cat(preds);auxdiag={'dev_ordered_mse':float(F.mse_loss(ap,devtarget[:,0])),'dev_matched_mse':float(matched_mse(ap,devtarget))}
        row={'epoch':ep+1,'train_BCE':sumbce/len(indices),'train_aux':sumaux/len(indices),'dev_BCE':devbce,'gradient_norms':gn,'seconds':time.monotonic()-started,**auxdiag};history.append(row)
        if devbce<best:best=devbce;beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};bestep=ep+1
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps({'arm':arm,'seed':seed,**row}),flush=True)
    net.load_state_dict(beststate);torch.save({'state':beststate,'arm':arm,'seed':seed,'res':res,'best_epoch':bestep,'target_stats':stats,'architecture':'ResNet18->10d->H-invariant MLP64->binary; no RGB bypass','backend':str(DEVICE),'dtype':'FP32','observation_protocol':'same-color canonical endpoint rendering, native64 bilinear224'},out/'model.pt')
    testz=logits(net,data['test_images'],res,batch)
    np.savez_compressed(out/'test_single_predictions.npz',logits=testz,
                        labels=data['test_labels'],parent=data['test_parents'],clean=data['test_clean'])
    if True:
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
    oracle_diagnostic(net,data,stats,out,res,batch)
    return result
