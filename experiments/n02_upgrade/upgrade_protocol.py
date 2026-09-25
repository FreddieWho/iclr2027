"""N02 continuous-render, fixed-information and early-sampling factorial (portable)."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from wide_protocol import net_for_wide

RADIUS=5/64 # normalized coordinates [-1,1], diameter equals five native64 pixels
SS=4
MEAN=torch.tensor([.485,.456,.406]).reshape(1,3,1,1)
STD=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def canonical_coords(coords):
    x=np.asarray(coords,dtype=np.float32).reshape(-1,4,2).copy()
    for i in range(len(x)):
        x[i,:2]=sorted(map(tuple,x[i,:2]));x[i,2:]=sorted(map(tuple,x[i,2:]))
    return x

def render_native(coords,bg,res,device='cpu',ss=SS):
    """Direct continuous capsules at requested native grid, supersampled per pixel.

    No image resize occurs here. Same physical radius, FOV, colors, occlusion
    order and symmetric pixel integration rule apply at every resolution.
    """
    x=torch.as_tensor(canonical_coords(coords),device=device)
    b=torch.as_tensor(bg,dtype=torch.float32,device=device).reshape(-1,1,1,1)
    axis=-1+(torch.arange(res*ss,device=device,dtype=torch.float32)+.5)*(2/(res*ss))
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    image=b.expand(-1,3,res*ss,res*ss).clone()
    for start,color in [(0,[.9,.1,.1]),(2,[.1,.1,.9])]:
        p=x[:,start];v=x[:,start+1]-p
        dx=xx[None]-p[:,0,None,None];dy=yy[None]-p[:,1,None,None]
        t=((dx*v[:,0,None,None]+dy*v[:,1,None,None])/(v.square().sum(1)[:,None,None].clamp_min(1e-12))).clamp(0,1)
        dist=(dx-t*v[:,0,None,None]).square()+(dy-t*v[:,1,None,None]).square()
        inside=dist<=RADIUS**2
        image=torch.where(inside[:,None],torch.tensor(color,device=device).reshape(1,3,1,1),image)
    return F.avg_pool2d(image,ss)

def net_for(init,stem,seed,weight_file,observation='A'):
    if observation=='W':
        return net_for_wide(init,stem,seed)  # random-only; asserts inside
    torch.manual_seed(seed);net=resnet18(weights=None)
    if init=='pretrained':net.load_state_dict(torch.load(weight_file,map_location='cpu',weights_only=True))
    net.fc=torch.nn.Linear(512,1)
    if stem=='lowstride':net.maxpool=torch.nn.Identity()
    return net

def make_arms():
    arms=[]
    for observation in ['A','B']:
        for stem in ['standard','lowstride']:
            for init in ['pretrained','random']:
                arms.append(dict(observation=observation,stem=stem,init=init,supervision='flip'))
    for stem in ['standard','lowstride']:
        for init in ['random']:
            arms.append(dict(observation='W',stem=stem,init=init,supervision='flip'))
    return arms

def arm_name(arm):return '_'.join(arm[k] for k in ['observation','stem','init','supervision'])
def resolution(obs):return 224 if obs in ['B','C'] else 64

def prepare_cache(data,cache,device='cuda'):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=False)
    result={};maps={}
    for split in ['train','dev','test','quartet']:
        coords=data[split+'_coords'].reshape(-1,4,2);bg=data[split+'_bg'].reshape(-1)
        for obs,res in [('A',64),('C',224)]:
            path=cache/f'{split}_{obs}.npy';arr=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(coords),3,res,res))
            start=time.monotonic()
            for i in range(0,len(coords),8):
                arr[i:i+8]=render_native(coords[i:i+8],bg[i:i+8],res,device).cpu().numpy()
            arr.flush();result[f'{split}_{obs}']={'sha256':filehash(path),'shape':list(arr.shape),'seconds':time.monotonic()-start,'image_sha256':[digest(im) for im in arr]}
            maps[f'{split}_{obs}']=np.load(path,mmap_mode='r')
    return maps,result

def input_batch(maps,split,ix,observation,device):
    source='A' if observation in ['A','B','W'] else 'C'
    x=torch.from_numpy(np.array(maps[f'{split}_{source}'][ix],copy=True)).to(device)
    if observation=='B':x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
    elif observation=='D':x=F.interpolate(x,size=(64,64),mode='area')
    return (x-MEAN.to(device))/STD.to(device)

def logits(net,maps,split,observation,batch,device):
    net.eval();out=[];n=len(maps[f'{split}_A'])
    with torch.inference_mode():
        for i in range(0,n,batch):out.append(net(input_batch(maps,split,slice(i,i+batch),observation,device)).flatten().cpu().numpy())
    return np.concatenate(out)

def metrics(z,y):
    ok=(z>0)==y;atom=ok[:,1]&ok[:,2];joint=atom&ok[:,3]
    cond_success=float(joint.sum()/atom.sum()) if atom.sum() else None
    composition_miss=float((atom&~ok[:,3]).sum()/atom.sum()) if atom.sum() else None
    return dict(metric_schema='n02_upgrade_metrics_v2_ccm_miss',n=len(y),P=float(ok[:,0].mean()),A=float(ok[:,1].mean()),B=float(ok[:,2].mean()),AB=float(ok[:,3].mean()),atomic_joint=float(atom.mean()),J=float(joint.mean()),J4=float((joint&ok[:,0]).mean()),conditional_joint_success=cond_success,composition_miss=composition_miss,CCM=composition_miss,legacy_CCM_success=cond_success,CCM_denominator=int(atom.sum()))

def train_arm(data,maps,out,arm,seed,weight_file,epochs=20,batch=32,device='cuda'):
    out=Path(out);out.mkdir(exist_ok=False);obs=arm['observation']
    net=net_for(arm['init'],arm['stem'],seed,weight_file,observation=obs).to(device)
    labels=data['train_labels'];indices=np.arange(len(labels))
    if arm['supervision']=='static':indices=np.resize(np.flatnonzero(data['train_clean']),len(labels))
    opt=torch.optim.Adam(net.parameters(),lr=3e-4);gen=torch.Generator().manual_seed(seed+100000)
    best=float('inf');history=[];beststate=None
    for ep in range(epochs):
        net.train();order=torch.randperm(len(indices),generator=gen).numpy();loss_sum=0.;start=time.monotonic()
        for i in range(0,len(order),batch):
            ix=indices[order[i:i+batch]];x=input_batch(maps,'train',ix,obs,device);y=torch.as_tensor(labels[ix],device=device)
            opt.zero_grad(set_to_none=True);z=net(x).flatten();loss=F.binary_cross_entropy_with_logits(z,y);loss.backward();opt.step();loss_sum+=float(loss.detach())*len(ix)
        dz=logits(net,maps,'dev',obs,batch,device)
        dev=float(F.binary_cross_entropy_with_logits(torch.tensor(dz),torch.tensor(data['dev_labels'])))
        if dev<best:best=dev;bestep=ep+1;beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        row=dict(epoch=ep+1,train_BCE=loss_sum/len(indices),dev_BCE=dev,seconds=time.monotonic()-start);history.append(row)
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(dict(arm=arm_name(arm),seed=seed,**row)),flush=True)
    net.load_state_dict(beststate)
    torch.save(dict(state=beststate,arm=arm,seed=seed,best_epoch=bestep,epochs=epochs,batch=batch,observation_contract='continuous capsules native64/native224; B=bilinear(A), D=area(C)',selection='single-dev BCE only'),out/'model.pt')
    tz=logits(net,maps,'test',obs,batch,device);qz=logits(net,maps,'quartet',obs,batch,device).reshape(-1,4)
    np.savez_compressed(out/'test_single_predictions.npz',logits=tz,labels=data['test_labels'],parent=data['test_parents'],clean=data['test_clean'])
    np.savez_compressed(out/'predictions.npz',logits=qz,labels=data['quartet_labels'],parent=data['quartet_parents'],small_edit=data['quartet_small_edit'])
    sm_result=None
    if 'smaledit_coords' in data:
        res=resolution(obs);smz_parts=[]
        sc=data['smaledit_coords'].reshape(-1,4,2);sbg=data['smaledit_bg'].reshape(-1)
        net.eval();torch.cuda.empty_cache()
        with torch.inference_mode():
            for i in range(0,len(sc),32):
                ci, bi = sc[i:i+32], sbg[i:i+32]
                parts=[render_native(ci[j:j+8],bi[j:j+8],64,device) for j in range(0,len(ci),8)]
                imgs=torch.cat(parts)
                if obs=='B':imgs=F.interpolate(imgs,size=(224,224),mode='bilinear',align_corners=False)
                xn=(imgs-MEAN.to(device))/STD.to(device)
                smz_parts.append(net(xn).flatten())
                del parts,imgs,xn;torch.cuda.empty_cache()
            smz=torch.cat(smz_parts).cpu().numpy()
        smz=smz.reshape(-1,4)
        np.savez_compressed(out/'smaledit_predictions.npz',logits=smz,labels=data['smaledit_labels'],parent=data['smaledit_parents'])
        sm_result=metrics(smz,data['smaledit_labels']);sm_result.update(n_smaledit=len(smz))
    result=metrics(qz,data['quartet_labels']);result.update(arm=arm,seed=seed,best_epoch=bestep,selection='single-dev BCE only',checkpoint_sha256=filehash(out/'model.pt'),train_exposure_indices_sha256=digest(indices),train_updates=epochs*int(np.ceil(len(indices)/batch)),train_exposure_count=epochs*len(indices),static_single_acc=float(((tz[data['test_clean']]>0)==data['test_labels'][data['test_clean']]).mean()),flip_single_acc=float(((tz[~data['test_clean']]>0)==data['test_labels'][~data['test_clean']]).mean()),training_seconds=sum(x['seconds'] for x in history))
    (out/'result.json').write_text(json.dumps(result,indent=2));del net,opt;torch.cuda.empty_cache()
    return result

def paired(a,b,y,parent,mask=None):
    if mask is None:mask=np.ones(len(y),bool)
    if not mask.any():return dict(n=0,status='NO_ELIGIBLE_QUARTETS')
    a,b,y,parent=a[mask],b[mask],y[mask],parent[mask]
    ao=(a>0)==y;bo=(b>0)==y;aj=ao[:,1:].all(1);bj=bo[:,1:].all(1);h=ao[:,1]&ao[:,2]&~ao[:,3];d=bj.astype(float)-aj
    unique=np.unique(parent);pd=np.array([d[parent==p].mean() for p in unique]);rng=np.random.default_rng(962009);boot=pd[rng.integers(len(pd),size=(4000,len(pd)))].mean(1)
    return dict(n=len(y),parent_n=len(unique),delta_J_row=float(d.mean()),delta_J_parent=float(pd.mean()),parent_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),baseline110_n=int(h.sum()),full_repair_n=int((h&bj).sum()),migration_n=int((h&bo[:,3]&~(bo[:,1]&bo[:,2])).sum()),baseline111_n=int(aj.sum()),regression111_n=int((aj&~bj).sum()))

def analyze(out,seeds):
    out=Path(out);results={};contrasts=[]
    for init in ['pretrained','random']:
        for stem in ['standard','lowstride']:contrasts.append((f'A_{stem}_{init}_flip',f'B_{stem}_{init}_flip'))
        for obs in ['A','B']:contrasts.append((f'{obs}_standard_{init}_flip',f'{obs}_lowstride_{init}_flip'));contrasts.append((f'{obs}_standard_{init}_static',f'{obs}_standard_{init}_flip'))
        contrasts.extend([(f'B_standard_{init}_flip',f'C_standard_{init}_flip'),(f'A_standard_{init}_flip',f'D_standard_{init}_flip'),(f'D_standard_{init}_flip',f'C_standard_{init}_flip')])
    for base,cand in contrasts:
        rows=[]
        for seed in seeds:
            a=np.load(out/f'{base}_s{seed}'/'predictions.npz');b=np.load(out/f'{cand}_s{seed}'/'predictions.npz');assert np.array_equal(a['parent'],b['parent']) and np.array_equal(a['labels'],b['labels'])
            rows.append(dict(seed=seed,overall=paired(a['logits'],b['logits'],a['labels'],a['parent']),small_edit=paired(a['logits'],b['logits'],a['labels'],a['parent'],a['small_edit']),other_edit=paired(a['logits'],b['logits'],a['labels'],a['parent'],~a['small_edit'])))
        results[cand+' minus '+base]=rows
    # Frozen structural prediction: lowstride reduces B-A advantage, especially <2px edits.
    interactions=[]
    for init in ['pretrained','random']:
        for seed in seeds:
            files={f'{o}_{s}':np.load(out/f'{o}_{s}_{init}_flip_s{seed}'/'predictions.npz') for o in ['A','B'] for s in ['standard','lowstride']}
            first=files['A_standard'];joint={k:((v['logits']>0)==v['labels'])[:,1:].all(1).astype(float) for k,v in files.items()}
            delta=(joint['B_lowstride']-joint['A_lowstride'])-(joint['B_standard']-joint['A_standard'])
            row=dict(init=init,seed=seed,predicted_sign='negative, particularly small_edit',n=len(delta))
            for label,mask in [('all',np.ones(len(delta),bool)),('small_edit',first['small_edit']),('other_edit',~first['small_edit'])]:
                ids=np.unique(first['parent'][mask]);vals=np.array([delta[mask&(first['parent']==p)].mean() for p in ids])
                if len(vals):
                    rng=np.random.default_rng(962010);ci=np.quantile(vals[rng.integers(len(vals),size=(4000,len(vals)))].mean(1),[.025,.975]).tolist()
                    row[label]=dict(n=int(mask.sum()),parent_n=len(ids),interaction_row=float(delta[mask].mean()),interaction_parent=float(vals.mean()),parent_bootstrap95=ci)
                else:row[label]=dict(n=0,status='NO_ELIGIBLE_QUARTETS')
            interactions.append(row)
    (out/'paired_analysis.json').write_text(json.dumps(results,indent=2));(out/'structural_prediction.json').write_text(json.dumps(interactions,indent=2))

def joint_of(d):
    return ((d['logits'] > 0) == d['labels'])[:, 1:].all(1).astype(float)


def boot_ci(vals, seed=962010):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    ci = np.quantile(vals[rng.integers(len(vals), size=(4000, len(vals)))].mean(1),
                     [.025, .975]).tolist()
    return float(vals.mean()), ci


def analyze_upgrade(out, seeds):
    """P0/P1/P2 verdict inputs on old bank + smaledit bank. No thresholds."""
    out = Path(out)
    rep = {}
    # P0: B-A standard flip replication per init (old bank).
    p0 = {}
    for init in ['pretrained', 'random']:
        rows = []
        for seed in seeds:
            a = np.load(out / f'A_standard_{init}_flip_s{seed}' / 'predictions.npz')
            b = np.load(out / f'B_standard_{init}_flip_s{seed}' / 'predictions.npz')
            rows.append(paired(a['logits'], b['logits'], a['labels'], a['parent']))
        p0[init] = rows
    rep['P0_B_minus_A_standard_oldbank'] = p0
    # P1: (W-A)/(B-A) on random-init, standard stem, old bank.
    p1 = []
    for seed in seeds:
        a = np.load(out / f'A_standard_random_flip_s{seed}' / 'predictions.npz')
        w = np.load(out / f'W_standard_random_flip_s{seed}' / 'predictions.npz')
        b = np.load(out / f'B_standard_random_flip_s{seed}' / 'predictions.npz')
        ja, jw, jb = joint_of(a).mean(), joint_of(w).mean(), joint_of(a).mean()
        ja2, jw2, jb2 = joint_of(a), joint_of(w), joint_of(b)
        # parent-equal means for the fraction
        pa, pw, pb = a['parent'], w['parent'], b['parent']
        ids = np.unique(pa)
        ma = np.array([ja2[pa == p].mean() for p in ids])
        mw = np.array([jw2[pw == p].mean() for p in ids])
        mb = np.array([jb2[pb == p].mean() for p in ids])
        denom = (mb - ma).mean()
        f = float((mw - ma).mean() / denom) if denom != 0 else None
        p1.append(dict(seed=seed, J_A=float(ja), J_W=float(jw), J_B=float(jb2.mean()),
                       f_WA_over_BA=f))
    rep['P1_compute_fraction_random_standard_oldbank'] = p1
    # P2: lowstride interaction on smaledit bank per init.
    p2 = []
    for init in ['pretrained', 'random']:
        for seed in seeds:
            try:
                files = {f'{o}_{s}': np.load(out / f'{o}_{s}_{init}_flip_s{seed}'
                                             / 'smaledit_predictions.npz')
                         for o in ['A', 'B'] for s in ['standard', 'lowstride']}
            except FileNotFoundError:
                p2.append(dict(init=init, seed=seed, status='MISSING_SMALEDIT_PREDS'))
                continue
            first = files['A_standard']
            joint = {k: joint_of(v) for k, v in files.items()}
            delta = (joint['B_lowstride'] - joint['A_lowstride']) - \
                    (joint['B_standard'] - joint['A_standard'])
            ids = np.unique(first['parent'])
            vals = np.array([delta[first['parent'] == p].mean() for p in ids])
            m, ci = boot_ci(vals)
            p2.append(dict(init=init, seed=seed, n=len(delta), parent_n=len(ids),
                           interaction_row=float(delta.mean()),
                           interaction_parent=m, parent_bootstrap95=ci))
    rep['P2_lowstride_interaction_smaledit'] = p2
    # Same interaction on the old 178 bank (predictions.npz) for comparison.
    p2old = []
    for init in ['pretrained', 'random']:
        for seed in seeds:
            files = {f'{o}_{s}': np.load(out / f'{o}_{s}_{init}_flip_s{seed}'
                                         / 'predictions.npz')
                     for o in ['A', 'B'] for s in ['standard', 'lowstride']}
            first = files['A_standard']
            joint = {k: joint_of(v) for k, v in files.items()}
            delta = (joint['B_lowstride'] - joint['A_lowstride']) - \
                    (joint['B_standard'] - joint['A_standard'])
            ids = np.unique(first['parent'])
            vals = np.array([delta[first['parent'] == p].mean() for p in ids])
            m, ci = boot_ci(vals)
            p2old.append(dict(init=init, seed=seed, n=len(delta), parent_n=len(ids),
                              interaction_row=float(delta.mean()),
                              interaction_parent=m, parent_bootstrap95=ci))
    rep['P2_lowstride_interaction_oldbank'] = p2old
    (out / 'upgrade_analysis.json').write_text(json.dumps(rep, indent=2))
