#!/usr/bin/env python3
"""Freeze N02 geometry, renderer, 48-arm configuration and portable GPU bundle."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for d in ['experiments/f095_campaign','experiments/next6_ef0f7a3','experiments/last15h','experiments/last15h/shared']:
    sys.path.insert(0,str(ROOT/d))
from paths import atomic_edits,oracle_at
from core.relations import make_relational_scenes
from u01_quartet import mine_quartets
from sampling_gpu_protocol import RADIUS,SS,canonical_coords,digest,filehash,render_native,make_arms

def can(x):return canonical_coords(np.asarray(x).reshape(-1,4,2)).reshape(-1,8).astype(float)
def parenthash(x):return digest(can(x))
def in_fov(x):return bool(np.max(np.abs(x))<=1-RADIUS)
def generate():
    data={};meta={};parents={}
    for split,n,seed in [('train',512,962001),('dev',128,962002),('test',512,962003)]:
        pool=np.asarray(make_relational_scenes(n*4,seed=seed)[0]);xs=np.asarray([x for x in pool if in_fov(x)][:n]);assert len(xs)==n
        parents[split]=xs;coords=[];labels=[];pids=[];clean=[];background=[]
        for pi,x in enumerate(xs):
            y0=oracle_at(x)[0]
            for view in range(4):
                coords.append(x);labels.append(y0);pids.append(pi);clean.append(True);background.append(.5+np.random.default_rng(seed*10+pi*100+view).uniform(-.05,.05))
            count=0;candidates=[]
            for trial in range(16):candidates.extend(atomic_edits(x,np.random.default_rng(seed*1000+pi*20+trial),radius=[.1,.2,.35,.5][trial%4]))
            for e,_ in candidates:
                edited=x+e
                if not in_fov(edited):continue
                try:y,m,_=oracle_at(edited)
                except ValueError:continue
                if y==y0 or m<.02:continue
                bg=.5+np.random.default_rng(seed*10+pi*100+4+count).uniform(-.05,.05)
                im=render_native(np.stack([x,edited]),[bg,bg],64).numpy()
                if int((np.abs(im[1]-im[0]).max(0)>.2).sum())<20:continue
                coords.append(edited);labels.append(y);pids.append(pi);clean.append(False);background.append(bg);count+=1
                if count==2:break
        data[split+'_coords']=np.asarray(coords);data[split+'_labels']=np.asarray(labels,dtype=np.float32);data[split+'_parents']=np.asarray(pids);data[split+'_clean']=np.asarray(clean);data[split+'_bg']=np.asarray(background,dtype=np.float32);data[split+'_base_coords']=xs
        meta[split]=dict(seed=seed,parent_count=n,draw_pool=n*4,rows=len(coords),clean_count=int(np.sum(clean)),flip_count=int((~np.asarray(clean)).sum()),parent_hashes=[parenthash(x) for x in xs],coords_sha256=digest(data[split+'_coords']))
        print('split',split,'parents',n,'rows',len(coords),flush=True)
    lookup={parenthash(x):i for i,x in enumerate(parents['test'])};qcoords=[];qy=[];qp=[];qbg=[];counts={};small=[];fovreject=0
    for x,ea,eb,y0,ya,yb,yab in mine_quartets(parents['test'],np.random.default_rng(962004)):
        states=np.stack([x,x+ea,x+eb,x+ea+eb]);pi=lookup[parenthash(x)]
        if not in_fov(states):fovreject+=1;continue
        if counts.get(pi,0)>=4:continue
        counts[pi]=counts.get(pi,0)+1;qcoords.append(states);qy.append([y0,ya,yb,yab]);qp.append(pi)
        bg=.5+np.random.default_rng(962005+pi).uniform(-.05,.05);qbg.append([bg]*4)
        # Frozen pixel-space structural stratum: the smaller constituent edit <=2 pixels.
        small.append(min(np.linalg.norm(ea,axis=-1).max(),np.linalg.norm(eb,axis=-1).max())*32<=2)
    data['quartet_coords']=np.asarray(qcoords);data['quartet_labels']=np.asarray(qy);data['quartet_parents']=np.asarray(qp);data['quartet_bg']=np.asarray(qbg,dtype=np.float32);data['quartet_small_edit']=np.asarray(small)
    assert len(qy)>0
    meta['quartet']=dict(seed=962004,rows=len(qy),parents=len(set(qp)),fov_rejected_before_max4_cap=fovreject,small_edit_count=int(np.sum(small)),parent_cap=4,coords_sha256=digest(data['quartet_coords']))
    # Parent splits precede all edits; nearest physical-state checks include endpoint reversal.
    meta['split_overlap']={}
    for s,t in [('train','dev'),('train','test'),('dev','test')]:
        dist,_=cKDTree(can(parents[s])).query(can(parents[t]),p=np.inf)
        assert dist.min()>1e-6;meta['split_overlap'][s+'_'+t]=dict(min_parent_linf=float(dist.min()),matches_within_1e6=int((dist<=1e-6).sum()))
    tr=can(data['train_coords']);teststates=np.concatenate([can(data['test_coords']),can(data['quartet_coords'])]);dist,_=cKDTree(tr).query(teststates,p=np.inf);assert dist.min()>1e-6
    meta['all_train_state_vs_test_state']=dict(min_linf=float(dist.min()),matches_within_1e6=int((dist<=1e-6).sum()))
    return data,meta,parents

def old_overlap(parents):
    rows=[];fresh=np.concatenate([can(parents[s]) for s in ['train','dev','test']]);test=can(parents['test'])
    paths=list((ROOT/'artifacts/discovery_campaign/scenes').glob('*/scenes.npz'))+list((ROOT/'artifacts/f095_campaign/D01').glob('scenes_*/scenes.npz'))+list((ROOT/'artifacts/e1a933_review').glob('vision*/data.npz'))
    paths += [ROOT/'artifacts/next6_ef0f7a3/u01/quartets_eval.npz']
    for path in sorted(set(paths)):
        b=np.load(path);values=[]
        if 'positions' in b:values.append(b['positions'])
        for key in ['train_coords','dev_coords','test_coords','quartet_coords']:
            if key in b:values.append(b[key].reshape(-1,4,2))
        if 'q0' in b:values.extend([b[k][0].reshape(1,4,2) for k in b.files if k.startswith('q')])
        if not values:continue
        old=can(np.concatenate(values));tree=cKDTree(old);dist,_=tree.query(fresh,p=np.inf);dt,_=tree.query(test,p=np.inf)
        rows.append(dict(path=str(path.relative_to(ROOT)),sha256=filehash(path),old_states=len(old),fresh_parent_matches_within_1e6=int((dist<=1e-6).sum()),fresh_test_matches_within_1e6=int((dt<=1e-6).sum()),min_parent_linf=float(dist.min())))
        assert dist.min()>1e-6,(path,dist.min())
    assert rows
    return rows

def audit_renderer():
    x=np.array([[-.5,-.35],[.5,-.35],[-.4,.4],[.4,.4]])
    a=render_native(x[None],[.5],64).numpy()[0];c=render_native(x[None],[.5],224).numpy()[0]
    rev=x[[1,0,3,2]];assert np.array_equal(a,render_native(rev[None],[.5],64).numpy()[0])
    # Independently inspect native-grid cross-sections; physical width should match.
    widths={}
    for res,im in [(64,a),(224,c)]:
        red=(im[0]-im[1])/.8;blue=(im[2]-im[1])/.8
        width=float(red[:,res//2].sum()*2/res)
        assert abs(width-2*RADIUS)<=2/res
        assert np.allclose(im[:,0,0],.5)
        assert abs(float(im[0].max())-.9)<1e-6 and abs(float(im[1].min())-.1)<1e-6
        widths[str(res)]=dict(measured_physical_width=width,expected_physical_width=2*RADIUS,image_sha256=digest(im))
    b=torch.nn.functional.interpolate(torch.tensor(a[None]),(224,224),mode='bilinear',align_corners=False).numpy()[0]
    native_vs_interpolated=float(np.abs(c-b).max());assert native_vs_interpolated>1e-4
    return dict(native224_vs_bilinear64_max_abs=native_vs_interpolated,protocol='continuous round-capped Euclidean segments; red first, blue second; same-color endpoint canonicalization; per-pixel 4x4 supersampling',FOV=[-1,1,-1,1],radius=RADIUS,SS=SS,width_audit=widths,endpoint_reversal_exact=True,native224_is_not_resized64=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--weights',type=Path,default=ROOT/'artifacts/e1a933_review/vision_cuda_bundle/torch_cache/hub/checkpoints/resnet18-f37072fd.pth');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    data,meta,parents=generate();meta['old_exposed_bank_overlap']=old_overlap(parents);meta['renderer']=audit_renderer()
    meta['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();meta['created_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());meta['command']=sys.argv
    meta['training_contract']='512 fresh train parents; 4 static nuisance views + max2 visible single flips per parent; margin>=.02;20 changed native64 pixels; no AB in train/dev; all capsules inside FOV. test quartet cap4/parent, all denominators preserved after declared geometry eligibility.'
    meta['new_observation_contract']='New continuous capsule renderer used consistently by all new arms. Different from old integer-square-brush D04; old checkpoint results stay separate.'
    np.savez_compressed(a.out/'data.npz',**data);(a.out/'data_manifest.json').write_text(json.dumps(meta,indent=2))
    config=dict(seeds=[803,805,806],epochs=20,batch=32,optimizer='Adam',lr=3e-4,arms=make_arms(),expected_completed=48,dtype='FP32',TF32=False,AMP=False,selection='minimum single-dev BCE; no target AB selection',frozen_prediction='removing maxpool reduces (J_B-J_A) relative to standard; predicted strongest for min constituent endpoint displacement <=2 native64 pixels; report interaction even if opposite, no threshold search',A='direct native64 continuous render',B='bilinear(A,224), align_corners=False',C='direct native224 continuous render',D='area(C,64): deterministic low-pass area downsample',lowstride='replace maxpool by Identity only; conv1 stride2 and all other layers unchanged',static_contract='repeat clean train indices deterministically to same per-epoch exposure count as flip; same batches, optimizer, dev criterion',seed_replication='3 model initializations, not independent tasks',checkpoint_policy='one selected-by-single-dev checkpoint per arm, no target-J selection',cache_policy='native A/C memmap outside output; retain per-image hashes, geometry and exact renderer; output excludes reconstructible multi-GB cache')
    (a.out/'config.json').write_text(json.dumps(config,indent=2));shutil.copyfile(a.weights,a.out/'resnet18-f37072fd.pth')
    for name in ['sampling_gpu_protocol.py','sampling_gpu_run.py','sampling_gpu_prepare.py']:
        shutil.copyfile(Path(__file__).parent/name,a.out/name)
    files=[dict(path=str(x.relative_to(a.out)),sha256=filehash(x),bytes=x.stat().st_size) for x in sorted(a.out.iterdir()) if x.is_file()]
    (a.out/'manifest.json').write_text(json.dumps(dict(files=files,expected_completed=48,source_commit=meta['source_commit']),indent=2));print('BUNDLE_FROZEN',a.out,'quartets',len(data['quartet_labels']),flush=True)
if __name__=='__main__':main()
