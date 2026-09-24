"""Minimal counterexamples based on e1a933 source contracts.
Not a rerun of the repository's trained models or data banks.
Source functions: u10_targets.fit_stats/featurize, u10_models.TypedTriMLP,
n08_visual.render, u06_relation_distill.rel10. Network/renderer bodies below
mirror the fetched source. Fixtures are generated locally, not study data.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)
class TypedTriMLP(nn.Module):
    def __init__(self,pw=48,ph=48,sh=48):
        super().__init__()
        self.phi=nn.Sequential(nn.Linear(2,pw),nn.ReLU(),nn.Linear(pw,ph),nn.ReLU())
        self.psi=nn.Sequential(nn.Linear(2*ph,sh),nn.ReLU(),nn.Linear(sh,sh),nn.ReLU())
        self.rho=nn.Linear(sh,1)
    def forward(self,x):
        p=x.reshape(-1,4,2)
        ht=self.phi(p[...,:3,:]).sum(dim=-2)
        hq=self.phi(p[...,3,:])
        return self.rho(self.psi(torch.cat([ht,hq],dim=-1))).reshape(-1)

def render(x,rng,lw=2):
    IMG=64
    img=np.full((IMG,IMG,3),.5+rng.uniform(-.05,.05),np.float32)
    px=((np.asarray(x,float)+1)/2*(IMG-1)).astype(int).clip(0,IMG-1)
    cols=[(.9,.1,.1)]*2+[(.1,.1,.9)]*2
    for s,(p,q) in enumerate([(px[0],px[1]),(px[2],px[3])]):
        n=int(np.hypot(*(q-p))*2)+1
        for t in np.linspace(0,1,max(n,2)):
            r,c=int(round(p[0]+t*(q[0]-p[0]))),int(round(p[1]+t*(q[1]-p[1])))
            r0,r1=max(0,r-lw),min(IMG,r+lw+1)
            c0,c1=max(0,c-lw),min(IMG,c+lw+1)
            img[c0:c1,r0:r1]=cols[2*s]
    return img.transpose(2,0,1)

def rel10(x):
    x=np.asarray(x,float).reshape(-1,4,2)
    i,j=np.triu_indices(4,1)
    cen=x.mean(1,keepdims=True)
    return np.concatenate([np.linalg.norm(x[:,i]-x[:,j],axis=-1),
                           np.linalg.norm(x-cen,axis=-1)],axis=-1)

def preprocess(x,mu,sd):
    return torch.from_numpy(((np.asarray(x,np.float32).reshape(len(x),-1)-mu)/sd).astype(np.float32))


def source_isotonic(x, y):
    # Body equivalent to d10_frontier.fit_isotonic + apply_isotonic.
    order=np.argsort(x,kind="stable")
    out=[]
    for xx, yy in zip(np.asarray(x)[order],np.asarray(y)[order]):
        out.append([xx,yy,1])
        while len(out)>=2 and out[-2][1]>out[-1][1]:
            x2,y2,w2=out.pop();x1,y1,w1=out.pop();w=w1+w2
            out.append([(x1*w1+x2*w2)/w,(y1*w1+y2*w2)/w,w])
    return np.interp(x,[b[0] for b in out],[b[1] for b in out])

def run():
    rng=np.random.default_rng(20260924)
    train=rng.uniform(-.8,.8,(512,4,2)).astype(np.float32)
    x=rng.uniform(-.8,.8,(128,4,2)).astype(np.float32)
    p=[1,0,2,3]
    mu=train.reshape(512,-1).mean(0);sd=train.reshape(512,-1).std(0)+1e-8
    torch.manual_seed(11);m=TypedTriMLP().eval()
    # invariant raw core vs non-invariant complete preprocessing pipeline
    with torch.no_grad():
        z=preprocess(x,mu,sd).reshape(-1,4,2)
        core_diff=(m(z)-m(z[:,p])).abs().max().item()
        pipeline_diff=(m(preprocess(x,mu,sd))-m(preprocess(x[:,p],mu,sd))).abs().max().item()
        # share statistics across the three exchangeable vertices (Q separate)
        muc=train[:,:3].mean((0,1));sdc=train[:,:3].std((0,1))+1e-8
        mu_fixed=np.concatenate([np.tile(muc,3),train[:,3].mean(0)])
        sd_fixed=np.concatenate([np.tile(sdc,3),train[:,3].std(0)+1e-8])
        fixed_diff=(m(preprocess(x,mu_fixed,sd_fixed))-m(preprocess(x[:,p],mu_fixed,sd_fixed))).abs().max().item()
    assert core_diff<1e-6 and pipeline_diff>1e-5 and fixed_diff<1e-6
    fixture=np.array([[-.72,-.28],[.61,.46],[-.46,.68],[.38,-.61]])
    a=render(fixture,np.random.default_rng(123))
    b=render(fixture[p],np.random.default_rng(123))
    t1,t2=rel10(fixture)[0],rel10(fixture[p])[0]
    identical=bool(np.array_equal(a,b));td=float(np.linalg.norm(t1-t2))
    assert identical and td>1e-3
    # For a repeated input with two deterministic incompatible targets,
    # minimal mean squared error (averaged over features and two rows).
    irreducible_mse=float(np.mean((t1-t2)**2)/4)
    torch.manual_seed(17);nn.Linear(512,1)
    order_a=torch.randperm(100)
    torch.manual_seed(17);nn.Linear(512,1);nn.Linear(512,10)
    order_c=torch.randperm(100)
    order_same=bool(torch.equal(order_a,order_c));assert not order_same
    # Local demonstrations of additional metric/interpretation errors.
    start=np.array([10.,1.]);end=np.array([.1,10.]);cut=5
    assert (start>=cut).tolist()!= (end>=cut).tolist()
    bn=nn.BatchNorm2d(3)
    for w in bn.parameters():w.requires_grad=False
    bn.train();before=bn.running_mean.clone();bn(torch.ones(4,3,8,8)*5)
    bn_changed=not torch.equal(before,bn.running_mean);assert bn_changed
    pre=np.array([.720,.737,.709]);rnd=np.array([.000,.662,.611])
    # D10: a max-only threshold ladder misses action changes at fixed coverage.
    scores=np.array([.4,.9]); costs=np.array([1.,2.]); feasible=np.array([True,False])
    def choose(t):
        idx=np.flatnonzero(scores>=t)
        return int(idx[np.argmin(costs[idx])]) if len(idx) else None
    sparse=choose(scores.max()); complete=choose(.4)
    assert sparse==1 and complete==0 and not feasible[sparse] and feasible[complete]
    # D10: block-centre interpolation fails even on training x values.
    ix=np.array([0.,1.,2.]);iy=np.array([1.,0.,1.])
    wrong=source_isotonic(ix,iy);right=np.array([.5,.5,1.])
    assert np.allclose(wrong,[.5,2/3,1.]) and not np.allclose(wrong,right)
    # D06: exists keep AND exists flip is neither sufficient nor necessary for E.
    def flags(edits,cut):
        lab=lambda u: int(u>cut)
        y0=lab(0.)
        keep=any(lab(e)==y0 for e in edits)
        flip=any(lab(e)!=y0 for e in edits)
        E=any(lab(edits[i])==y0 and lab(edits[j])==y0 and lab(edits[i]+edits[j])!=y0
              for i in range(len(edits)) for j in range(i+1,len(edits)))
        return [bool(keep and flip),bool(E)]
    fpos=flags([-.1,1.],.5);fneg=flags([.6,.6],1.)
    assert fpos==[True,False] and fneg==[False,True]
    out={
      'scope':'local synthetic contract tests; NOT original-model empirical replication',
      'typed_normalization':{'bare_core_max_abs_diff':core_diff,
        'source_coordinatewise_pipeline_max_abs_diff':pipeline_diff,
        'role_shared_pipeline_max_abs_diff':fixed_diff},
      'unobservable_aux_target':{'same_render_after_invisible_endpoint_swap':identical,
        'ordered_rel10_l2_difference':td,'two_copy_mse_minimum_raw_units':irreducible_mse,
        'interpretation':'A deterministic image-only head cannot recover an arbitrary endpoint indexing from an identical image. Not proof this causes the observed U06 null.'},
      'aux_init_changes_batch_order_without_independent_rng':not order_same,
      'pre_vs_post_confidence_masks':{'correct_start_mask':(start>=cut).tolist(),
                                    'source_endpoint_mask':(end>=cut).tolist()},
      'requires_grad_false_does_not_freeze_bn_running_mean':bn_changed,
      'reported_D04_224_pretraining_minus_random_pp':((pre-rnd)*100).round(3).tolist(),
      'D04_mean_delta_all_seeds_pp':float((pre-rnd).mean()*100),
      'max_only_policy_ladder':{'chosen_action_sparse':sparse,'chosen_action_complete':complete,
          'coverage_both':1.,'sparse_success':False,'complete_success':True},
      'isotonic_training_fit':{'source':wrong.tolist(),'correct_L2_fit':right.tolist()},
      'football_keep_flip_vs_E':{'false_positive_legacy_E':fpos,'false_negative_legacy_E':fneg},
      'tests_passed':9}
    outpath=Path(__file__).resolve().parents[1]/'evidence'/'CONTRACT_TEST_RESULTS.json'
    outpath.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
