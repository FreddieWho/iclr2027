import io
import itertools
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/f095_campaign'))
from u10_targets import build_arch, fit_stats, featurize
from d03_eval import start_confidence_mask
from d02_orbits import groupavg_orbit_metrics


def test_checkpoint_full_pipeline_symmetry_and_bad_stats_detected():
    rng=np.random.default_rng(42)
    for task,n in [('T1',4),('T2',3)]:
        x=rng.normal(size=(80,n,2))*np.arange(1,n+1)[None,:,None]+np.arange(n)[None,:,None]
        stats=fit_stats(task,'typed',x)
        torch.manual_seed(5); model=build_arch(task,'typed')
        buf=io.BytesIO();torch.save(dict(state=model.state_dict(),stats=stats),buf);buf.seek(0)
        ck=torch.load(buf,weights_only=False); restored=build_arch(task,'typed');restored.load_state_dict(ck['state'])
        with torch.no_grad():
            ref=restored(featurize(task,'typed',x,ck['stats']))
            for p in itertools.permutations(range(n-1)):
                g=(*p,n-1)
                actual=restored(featurize(task,'typed',x[:,g,:],ck['stats']))
                torch.testing.assert_close(actual,ref,rtol=1e-5,atol=1e-6)
            flat=x.reshape(len(x),-1)
            bad=(flat.mean(0),flat.std(0),'indexwise_bad')
            g=tuple(reversed(range(n-1)))+(n-1,)
            a=restored(featurize(task,'typed',x,bad));b=restored(featurize(task,'typed',x[:,g,:],bad))
            assert (a-b).abs().max()>1e-4


def test_high_confidence_uses_start_and_is_parent_order_invariant():
    start=np.array([10.,1.]); parents=np.array([11,23]); mapping={11:0,23:1}
    assert start_confidence_mask(start,parents,mapping,5).tolist()==[True,False]
    assert start_confidence_mask(start,parents[::-1],mapping,5).tolist()==[False,True]


def test_group_averaged_model_all8_is_identity_and_raw_can_disagree():
    logits=np.full((2,3,8),2.)
    logits[0,:,0]=-1.
    labels=np.ones((2,3),int)
    all8,delta=groupavg_orbit_metrics(logits,labels)
    assert all8==1 and delta<1e-12
    assert ((logits>0)==labels[:,:,None]).all(axis=(1,2)).mean()==.5
