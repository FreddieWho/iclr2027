"""Production-path regression plus observational audit on all 512 legacy train parents."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'f095_campaign'))
from d04_vision_train import make_net,train_mode,backbone_hash,render
from u06_relation_distill import target_orbit,normalize_orbit,matched_mse,H_ENDPOINTS,shuffled_indices
out=Path('artifacts/e1a933_review/vision_regression');out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
checks={}
net=make_net('headonly',803);h=backbone_hash(net);train_mode(net,'headonly')
opt=torch.optim.SGD(net.fc.parameters(),lr=.1);x=torch.rand(4,3,64,64)
net(x).mean().backward();opt.step();checks['frozen_params_and_buffers']=backbone_hash(net)==h
# Replay the old net.train bug, against the same production-created backbone.
net.train();net(x);checks['old_bn_bug_detected']=backbone_hash(net)!=h
net=make_net('bnadapt',803);h=backbone_hash(net);train_mode(net,'bnadapt');net(x)
checks['explicit_bn_adaptation_changes_buffers']=backbone_hash(net)!=h
coords=np.load('artifacts/discovery_campaign/scenes/train_101/scenes.npz')['positions']
orbit=target_orbit(coords);normalized,stats=normalize_orbit(orbit)
pred=torch.tensor(normalized[:,1],dtype=torch.float32);targets=torch.tensor(normalized,dtype=torch.float32)
checks['one_consistent_matching_zero']=float(matched_mse(pred,targets))<1e-12
checks['old_ordered_mse_conflict_detected']=float(((pred-targets[:,0])**2).mean())>1e-4
for p in H_ENDPOINTS:
    transformed,_=normalize_orbit(target_orbit(coords[:,p]),stats)
    checks['normalization_commutes_'+str(p.tolist())]=np.allclose(np.sort(normalized,axis=1),np.sort(transformed,axis=1))
# Matching must choose one group element for the whole structure.
chimeras=targets[:,0].clone();chimeras[:,:3]=targets[:,1,:3]
checks['independent_coordinate_sort_not_used']=float(matched_mse(chimeras,targets))>1e-4
sh=shuffled_indices(len(coords),803);torch.rand(1000);sh2=shuffled_indices(len(coords),803)
checks['shuffle_independent_of_global_rng']=torch.equal(sh,sh2)
maxdiff=[];labelstable=[]
from paths import oracle_at
for i,x in enumerate(coords):
    base=render(x,np.random.default_rng(77000+i));d=[];ys=[]
    for p in H_ENDPOINTS:
        d.append(float(np.abs(base-render(x[p],np.random.default_rng(77000+i))).max()))
        ys.append(oracle_at(x[p])[0])
    maxdiff.append(d);labelstable.append(len(set(ys))==1)
maxdiff=np.array(maxdiff);var=orbit.var(axis=1).mean(axis=1);zvar=normalized.var(axis=1).mean(axis=1)
np.savez_compressed(out/'per_parent_observability.npz',max_pixel_diff=maxdiff,physical_target_variance=var,normalized_target_variance=zvar,label_constant=labelstable)
result={'checks':checks,'all_checks_pass':all(checks.values()),'parent_count':len(coords),'pixel_equal_orbit_fraction':float((maxdiff==0).all(1).mean()),'pixel_equal_transform_fraction':float((maxdiff==0).mean()),'maximum_pixel_difference':float(maxdiff.max()),'label_constant_fraction':float(np.mean(labelstable)),'physical_orbit_variance_mean':float(var.mean()),'normalized_orbit_variance_mean':float(zvar.mean()),'interpretation':'Variance is an irreducible squared-error bound only for exactly equal render orbits and uniform hidden naming; measured orbit ambiguity, not full training loss attribution.'}
from vision_protocol import render as canonical_render
worst=0.
for i,x in enumerate(coords):
    base=canonical_render(x,np.random.default_rng(i))
    for p in H_ENDPOINTS:
        worst=max(worst,float(np.abs(base-canonical_render(x[p],np.random.default_rng(i))).max()))
assert worst==0
result['canonical_renderer_all_512_orbits_pixel_equal']=True
result['canonical_renderer_max_pixel_difference']=worst
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert all(checks.values())
