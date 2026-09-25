#!/usr/bin/env python3
"""Independent acceptance of collected M2 predictions; no training or remote access."""
from pathlib import Path
import csv, hashlib, itertools, json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/submission_audit_20260925/gpu_m2'
OUT=ROOT/'artifacts/submission_audit_20260925/m2_acceptance'
SEEDS=(803,805,806)
DRAWS=10000
BOOT_SEED=20260925

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def tensor_sha(t):
 t=t.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(t.dtype).encode());h.update(json.dumps(list(t.shape)).encode());h.update(t.numpy().tobytes());return h.hexdigest()

def crossing(x):
 a,b,c,d=[x[:,i] for i in range(4)]
 def orient(a,b,c):
  u=b-a;v=c-a;return u[:,0]*v[:,1]-u[:,1]*v[:,0]
 return (orient(a,b,c)*orient(a,b,d)<0)&(orient(c,d,a)*orient(c,d,b)<0)

def main():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
 checks=[];hashes={}
 def check(name,condition,detail=None):
  checks.append({'check':name,'pass':bool(condition),'detail':detail})
  if not condition:raise AssertionError(name+': '+str(detail))
 def record(p):hashes[str(p.relative_to(ROOT))]=sha(p)
 job=json.loads((ROOT/'artifacts/submission_audit_20260925/GPU_M2_JOB.json').read_text())
 for entry in job['input_files']:
  p=BASE/'inputs'/Path(entry['path']).name;record(p);check('input_hash:'+p.name,sha(p)==entry['sha256'])
 run=json.loads((BASE/'out/RUN_RECEIPT.json').read_text())
 check('complete_16_cells',run['status']=='SUCCEEDED' and len(run['steps'])==16 and all(s['status']=='SUCCEEDED' for s in run['steps']))
 for entry in run['outputs']:
  p=BASE/'out'/entry['path'];record(p);check('output_hash:'+entry['path'],p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'])
 bank=np.load(BASE/'inputs/data.npz',allow_pickle=False)
 y=bank['quartet_labels'].astype(bool);parents=bank['quartet_parents'];unique,inv=np.unique(parents,return_inverse=True);n=len(y);g=len(unique)
 check('denominator',y.shape==(692,4) and g==351)
 check('oracle_eligibility',np.all(y[:,0]==y[:,1]) and np.all(y[:,0]==y[:,2]) and np.all(y[:,0]!=y[:,3]))
 check('quartet_keys_are_parent_lineage',np.array_equal(bank['quartet_keys'],bank['test_parent_keys'][parents]))
 seen={};keep=np.ones(n,dtype=bool);duplicates=[]
 for q,parent in enumerate(parents):
  a=bank['quartet_edits_a'][q].tobytes();b=bank['quartet_edits_b'][q].tobytes();key=(int(parent),*sorted([a,b]));order=[0,1,2,3] if a<=b else [0,2,1,3]
  if key in seen:
   first,old_order=seen[key];keep[q]=False
   identical=np.array_equal(bank['quartet_images'][q,order],bank['quartet_images'][first,old_order])
   check('duplicate_geometry_images:'+str(q),identical)
   duplicates.append({'retained_row':first,'dropped_row':q,'parent':int(parent),'identical_images_up_to_atomic_order':bool(identical)})
  else:seen[key]=(q,order)
 check('deduplicated_bank',int(keep.sum())==685 and len(np.unique(parents[keep]))==351)

 for a,b in itertools.combinations(['train','dev','test'],2):
  check('parent_split_disjoint:'+a+':'+b,not(set(bank[a+'_parent_keys'])&set(bank[b+'_parent_keys'])))
 base=bank['test_parent_coords'][parents];states=np.stack([base,base+bank['quartet_edits_a'],base+bank['quartet_edits_b'],base+bank['quartet_edits_a']+bank['quartet_edits_b']],axis=1)
 oracle=crossing(states.reshape(-1,4,2)).reshape(n,4)
 # The convention may label clearance rather than intersection; require one globally consistent polarity.
 oracle_polarity='intersection_is_one' if np.array_equal(oracle,y) else 'clearance_is_one'
 check('independent_geometric_oracle',np.array_equal(oracle,y) or np.array_equal(~oracle,y),oracle_polarity)
 def image_hashes(images):return {hashlib.sha256(im.tobytes()).hexdigest() for im in images}
 check('no_train_quartet_image_duplicates',not(image_hashes(bank['train_images'])&image_hashes(bank['quartet_images'].reshape(-1,3,64,64))))
 sizes=np.bincount(inv,minlength=g).astype(float)
 weights=np.random.default_rng(BOOT_SEED).multinomial(g,np.ones(g)/g,size=DRAWS).astype(float)
 den=weights@sizes
 def group_sum(v):return np.bincount(inv,weights=np.asarray(v,dtype=float),minlength=g)
 def interval(v,eligible=None):
  num=group_sum(v)
  d=sizes if eligible is None else group_sum(eligible)
  bs_d=den if eligible is None else weights@d
  ok=bs_d>0;boot=(weights@num)[ok]/bs_d[ok]
  return {'estimate':float(np.sum(v)/np.sum(d)) if np.sum(d)>0 else None,'ci95':np.quantile(boot,[.025,.975]).tolist() if len(boot) else None,'denominator':int(np.sum(d)),'parents':int(np.count_nonzero(d)),'valid_bootstrap_draws':int(ok.sum())}
 cells={};dedup_cells={};bits={};audits={};results={}
 csv_rows=[]
 for folder in sorted((BASE/'out/m2').iterdir()):
  if not folder.is_dir():continue
  name=folder.name;r=json.loads((folder/'result.json').read_text());results[name]=r
  a=np.load(folder/'predictions.npz',allow_pickle=False);logits=a['logits'].reshape(n,4)
  check(name+':prediction_alignment',np.array_equal(a['labels'],y) and np.array_equal(a['parents'],parents))
  check(name+':finite_logits',np.isfinite(logits).all())
  correct=(logits>0)==y;bits[name]=correct
  for dup in duplicates:
   q,j=dup['dropped_row'],dup['retained_row']
   same=np.array_equal(bank['quartet_edits_a'][q],bank['quartet_edits_a'][j])
   order=[0,1,2,3] if same else [0,2,1,3]
   check(name+':duplicate_prediction:'+str(q),np.array_equal(correct[q],correct[j,order]))
  atomic=correct[:,1]&correct[:,2];j3=atomic&correct[:,3];j4=j3&correct[:,0]
  metrics={'P':float(correct[:,0].mean()),'A':float(correct[:,1].mean()),'B':float(correct[:,2].mean()),'AB':float(correct[:,3].mean()),'atomic_joint':float(atomic.mean()),'J3':float(j3.mean()),'J4':float(j4.mean())}
  for k,v in metrics.items():check(name+':reported_'+k,abs(r['quartet_table'][k]-v)<1e-12)
  metrics.update({'J3_ci':interval(j3),'J4_ci':interval(j4),'CCM':interval(atomic&~correct[:,3],atomic),'equal_parent_J3':float(np.mean(group_sum(j3)/sizes)),'dev_geometry_mse':r.get('dev_geometry_mse')})
  cells[name]=metrics
  dedup_cells[name]={'J3':float(j3[keep].mean()),'J4':float(j4[keep].mean()),'atomic_joint':float(atomic[keep].mean()),'AB':float(correct[keep,3].mean()),'J3_ci':interval(j3&keep,keep),'J4_ci':interval(j4&keep,keep),'CCM':interval(atomic&~correct[:,3]&keep,atomic&keep),'dev_geometry_mse':r.get('dev_geometry_mse')}

  if r.get('history'):
   history=r['history'];check(name+':20_epochs',len(history)==20 and [h['epoch'] for h in history]==list(range(20)))
   check(name+':dev_selection',abs(min(h['dev_loss'] for h in history)-r['train']['best_dev_loss'])<1e-12)
   check(name+':matched_exposure',r['train']['train_row_exposures_per_epoch']==2532 and r['train']['train_row_exposures_total']==50640)
  if name.startswith('geom_front'):
   audit=r['backbone_update_audit'];before=audit['before_training'];after=audit['after_selected_dev_checkpoint'];frozen=audit['freeze_backbone'];grad=audit['first_batch_gradient']
   cp=torch.load(folder/'model_probe.pt',map_location='cpu',weights_only=True)
   for kind in ['parameters','buffers']:
    check(name+':checkpoint_'+kind,all(tensor_sha(cp['backbone.'+k])==h for k,h in after[kind].items()))
   check(name+':probe_trained',any(tensor_sha(cp['probe.'+k])!=h for k,h in audit['probe_parameters_before_training'].items()))
   changed_p=sum(before['parameters'][k]!=v for k,v in after['parameters'].items());changed_bn=sum(before['batchnorm_buffers'][k]!=v for k,v in after['batchnorm_buffers'].items())
   check(name+':freeze_semantics',(changed_p==0 and changed_bn==0 and grad['gradient_tensor_count']==0 and not audit['training_mode_at_first_batch']) if frozen else (changed_p>0 and grad['gradient_l2_norm']>0 and audit['training_mode_at_first_batch']))
   audits[name]={'frozen':frozen,'changed_parameter_tensors':changed_p,'changed_batchnorm_buffers':changed_bn,'gradient_l2_norm':grad['gradient_l2_norm'],'checkpoint_hashes_match':True,'selected_epoch':int(np.argmin([h['dev_loss'] for h in r['history']]))}
  for q in range(n):csv_rows.append([name,q,int(parents[q]),*y[q].astype(int).tolist(),*logits[q].astype(float).tolist(),*correct[q].astype(int).tolist(),int(keep[q])])
 for seed in SEEDS:
  f=results[f'geom_front_{seed}']['backbone_update_audit'];t=results[f'geom_front_tuned_{seed}']['backbone_update_audit']
  check(f'paired_initialization_{seed}',f['probe_parameters_before_training']==t['probe_parameters_before_training'] and f['before_training']['parameters']==t['before_training']['parameters'] and f['before_training']['buffers']==t['before_training']['buffers'])
 # Independent geometric-feature reconstruction and fixed-head CPU replay.
 perms=list(itertools.permutations(range(4)))
 perms=[p for p in perms if {frozenset(p[:2]),frozenset(p[2:])}=={frozenset([0,1]),frozenset([2,3])}]
 ij=list(itertools.combinations(range(4),2));feats=[]
 for x in states.reshape(-1,4,2):
  candidates=[[float(np.linalg.norm(x[p[a]]-x[p[b]])) for a,b in ij] for p in perms];feats.append(min(candidates))
 payload=torch.load(BASE/'inputs/model.pt',map_location='cpu',weights_only=False)
 st=payload['state'];z=torch.tensor((np.asarray(feats)-np.asarray(payload['stats']['mu']))/np.asarray(payload['stats']['sd']),dtype=torch.float32)
 with torch.no_grad():
  z=torch.relu(torch.nn.functional.linear(z,st['net.0.weight'],st['net.0.bias']))
  z=torch.relu(torch.nn.functional.linear(z,st['net.2.weight'],st['net.2.bias']))
  z=torch.nn.functional.linear(z,st['feat_head.weight'],st['feat_head.bias'])
  z=torch.nn.functional.linear(z,st['cls.weight'],st['cls.bias']).numpy().reshape(n,4)
 stored=np.load(BASE/'out/m2/true_geometry_803/predictions.npz')['logits'].reshape(n,4)
 check('true_geometry_head_replay_decisions',np.array_equal(z>0,stored>0))
 replay={'states':int(z.size),'max_logit_difference':float(np.max(np.abs(z-stored))),'decision_mismatches':int(np.sum((z>0)!=(stored>0)))}
 pairs={}
 for seed in SEEDS:
  for newer,older in [(f'geom_front_tuned_{seed}',f'geom_front_{seed}'),(f'direct_full_{seed}',f'direct_clean_{seed}'),(f'direct_full_{seed}',f'geom_front_tuned_{seed}'),('true_geometry_803',f'geom_front_tuned_{seed}'),('true_geometry_803',f'random_head_{seed}')]:
   b=bits[older];c=bits[newer];h=b[:,1]&b[:,2]&~b[:,3];j0=b[:,1:].all(1);j1=c[:,1:].all(1)
   full=h&j1;migration=h&c[:,3]&~c[:,1:3].all(1);endpoint=h&c[:,3]
   check(newer+' minus '+older+':flow_partition',np.array_equal(full|migration,endpoint) and not np.any(full&migration))
   both0=b[:,1:3].all(1);both1=c[:,1:3].all(1)
   pairs[newer+' minus '+older]={'delta_J3':interval(j1.astype(float)-j0.astype(float)),'delta_J4':interval(c.all(1).astype(float)-b.all(1).astype(float)),'H_n':int(h.sum()),'H_parents':int(len(np.unique(parents[h]))),'full_repair_n':int(full.sum()),'relocation_n':int(migration.sum()),'endpoint_repair_n':int(endpoint.sum()),'full_repair':interval(full,h),'relocation':interval(migration,h),'joint_regression_n':int((j0&~j1).sum()),'joint_gain_n':int((~j0&j1).sum()),'atomic_regression_n':int((both0&~both1).sum())}
   pairs[newer+' minus '+older]['deduplicated']={'delta_J3':interval((j1.astype(float)-j0.astype(float))*keep,keep),'delta_J4':interval((c.all(1).astype(float)-b.all(1).astype(float))*keep,keep),'H_n':int((h&keep).sum()),'H_parents':int(len(np.unique(parents[h&keep]))),'full_repair_n':int((full&keep).sum()),'relocation_n':int((migration&keep).sum()),'endpoint_repair_n':int((endpoint&keep).sum()),'full_repair':interval(full&keep,h&keep),'relocation':interval(migration&keep,h&keep),'joint_regression_n':int((j0&~j1&keep).sum()),'joint_gain_n':int((~j0&j1&keep).sum())}

 with (OUT/'PREDICTIONS.csv').open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['cell','quartet','parent','label_P','label_A','label_B','label_AB','logit_P','logit_A','logit_B','logit_AB','correct_P','correct_A','correct_B','correct_AB','included_in_primary']);w.writerows(csv_rows)
 report={'status':'PASS_WITH_DEDUPLICATED_PRIMARY', 'n_unique_quartets':int(keep.sum()),'duplicate_records':duplicates,'primary_estimand':'One row per unique parent plus unordered exact edit pair; 685 quartets / 351 parents; first occurrence retained without using model outcomes','deduplicated_cells':dedup_cells,'scope':'Collected 16-cell corrected M2 run; scoped scientific acceptance, not completion of every original M2 proposal item','n_quartets':n,'n_parents':g,'test_parent_pool_size':384,'oracle_polarity':oracle_polarity,'bootstrap':{'draws':DRAWS,'seed':BOOT_SEED,'unit':'parent','point_estimand':'quartet row-weighted','interval':'percentile 95%; per seed; no multiplicity adjustment','comparison_registration':'acceptance analysis; not a new preregistered superiority test'},'cells':cells,'paired_comparisons':pairs,'backbone_audits':audits,'true_geometry_replay':replay,'checks':checks,'input_sha256':hashes,'limitations':['One previously evaluated bank and renderer; 3 training seeds are not 3 independent datasets.','Geometry frontends receive privileged coordinate-derived supervision; direct models receive relation labels. Equal image exposures are not equal information or matched objectives.','Tuning changes both learnable weights and BatchNorm behavior; this is a bundled recipe contrast.','Predicted six-distance vectors are unconstrained; physical-validity and error-matched interface interventions were not run.','No cross-renderer validation or proof of global unlearnability.','Full ResNet forward replay of every image is not run; accepted predictions are checked against hashes, labels, metrics, selected checkpoint tensors, and training receipts.']}
 record(Path(__file__).resolve())
 (OUT/'ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'checks':len(checks),'cells':len(cells),'replay':replay,'J3_primary':{k:round(v['J3'],4) for k,v in dedup_cells.items()},'paired_primary':{k:v['deduplicated']['delta_J3'] for k,v in pairs.items()}}))

if __name__=='__main__':main()
