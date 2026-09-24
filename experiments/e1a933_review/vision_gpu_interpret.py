"""Independent read-only O03/N01 metrics, selection and auxiliary-learning audit."""
from pathlib import Path
import json,numpy as np
ROOT=Path('artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix')
rows=[];aux=[];reference=None
for arm in ['pretrained_static','pretrained_flip','random_static','random_flip','ordered','matched','shuffled_matched']:
    for seed in [803,805,806]:
        p=ROOT/f'{arm}_s{seed}';r=json.loads((p/'result.json').read_text());h=json.loads((p/'history.json').read_text());pred=np.load(p/'predictions.npz')
        assert len(h)==20 and r['best_epoch']==int(np.argmin([x['dev_BCE'] for x in h]))+1
        key=(pred['labels'],pred['parent'])
        if reference is None:reference=key
        assert all(np.array_equal(a,b) for a,b in zip(reference,key))
        j=float(((pred['logits'][:,1:]>0)==pred['labels'][:,1:]).all(1).mean());assert abs(j-r['J'])<1e-12
        rows.append({'arm':arm,'seed':seed,**r})
        if (p/'dev_aux_predictions.npz').exists():
            d=np.load(p/'dev_aux_predictions.npz');sd=d['normalization_std'];physical=(((d['prediction'][:,None]-d['target_orbit'])*sd)**2).mean(-1).min(1)
            aux.append({'arm':arm,'seed':seed,'selected_epoch':r['best_epoch'],'matched_MSE':float(d['matched_error'].mean()),'ordered_MSE':float(d['ordered_error'].mean()),'physical_matched_MSE':float(physical.mean()),'constant_train_mean_MSE':float((d['target_orbit'][:,0]**2).mean()),'dev_parent_count':len(np.unique(d['parent'])),'dev_sample_count':len(d['parent']),'epoch1_matched_MSE':h[0]['dev_matched_mse'],'epoch20_matched_MSE':h[-1]['dev_matched_mse']})
summary={arm:{'J_mean':float(np.mean([r['J'] for r in rows if r['arm']==arm])),'J_seeds':[r['J'] for r in rows if r['arm']==arm]} for arm in sorted(set(r['arm'] for r in rows))}
auxpairs=[]
for baseline in ['ordered','shuffled_matched','pretrained_flip']:
    for seed in [803,805,806]:
        a=np.load(ROOT/f'{baseline}_s{seed}'/'dev_aux_predictions.npz');b=np.load(ROOT/f'matched_s{seed}'/'dev_aux_predictions.npz')
        assert np.array_equal(a['parent'],b['parent']) and np.allclose(a['target_orbit'],b['target_orbit'])
        delta=b['matched_error']-a['matched_error'];parent=a['parent'];unique=np.unique(parent);rng=np.random.default_rng(958000+seed);boot=[]
        for _ in range(2000):
            ix=np.concatenate([np.flatnonzero(parent==v) for v in rng.choice(unique,len(unique),replace=True)]);boot.append(float(delta[ix].mean()))
        auxpairs.append({'baseline':baseline,'seed':seed,'mean_matched_MSE_difference':float(delta.mean()),'parent_bootstrap95':np.quantile(boot,[.025,.975]).tolist()})
result={'status':'21_RUN_INDEPENDENT_ARRAY_AND_SELECTION_CHECK_PASS','classification':summary,'auxiliary':aux,'auxiliary_parent_paired':auxpairs,'per_run':rows,'N03_gate':'SUPPORTED_LEARNING_WITHOUT_CONSISTENT_J_BENEFIT; geometry is approximate, especially selected seed805 epoch2; no claim of exact recovery','scientific_N01':'No consistent repair benefit of matched targets over ordered/BCE/shuffled; observational matching improves auxiliary learnability but is not shown sufficient for task repair.'}
(ROOT.parent/'independent_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'auxiliary_parent_paired':auxpairs},indent=2))
