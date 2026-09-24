"""Read-only three-seed reanalysis of rounded legacy D04 J; no re-training."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
base=Path('artifacts/f095_campaign/D04');rows=[]
for seed in [803,805,806]:
    d=json.loads((base/f'resnet_s{seed}/result.json').read_text())
    p=json.loads((base/f'resnet224_s{seed}/result.json').read_text())['pretrained']['J']
    r=json.loads((base/f'resnet224x_s{seed}/result.json').read_text())['random']['J']
    p64=d['pretrained']['J'];r64=d['random']['J']
    rows.append({'seed':seed,'pretrained64':p64,'random64':r64,'pretrained224':p,'random224':r,'init_effect64':p64-r64,'init_effect224':p-r,'scale_effect_pretrained':p-p64,'scale_effect_random':r-r64,'init_by_scale_interaction':(p-r)-(p64-r64)})
summary={}
for key in rows[0]:
    if key=='seed':continue
    a=np.array([r[key] for r in rows]);h=float(t.ppf(.975,2)*a.std(ddof=1)/np.sqrt(3))
    summary[key]={'mean':float(a.mean()),'seed_range':[float(a.min()),float(a.max())],'t95_seed_interval':[float(a.mean()-h),float(a.mean()+h)]}
out=Path('artifacts/e1a933_review/vision_regression');out.mkdir(exist_ok=True,parents=True)
(out/'legacy_factorial.json').write_text(json.dumps({'source':'Archived rounded J; no new scientific rerun; n=3 seeds; collapsed803 retained','per_seed':rows,'summary':summary},indent=2))
