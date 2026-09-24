"""Arithmetic on transcribed archived summaries, not an empirical rerun.
Source commit: e1a933e6b32dd07c6895c7925cbccd9004da7f54.
Inputs: artifacts/f095_campaign/U10/U10_T{1,2}_EVAL.json,
        reports/f095_campaign/D04.md.
Rounded J values imply rounded contrasts; parent-level CIs are NOT computed.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = {
 'T1': {'raw_clean':[.0962,.0686,.0742], 'raw_flip':[.2489,.2100,.1927],
        'six_clean':[.2026,.1924,.1960], 'six_flip':[.7489,.7731,.7272]},
 'T2': {'raw_clean':[.1516,.1595,.1313], 'raw_flip':[.6059,.6396,.6433],
        'six_clean':[.7325,.6889,.7380], 'six_flip':[.9977,.9971,.9997]},
}
out={'scope':'Arithmetic on rounded, transcribed archived summaries. No new training or re-evaluation.',
     'commit':'e1a933e6b32dd07c6895c7925cbccd9004da7f54','tasks':{}}
for task,d in data.items():
    a={k:np.array(v) for k,v in d.items()}
    I=(a['six_flip']-a['six_clean'])-(a['raw_flip']-a['raw_clean'])
    out['tasks'][task]={'inputs':d,'interaction_pp':(100*I).round(4).tolist(),
                      'feature_advantage_under_flip_pp':(100*(a['six_flip']-a['raw_flip'])).round(4).tolist(),
                      'raw_flip_J_mean':float(a['raw_flip'].mean()),
                      'six_flip_J_mean':float(a['six_flip'].mean())}
pre=np.array([.720,.737,.709]); rnd=np.array([.000,.662,.611])
out['D04_224']={'pretrained':pre.tolist(),'random':rnd.tolist(),
               'paired_pretraining_delta_pp':(100*(pre-rnd)).round(4).tolist(),
               'mean_delta_pp':float(100*(pre-rnd).mean()),
               'warning':'No equivalence test; do not remove the collapsed seed from primary comparison.'}
(ROOT/'evidence'/'ARCHIVED_CONTRASTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
