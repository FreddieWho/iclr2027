"""Noise-independent collision and coordinate proximity audit on formal bank."""
from pathlib import Path
import numpy as np,json,hashlib
from scipy.spatial import cKDTree
p=Path('artifacts/e1a933_review/vision_canonical224_v2');d=dict(np.load(p/'data.npz'));result={}
def shapes(images):
    masks=(images[:,0]>.8).astype(np.uint8)+2*(images[:,2]>.8).astype(np.uint8)
    return {hashlib.sha256(m.tobytes()).hexdigest() for m in masks}
sets={s:shapes(d[s+'_images']) for s in ['train','dev','test']}
sets['quartet']=shapes(d['quartet_images'].reshape(-1,3,64,64))
for a,b in [('train','dev'),('train','test'),('train','quartet'),('dev','test'),('dev','quartet')]:
    result[a+'_vs_'+b+'_identical_color_masks']=len(sets[a]&sets[b])
def canonical(a):
    return np.array([sorted(map(tuple,x[:2]))+sorted(map(tuple,x[2:])) for x in a.reshape(-1,4,2)]).reshape(-1,8)
tree=cKDTree(canonical(d['train_coords']))
for split in ['dev','test','quartet']:
    dist=tree.query(canonical(d[split+'_coords']),p=np.inf)[0]
    result['train_vs_'+split+'_minimum_canonical_coordinate_Linf']=float(dist.min())
    result['train_vs_'+split+'_state_count_within_Linf_0.01']=int((dist<=.01).sum())
result['near_duplicate_boundary']='Noise-independent exact visible color-mask collision checked. Physical coordinate nearest-neighbor is diagnostic, not proof that perceptually similar scenes are absent.'
(p/'split_visual_duplicate_audit.json').write_text(json.dumps(result,indent=2))
