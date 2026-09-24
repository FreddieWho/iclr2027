"""Image-only color segmentation/PCA line fit, fixed known renderer stroke width.
No train/dev/test labels enter fitting. Invalid fits are counted, never dropped.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from vision_cuda_protocol import metrics
DATA=Path('artifacts/e1a933_review/vision_canonical224_v2/data.npz')
OUT=Path('artifacts/e1a933_review/vision_n03_analytic');OUT.mkdir(exist_ok=False)
def fit(image):
    segments=[]
    for channel in [0,2]:
        yy,xx=np.where(image[channel]>.8)
        if len(xx)<3:return np.zeros((4,2)),False
        pts=np.column_stack([xx,yy]).astype(float);center=pts.mean(0)
        cov=(pts-center).T@(pts-center);_,vec=np.linalg.eigh(cov);axis=vec[:,-1]
        projected=(pts-center)@axis
        # n08 renders an axis-aligned square of half-width2 about the centerline.
        cap=2*np.abs(axis).sum();lo=projected.min()+cap;hi=projected.max()-cap
        if hi<=lo:return np.zeros((4,2)),False
        segments.extend([center+lo*axis,center+hi*axis])
    return np.asarray(segments)/63*2-1,True

def proper_cross(x):
    a,b,c,d=x
    def orient(a,b,c):
        u=b-a;v=c-a;return u[0]*v[1]-u[1]*v[0]
    return int(orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0)

def predict(images):
    coords=[];valid=[];pred=[]
    for image in images:
        x,ok=fit(image);coords.append(x);valid.append(ok);pred.append(proper_cross(x) if ok else 0)
    return np.asarray(pred),np.asarray(coords),np.asarray(valid)

data=np.load(DATA);summary={'algorithm':'threshold visible red/blue at.8; PCA axis; extent corrected for known2px square stroke; analytic proper crossing','no_label_tuning':True,'invalid_policy':'record invalid; fixed negative fallback; never exclude','data_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest()}
for split in ['dev','test']:
    pred,coords,valid=predict(data[split+'_images']);labels=data[split+'_labels'];clean=data[split+'_clean']
    np.savez_compressed(OUT/(split+'_predictions.npz'),prediction=pred,labels=labels,estimated_geometry=coords,valid_fit=valid,parent=data[split+'_parents'],clean=clean)
    summary[split]={'n':len(pred),'accuracy':float((pred==labels).mean()),'static_accuracy':float((pred[clean]==labels[clean]).mean()),'single_flip_accuracy':float((pred[~clean]==labels[~clean]).mean()),'invalid_n':int((~valid).sum())}
q=data['quartet_images'];pred,coords,valid=predict(q.reshape(-1,3,64,64));z=(pred.reshape(-1,4)*2-1).astype(float)
summary['quartets']=metrics(z,data['quartet_labels']);summary['quartets']['invalid_state_n']=int((~valid).sum())
np.savez_compressed(OUT/'predictions.npz',logits=z,labels=data['quartet_labels'],parent=data['quartet_parents'],estimated_geometry=coords.reshape(-1,4,4,2),valid_fit=valid.reshape(-1,4))
summary['logit_boundary']='Saved +/-1 is a hard decision encoding, not calibrated confidence.'
(OUT/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
