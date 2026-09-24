"""O05 real histories, causal controlled masks, physically consistent edit copies."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/f095_campaign'))
from d06_oracle import channel_label,point_segment_dist

def observation(history,hidden):
 # history oldest(-.4), recent(-.2), current. Two masked nodes use recent data.
 z=history[2].copy();z[hidden]=history[1,hidden]
 velocity=(history[2]-history[1])/.2;velocity[hidden]=(history[1,hidden]-history[0,hidden])/.2
 mask=np.zeros((13,1));mask[hidden]=1
 return np.concatenate([z/52.5,velocity/10,mask],1).astype(np.float32)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 source=ROOT/'artifacts/e1a933_review/football/events_candidates.jsonl';records=[json.loads(s) for s in source.read_text().splitlines()]
 arrays=collections.defaultdict(list);counts={};qmeta=[]
 for match,split in [('J03WOY','train'),('J03WMX','dev'),('J03WN1','test')]:
  fr=pd.read_parquet(ROOT/f'artifacts/data_v2/idsse/canonical/frames/{match}.parquet',columns=['game_section','timestamp_ms','entity_type','person_id','x','y'])
  fr=fr[fr.entity_type=='player'];sections={sec:g.set_index('timestamp_ms',drop=False).sort_index() for sec,g in fr.groupby('game_section')};times={sec:np.unique(g.index) for sec,g in sections.items()};cache={}
  def get(sec,t,ids):
   key=(sec,t)
   if key not in cache:
    tt=times[sec];j=np.searchsorted(tt,t,side='right')-1
    if j<0 or t-tt[j]>40:raise ValueError('gap_or_period_boundary')
    f=sections[sec].loc[[tt[j]]]
    if f.person_id.duplicated().any():raise ValueError('duplicate_person')
    cache[key]=f.set_index('person_id')[['x','y']]
   f=cache[key]
   if not set(ids)<=set(f.index):raise ValueError('missing_identity_in_history')
   z=f.loc[ids].to_numpy(float)
   if not np.isfinite(z).all():raise ValueError('nonfinite_history')
   return z
  skipped=collections.Counter();n_before=len(arrays[split+'_x']);group=0
  for row in [r for r in records if r['match']==match]:
   ids=[row['passer_id'],row['recipient_id']]+row['defender_ids'];sec=row['period'];end=row['frame_timestamp_ms'];trajectory=[]
   # Natural history is a different task from synthetic witness states.
   for offset in range(-2000,1,200):
    t=end+offset
    try:
     # Check every intermediate native frame: cannot jump across missing IDs.
     for intermediate in range(t-400,t+1,40):get(sec,intermediate,ids)
     h=np.stack([get(sec,t-400,ids),get(sec,t-200,ids),get(sec,t,ids)])
    except ValueError as err:skipped[str(err)]+=1;continue
    old=h[1];k=2+int(np.argmin([point_segment_dist(d,old[0],old[1]) for d in old[2:]]));hidden=[1,k]
    x=observation(h,hidden);y=channel_label(h[2,0],h[2,1],h[2,2:],.5,1.)['label']
    arrays[split+'_x'].append(x);arrays[split+'_y'].append(y);arrays[split+'_group'].append(group);arrays[split+'_seconds'].append(offset/1000)
    inferred=x[:,:2]*52.5;cv=inferred+x[:,2:4]*10*.2*x[:,4:5]
    arrays[split+'_last'].append(channel_label(inferred[0],inferred[1],inferred[2:],.5,1.)['label']);arrays[split+'_cv'].append(channel_label(cv[0],cv[1],cv[2:],.5,1.)['label'])
    if offset==0 and row['probe'].get('E_found'):
     w=row['probe']['witness'];da=np.zeros((13,2));db=da.copy();da[1]=w['receiver_delta'];db[2+row['probe']['defender_index']]=w['defender_delta']
     states=[h,h+da[None],h+db[None],h+da[None]+db[None]]
     qx=np.array([observation(s,hidden) for s in states]);qy=[channel_label(s[2,0],s[2,1],s[2,2:],.5,1.)['label'] for s in states]
     assert qy[0]==qy[1]==qy[2] and qy[3]!=qy[0]
     arrays[split+'_qx'].append(qx);arrays[split+'_qy'].append(qy);arrays[split+'_qgroup'].append(group)
     if split=='train':
      for xx,yy in zip(qx[1:],qy[1:]):arrays['change_x'].append(xx);arrays['change_y'].append(yy);arrays['repeat_x'].append(x);arrays['repeat_y'].append(y)
    trajectory.append(offset)
   group+=1
  counts[match]={'split':split,'candidate_groups':group,'natural_snapshots':len(arrays[split+'_x'])-n_before,'controlled_E_quartets':len(arrays[split+'_qx']),'skipped_windows':dict(skipped)}
  print(match,counts[match],flush=True)
 for k,v in arrays.items():arrays[k]=np.asarray(v,dtype=np.float32 if k.endswith('_x') or k.endswith('_qx') else None)
 np.savez_compressed(a.out/'data.npz',**arrays)
 manifest={'status':'FROZEN','counts':counts,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((a.out/'data.npz').read_bytes()).hexdigest(),
  'split':'J03WOY train / J03WMX dev / J03WN1 test; 3 matches only',
  'history':'2 seconds natural trajectories, snapshots each .2s; every native .04s identity checked within each .4s history; no cross-period/gap fill',
  'mask':'receiver plus defender closest to channel in last-observed frame; selected using past coordinates only; no arbitrary Gaussian noise',
  'features':'13 nodes [passer,receiver,11defenders] x [x/52.5,y/52.5,vx/10,vy/10,mask]; hidden positions from t-.2, velocity from t-.4 to t-.2',
  'change':'train only witnessed A/B/AB edits applied as same offset to every history frame, preserving observability; static repeats matched base rows to same sample count',
  'configs':{'seeds':[11,23,47],'encoders':['raw','typed'],'augment':['static','change'],'epochs':300,'batch':512,'optimizer':'Adam','lr':.001,'checkpoint':'lowest dev BCE every10epochs; dev has natural snapshots only','threshold':0.,'cpu_threads':4},
  'scope':'controlled causal partial-observation stress test with real natural trajectories; not native missingness learning or pass success',
  'estimated_gpu_memory':'<2GB; batch512 small networks; must measure'}
 (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
