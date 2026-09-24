#!/usr/bin/env python3
"""Role/time-audited football geometry. E means yA=yB=y0 and yAB!=y0.

Historical D06 files remain immutable. Their `paired` counted existence of
single keep and flip, named supported_single_keep_and_flip in new receipts.
The oracle is AND_d(not blocked_d). Moving different blockers independently
cannot make open -> blocked when each single edit preserves open. A receiver
endpoint and a constraining defender can interact through the same distance.
"""
import numpy as np
import pandas as pd
from d06_oracle import channel_label,point_segment_dist


def select_frame(frames,match,period,timestamp_ms,max_gap_ms=500):
    """Exactly one match/period/time. Earlier frame wins equidistant ties."""
    subset=frames[(frames.source_match_id==match)&(frames.game_section==period)]
    if subset.empty:raise ValueError('no_frame_in_period')
    times=np.sort(subset.timestamp_ms.unique());distance=np.abs(times-timestamp_ms)
    t=times[np.flatnonzero(distance==distance.min())[0]]
    if abs(t-timestamp_ms)>max_gap_ms:raise ValueError('no_frame_within_tolerance')
    rows=subset[subset.timestamp_ms==t].copy()
    if rows.person_id.duplicated().any():raise ValueError('duplicate_person_frame')
    return rows


def audit_roles(rows,passer,recipient,event_team):
    nonplayers=int((rows.entity_type!='player').sum())
    players=rows[rows.entity_type=='player'].copy()
    invalid=players[['x','y']].isna().any(axis=1)|players.team_id.isna()|players.person_id.isna()
    if invalid.any():raise ValueError('missing_player_coordinates_or_identity')
    by=players.set_index('person_id')
    if passer not in by.index or recipient not in by.index:raise ValueError('missing_endpoint_player')
    if by.loc[passer,'team_id']!=by.loc[recipient,'team_id']:raise ValueError('recipient_wrong_team')
    if by.loc[passer,'team_id']!=event_team:raise ValueError('event_team_mismatch')
    if len(players.team_id.unique())!=2:raise ValueError('not_two_teams')
    defenders=players[players.team_id!=event_team]
    if len(defenders)!=11:raise ValueError('incomplete_opponent_team')
    p=by.loc[passer,['x','y']].to_numpy(float)
    r=by.loc[recipient,['x','y']].to_numpy(float)
    D=defenders[['x','y']].to_numpy(float)
    return p,r,D,defenders.person_id.tolist(),nonplayers


def motion_cap(frames,horizon_seconds=.2):
    """Identity/period-linked finite differences; gaps over 120ms are broken."""
    d=frames[frames.entity_type=='player'].sort_values(['source_match_id','game_section','person_id','timestamp_ms'])
    g=d.groupby(['source_match_id','game_section','person_id'],sort=False)
    dt=g.timestamp_ms.diff()/1000;dx=g.x.diff();dy=g.y.diff()
    valid=dt.gt(0)&dt.le(.12)&np.isfinite(dx)&np.isfinite(dy)
    vx=(dx/dt).where(valid);vy=(dy/dt).where(valid)
    speed=np.hypot(vx,vy)
    tmp=d[['source_match_id','game_section','person_id']].copy();tmp['vx']=vx;tmp['vy']=vy
    gg=tmp.groupby(['source_match_id','game_section','person_id'],sort=False)
    acceleration=np.hypot(gg.vx.diff()/dt,gg.vy.diff()/dt).where(valid)
    vs=speed.dropna().to_numpy();aa=acceleration.dropna().to_numpy()
    v95=float(np.quantile(vs,.95));a95=float(np.quantile(aa,.95))
    cap=v95*horizon_seconds+.5*a95*horizon_seconds**2
    return cap,{'horizon_seconds':horizon_seconds,'consecutive_dt_seconds_max':.12,
                'valid_velocity_pairs':len(vs),'valid_acceleration_triples':len(aa),
                'speed_p95_m_s':v95,'acceleration_p95_m_s2':a95,'cap_m':cap,
                'invalid_or_gap_pair_count':int((~valid).sum()),
                'interpretation':'empirical displacement envelope, not joint dynamic feasibility proof'}


def labels_batch(p,receivers,defenders,radius=.5,end_excl=1.):
    r=np.asarray(receivers);D=np.asarray(defenders);p=np.asarray(p)
    v=r-p;den=(v*v).sum(1);t=np.clip(((D-p)*v[:,None,:]).sum(2)/np.maximum(den[:,None],1e-12),0,1)
    dist=np.linalg.norm(D-(p+v[:,None,:]*t[:,:,None]),axis=2)
    valid=(np.linalg.norm(D-p,axis=2)>end_excl)&(np.linalg.norm(D-r[:,None,:],axis=2)>end_excl)
    return (~((dist<radius)&valid).any(1)).astype(int)


def quartet_search(p,r,D,cap,rng,radius=.5,end_excl=1.,random_tries=128):
    """Bounded model-blind endpoint/blocker edits; guided boundary + random control."""
    p,r,D=map(lambda a:np.asarray(a,float),(p,r,D))
    active=(np.linalg.norm(D-p,axis=1)>end_excl)&(np.linalg.norm(D-r,axis=1)>end_excl)
    if not active.any():return {'eligible':False,'reason':'no_interior_defender'}
    distances=np.array([point_segment_dist(z,p,r) for z in D]);distances[~active]=np.inf
    k=int(distances.argmin());v=r-p;t=float(np.clip((D[k]-p)@v/(v@v),0,1));q=p+t*v
    normal=(D[k]-q)/(np.linalg.norm(D[k]-q)+1e-12)
    base=channel_label(p,r,D,radius,end_excl)['label'];sign=1 if base else -1
    ar=[];bd=[]
    # Fixed, finite cap fractions; no model feedback and no cap enlargement.
    for a in [.05,.1,.2,.35,.5,.7,1.]:
      for b in [.05,.1,.2,.35,.5,.7,1.]:ar.append(sign*normal*cap*a);bd.append(-sign*normal*cap*b)
    guided_n=len(ar)
    for _ in range(random_tries):
      a=rng.normal(size=2);b=rng.normal(size=2)
      ar.append(a/(np.linalg.norm(a)+1e-12)*cap*rng.random());bd.append(b/(np.linalg.norm(b)+1e-12)*cap*rng.random())
    ar=np.array(ar);bd=np.array(bd);n=len(ar)
    RR=r+ar;DD=np.broadcast_to(D,(n,*D.shape)).copy();DD[:,k]+=bd
    legal=(np.abs(RR[:,0])<=52.5)&(np.abs(RR[:,1])<=34)&(np.abs(DD[:,k,0])<=52.5)&(np.abs(DD[:,k,1])<=34)
    A=labels_batch(p,RR,np.broadcast_to(D,(n,*D.shape)),radius,end_excl)
    B=labels_batch(p,np.broadcast_to(r,(n,2)),DD,radius,end_excl)
    AB=labels_batch(p,RR,DD,radius,end_excl)
    E=legal&(A==base)&(B==base)&(AB!=base)
    result={'eligible':True,'base_label':int(base),'defender_index':k,'projection_fraction':t,
            'guided_E':bool(E[:guided_n].any()),'random_E':bool(E[guided_n:].any()),'E_found':bool(E.any()),
            'guided_trials':guided_n,'random_trials':random_tries,
            'supported_single_keep_and_flip':bool(np.any((A==base)&legal)&np.any((A!=base)&legal)),
            'single_keep_pairs':int((legal&(A==base)&(B==base)).sum()),'E_pairs':int(E.sum()),'legal_pairs':int(legal.sum())}
    if E.any():
      i=int(np.flatnonzero(E)[0]);lo,hi=0.,1.
      # Refine a witnessed combined boundary while never expanding either cap.
      for _ in range(24):
        mid=(lo+hi)/2;dd=D.copy();dd[k]+=mid*bd[i]
        if channel_label(p,r+mid*ar[i],dd,radius,end_excl)['label']==base:lo=mid
        else:hi=mid
      scale=min(1.,hi+1e-6);dd=D.copy();dd[k]+=scale*bd[i]
      check=[channel_label(p,a,b,radius,end_excl)['label'] for a,b in
             [(r,D),(r+scale*ar[i],D),(r,dd),(r+scale*ar[i],dd)]]
      if check[0]==check[1]==check[2] and check[3]!=check[0]:
        da,db=scale*ar[i],scale*bd[i];refined=True
      else:
        da,db=ar[i],bd[i];check=[int(base),int(A[i]),int(B[i]),int(AB[i])];refined=False
      result['witness']={'receiver_delta':da.tolist(),'defender_delta':db.tolist(),
          'labels_0_A_B_AB':check,'method':'guided' if i<guided_n else 'random',
          'boundary_bisection_steps':24,'boundary_refined':refined}

    return result


if __name__=='__main__':
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'e1a933_review/football_repair.py'),run_name='__main__')
