"""R08/O05: finite three-match real-coordinate proxy and causal mask baselines."""
import argparse,collections,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/f095_campaign'))
from d06_roles import select_frame,audit_roles,motion_cap,quartet_search,labels_batch
from d06_oracle import channel_label,point_segment_dist
OUT=ROOT/'artifacts/e1a933_review/football'

def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['probe','natural','verify-natural','natural-report','legacy-recheck'],default='probe')
    ap.add_argument('--events-per-match',type=int,default=80);ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--situations',type=Path,default=None,help='natural modes: situation table (default OUT/events_candidates.jsonl)')
    ap.add_argument('--natural-out',type=Path,default=None,help='natural modes: NATURAL_E_RATE.json path (default OUT/NATURAL_E_RATE.json)')
    ap.add_argument('--windows',default='0.2,0.4,1.0,2.0',help='natural modes: forward windows in seconds')
    ap.add_argument('--bootstrap-reps',type=int,default=2000)
    args=ap.parse_args()
    OUT=args.out
    if args.mode!='probe':return MODES[args.mode](args)
    if (OUT/'FOOTBALL_RESULTS.json').exists():raise FileExistsError('Immutable completed/partial run exists; use --out NEW_DIRECTORY')
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();rng=np.random.default_rng(260924);summary={'scope':'existing canonical real coordinates; geometry proxy, no counterfactual pass-success labels',
      'mask_baseline':'controlled receiver and chosen defender stale for 0.2 seconds; both use only past coordinates; not natural missingness learning',
      'match_split':{'J03WOY':'train','J03WMX':'dev','J03WN1':'test'},'matches':{}}
    records=[]
    for match,split in summary['match_split'].items():
      source=ROOT/f'artifacts/data_v2/idsse/canonical/frames/{match}.parquet'
      fr=pd.read_parquet(source);ev=pd.read_parquet(ROOT/f'artifacts/data_v2/idsse/canonical/events/{match}.parquet')
      ev=ev.sort_values('timestamp_ms');missing_before=int(ev.game_section.isna().sum())
      ev['period_source']=np.where(ev.game_section.isna(),'forward_filled_event_boundary','explicit')
      ev['game_section']=ev.game_section.ffill()
      cap,kin=motion_cap(fr)
      bysection={sec:df.set_index('timestamp_ms',drop=False).sort_index() for sec,df in fr.groupby('game_section')}
      times={sec:np.unique(df.index) for sec,df in bysection.items()}
      def frame(sec,t,causal=False):
        if sec not in times:raise ValueError('no_period')
        tt=times[sec]
        if causal:
          v=tt[tt<=t]
          if not len(v):raise ValueError('no_past_frame')
          chosen=v[-1]
        else:chosen=tt[np.abs(tt-t).argmin()]
        if abs(chosen-t)>500:raise ValueError('frame_gap')
        chunk=bysection[sec].loc[[chosen]].reset_index(drop=True)
        return select_frame(chunk,match,sec,t)
      passes=ev[(ev.event_type=='Play:Pass')&ev.recipient_id.notna()]
      # Spread fixed sample over the whole match; no model/label enrichment here.
      chosen=passes.iloc[np.unique(np.linspace(0,len(passes)-1,min(args.events_per_match,len(passes))).astype(int))]
      skips=collections.Counter();used=0;natural=[];candidate=[];baseline=collections.defaultdict(list)
      entity=fr.entity_type.value_counts(dropna=False).to_dict()
      player_counts=fr[fr.entity_type=='player'].groupby(['game_section','timestamp_ms']).size()
      for _,e in chosen.iterrows():
        try:
          rows=frame(e.game_section,e.timestamp_ms)
          p,r,D,ids,nonplayers=audit_roles(rows,e.player_id,e.recipient_id,e.team_id)
        except ValueError as err:skips[str(err)]+=1;continue
        used+=1;sec=e.game_section;timestamp=int(rows.timestamp_ms.iloc[0])
        team=rows[(rows.entity_type=='player')&(rows.team_id==e.team_id)&(rows.person_id!=e.player_id)]
        try:past=frame(sec,timestamp-200,True).set_index('person_id');past2=frame(sec,timestamp-400,True).set_index('person_id')
        except ValueError:past=past2=None
        for _,recipient in team.iterrows():
          rr=np.array([recipient.x,recipient.y]);dist=float(np.linalg.norm(rr-p))
          if dist<2 or dist>45 or abs(rr[0])>52.5 or abs(rr[1])>34:continue
          result=quartet_search(p,rr,D,cap,rng)
          selected=recipient.person_id==e.recipient_id
          row={'match':match,'split':split,'event_id':str(e.event_id),'period':sec,'period_source':e.period_source,
               'event_timestamp_ms':int(e.timestamp_ms),'frame_timestamp_ms':timestamp,'alignment_ms':timestamp-int(e.timestamp_ms),
               'passer_id':e.player_id,'recipient_id':recipient.person_id,'defender_ids':ids,'nonplayer_rows_excluded':nonplayers,
               'passer_xy':p.tolist(),'receiver_xy':rr.tolist(),'defenders_xy':D.tolist(),'actual_selected_pass':bool(selected),
               'candidate_inclusion_probability_given_sampled_event':1.,'candidate_weight':1.,
               'observed_outcome':str(e.outcome) if selected else None,'probe':result}
          candidate.append(result)
          if selected:natural.append(result)
          if result['eligible'] and past is not None:
            k=result['defender_index'];pid=recipient.person_id;did=ids[k]
            if all(z in past.index and z in past2.index for z in [pid,did]):
              # Two explicitly masked entities; complete coordinates define all labels.
              complete=channel_label(p,rr,D,.5,1.)['label'];pr=past.loc[pid,['x','y']].to_numpy(float);pd_=past.loc[did,['x','y']].to_numpy(float)
              prevtime=float(past.loc[pid,'timestamp_ms']);older=float(past2.loc[pid,'timestamp_ms']);delta=prevtime-older
              if 0<delta<=500 and timestamp-prevtime<=500:
                v_r=(pr-past2.loc[pid,['x','y']].to_numpy(float))/delta
                v_d=(pd_-past2.loc[did,['x','y']].to_numpy(float))/delta
                rates={}
                for name,rhat,dhat in [('last_value',pr,pd_),('constant_velocity',pr+v_r*(timestamp-prevtime),pd_+v_d*(timestamp-prevtime))]:
                  DD=D.copy();DD[k]=dhat;pred=channel_label(p,rhat,DD,.5,1.)['label']
                  baseline[name].append([int(complete),int(pred)]);rates[name]=int(pred)
                row['controlled_mask']={'truth':int(complete),'predictions':rates,'past_timestamp_ms':int(prevtime),'older_timestamp_ms':int(older)}
          records.append(row)
      def aggregate(results):
        eligible=[r for r in results if r['eligible']];nf=sum(r['E_found'] for r in eligible)
        keep=sum(r['single_keep_pairs'] for r in eligible);ep=sum(r['E_pairs'] for r in eligible)
        return {'candidate_n':len(results),'eligible_n':len(eligible),'E_found_n':nf,'E_found_per_candidate':nf/len(results) if results else None,
                'guided_found_n':sum(r['guided_E'] for r in eligible),'random_found_n':sum(r['random_E'] for r in eligible),
                'tested_legal_pairs':sum(r['legal_pairs'] for r in eligible),'single_keep_pair_n':keep,'E_pair_n':ep,'E_conditional_on_single_keep':ep/keep if keep else None}
      reports={}
      for name,vals in baseline.items():
        a=np.array(vals);reports[name]={'n':len(a),'accuracy':float((a[:,0]==a[:,1]).mean()),'false_open':int(((a[:,0]==0)&(a[:,1]==1)).sum()),'false_blocked':int(((a[:,0]==1)&(a[:,1]==0)).sum())}
      summary['matches'][match]={'split':split,'frame_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'event_period_missing_before_ffill':missing_before,'event_period_missing_after_ffill':int(ev.game_section.isna().sum()),
        'motion':kin,'entity_rows':entity,'unique_frames':len(player_counts),'frames_with_player_count_not_22':int((player_counts!=22).sum()),
        'nonfinite_player_coordinate_rows':int(fr.loc[fr.entity_type=='player',['x','y']].isna().any(axis=1).sum()),
        'passes_total':len(passes),'passes_sampled':len(chosen),'passes_used':used,'skips':dict(skips),
        'observed_selected_passes':aggregate(natural),'all_legal_receiver_candidates':aggregate(candidate),'controlled_mask_baselines':reports}
      print(match,'used',used,'natural',aggregate(natural),'candidates',aggregate(candidate),flush=True)
      (OUT/'FOOTBALL_RESULTS.json').write_text(json.dumps(summary,indent=2))
    (OUT/'events_candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary['wall_seconds']=time.monotonic()-started
    summary['NOT_RUN']=['natural-time E quartet prevalence (probe uses controlled edits)','learned partial-observation relation model','natural missingness state-estimator fit','native multimodal O06 (no approved provider/budget established)']
    (OUT/'FOOTBALL_RESULTS.json').write_text(json.dumps(summary,indent=2))
# --------------------------------------------------------------- R08 natural E
# Natural-E: the same oracle and the same 1449 legal candidate situations as the
# controlled probe, but the two action variables take values that real players
# actually reached (identity-linked, same match and period). A = the candidate
# receiver's position at a later real frame; B = the t1-time constraining
# defender's position at that same later frame; the passer and the other ten
# defenders stay at t1. Only the A-only and B-only cells are counterfactual; the
# joint cell crosses two real observed single-player moves. The controlled probe
# instead searched cap-bounded guided/random displacements, which is a search
# constructibility rate, not a natural occurrence rate.

PRIMARY_WINDOW_SECONDS=.2
RADIUS,END_EXCL=.5,1.
PRIMARY_CELL=(PRIMARY_WINDOW_SECONDS,'contract','forward')


def constraining_defender(p,r,D,radius=RADIUS,end_excl=END_EXCL):
    """Index of the nearest interior defender to segment p->r.

    Mirrors the rule inside the frozen f095 quartet_search; every call here is
    asserted against the stored probe defender_index, so the two spellings are
    checked to agree rather than assumed.
    """
    p,r,D=map(lambda a:np.asarray(a,float),(p,r,D))
    active=(np.linalg.norm(D-p,axis=1)>end_excl)&(np.linalg.norm(D-r,axis=1)>end_excl)
    if not active.any():return None
    distances=np.array([point_segment_dist(z,p,r) for z in D]);distances[~active]=np.inf
    return int(distances.argmin())


def _player_grid(players):
    """period -> (times_ms,{person_id:column},X,Y); NaN where a player is absent."""
    grid={}
    for period,df in players.groupby('game_section',sort=False):
        times=np.unique(df.timestamp_ms.to_numpy().astype(np.int64))
        ids=np.sort(df.person_id.unique());col={p:int(i) for i,p in enumerate(ids)}
        X=np.full((len(times),len(ids)),np.nan);Y=np.full((len(times),len(ids)),np.nan)
        ri=np.searchsorted(times,df.timestamp_ms.to_numpy().astype(np.int64));ci=df.person_id.map(col).to_numpy()
        X[ri,ci]=df.x.to_numpy();Y[ri,ci]=df.y.to_numpy()
        grid[period]=(times,col,X,Y)
    return grid


def _lag_displacement(g,window_ms):
    """Identity-linked real displacements at lag window_ms within one period."""
    times,col,X,Y=g;t=times.astype(float)+window_ms
    j=np.clip(np.searchsorted(times,t),0,len(times)-1);jm=np.clip(j-1,0,len(times)-1)
    pick=np.where(np.abs(times[j]-t)<=np.abs(t-times[jm]),j,jm).astype(int)
    ok=np.abs(times[pick]-t)<=21.
    return np.hypot(X[pick]-X,Y[pick]-Y)[ok].ravel()


def natural_E_truth(y0,yA,yB,yAB):
    """E iff both single edits preserve the base label and the joint edit flips it.

    Label-agnostic: y0=0 (blocked -> open) counts as well. This is the predicate
    the historical `supported_single_keep_and_flip` statistic is NOT.
    """
    return bool(yA==y0 and yB==y0 and yAB!=y0)


def natural_cell_labels(p,r,D,RR,DD,base,radius=RADIUS,end_excl=END_EXCL):
    """(y0,yA,yB,yAB) for one situation: passer fixed, only receiver/defender k moved."""
    n=len(RR);fixed=np.broadcast_to(D,(n,*D.shape))
    y0=np.full(n,int(base))
    yA=labels_batch(p,RR,fixed,radius,end_excl)
    yB=labels_batch(p,np.broadcast_to(r,(n,2)),DD,radius,end_excl)
    yAB=labels_batch(p,RR,DD,radius,end_excl)
    return y0,yA,yB,yAB


def _labels_pairwise_p(p,receivers,defenders,radius=RADIUS,end_excl=END_EXCL):
    """d06_roles.labels_batch generalised to per-pair passer positions (real-state label)."""
    p=np.asarray(p,float);r=np.asarray(receivers,float);D=np.asarray(defenders,float)
    v=r-p;den=(v*v).sum(1)
    t=np.clip(((D-p[:,None,:])*v[:,None,:]).sum(2)/np.maximum(den[:,None],1e-12),0,1)
    dist=np.linalg.norm(D-(p[:,None,:]+v[:,None,:]*t[:,:,None]),axis=2)
    valid=(np.linalg.norm(D-p[:,None,:],axis=2)>end_excl)&(np.linalg.norm(D-r[:,None,:],axis=2)>end_excl)
    return (~((dist<radius)&valid).any(1)).astype(int)


def load_situations(path,radius=RADIUS,end_excl=END_EXCL):
    """Read the frozen probe situation table and re-check every base label/index."""
    rows=[json.loads(l) for l in path.open()]
    audit={'source':str(path.relative_to(ROOT)),'n_records':len(rows),'base_label_mismatch':0,
           'defender_index_mismatch':0,'vector_vs_scalar_base_mismatch':0,'missing_required_fields':0}
    fields=['match','period','frame_timestamp_ms','passer_id','recipient_id','defender_ids',
            'passer_xy','receiver_xy','defenders_xy','probe']
    for rec in rows:
        audit['missing_required_fields']+=int(any(f not in rec for f in fields))
        p=np.asarray(rec['passer_xy'],float);r=np.asarray(rec['receiver_xy'],float)
        D=np.asarray(rec['defenders_xy'],float)
        rec['_base']=int(channel_label(p,r,D,radius,end_excl)['label'])
        rec['_k']=constraining_defender(p,r,D,radius,end_excl)
        audit['base_label_mismatch']+=int(rec['_base']!=int(rec['probe']['base_label']))
        audit['defender_index_mismatch']+=int(rec['_k']!=int(rec['probe']['defender_index']))
        audit['vector_vs_scalar_base_mismatch']+=int(int(labels_batch(p,r[None],D[None],radius,end_excl)[0])!=rec['_base'])
        rec['_sid']=f"{rec['match']}:{rec['event_id']}:{rec['recipient_id']}"
    assert audit['base_label_mismatch']==0 and audit['defender_index_mismatch']==0,audit
    assert audit['vector_vs_scalar_base_mismatch']==0 and audit['missing_required_fields']==0,audit
    return rows,audit


def _cluster_bootstrap(events,flags,reps,rng):
    """Percentile CI resampling whole events (the sampling unit), not situations."""
    uniq,inv=np.unique(events,return_inverse=True)
    by=[np.flatnonzero(inv==i) for i in range(len(uniq))]
    rates=np.empty(reps)
    for b in range(reps):
        pick=rng.integers(0,len(uniq),len(uniq))
        rates[b]=flags[np.concatenate([by[i] for i in pick])].mean()
    return [float(np.quantile(rates,.025)),float(np.quantile(rates,.975))]


def natural_e_main(args):
    sit_path=args.situations or (args.out/'events_candidates.jsonl')
    res_path=args.out/'FOOTBALL_RESULTS.json'
    natural_out=args.natural_out or (args.out/'NATURAL_E_RATE.json')
    pairs_path=natural_out.with_name('natural_pairs_primary.jsonl')
    if natural_out.exists() or pairs_path.exists():
        raise FileExistsError('natural output exists; write to a new path with --natural-out')
    windows=[float(x) for x in args.windows.split(',')]
    results=json.loads(res_path.read_text())
    rows,audit=load_situations(sit_path)
    by_match=collections.OrderedDict()
    for rec in rows:by_match.setdefault(rec['match'],[]).append(rec)
    rng=np.random.default_rng(260925)
    out={'scope':'natural E occurrence on real football tracking coordinates; geometry oracle, no pass-success labels',
         'mode':'natural','primary_cell':{'window_seconds':PRIMARY_CELL[0],'cap_rule':PRIMARY_CELL[1],'direction':PRIMARY_CELL[2]},
         'situation_source':audit,'windows_seconds':windows,
         'cap_rules':{'contract':'speed_p95*W + 0.5*accel_p95*W^2 from same player/match/period consecutive diffs 0<dt<=0.12s (R08 motion_cap formula)',
                      'empirical':'directly measured identity-linked same-period displacement p95 at lag W'},
         'directions':{'forward':'later real frames only (0<dt<=W)','both':'past and later real frames (0<|dt|<=W)'},
         'oracle':'AND over defenders of not-blocked; open iff no defender disc (radius=0.5, endpoint exclusion 1.0m) covers the segment interior',
         'E_definition':'y(x+eA)=y(x+eB)=y(x) and y(x+eA+eB)!=y(x), label-agnostic (y0=0 works too)',
         'variables':'A = candidate receiver real displacement to a later frame; B = t1-time constraining defender real displacement to the same frame; passer and the other ten defenders held at t1',
         'real_vs_analytic':'x and both real single-player moves are real observations; the A-only, B-only and joint cell labels are model-independent analytic oracle evaluations; y_real uses the real positions of passer/receiver/all eleven defenders at the later frame',
         'matches':collections.OrderedDict(),'pair_rows_written':0}
    pair_rows=[]
    for match,recs in by_match.items():
        frames=pd.read_parquet(ROOT/f'artifacts/data_v2/idsse/canonical/frames/{match}.parquet')
        players=frames[frames.entity_type=='player']
        cap,kin=motion_cap(frames,PRIMARY_WINDOW_SECONDS)
        grid=_player_grid(players)
        cap_contract={W:kin['speed_p95_m_s']*W+.5*kin['acceleration_p95_m_s2']*W**2 for W in windows}
        cap_emp={W:float(np.nanquantile(np.concatenate([_lag_displacement(g,int(round(W*1000))) for g in grid.values()]),.95)) for W in windows}
        cells={};sit_E={};sit_ev=[];sit_sel=[]
        for rec in recs:
            p=np.asarray(rec['passer_xy'],float);r=np.asarray(rec['receiver_xy'],float)
            D=np.asarray(rec['defenders_xy'],float);k=rec['_k'];base=rec['_base']
            g=grid[rec['period']];times,col,X,Y=g
            t1=int(rec['frame_timestamp_ms']);i1=int(np.searchsorted(times,t1))
            assert times[i1]==t1,(match,t1)
            cr=col.get(rec['recipient_id']);cd=col.get(rec['defender_ids'][k]);cp=col.get(rec['passer_id'])
            dcols=np.array([col.get(d,-1) for d in rec['defender_ids']])
            assert cr is not None and cd is not None and cp is not None and (dcols>=0).all(),rec['_sid']
            sit_ev.append(f"{match}:{rec['event_id']}");sit_sel.append(bool(rec['actual_selected_pass']))
            dt=(times-t1)/1000.
            for W in windows:
                for direction in ('forward','both'):
                    sel=(dt>0)&(dt<=W) if direction=='forward' else (dt!=0)&(np.abs(dt)<=W)
                    idx=np.flatnonzero(sel)
                    base_key=(W,direction)
                    for cap_rule,caps in (('contract',cap_contract),('empirical',cap_emp)):
                        key=(W,cap_rule,direction)
                        c=cells.setdefault(key,{'window_seconds':W,'direction':direction,'cap_rule':cap_rule,
                            'cap_m':caps[W],'n_pairs_in_window':0,'excluded_absent_at_t2':0,
                            'excluded_out_of_field':0,'excluded_over_cap':0,'n_pairs_in_cap':0,
                            'truth_table':collections.Counter(),'E_pairs_by_base_label':collections.Counter(),
                            'single_keep_pairs_by_base_label':collections.Counter(),'situations_with_E':0,
                            'situations_with_E_selected':0,'dr_m':[],'dk_m':[],'real_state':collections.Counter(),
                            'real_state_available':0,'E_pairs_with_real_flip':0,'real_flip_pairs':0,'_sit_E':[]})
                        c['n_pairs_in_window']+=int(len(idx))
                        if cr is None or cd is None or not len(idx):c['_sit_E'].append(0);continue
                        dr=np.hypot(X[idx,cr]-r[0],Y[idx,cr]-r[1]);dk=np.hypot(X[idx,cd]-D[k,0],Y[idx,cd]-D[k,1])
                        absent=~(np.isfinite(dr)&np.isfinite(dk))
                        c['excluded_absent_at_t2']+=int(absent.sum())
                        r2=np.stack([X[idx,cr],Y[idx,cr]],1);d2=np.stack([X[idx,cd],Y[idx,cd]],1)
                        infield=(np.abs(r2[:,0])<=52.5)&(np.abs(r2[:,1])<=34)&(np.abs(d2[:,0])<=52.5)&(np.abs(d2[:,1])<=34)
                        c['excluded_out_of_field']+=int((~infield&~absent).sum())
                        over=(dr>caps[W])|(dk>caps[W]);c['excluded_over_cap']+=int((over&~absent).sum())
                        keep=np.flatnonzero(~absent&infield&~over)
                        if not len(keep):c['_sit_E'].append(0);continue
                        c['n_pairs_in_cap']+=int(len(keep))
                        RR=r2[keep];DD=np.broadcast_to(D,(len(keep),*D.shape)).copy();DD[:,k]=d2[keep]
                        _,A,B,AB=natural_cell_labels(p,r,D,RR,DD,base)
                        real_ok=np.isfinite(X[idx[keep]][:,cp])&np.isfinite(X[idx[keep]][:,dcols]).all(1) if cp is not None else np.zeros(len(keep),bool)
                        real_lab=np.full(len(keep),-1)
                        if real_ok.any():
                            D2=np.stack([X[idx[keep]][:,dcols],Y[idx[keep]][:,dcols]],2)[real_ok]
                            p2=np.stack([X[idx[keep]][:,cp],Y[idx[keep]][:,cp]],1)[real_ok]
                            real_lab[real_ok]=_labels_pairwise_p(p2,RR[real_ok],D2)
                        c['dr_m']+=dr[keep].tolist();c['dk_m']+=dk[keep].tolist()
                        for a,b,ab,rl in zip(A,B,AB,real_lab):
                            c['truth_table'][f'{base}{a}{b}{ab}']+=1
                            if (a==base)and(b==base):
                                c['single_keep_pairs_by_base_label'][base]+=1
                            if natural_E_truth(base,a,b,ab):
                                c['E_pairs_by_base_label'][base]+=1
                                c['E_pairs_with_real_flip']+=int(rl!=-1 and rl!=base)
                            if rl!=-1:
                                c['real_state_available']+=1;c['real_state'][f'{base}{rl}']+=1
                                c['real_flip_pairs']+=int(rl!=base)
                        E=np.array([natural_E_truth(base,a,b,ab) for a,b,ab in zip(A,B,AB)])
                        c['_sit_E'].append(int(E.any()))
                        if E.any():
                            c['situations_with_E']+=1
                            c['situations_with_E_selected']+=int(bool(rec['actual_selected_pass']))
                        if key==PRIMARY_CELL:
                            for j,i2 in enumerate(keep):
                                pair_rows.append({'situation_id':rec['_sid'],'match':match,'split':rec['split'],
                                    'event_id':rec['event_id'],'period':rec['period'],'frame_timestamp_ms':t1,
                                    't2_timestamp_ms':int(times[idx[i2]]),'dt_seconds':float(dt[idx[i2]]),
                                    'passer_id':rec['passer_id'],'receiver_id':rec['recipient_id'],
                                    'defender_id':rec['defender_ids'][k],'defender_index':k,
                                    'receiver_displacement_m':float(dr[i2]),'defender_displacement_m':float(dk[i2]),
                                    'cap_m':caps[W],'actual_selected_pass':bool(rec['actual_selected_pass']),
                                    'labels_y0_yA_yB_yAB':[int(base),int(A[j]),int(B[j]),int(AB[j])],
                                    'real_state_label_at_t2':int(real_lab[j])})
        match_out={'split':results['matches'][match]['split'],'frame_source_sha256':results['matches'][match]['frame_source_sha256'],
            'n_situations':len(recs),'n_events':len(set(sit_ev)),'n_selected_situations':int(sum(sit_sel)),
            'kinematics':kin,'cap_contract_m':cap_contract,'empirical_displacement_p95_m':cap_emp,'cells':{}}
        for key,c in cells.items():
            sit_E=np.array(c.pop('_sit_E'));ev=np.array(sit_ev);sel=np.array(sit_sel)
            n_sel=int(sel.sum());E_sel=int(sit_E[sel].sum())
            ev_uniq=np.unique(ev);ev_flag=np.array([sit_E[ev==u].any() for u in ev_uniq])
            single_keep=sum(c['single_keep_pairs_by_base_label'].values());E_pairs=sum(c['E_pairs_by_base_label'].values())
            n_in=c['n_pairs_in_cap']
            c['truth_table']=dict(sorted(c['truth_table'].items()))
            c['E_pairs_by_base_label']=dict(sorted(c['E_pairs_by_base_label'].items()))
            c['single_keep_pairs_by_base_label']=dict(sorted(c['single_keep_pairs_by_base_label'].items()))
            c['real_state']=dict(sorted(c['real_state'].items()))
            c['natural_E_pairs']=E_pairs;c['single_keep_pairs']=single_keep
            c['natural_E_pair_rate']=E_pairs/n_in if n_in else None
            c['natural_E_conditional_on_single_keep']=E_pairs/single_keep if single_keep else None
            c['natural_E_situation_rate']=float(sit_E.mean()) if len(sit_E) else None
            c['natural_E_situation_rate_selected_receivers']=E_sel/n_sel if n_sel else None
            c['natural_E_event_rate']=float(ev_flag.mean()) if len(ev_flag) else None
            c['n_situations_with_E']=int(sit_E.sum());c['n_events_with_E']=int(ev_flag.sum());c['n_events']=int(len(ev_uniq))
            c['dr_quantiles_m']={q:float(np.quantile(c['dr_m'],q)) for q in (.5,.95,1.)} if c['dr_m'] else None
            c['dk_quantiles_m']={q:float(np.quantile(c['dk_m'],q)) for q in (.5,.95,1.)} if c['dk_m'] else None
            c.pop('dr_m');c.pop('dk_m')
            if key==PRIMARY_CELL:
                c['situation_rate_ci95_cluster_bootstrap_events']=_cluster_bootstrap(ev,sit_E,args.bootstrap_reps,rng)
                c['event_rate_ci95_cluster_bootstrap_events']=_cluster_bootstrap(ev_uniq,ev_flag.astype(float),args.bootstrap_reps,rng)
            match_out['cells'][f"W{c['window_seconds']}|{c['cap_rule']}|{c['direction']}"]=c
        out['matches'][match]=match_out
        print(match,'situations',len(recs),'events',len(set(sit_ev)),flush=True)
    # pooled rates over the same cell definition; caps differ per match by design
    pooled={'n_situations':int(sum(m['n_situations'] for m in out['matches'].values())),
            'n_selected_situations':int(sum(m['n_selected_situations'] for m in out['matches'].values())),
            'n_events':int(sum(m['n_events'] for m in out['matches'].values())),'cells':{}}
    for key in out['matches'][list(out['matches'])[0]]['cells']:
        pcell={};pairs=sum(m['cells'][key]['n_pairs_in_cap'] for m in out['matches'].values())
        Ep=sum(m['cells'][key]['natural_E_pairs'] for m in out['matches'].values())
        sk=sum(m['cells'][key]['single_keep_pairs'] for m in out['matches'].values())
        nsit=sum(m['cells'][key]['n_situations_with_E'] for m in out['matches'].values())
        nev=sum(m['cells'][key]['n_events_with_E'] for m in out['matches'].values())
        pcell={'n_pairs_in_cap':pairs,'natural_E_pairs':Ep,'single_keep_pairs':sk,
               'natural_E_pair_rate':Ep/pairs if pairs else None,
               'natural_E_conditional_on_single_keep':Ep/sk if sk else None,
               'n_situations_with_E':nsit,'natural_E_situation_rate':nsit/pooled['n_situations'],
               'n_events_with_E':nev,'natural_E_event_rate':nev/pooled['n_events']}
        pooled['cells'][key]=pcell
    out['pooled']=pooled
    ref={}
    for match,m in results['matches'].items():
        a=m['all_legal_receiver_candidates'];b=m['observed_selected_passes']
        ref[match]={'sampled_candidates':{'E_found':a['E_found_n'],'n':a['candidate_n'],'rate':a['E_found_per_candidate'],
                      'E_pairs':a['E_pair_n'],'single_keep_pairs':a['single_keep_pair_n'],
                      'E_conditional_on_single_keep':a['E_conditional_on_single_keep']},
                    'selected_receivers':{'E_found':b['E_found_n'],'n':b['candidate_n'],'rate':b['E_found_per_candidate'],
                      'E_pairs':b['E_pair_n'],'single_keep_pairs':b['single_keep_pair_n'],
                      'E_conditional_on_single_keep':b['E_conditional_on_single_keep']}}
    out['controlled_search_reference']={'source':str(res_path.relative_to(ROOT)),'note':'immutable R08 probe results, read not recomputed',
        'matches':ref,'pooled':{'sampled_candidates':{'E_found':670,'n':1449},'selected_receivers':{'E_found':48,'n':155}}}
    prim=f'W{PRIMARY_CELL[0]}|{PRIMARY_CELL[1]}|{PRIMARY_CELL[2]}'
    out['enrichment_primary_cell']={'cell':prim,'per_match':{},'note':'ratio = controlled search constructibility rate / natural situation rate; both on the same 1449 situations'}
    for match,m in out['matches'].items():
        nat=m['cells'][prim]['natural_E_situation_rate'];sr=ref[match]['sampled_candidates']['rate']
        out['enrichment_primary_cell']['per_match'][match]={'natural_situation_rate':nat,'search_constructibility_rate':sr,
            'ratio':(sr/nat) if nat else None}
    nat=out['pooled']['cells'][prim]['natural_E_situation_rate']
    out['enrichment_primary_cell']['pooled']={'natural_situation_rate':nat,'search_constructibility_rate':670/1449,
        'ratio':(670/1449)/nat if nat else None}
    out['boundaries']=['caps and windows are empirical motion envelopes, not joint dynamic-feasibility proofs',
        'the joint cell crosses two real single-player moves and is not a real observed frame',
        'y_real requires passer, receiver and all eleven t1-identified defenders present at the later frame',
        'forward windows only in the primary cell; situation set is the R08 240-event sample (162 audited events), not every pass of the three matches',
        'no threshold is applied to call anything infeasible; a low natural rate is a rare-stress-test statement',
        'reports/e1a933_review/FOOTBALL_E_FEASIBILITY_V2.md is generated by football_summarize.py and still lists the natural rate as NOT_RUN; this lane did not rewrite that generator, so its line is stale',
        'the situation table is the frozen R08 probe output (re-read, not re-searched); the probe path of this script is unchanged']
    out['commands']=[f"OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 .venv/bin/python experiments/e1a933_review/football_repair.py --mode natural --windows {args.windows}",
        'OPENBLAS_NUM_THREADS=6 .venv/bin/python experiments/e1a933_review/football_repair.py --mode verify-natural',
        'OPENBLAS_NUM_THREADS=6 .venv/bin/python experiments/e1a933_review/football_repair.py --mode legacy-recheck',
        'OPENBLAS_NUM_THREADS=6 .venv/bin/python experiments/e1a933_review/football_repair.py --mode natural-report']
    out['NOT_RUN']=['real pass-success labels for counterfactual cells (the oracle is geometric, not outcome-based)',
        'natural E rate over every pass of the three matches (only the R08 240-event sample is measured)',
        'learned partial-observation relation model and natural-missingness state estimation (O05 scope)',
        'other four available matches (outside the R08 train/dev/test naming; J03WQQ is sealed)']
    out['pair_rows_written']=len(pair_rows)
    natural_out.write_text(json.dumps(out,indent=2))
    pairs_path.write_text(''.join(json.dumps(r)+'\n' for r in pair_rows))
    print('wrote',natural_out,pairs_path,len(pair_rows))


def verify_natural_main(args):
    """Re-derive every primary-cell pair and re-check every natural E witness with
    the scalar production oracle (channel_label), not the vectorised path."""
    natural_out=args.natural_out or (args.out/'NATURAL_E_RATE.json')
    pairs_path=natural_out.with_name('natural_pairs_primary.jsonl')
    out=json.loads(natural_out.read_text());rows=[json.loads(l) for l in pairs_path.open()]
    sit={}
    for line in (args.situations or (args.out/'events_candidates.jsonl')).open():
        rec=json.loads(line);sit[f"{rec['match']}:{rec['event_id']}:{rec['recipient_id']}"]=rec
    frames_cache={}
    def frame(match,period,t):
        if match not in frames_cache:
            fr=pd.read_parquet(ROOT/f'artifacts/data_v2/idsse/canonical/frames/{match}.parquet')
            fr=fr[fr.entity_type=='player']
            frames_cache[match]={(p,int(tt)):g.set_index('person_id')[['x','y']]
                                 for (p,tt),g in fr.groupby(['game_section','timestamp_ms'])}
        return frames_cache[match][(period,int(t))]
    prim=f"W{PRIMARY_CELL[0]}|{PRIMARY_CELL[1]}|{PRIMARY_CELL[2]}"
    counts=collections.Counter();by_match=collections.defaultdict(collections.Counter);invalid=[]
    for row in rows:
        y0,ya,yb,yab=row['labels_y0_yA_yB_yAB']
        counts['pairs']+=1;by_match[row['match']]['pairs']+=1
        E=bool(ya==y0 and yb==y0 and yab!=y0);keep=bool(ya==y0 and yb==y0)
        counts['E_pairs']+=int(E);by_match[row['match']]['E_pairs']+=int(E)
        counts['single_keep_pairs']+=int(keep);by_match[row['match']]['single_keep_pairs']+=int(keep)
        by_match[row['match']][f'{y0}{ya}{yb}{yab}']+=1
        if not E:continue
        rec=sit[row['situation_id']]
        p=np.asarray(rec['passer_xy'],float);r=np.asarray(rec['receiver_xy'],float);D=np.asarray(rec['defenders_xy'],float)
        f2=frame(row['match'],row['period'],row['t2_timestamp_ms'])
        missing=[q for q in (row['receiver_id'],row['defender_id']) if q not in f2.index]
        if missing:counts['witness_missing_at_t2']+=1;invalid.append(row['situation_id']);continue
        r2=f2.loc[row['receiver_id'],['x','y']].to_numpy(float);d2=f2.loc[row['defender_id'],['x','y']].to_numpy(float)
        DD=D.copy();DD[row['defender_index']]=d2
        got=[channel_label(p,r,D,RADIUS,END_EXCL)['label'],channel_label(p,r2,D,RADIUS,END_EXCL)['label'],
             channel_label(p,r,DD,RADIUS,END_EXCL)['label'],channel_label(p,r2,DD,RADIUS,END_EXCL)['label']]
        cap=row['cap_m']
        good=(got==[y0,ya,yb,yab] and got[0]==got[1]==got[2] and got[3]!=got[0]
              and np.linalg.norm(r2-r)<=cap+1e-9 and np.linalg.norm(d2-D[row['defender_index']])<=cap+1e-9
              and np.isclose(row['receiver_displacement_m'],np.linalg.norm(r2-r),atol=1e-9)
              and np.isclose(row['defender_displacement_m'],np.linalg.norm(d2-D[row['defender_index']]),atol=1e-9))
        counts['witnesses_checked']+=1
        if not good:counts['witnesses_invalid']+=1;invalid.append(row['situation_id'])
    reported={m:out['matches'][m]['cells'][prim]['natural_E_pairs'] for m in out['matches']}
    truth_matches=all({k:v for k,v in by_match[m].items() if k!='pairs' and k!='E_pairs' and k!='single_keep_pairs'}
                      ==out['matches'][m]['cells'][prim]['truth_table'] for m in out['matches'])
    verification={'primary_cell':out['primary_cell'],'pair_rows':len(rows),
        'pair_row_counts':{k:dict(v) for k,v in by_match.items()},
        'scalar_oracle_witness_recheck':dict(counts),'invalid_situation_ids':invalid[:20],
        'pair_rows_match_reported':all(by_match[m]['E_pairs']==reported[m] for m in reported),
        'full_truth_table_matches_json':truth_matches,
        'reported_primary_cell_E_pairs_by_match':reported,
        'scalar_production_oracle':True,
        'cap_matches_json':all(np.isclose(row['cap_m'],out['matches'][row['match']]['cells'][prim]['cap_m']) for row in rows)}
    assert not invalid and verification['pair_rows_match_reported'] and verification['cap_matches_json'] and truth_matches,verification
    (natural_out.with_name('NATURAL_E_VERIFICATION.json')).write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))


def legacy_recheck_main(args):
    """Re-run the frozen HEAD d06_roles.py probe into a temp directory and compare
    its `paired` statistic with the committed artifact, so the renamed statistic
    (supported_single_keep_and_flip) is shown to keep the old numbers reproducible.
    Nothing inside the repository is written except this receipt."""
    import os,subprocess,tempfile
    head=subprocess.run(['git','show','HEAD:experiments/f095_campaign/d06_roles.py'],cwd=ROOT,
                        capture_output=True,text=True,check=True).stdout
    old=json.loads((ROOT/'artifacts/f095_campaign/D06/D06_ROLES_J03WOY.json').read_text())
    with tempfile.TemporaryDirectory() as td:
        src=head.replace('ROOT = Path(__file__).resolve().parents[2]',f'ROOT = Path({str(ROOT)!r})')
        src=src.replace('OUT = ROOT / "artifacts" / "f095_campaign" / "D06"',f'OUT = Path({td!r})')
        assert 'OUT = Path(' in src and f'ROOT = Path({str(ROOT)!r})' in src
        script=Path(td)/'d06_roles_legacy.py';script.write_text(src)
        subprocess.run([str(ROOT/'.venv/bin/python'),str(script),'--match','J03WOY'],cwd=ROOT,check=True,
                       capture_output=True,text=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'6','OMP_NUM_THREADS':'6'})
        new=json.loads((Path(td)/'D06_ROLES_J03WOY.json').read_text())
    keys=['n_pass','used','paired','keep_only','flip_only','neither','paired_rate','margin_median','margin_frac_open']
    cmp={k:{'committed':old[k],'rerun':new[k],'match':old[k]==new[k]} for k in keys}
    receipt={'statistic':'paired (historical name) == supported_single_keep_and_flip (new name in experiments/f095_campaign/d06_roles.py)',
        'renamed_in_current_code':'supported_single_keep_and_flip' in (ROOT/'experiments/f095_campaign/d06_roles.py').read_text(),
        'old_artifact_preserved':'paired' in old,
        'committed_artifact':'artifacts/f095_campaign/D06/D06_ROLES_J03WOY.json',
        'head_script':'git show HEAD:experiments/f095_campaign/d06_roles.py','match':'J03WOY',
        'comparison':cmp,'all_match':all(v['match'] for v in cmp.values()),
        'contested':{'committed':old['contested'],'rerun':new['contested']},
        'by_outcome':{'committed':old['by_outcome'],'rerun':new['by_outcome']},
        'note':'the re-run writes only into a temporary directory; the frozen f095 artifact is untouched'}
    (args.out/'LEGACY_STAT_RECHECK.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt['comparison'],indent=2))


def natural_report_main(args):
    natural_out=args.natural_out or (args.out/'NATURAL_E_RATE.json')
    r=json.loads(natural_out.read_text());prim=f"W{PRIMARY_CELL[0]}|{PRIMARY_CELL[1]}|{PRIMARY_CELL[2]}"
    pct=lambda v:f"{100*v:.2f}%" if v is not None else 'NA'
    cell=lambda m,key=prim:r['matches'][m]['cells'][key]
    refs=r['controlled_search_reference']['matches']
    search_pairs=sum(v['sampled_candidates']['E_pairs'] for v in refs.values())
    search_keep=sum(v['sampled_candidates']['single_keep_pairs'] for v in refs.values())
    pc=r['pooled']['cells'][prim];pe=r['enrichment_primary_cell']['pooled']
    L=['# R08 补验：自然 E 发生率（真实比赛坐标）','',
      '状态：自然 E 发生率已独立完成（本文件）；受控搜索可构率沿用 R08 只读产物；旧统计更名与可复现性见第 7 节。本 lane 的完成不等于整个施工包完成。','',
      '## 0. 一句话结论','',
      f"在同一条几何 oracle、同一批 {r['pooled']['n_situations']} 个合法情境上，把两个作用变量换成真实观测位移后，自然 E 情境率为 "
      f"{pct(pc['natural_E_situation_rate'])}（{pc['n_situations_with_E']}/{r['pooled']['n_situations']}），而受控搜索在同一批情境上的可构率为 "
      f"{pct(670/1449)}（{670}/1449），富集约 {pe['ratio']:.0f} 倍。自然 E 存在，但属于稀有压力测试，不是常见情形。",'',
      '## 1. 这条补的是什么','',
      '施工卡 R08 要求三个分母分开。原产物只有「搜索可构率」：在 cap 内用解析法向引导（49 对）＋随机对照（128 对）**搜索** witness，得到 670/1449。',
      '搜索可以沿法向反复试探并二分边界，因此它是「在给定 cap 内是否存在 witness」的构造率，不是「自然情境中是否出现 E」的发生率。',
      '本文件把同一条 oracle、同一批合法情境上的两个作用变量改成**真实观测值**：',
      'A = 接球候选者在后续真实帧上的位置；B = t1 时刻真正约束通道的防守者（身份固定）在同一后续真实帧上的位置；传球者与其余 10 名防守者保持 t1 坐标。',
      '四个格子中只有 A-only、B-only 是反事实，联合格是两个**真实单人位移**的交叉；没有任何引导、随机搜索或 cap 放大。','',
      '## 2. 分子/分母定义（可核）','',
      f"- 情境集合：`{r['situation_source']['source']}`，{r['situation_source']['n_records']} 条记录（240 个抽样传球事件中通过角色/时间审计的 {r['pooled']['n_events']} 个事件内全部同队 2–45m 场内接球候选）。载入时逐条用标量 oracle 复核 base 标签与 defender_index：不一致 {r['situation_source']['base_label_mismatch']} / {r['situation_source']['defender_index_mismatch']} 条。",
      '- 自然对：同一 match+period 的真实帧对 (t1,t2)，dt 落在窗口内，接球者与防守者 k 在 t2 都真实存在（身份链接、坐标有限），两人位移都 ≤ cap，且移动后位置仍在场内（|x|≤52.5, |y|≤34）。',
      f"- cap 合同：{r['cap_rules']['contract']}。替代 cap：{r['cap_rules']['empirical']}。",
      f"- 方向：{r['directions']['forward']}；{r['directions']['both']}。",
      f"- 主格（开工前指定，非事后挑选）：窗口 {PRIMARY_CELL[0]}s、合同 cap、forward。",
      f"- E 判定：{r['E_definition']}。",
      f"- 真实观测 vs 解析计算：{r['real_vs_analytic']}。",
      f"- 三个分母：(i) 自然 E 率，分母 = 全部合法情境 {r['pooled']['n_situations']}（另给实际被选中接球者的子集）；(ii) 候选抽样富集率，分母 = 被抽样的合法候选 1449，分子 = 受控搜索找到 witness 的候选 670（沿用只读产物）；(iii) E 条件率，分母 = 两个单编辑都保持 base 的编辑对，分子 = 其中联合编辑翻转的编辑对。",'',
      '## 3. 主表：三个分母（逐场，主格）','',
      '| 比赛 / 用途 | 情境数 | (i) 自然E情境率 | (i) 自然E事件率 | (i) 自然E率(实际被选中接球者) | (ii) 搜索可构率 | 富集比 (ii)/(i) | (iii) 搜索E条件率 | (iii) 自然E条件率 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in r['matches']:
        c=cell(m);e=r['enrichment_primary_cell']['per_match'][m];ref=refs[m]['sampled_candidates'];mo=r['matches'][m]
        L.append(f"| {m} / {mo['split']} | {mo['n_situations']} | {pct(c['natural_E_situation_rate'])} ({c['n_situations_with_E']}/{mo['n_situations']}) | "
                 f"{pct(c['natural_E_event_rate'])} ({c['n_events_with_E']}/{c['n_events']}) | "
                 f"{pct(c['natural_E_situation_rate_selected_receivers'])} ({c['situations_with_E_selected']}/{mo['n_selected_situations']}) | "
                 f"{pct(ref['rate'])} ({ref['E_found']}/{ref['n']}) | {e['ratio']:.0f}x | "
                 f"{pct(ref['E_conditional_on_single_keep'])} ({ref['E_pairs']}/{ref['single_keep_pairs']}) | "
                 f"{pct(c['natural_E_conditional_on_single_keep'])} ({c['natural_E_pairs']}/{c['single_keep_pairs']}) |")
    L.append(f"| **合计** | {r['pooled']['n_situations']} | {pct(pc['natural_E_situation_rate'])} ({pc['n_situations_with_E']}/{r['pooled']['n_situations']}) | "
             f"{pct(pc['natural_E_event_rate'])} ({pc['n_events_with_E']}/{r['pooled']['n_events']}) | "
             f"{pct(sum(cell(m)['situations_with_E_selected'] for m in r['matches'])/r['pooled']['n_selected_situations'])} "
             f"({sum(cell(m)['situations_with_E_selected'] for m in r['matches'])}/{r['pooled']['n_selected_situations']}) | "
             f"{pct(670/1449)} ({670}/1449) | {pe['ratio']:.0f}x | {pct(search_pairs/search_keep)} ({search_pairs}/{search_keep}) | "
             f"{pct(pc['natural_E_conditional_on_single_keep'])} ({pc['natural_E_pairs']}/{pc['single_keep_pairs']}) |")
    L+=['',f"- (i) 的自然 E 情境率 95% 事件簇 bootstrap 区间（按事件重抽，事件是抽样单位）："+
        '；'.join(f"{m} {pct(cell(m)['situation_rate_ci95_cluster_bootstrap_events'][0])}–{pct(cell(m)['situation_rate_ci95_cluster_bootstrap_events'][1])}" for m in r['matches'])+'。',
      f"- (i) 的自然 E 事件率 95% 区间："+'；'.join(f"{m} {pct(cell(m)['event_rate_ci95_cluster_bootstrap_events'][0])}–{pct(cell(m)['event_rate_ci95_cluster_bootstrap_events'][1])}" for m in r['matches'])+'。',
      f"- (ii) 与 (i) 的分子不同、分母也不同：(ii) 的分母是 1449 个被抽样的合法候选，(i) 的分母是同一批情境；两者都用同一批情境与同一条 oracle，差别只在编辑生成方式（搜索 vs 真实位移）。",
      '- 富集比 = (ii)/(i)，即同一批情境上「搜索能找到 witness」与「自然位移里真的出现 E」的比值；它说明搜索是构造性的，不能反过来当自然发生率。',
      '- 阈值裁决：无。本文件不设任何「必须够大」的门槛，也不把低自然率写成不可行；低自然率读作稀有压力测试，分母逐场给出。','',
      '## 4. oracle 布尔分解（主格真值表，逐场）','',
      '| 比赛 | 窗口内真实帧对 | 入 cap 对 | t2 缺测排除 | 场内排除 | 超 cap 排除 | y0yAyByAB 计数 |','|---|---:|---:|---:|---:|---:|---|']
    for m in r['matches']:
        c=cell(m);tt=' '.join(f"{k}:{v}" for k,v in c['truth_table'].items())
        L.append(f"| {m} | {c['n_pairs_in_window']} | {c['n_pairs_in_cap']} | {c['excluded_absent_at_t2']} | {c['excluded_out_of_field']} | {c['excluded_over_cap']} | {tt} |")
    L+=['','E 格是 y0=yA=yB 且 yAB≠y0 的模式（1110 与 0001）；其余 14 格是单编辑即翻转、或联合也不翻转。','',
      '## 5. 窗口 / cap / 方向敏感性（并列报告，不做取舍）','',
      '| 比赛 | 窗口(s) | cap(m) | 方向/cap规则 | 入 cap 对 | 自然E对 | 自然E对率 | 自然E条件率 | 自然E情境率 |','|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for m in r['matches']:
        for key,c in r['matches'][m]['cells'].items():
            L.append(f"| {m} | {c['window_seconds']} | {c['cap_m']:.3f} | {c['direction']}/{c['cap_rule']} | {c['n_pairs_in_cap']} | {c['natural_E_pairs']} | "
                     f"{pct(c['natural_E_pair_rate'])} | {pct(c['natural_E_conditional_on_single_keep'])} | {pct(c['natural_E_situation_rate'])} |")
    L+=['','合同 cap 随窗口按 v95·W+0.5·a95·W² 二次增长，窗口 ≥1s 时已不是有意义的运动学界（表内直接给出 cap 值）；empirical 行用同一窗口上直接测得的同球员位移 p95。两套 cap 与两个方向全部列出，不选一个当结论。','',
      '## 6. 真实状态对照','',
      '| 比赛 | 入 cap 对 | t2 有完整帧的对 | 真实帧翻转对 | 真实帧翻转率 | 自然E对中真实帧也翻转 | 以真实 t2 帧为联合格的 E 率 |','|---|---:|---:|---:|---:|---:|---:|']
    for m in r['matches']:
        c=cell(m);n=c['real_state_available']
        L.append(f"| {m} | {c['n_pairs_in_cap']} | {n} | {c['real_flip_pairs']} | {pct(c['real_flip_pairs']/n if n else None)} | "
                 f"{c['E_pairs_with_real_flip']} | {pct(c['E_pairs_with_real_flip']/n if n else None)} |")
    L+=['','最后一列把联合格换成 t2 的**真实整帧**（传球者、接球者与 t1 身份链接的 11 名对手都在 t2 的真实位置）：这时四格里有三格是真实观测，只有 A-only/B-only 仍是反事实。它要求 t2 帧里这些身份都在，覆盖率见表。','',
      '## 7. 旧统计更名与可复现','']
    lr_path=args.out/'LEGACY_STAT_RECHECK.json'
    if lr_path.exists():
        lr=json.loads(lr_path.read_text())
        L+=['历史 `paired`（至少一个 keep 且至少一个 flip）在 `experiments/f095_campaign/d06_roles.py` 中已更名为 `supported_single_keep_and_flip`，旧产物 `artifacts/f095_campaign/D06/D06_ROLES_J03WOY.json` 原样保留。',
            f"复现检查（`--mode legacy-recheck`，把 HEAD 版 d06_roles.py 复制到临时目录运行，只写临时目录）：committed 与重跑的 `paired`={lr['comparison']['paired']['committed']} vs {lr['comparison']['paired']['rerun']}、`used`={lr['comparison']['used']['committed']} vs {lr['comparison']['used']['rerun']}、`paired_rate`={lr['comparison']['paired_rate']['committed']} vs {lr['comparison']['paired_rate']['rerun']}，全部字段一致={lr['all_match']}。",
            '该旧统计只能称 supported_single_keep_and_flip，不能称 E 可构率；它也不是自然发生率。','']
    else:
        L+=['`LEGACY_STAT_RECHECK.json` 未生成，本节 NOT_RUN。','']
    L+=['## 8. 证据与复现命令','']
    for c in r['commands']:L.append(f'```\n{c}\n```')
    L+=['','产物：`artifacts/e1a933_review/football/NATURAL_E_RATE.json`（三分母、逐场计数、真值表、bootstrap 区间）、`natural_pairs_primary.jsonl`（主格逐对标签）、`NATURAL_E_VERIFICATION.json`（标量 oracle 逐 witness 复核）、`LEGACY_STAT_RECHECK.json`。',
      '`experiments/e1a933_review/football_repair.py` 原有 `--mode probe` 路径未改动；旧产物 `artifacts/f095_campaign/`、`experiments/f095_campaign/` 未被写入。','',
      '## 9. 未完成与边界','']
    for b in r['boundaries']:L.append(f'- {b}')
    for b in r['NOT_RUN']:L.append(f'- NOT_RUN：{b}')
    (ROOT/'reports/e1a933_review/R08_NATURAL_E_RATE.md').write_text('\n'.join(L)+'\n')
    print('wrote report')


MODES={'natural':natural_e_main,'verify-natural':verify_natural_main,
       'natural-report':natural_report_main,'legacy-recheck':legacy_recheck_main}

if __name__=='__main__':main()
