"""Coordinate-based model-specific bank lineage; never compare naked integer ids.

Modes:
  matrix        (default) rebuild MODEL_BANK_OVERLAP.csv / SAMPLE_LINEAGE.csv and
                the namespace assertions into a fresh --out-dir.
  augmentation  audit every checkpoint's training-time augmentation states against
                the three evaluation banks' static states, resolve every
                checkpoint's training source from manifests / run receipts /
                scripts plus a checkpoint-embedded fingerprint, and fill the
                augmentation_exact_state_audit / train_source_resolution columns
                of an existing MODEL_BANK_OVERLAP.csv in place.
"""
import argparse
import functools
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/e1a933_review/data_lineage'

# --- augmentation-state audit tolerances (Linf over flattened [4,2] coordinates)
# EXACT: two serializations of the *same* point.  The pool coordinates are float64
#   on disk, the trainer casts them (and every edit) to float32, and the bank
#   stores its states as float32; the largest measured gap between two
#   representations of one point is <2e-7 (AUGMENTATION_STATE_AUDIT.json ->
#   serialization_noise).  1e-6 is 5x that and ~1.5e4x below the smallest
#   separation between two *distinct* designed candidate edits (0.015 Linf,
#   measured over the deterministic grid of r02_search.candidates_for_scene).
# NEAR: same point inside the near-duplicate band; still ~150x below 0.015, so it
#   cannot merge two genuinely distinct designed samples.  No slot-adjacency p99
#   is used anywhere.
TOL_EXACT = 1e-6
TOL_NEAR = 1e-4
CAP_PAIRS = 5000
WORKERS = 6

POOL_PATHS = {
    'N':   ROOT/'artifacts/f095_campaign/D01/scenes_N/scenes.npz',
    '4N':  ROOT/'artifacts/f095_campaign/D01/scenes_4N/scenes.npz',
    '16N': ROOT/'artifacts/f095_campaign/D01/scenes_16N/scenes.npz',
}
BANKS = ['dev512', 'd03E', 'd09fresh665']
FLIP_METHODS = {'flipmine', 'flip_fx', 'supmine', 'fliprand', 'unifmatched'}
METHOD_NAMES = {'clean', 'flipmine', 'flip_fx', 'supmine', 'fliprand', 'auxmargin',
                'relfeat', 'unifmatched'}
# Local run receipts that record the exact training command of the 15 checkpoints
# whose run manifest carries no train_dir/data_sha256.  Each key is a literal
# substring that must appear in a receipt command; absence is reported as such.
RECEIPT_KEYS = {
    'artifacts/discovery_campaign/r04b_s11': ('r04b_s11',),
    'artifacts/discovery_campaign/r04b_s23': ('r04b_s23', 'for sd in 23 47'),
    'artifacts/discovery_campaign/r04b_s47': ('r04b_s47', 'for sd in 23 47'),
    'artifacts/discovery_campaign/r04b_s23_relfeat': ('r04b_s23_relfeat',),
    'artifacts/discovery_campaign/r04b_s47_relfeat': ('r04b_s47_relfeat',),
    'artifacts/next_novelty/relflip/s11': ('relflip_train.py --seed 11',),
    'artifacts/next_novelty/relflip/s23': ('relflip_train.py --seed 23',),
    'artifacts/next_novelty/relflip/s47': ('relflip_train.py --seed 47',),
}
# The exact training invocation of each of those 15 checkpoints, verified by reading
# the receipt command and its .output log ("saved <run dir>" / "TRAIN_DONE").
TRAINING_RECEIPT = {
    'artifacts/discovery_campaign/r04b_s11': '.pi/tasks/session-2971751-2971751/bae255145.json',
    'artifacts/discovery_campaign/r04b_s23': '.pi/tasks/session-2971751-2971751/b0d835971.json',
    'artifacts/discovery_campaign/r04b_s47': '.pi/tasks/session-2971751-2971751/b0d835971.json',
    'artifacts/discovery_campaign/r04b_s23_relfeat': '.pi/tasks/session-2571586-2571586/b28952361.json',
    'artifacts/discovery_campaign/r04b_s47_relfeat': '.pi/tasks/session-2571586-2571586/b83386ed0.json',
    'artifacts/next_novelty/relflip/s11': '.pi/tasks/session-2571586-2571586/bb8c3f344.json',
    'artifacts/next_novelty/relflip/s23': '.pi/tasks/session-2571586-2571586/bed44b19c.json',
    'artifacts/next_novelty/relflip/s47': '.pi/tasks/session-2571586-2571586/b447e5cec.json',
}


@functools.lru_cache(maxsize=None)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def coord_hash(x): return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()


def bank_coordinates(path):
    b=np.load(path);meta=json.loads(str(b['Qmeta']));groups={}
    for i,m in enumerate(meta):groups.setdefault(m['qid'],[]).append(i)
    rows=[]
    for qid,indices in groups.items():
        if len(indices)!=2:continue
        a=next(i for i in indices if meta[i]['ptype']=='A');bb=next(i for i in indices if meta[i]['ptype']=='B')
        xa=b['Qx'][a].astype(float);xb=b['Qx'][bb].astype(float);xab=xa+b['Qe'][a].astype(float)
        # Qx_B=x0+eB and Qe_A=eB; reconstructed x0 has float32 serialization error.
        x0=xb-b['Qe'][a].astype(float)
        rows.append((qid,meta[a]['parent'],x0,np.stack([xa,xb,xab])))
    return rows


def build_matrix(OUT):
    if (OUT/"MODEL_BANK_OVERLAP.csv").exists():raise FileExistsError("Use a fresh --out-dir")
    OUT.mkdir(parents=True,exist_ok=True)
    src=ROOT/'artifacts/f095_campaign/D01'
    train={n:src/f'scenes_{n}/scenes.npz' for n in ['N','4N','16N']}
    xx={n:np.load(p)['positions'].reshape(-1,8) for n,p in train.items()}
    candidates=[]
    for family in ['D01','D02','U01','U02','U07']:
        for mp in (ROOT/f'artifacts/f095_campaign/{family}').glob('**/model.pt'):
            size='16N' if '16N' in str(mp) else '4N' if '4N' in str(mp) else 'N'
            candidates.append((mp,size))
    for mp in (ROOT/'artifacts/discovery_campaign').glob('r04b_s*/**/model.pt'):candidates.append((mp,'N'))
    for mp in (ROOT/'artifacts/next_novelty/relflip').glob('**/model.pt'):candidates.append((mp,'N'))
    banks=['dev512','d03E','d09fresh665']
    output=[];samples=[]
    for bank in banks:
        bp=ROOT/f'artifacts/p123_upgrade/bank/bank_{bank}.npz';rows=bank_coordinates(bp)
        x0=np.stack([r[2] for r in rows]).reshape(-1,8)
        states=np.stack([r[3] for r in rows]).reshape(-1,8)
        # Best matching full original float64 scene supplies canonical coordinate identity.
        dist16,ix16=cKDTree(xx['16N']).query(x0,p=np.inf)
        for j,r in enumerate(rows):
            samples.append(dict(bank=bank,bank_sha256=sha(bp),sample_id=f'{sha(bp)}:{r[0]}',
                qid=r[0],original_parent_id=r[1],reconstructed_parent_hash=coord_hash(r[2]),
                matched_16N_parent=int(ix16[j]) if dist16[j]<=1e-6 else '',
                canonical_parent_hash=coord_hash(xx['16N'][ix16[j]]) if dist16[j]<=1e-6 else '',
                nearest_16N_linf=float(dist16[j])))
        for size in train:
            dt,ix=cKDTree(xx[size]).query(x0,p=np.inf)
            st,_=cKDTree(xx[size]).query(states,p=np.inf)
            seen=dt<=1e-6
            for mp,sz in candidates:
                if sz!=size:continue
                manifest = mp.parent.parent/'run_manifest.json'
                resolution = 'source-family documented N history; no embedded data hash'
                if manifest.exists():
                    md=json.loads(manifest.read_text())
                    declared=ROOT/md['train_dir']/'scenes.npz'
                    assert sha(declared)==md['data_sha256']==sha(train[size]), (mp, declared, size)
                    resolution='run_manifest train_dir + data_sha256 verified'
                output.append(dict(checkpoint=str(mp.relative_to(ROOT)),checkpoint_sha256=sha(mp),
                    train_source=str(train[size].relative_to(ROOT)),train_source_sha256=sha(train[size]),
                    train_n=len(xx[size]),bank=bank,bank_sha256=sha(bp),n_quartets=len(rows),
                    old_parent_new_edit_quartets=int(seen.sum()),unseen_parent_quartets=int((~seen).sum()),
                    exact_static_state_duplicates=int((st==0).sum()),serialized_static_state_matches=int((st<=1e-6).sum()),
                    near_parent_quartets=int(((dt>1e-6)&(dt<=1e-4)).sum()),
                    augmentation_exact_state_audit='NOT_RUN; parent ancestry excludes all descendants of seen parents',
                    train_source_resolution=resolution))
    for name,rows in [('MODEL_BANK_OVERLAP.csv',output),('SAMPLE_LINEAGE.csv',samples)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Independently verify metadata namespace against original serialized coordinates.
    x16=xx['16N'].reshape(-1,4,2)
    source09=np.load(ROOT/'artifacts/discovery_campaign/scenes/d09_fresh/scenes.npz')['positions']
    assert np.array_equal(source09,x16[512:])
    assertions={'d09_pool_equals_X16_offset512':True}
    for bank,pool in [('dev512','eval_202'),('d09fresh665','d09_fresh')]:
        x=np.load(ROOT/f'artifacts/discovery_campaign/scenes/{pool}/scenes.npz')['positions']
        rr=bank_coordinates(ROOT/f'artifacts/p123_upgrade/bank/bank_{bank}.npz')
        err=max(float(np.max(np.abs(r[2]-x[r[1]]))) for r in rr)
        assert err<1e-6
        assertions[bank]=dict(parent_namespace=pool,max_reconstruction_linf=err)
    for task in ['T1','T2']:
        tr=np.load(ROOT/f'artifacts/f095_campaign/U10/{task}_train/scenes.npz')['positions']
        ev=np.load(ROOT/f'artifacts/f095_campaign/U10/{task}_eval/scenes.npz')['positions']
        dist,_=cKDTree(tr.reshape(len(tr),-1)).query(ev.reshape(len(ev),-1),p=np.inf)
        assertions[task]=dict(train_n=len(tr),eval_n=len(ev),nearest_parent_linf=float(dist.min()),serialized_matches=int((dist<=1e-6).sum()))
        assert dist.min()>1e-4
    (OUT/'SOURCE_NAMESPACE_ASSERTIONS.json').write_text(json.dumps(assertions,indent=2))
    aa={r['matched_16N_parent'] for r in samples if r['bank']=='d03E'}-{''}
    bb={r['matched_16N_parent'] for r in samples if r['bank']=='d09fresh665'}-{''}
    common=aa&bb
    cross=dict(shared_E_parent_count=len(common),canonical_X16_indices=sorted(common),
               d03_quartets_on_shared_parents=sum(r['bank']=='d03E' and r['matched_16N_parent'] in common for r in samples),
               d09_quartets_on_shared_parents=sum(r['bank']=='d09fresh665' and r['matched_16N_parent'] in common for r in samples))
    (OUT/'D03_D09_CROSSBANK.json').write_text(json.dumps(cross,indent=2))
    print('checkpoints',len(candidates),'rows',len(output),flush=True)
    for b in banks:
        for size in (512,2048,8192):
            z=next((r for r in output if r['bank']==b and r['train_n']==size),None)
            if z:print(b,size,z['old_parent_new_edit_quartets'],z['unseen_parent_quartets'],flush=True)


# ---------------------------------------------------------------------------
# augmentation-state audit (R03)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def pool_arrays(pool):
    """float32 model-visible coordinates of a training pool (as the trainers load them)."""
    d=np.load(POOL_PATHS[pool])
    return d['positions'].astype(np.float32), d['labels'].astype(np.int64)


@functools.lru_cache(maxsize=None)
def mined_flips(pool):
    """Rebuild the mined flip set exactly as r04b_methods.mine_flips(seed=5) did.

    Returns (parent_idx, edit_float32, pre-normalization float32 states, raw list).
    """
    sys.path.insert(0,str(ROOT/'experiments/discovery_campaign'))
    import r04b_methods
    X32,y=pool_arrays(pool)
    mined=r04b_methods.mine_flips(X32.astype(float),y,seed=5)
    ii=np.array([m['scene'] for m in mined],dtype=np.int64)
    ee=np.array([m['edit'] for m in mined],dtype=np.float32)
    states=(X32[ii]+ee).reshape(len(ii),8)      # float32+float32 -> what the model saw
    return ii,ee,states,mined


@functools.lru_cache(maxsize=None)
def orbit_states(pool,kind):
    """Full G8 role-permutation orbit of a state set (D02 --aug g8 draws one
    element per scene per epoch, so the 300-epoch union is this orbit or less)."""
    sys.path.insert(0,str(ROOT/'experiments/f095_campaign'))
    from symmetry import GROUP
    base=(pool_arrays(pool)[0] if kind=='clean' else mined_flips(pool)[2]).reshape(-1,4,2)
    return np.stack([base[:,list(g),:] for g in GROUP]).reshape(-1,8).astype(np.float64)


@functools.lru_cache(maxsize=None)
def bank_targets(bank):
    """Static states of an evaluation bank: 3 per quartet (A, B, AB) plus singles."""
    bp=ROOT/f'artifacts/p123_upgrade/bank/bank_{bank}.npz'
    b=np.load(bp,allow_pickle=True);meta=json.loads(str(b['Qmeta']))
    groups={}
    for i,m in enumerate(meta):groups.setdefault(m['qid'],[]).append(i)
    qs,qi=[],[]
    for qid,idx in groups.items():
        if len(idx)!=2:continue
        a=next(i for i in idx if meta[i]['ptype']=='A');bb=next(i for i in idx if meta[i]['ptype']=='B')
        xa=b['Qx'][a].astype(np.float64).reshape(-1)
        xb=b['Qx'][bb].astype(np.float64).reshape(-1)
        xab=xa+b['Qe'][a].astype(np.float64).reshape(-1)
        for tag,arr in (('A',xa),('B',xb),('AB',xab)):
            qs.append(arr);qi.append(f'{qid}:{tag}')
    smeta=json.loads(str(b['Smeta']))
    ss=np.asarray(b['Sx'],dtype=np.float64)
    ss=ss.reshape(len(ss),-1) if len(ss) else np.zeros((0,8))
    se=np.asarray(b['Se'],dtype=np.float64)
    se=se.reshape(len(se),-1) if len(se) else np.zeros((0,8))
    # build_bank.py stores Sx = the *start* (parent) state and Se = the edit, so the
    # evaluated atomic state is Sx+Se.  Comparing Sx alone compares parents, not
    # edited states.
    si=[f'S{m["eid"]}' for m in smeta]
    return dict(path=str(bp.relative_to(ROOT)),sha256=sha(bp),n_quartets=len(groups),
                quartet_states=np.stack(qs),quartet_ids=qi,
                single_states=ss+se,single_ids=si,
                single_start_states=ss,n_single_start=len(ss))


def query_states(states,targets,ids):
    """Nearest Linf target for each state, split into exact / near bands."""
    if len(states)==0 or len(targets)==0:
        return dict(n_states=len(states),n_targets=len(targets),n_exact=0,n_near=0,
                    min_linf=None,n_byte_identical=0,matches=[])
    d,ix=cKDTree(targets).query(states,k=1,p=np.inf,workers=WORKERS)
    exact=np.flatnonzero(d<=TOL_EXACT)
    near=np.flatnonzero((d>TOL_EXACT)&(d<=TOL_NEAR))
    byte=sum(coord_hash(states[j])==coord_hash(targets[ix[j]]) for j in exact)
    pairs=[dict(aug_index=int(j),target_index=int(ix[j]),bank_id=ids[int(ix[j])],linf=float(d[j]))
           for j in list(exact[:CAP_PAIRS])+list(near[:CAP_PAIRS])]
    return dict(n_states=len(states),n_targets=len(targets),n_exact=int(len(exact)),
                n_near=int(len(near)),min_linf=float(d.min()),
                n_byte_identical=int(byte),n_distinct_targets=int(len(set(ix[exact].tolist()))),
                n_exact_pairs_capped=bool(len(exact)>CAP_PAIRS),matches=pairs)


def ckpt_convention(ck):
    fz=ck.get('featurize')
    return 'raw8' if not fz else fz


@functools.lru_cache(maxsize=None)
def pool_fingerprint(pool,conv):
    """mu/sd exactly as the trainers computed them from the pool (float32 path)."""
    sys.path.insert(0,str(ROOT/'experiments/discovery_campaign'))
    from common import rel_features
    X32,_=pool_arrays(pool)
    if conv=='raw8':
        return X32.reshape(-1,8).mean(0),X32.reshape(-1,8).std(0)+1e-8
    if conv=='scalar8':
        gmu,gsd=float(X32.mean()),float(X32.std()+1e-8)
        return np.full(8,gmu,np.float32),np.full(8,gsd,np.float32)
    Fr=rel_features(X32)
    if conv=='rel12':
        return Fr.mean(0),Fr.std(0)+1e-8
    if conv=='sixdist':
        F6=Fr[:,:6]
        return F6.mean(0),F6.std(0)+1e-8
    raise ValueError(conv)


def fingerprint_match(ckpt):
    """Checkpoint-embedded mu/sd are a fingerprint of the exact training array."""
    import torch
    ck=torch.load(ckpt,map_location='cpu',weights_only=False)
    conv=ckpt_convention(ck);mu,sd=ck.get('mu'),ck.get('sd')
    if mu is None or sd is None:
        return dict(convention=conv,status='no_mu_sd',pools=[])
    hits=[p for p in POOL_PATHS
          if np.array_equal(np.asarray(mu),pool_fingerprint(p,conv)[0])
          and np.array_equal(np.asarray(sd),pool_fingerprint(p,conv)[1])]
    return dict(convention=conv,status='unique_match' if len(hits)==1 else ('ambiguous' if hits else 'no_match'),
                pools=hits,
                pool_sha256={p:sha(POOL_PATHS[p]) for p in hits})


def shell_env(cmd):
    return {k:v for k,v in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)=([^\s;&|]+)',cmd)}


def receipt_evidence(run_dir):
    """Exact training command from the local .pi/tasks receipt store."""
    keys=RECEIPT_KEYS.get(str(run_dir.relative_to(ROOT)))
    store=ROOT/'.pi'/'tasks'
    if keys is None:return dict(search_keys=None,status='no_key',commands=[])
    if not store.exists():return dict(search_keys=list(keys),status='receipt_store_absent',commands=[])
    cmds=[]
    for f in sorted(store.glob('*/*.json')):
        try:cmd=json.loads(f.read_text()).get('command') or ''
        except Exception:continue
        matched=[k for k in keys if k in cmd]
        if matched:
            env=shell_env(cmd)
            train=re.search(r'--train\s+(\S+)',cmd)
            script=re.search(r'(\S+\.py)',cmd)
            tok=train.group(1) if train else None
            resolved=tok
            stok=script.group(1) if script else None
            norm=cmd
            for k,v in env.items():
                if resolved:resolved=resolved.replace('$'+k,v)
                if stok:stok=stok.replace('$'+k,v)
                norm=norm.replace('$'+k,v)
            cmds.append(dict(receipt=str(f.relative_to(ROOT)),matched_keys=matched,train_arg=tok,
                             train_arg_resolved=resolved,script=stok,
                             command=cmd[:400],command_normalized=norm[:400]))
    # put the actual training invocation (--output + --train) first
    rel=str(run_dir.relative_to(ROOT)) if str(run_dir).startswith(str(ROOT)) else str(run_dir)
    auth=TRAINING_RECEIPT.get(rel)
    def rank(c):
        if c['receipt']==auth:return 0
        cmd=c['command_normalized']
        exact=bool(re.search(r'--output\s+'+re.escape(rel)+r'(\s|$)',cmd))
        contaminates=re.search(re.escape(rel)+r'_[A-Za-z0-9]',cmd) is not None
        if exact and not contaminates:return 1
        if c['train_arg'] and not contaminates:return 2
        if exact:return 3
        return 4 if c['train_arg'] else 5
    cmds.sort(key=rank)
    for c in cmds:c['authoritative']=c['receipt']==auth
    return dict(search_keys=list(keys),authoritative_receipt=auth,
                status='found' if cmds else 'no_receipt',commands=cmds[:4])


def script_literal_evidence(script_rel):
    """Lines of the training script that name the training data path."""
    if not script_rel:return []
    p=ROOT/script_rel
    if not p.exists():return []
    out=[]
    for i,ln in enumerate(p.read_text().splitlines(),1):
        if 'scenes' in ln and ('train' in ln or 'np.load' in ln):
            out.append(f'{script_rel}:{i}: {ln.strip()}')
    return out


def row_spec(checkpoint):
    """(run_dir, method, has_flip, aug) for one checkpoint path."""
    cp=ROOT/checkpoint
    name=cp.parent.name
    if name in METHOD_NAMES:
        run_dir,method=cp.parent.parent,name
    else:
        run_dir,method=cp.parent,'relflip'      # relflip/s{seed}/model.pt
    manifest=run_dir/'run_manifest.json'
    if manifest.exists():
        md=json.loads(manifest.read_text())
        methods=set(md.get('methods',[]))|{method}
        aug=md.get('aug') or 'none'
    else:                                        # r04b family (manifest has no methods/aug)
        methods={method}
        aug='none'
    # the flip term is per checkpoint, not per run: a run's `clean` arm trains on
    # clean inputs only (d01_capacity: loss = lb for name=='clean').
    has_flip=(method in FLIP_METHODS) or method=='relflip'
    return run_dir,method,has_flip,aug,methods


def augment_set(pool,has_flip,aug):
    """Augmentation state blocks a checkpoint actually trained on."""
    blocks=[]
    if aug=='g8':
        blocks.append(('g8_orbit_clean',orbit_states(pool,'clean')))
        if has_flip:blocks.append(('g8_orbit_mined',orbit_states(pool,'mined')))
    elif has_flip:
        blocks.append(('mined_flip',mined_flips(pool)[2].astype(np.float64)))
    return blocks


def singles_start_diagnostic(targets):
    """build_bank.py:82 stores Sx = the start/parent state, Se = the edit.

    Recording this explicitly so that no downstream reader mistakes a Sx-only
    comparison for an edited-state comparison.
    """
    X16=pool_arrays('16N')[0].astype(np.float64).reshape(-1,8)
    out={}
    for bank in BANKS:
        st=targets[bank]['single_start_states']
        if len(st)==0:out[bank]=dict(n=0);continue
        d,_=cKDTree(X16).query(st,k=1,p=np.inf,workers=WORKERS)
        out[bank]=dict(n=int(len(st)),match_16N_clean_le1e6=int((d<=TOL_EXACT).sum()),
                       match_16N_clean_le1e4=int((d<=TOL_NEAR).sum()),
                       note='start states are the parents by construction')
    return out


def detector_power(targets,banks=BANKS,n=400):
    """Positive/negative controls: the audit must flag planted duplicates.

    (a) bank states themselves -> exact;  (b) +5e-5 -> near;  (c) +5e-4 -> clear;
    (d) one 16N mined-flip state planted into the d03E target set -> exactly one
    exact hit for that block.
    """
    ctrl={}
    for bank in banks:
        t=targets[bank]['quartet_states'][:n]
        ctrl[bank]=dict(
            identical=query_states(t,t,np.arange(len(t)).astype(str))['n_exact'],
            perturb_5e_5_near=query_states(t+5e-5,t,np.arange(len(t)).astype(str))['n_near'],
            perturb_5e_4_flagged=query_states(t+5e-4,t,np.arange(len(t)).astype(str))['n_exact']
                                 +query_states(t+5e-4,t,np.arange(len(t)).astype(str))['n_near'],
            n_controls=len(t))
    mined16=mined_flips('16N')[2].astype(np.float64)
    planted=np.concatenate([targets['d03E']['quartet_states'],mined16[:1]])
    ids=list(targets['d03E']['quartet_ids'])+['PLANTED:16N:mined_flip:0']
    ctrl['planted_16N_mined_flip_in_d03E']=query_states(mined16[:1],planted,ids)
    return ctrl


def distinct_sets(targets,audit):
    """Every distinct augmentation block, audited once against every bank."""
    used={}
    for ck,v in audit.items():
        if v['setkey'] in ('none','unresolved_train_source'):continue
        for blk in v['setkey'].split('+'):
            used.setdefault(f"{v['pool']}:{blk}",[]).append(ck)
    out={}
    for pool in POOL_PATHS:
        for kind,states in (('mined_flip',mined_flips(pool)[2].astype(np.float64)),
                            ('g8_orbit_clean',orbit_states(pool,'clean')),
                            ('g8_orbit_mined',orbit_states(pool,'mined'))):
            key=f'{pool}:{kind}'
            cell={bank:dict(quartet=query_states(states,targets[bank]['quartet_states'],targets[bank]['quartet_ids']),
                            single=query_states(states,targets[bank]['single_states'],targets[bank]['single_ids']))
                  for bank in BANKS}
            for bank in BANKS:
                ex=explain_exact(pool,kind,bank,targets)
                if ex is not None:cell[bank]['exact_match_detail']=ex
            out[key]=dict(pool=pool,kind=kind,n_states=int(len(states)),
                          used_by_checkpoints=sorted(used.get(key,[])),banks=cell)
    return out


def explain_exact(pool,kind,bank,targets):
    """Parent/ family detail of every exact match of one augmentation block."""
    if kind=='mined_flip':
        states=mined_flips(pool)[2].astype(np.float64);raw=mined_flips(pool)[3];nbase=len(raw)
    else:
        base_kind='clean' if kind=='g8_orbit_clean' else 'mined'
        states=orbit_states(pool,base_kind);raw=None
        nbase=len(pool_arrays(pool)[0]) if base_kind=='clean' else len(mined_flips(pool)[0])
    t=targets[bank]
    res=query_states(states,t['quartet_states'],t['quartet_ids']);kindname='quartet'
    if res['n_exact']==0:
        res=query_states(states,t['single_states'],t['single_ids']);kindname='single'
    if res['n_exact']==0:return None
    b=np.load(ROOT/t['path'],allow_pickle=True)
    smeta={m['eid']:m for m in json.loads(str(b['Smeta']))} if kindname=='single' else {}
    qmeta={m['qid']:m for m in json.loads(str(b['Qmeta']))} if kindname=='quartet' else {}
    mined=mined_flips(pool)[3]
    det=[];augparents=set();bankparents=set()
    for p in res['matches']:
        if p['linf']>TOL_EXACT:continue
        j=p['aug_index'];tid=p['bank_id']
        row=dict(aug_index=j,linf=p['linf'],bank_id=tid)
        if kind=='mined_flip':
            src=mined[j]
        else:
            gi,bi=divmod(j,nbase)
            row.update(aug_group_index=gi,aug_base_index=bi)
            src=mined[bi] if base_kind=='mined' else None
        if src is not None:
            row.update(aug_parent_pool_index=int(src['scene']),aug_family=src.get('fam'),
                       aug_new_label=int(src['new']),aug_margin=float(src['margin']))
            augparents.add(int(src['scene']))
        if kindname=='single':
            m=smeta.get(int(tid[1:]),{})
            row.update(bank_parent=m.get('parent'),bank_family=m.get('fam'),bank_y0=m.get('y0'),
                       bank_y1=m.get('y1'),bank_flip=m.get('flip'))
            bankparents.add(m.get('parent'))
        else:
            qid=int(tid.split(':')[0]);m=qmeta.get(qid,{})
            row.update(bank_parent=m.get('parent'),bank_ptype=tid.split(':')[1],
                       bank_fam_A=m.get('fam_A'),bank_fam_B=m.get('fam_B'),bank_y0=m.get('y0'),
                       bank_y1=m.get('yAB'))
            bankparents.add(m.get('parent'))
        det.append(row)
    return dict(target_kind=kindname,n_exact=res['n_exact'],n_distinct_targets=res['n_distinct_targets'],
                n_byte_identical=res['n_byte_identical'],
                n_distinct_aug_parents=len(augparents),n_distinct_bank_parents=len(bankparents),
                parent_offset_consistent=(kindname=='single' and
                    all(d['bank_parent']+512==d['aug_parent_pool_index'] for d in det if 'aug_parent_pool_index' in d and d['bank_parent'] is not None)),
                details_capped=res['n_exact_pairs_capped'],details=det)


def mined_count_crosscheck():
    """The run manifests record n_mined_flips per training pool; the rebuild must agree."""
    out={}
    for p in POOL_PATHS:
        rel=str(POOL_PATHS[p].parent.relative_to(ROOT))
        obs=set()
        for f in (ROOT/'artifacts/f095_campaign').glob('**/run_manifest.json'):
            md=json.loads(f.read_text())
            if md.get('train_dir')==rel:obs.add(md['n_mined_flips'])
        out[p]=dict(manifest_n_mined_flips=sorted(obs),rebuilt_n=len(mined_flips(p)[0]),
                    matches=sorted(obs)==[len(mined_flips(p)[0])])
    for f in [ROOT/'artifacts/discovery_campaign/r04b_s11/manifest.json',
              ROOT/'artifacts/next_novelty/relflip/s11/manifest.json']:
        md=json.loads(f.read_text())
        n=md.get('n_mined_flips',md.get('n_flips'))
        out[str(f.relative_to(ROOT))]=dict(manifest_n_mined_flips=[n],rebuilt_n=len(mined_flips('N')[0]),
                                          matches=n==len(mined_flips('N')[0]))
    return out


def unif_matched_states(pool,seed,n):
    """Rebuild r04b_methods' --unif-n independent-sign control exactly (its own RNG)."""
    sys.path.insert(0,str(ROOT/'experiments/discovery_campaign'))
    import r04b_methods
    X32,_=pool_arrays(pool)
    ru=np.random.default_rng(seed+31337);radii=[0.03,0.06,0.10,0.16]
    UE,US,att=[],[],0
    while len(UE)<n and att<n*60:
        att+=1
        i=int(ru.integers(0,len(X32)))
        e=ru.normal(size=(4,2));e*=ru.choice(radii)*2/np.linalg.norm(e)
        try:o=r04b_methods.segment_relation(X32[i]+e)
        except ValueError:continue
        if o['ambiguous'] or o['margin']<0.03:continue
        UE.append(e.astype(np.float32));US.append(i)
    ii=np.array(US,dtype=np.int64)
    ee=np.stack(UE) if UE else np.zeros((0,4,2),np.float32)
    return dict(n=len(ii),attempts=att,states=(X32[ii]+ee).reshape(len(ii),8).astype(np.float64))


@functools.lru_cache(maxsize=None)
def dro_perturbation_states(pool,seed):
    """Rebuild r04_train.py's two perturbation banks exactly (eps=0.1, 35 modes)."""
    sys.path.insert(0,str(ROOT/'docs/iclr2027_discovery_campaign_20260917'))
    from core.perturbations import balanced_sign_patterns,antithetic_fields
    X32,_=pool_arrays(pool);N=len(X32)
    pats=balanced_sign_patterns((8,),max_patterns=64,seed=7)
    F=antithetic_fields(pats,0.1,np.array([1.0])).reshape(len(pats),2,4,2).astype(np.float32)
    anti=(X32[:,None,None,:,:]+F[None,:,:,:,:]).reshape(-1,8)
    K=len(pats)*2
    urng=np.random.default_rng(seed+1000)
    Us=urng.choice([-1.0,1.0],size=(N,K,8)).astype(np.float32)*(0.1/np.sqrt(8))
    unif=(X32.reshape(N,8).astype(float)[:,None,:]+Us).reshape(-1,8)
    return dict(n_modes=len(pats),antithetic=anti.astype(np.float64),unif_sign=unif.astype(np.float64))


# Geometry-quartet training sources outside the 96-checkpoint matrix, with the exact
# command that produced them.  Everything else in artifacts/ trains on a different
# task or a different input space (T5R3 football, U10 T1/T2, vision, soccer, p3d,
# l007) and never touches the three banks' coordinates.
EXTRA_SOURCES = [
    dict(key='N:dro_antithetic',dirs='artifacts/discovery_campaign/r04_*',
         methods='minimax/equal/adv arms of r04_train.py',pool='N',
         builder='antithetic_fields(balanced_sign_patterns((8,),seed=7),eps=0.1) x 2 arms x 35 modes'),
    dict(key='N:dro_unif_sign_s11',dirs='artifacts/discovery_campaign/r04_unif_s11',
         methods='unif arm',pool='N',builder='rng(seed+1000) independent +/- eps/sqrt(8) x 70'),
    dict(key='N:dro_unif_sign_s23',dirs='artifacts/discovery_campaign/r04_unif_s23',
         methods='unif arm',pool='N',builder='rng(seed+1000) independent +/- eps/sqrt(8) x 70'),
    dict(key='N:dro_unif_sign_s47',dirs='artifacts/discovery_campaign/r04_unif_s47',
         methods='unif arm',pool='N',builder='rng(seed+1000) independent +/- eps/sqrt(8) x 70'),
    dict(key='N:unifmatched_s11',dirs='artifacts/discovery_campaign/r04c_budget',
         methods='unifmatched arm (--unif-n 1534)',pool='N',builder='rng(seed+31337) candidate-scale random edits'),
    dict(key='N:unifmatched_s23',dirs='artifacts/discovery_campaign/r04c_unif_s23',
         methods='unifmatched arm',pool='N',builder='rng(seed+31337) candidate-scale random edits'),
    dict(key='N:unifmatched_s47',dirs='artifacts/discovery_campaign/r04c_unif_s47',
         methods='unifmatched arm',pool='N',builder='rng(seed+31337) candidate-scale random edits'),
]


def extra_geometry_sources(targets):
    """Same audit for the geometry-task training sources outside the 96-checkpoint matrix."""
    dro={s:dro_perturbation_states('N',s) for s in (11,23,47)}
    unif={s:unif_matched_states('N',s,1534) for s in (11,23,47)}
    out={}
    for spec in EXTRA_SOURCES:
        k=spec['key']
        if k=='N:dro_antithetic':states=dro[11]['antithetic']
        elif k.startswith('N:dro_unif_sign'):states=dro[int(k[-2:])]['unif_sign']
        else:states=unif[int(k[-2:])]['states']
        out[k]=dict(spec,dirs=sorted({str(p.parent.parent.relative_to(ROOT)) for p in
                                      ROOT.glob(spec['dirs']+'/**/model.pt')}),
                    n_states=int(len(states)),banks={
            bank:dict(quartet=query_states(states,targets[bank]['quartet_states'],targets[bank]['quartet_ids']),
                      single=query_states(states,targets[bank]['single_states'],targets[bank]['single_ids']))
            for bank in BANKS})
    return out


def fmt_linf(v):
    return 'n/a' if v is None else f'{v:.3g}'


def build_augmentation_audit(OUT):
    csv_path=OUT/'MODEL_BANK_OVERLAP.csv'
    if not csv_path.exists():raise FileNotFoundError(f'{csv_path} missing; run --mode matrix first')
    with csv_path.open() as f:
        reader=csv.DictReader(f);fieldnames=list(reader.fieldnames);old=list(reader)
    # 0. faithfulness of the reconstruction we are about to trust
    stored=np.load(ROOT/'artifacts/discovery_campaign/r04b_s11/mined.npz')
    ii,ee,states,mined=mined_flips('N')
    sed=np.stack([stored[f'edit_{k}'] for k in range(len(ii))])
    faithful=dict(n=len(ii),edit_bytes_identical=bool(np.array_equal(ee,sed)),
                  scene_ids_identical=bool(np.array_equal(ii,stored['meta'][:,0].astype(np.int64))),
                  new_labels_identical=bool(np.array_equal(np.array([m['new'] for m in mined]),stored['meta'][:,1].astype(np.int64))),
                  margin_max_abs_diff=float(np.max(np.abs(np.array([m['margin'] for m in mined])-stored['meta'][:,2].astype(np.float64)))),
                  source='artifacts/discovery_campaign/r04b_s11/mined.npz')
    assert faithful['edit_bytes_identical'] and faithful['scene_ids_identical']
    # 1. serialization noise that the tolerances have to absorb
    noise={}
    for p,path in POOL_PATHS.items():
        x64=np.load(path)['positions']
        noise[p]=dict(float32_rounding_linf=float(np.max(np.abs(x64-x64.astype(np.float32)))),
                      n=len(x64))
    noise['edit_float32_rounding_linf']=float(np.max(np.abs(np.array([m['edit'] for m in mined],dtype=np.float64)-ee.astype(np.float64))))
    noise['bank_parent_reconstruction_linf']=max(
        json.loads((OUT/'SOURCE_NAMESPACE_ASSERTIONS.json').read_text())[b]['max_reconstruction_linf']
        for b in ('dev512','d09fresh665'))
    # 2. per-checkpoint audit
    targets={b:bank_targets(b) for b in BANKS}
    audit={};resolution={}
    for checkpoint in sorted({r['checkpoint'] for r in old}):
        run_dir,method,has_flip,aug,methods=row_spec(checkpoint)
        fp=fingerprint_match(checkpoint)
        pool=fp['pools'][0] if fp['status']=='unique_match' else None
        # training-source resolution
        manifest=run_dir/'run_manifest.json'
        ev=[];status='unresolved'
        if manifest.exists():
            md=json.loads(manifest.read_text())
            if md.get('train_dir'):
                declared=ROOT/md['train_dir']/'scenes.npz'
                ok=(sha(declared)==md.get('data_sha256'))
                ev.append(dict(type='run_manifest',path=str(manifest.relative_to(ROOT)),
                               train_dir=md['train_dir'],data_sha256=md.get('data_sha256'),
                               file_sha256_matches=bool(ok)))
        rc=receipt_evidence(run_dir)
        train_arg=None
        if rc['status']=='found':
            for c in rc['commands']:
                if c['train_arg'] and train_arg is None:train_arg=c['train_arg_resolved']
                ev.append(dict(type='run_receipt',path=c['receipt'],train_arg=c['train_arg'],
                               train_arg_resolved=c['train_arg_resolved'],
                               authoritative=c.get('authoritative'),command=c['command']))
        else:
            ev.append(dict(type='run_receipt_lookup',status=rc['status'],search_keys=rc['search_keys']))
        scripts=[c['script'] for c in rc['commands'] if c.get('train_arg') and c.get('script')] or \
                [c['script'] for c in rc['commands'] if c.get('script')]
        for ln in script_literal_evidence(scripts[0] if scripts else None):
            ev.append(dict(type='script_literal',line=ln))
        ev.append(dict(type='checkpoint_fingerprint',convention=fp['convention'],status=fp['status'],
                       pools=fp['pools'],pool_sha256=fp['pool_sha256']))
        train_sha=sha(POOL_PATHS[pool]) if pool else None
        if fp['status']=='unique_match':status='resolved'
        resolution[checkpoint]=dict(run_dir=str(run_dir.relative_to(ROOT)),method=method,
                                    train_source=str(POOL_PATHS[pool].relative_to(ROOT)) if pool else None,
                                    train_source_sha256=train_sha,status=status,
                                    train_arg_from_receipt=train_arg,evidence=ev)
        # augmentation states
        blocks=augment_set(pool,has_flip,aug) if pool else []
        setkey='unresolved_train_source' if not pool else (
            'none' if not blocks else '+'.join(b[0] for b in blocks))
        ast=np.concatenate([b[1] for b in blocks]) if blocks else np.zeros((0,8))
        res={}
        for bank in BANKS:
            t=targets[bank]
            res[bank]=dict(quartet=query_states(ast,t['quartet_states'],t['quartet_ids']),
                           single=query_states(ast,t['single_states'],t['single_ids']))
        audit[checkpoint]=dict(run_dir=resolution[checkpoint]['run_dir'],method=method,
                               pool=pool,aug=aug,methods=sorted(methods),setkey=setkey,
                               n_aug_states=int(len(ast)),banks=res)
    # 3. CSV columns
    new=[]
    for r in old:
        a=audit[r['checkpoint']];res=resolution[r['checkpoint']]
        b=a['banks'][r['bank']]
        if a['n_aug_states']==0:
            col=('UNRESOLVED_TRAIN_SOURCE; no augmentation set could be built'
                 if a['setkey']=='unresolved_train_source' else
                 'NO_AUGMENTATION (clean-only inputs); see exact_static_state_duplicates / serialized_static_state_matches')
        else:
            q,s=b['quartet'],b['single']
            basis=('G8 role-permutation orbit (superset of the 300-epoch draws)'
                   if 'orbit' in a['setkey'] else 'mined_recomputed_seed=5')
            col=(f"AUDITED set={a['setkey']} n_states={a['n_aug_states']}; "
                 f"quartet exact_le1e-6={q['n_exact']} near_le1e-4={q['n_near']} min_linf={fmt_linf(q['min_linf'])}; "
                 f"singles exact_le1e-6={s['n_exact']} near_le1e-4={s['n_near']} "
                 f"min_linf={fmt_linf(s['min_linf'])}; byte_identical={q['n_byte_identical']+s['n_byte_identical']}; "
                 + (f"bank_states_hit={q['n_distinct_targets']+s['n_distinct_targets']}; "
                    if q['n_exact']+s['n_exact']>0 else '')
                 + f"{basis}")
        if res['status']=='resolved':
            keep={'run_manifest','run_receipt','script_literal','checkpoint_fingerprint'}
            legs=list(dict.fromkeys(e['type'] for e in res['evidence']
                                    if e['type'] in keep and (e['type']!='run_receipt' or e.get('path'))))
            extra=f"; receipt --train={res['train_arg_from_receipt']}" if res['train_arg_from_receipt'] else ''
            rescol=(f"resolved: {'+'.join(legs)}{extra}; train_source={res['train_source']} "
                    f"sha256={res['train_source_sha256'][:16]}...; ckpt mu/sd bit-exact")
        else:
            rescol=f"unresolved: {res['status']}"
        r2=dict(r);r2['augmentation_exact_state_audit']=col;r2['train_source_resolution']=rescol
        new.append(r2)
    for a,b in zip(old,new):
        for k in fieldnames:
            if k in ('augmentation_exact_state_audit','train_source_resolution'):continue
            assert a[k]==b[k],(k,a[k],b[k])
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(new)
    (OUT/'AUGMENTATION_STATE_AUDIT.json').write_text(json.dumps(dict(
        tol_exact=TOL_EXACT,tol_near=TOL_NEAR,
        tol_basis='exact=float32-serialization band of the same point; near=1e-4 is ~150x below the '
                  'smallest Linf separation between two distinct designed candidate edits (0.015)',
        reconstruction_faithfulness=faithful,serialization_noise=noise,
        mined_count_crosscheck=mined_count_crosscheck(),
        singles_start_state_diagnostic=singles_start_diagnostic(targets),
        detector_power=detector_power(targets),
        sets=distinct_sets(targets,audit),
        extra_geometry_sources=extra_geometry_sources(targets),
        bank_targets={b:dict(path=targets[b]['path'],sha256=targets[b]['sha256'],
                             n_quartets=targets[b]['n_quartets'],
                             n_quartet_states=len(targets[b]['quartet_states']),
                             n_single_states=len(targets[b]['single_states']),
                             n_single_start_states=targets[b]['n_single_start']) for b in BANKS},
        checkpoints=audit),indent=2,sort_keys=True))
    (OUT/'TRAIN_SOURCE_RESOLUTION.json').write_text(json.dumps(dict(
        n_checkpoints=len(resolution),
        resolved=sum(v['status']=='resolved' for v in resolution.values()),
        by_pool={p:sum(v['train_source']==str(POOL_PATHS[p].relative_to(ROOT)) for v in resolution.values()) for p in POOL_PATHS},
        checkpoints=resolution),indent=2,sort_keys=True))
    print('audited',len(audit),'checkpoints; resolved',
          sum(v['status']=='resolved' for v in resolution.values()),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['matrix','augmentation'],default='matrix')
    p.add_argument('--out-dir',type=Path,default=OUT)
    a=p.parse_args()
    if a.mode=='matrix':build_matrix(a.out_dir)
    else:build_augmentation_audit(a.out_dir)


if __name__=='__main__': main()
