"""R01/R02/O02: immutable-bank reanalysis and corrected typed retraining."""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/f095_campaign'))
from u10_targets import build_arch, fit_stats, featurize, TASKS, count_params
from data_stats import parent_bootstrap

OLD = ROOT / 'artifacts/f095_campaign/U10'
OUT = ROOT / 'artifacts/e1a933_review/data_u10'
torch.set_num_threads(2)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def groups(task):
    n = TASKS[task]['npts']
    return [(*p, n-1) for p in itertools.permutations(range(n-1))]


@torch.no_grad()
def predict(model, task, arch, x, stats):
    return np.concatenate([model(featurize(task, arch, x[i:i+512], stats)).numpy()
                           for i in range(0, len(x), 512)])


def train():
    for task in TASKS:
        d = np.load(OLD / f'{task}_train/scenes.npz')
        x, y = d['positions'], torch.tensor(d['labels'], dtype=torch.float32)
        fz = np.load(OLD / f'{task}_flips.npz', allow_pickle=True)
        meta = fz['meta']; pi = np.array([int(r[0]) for r in meta])
        edits = np.stack([fz[f'edit_{k}'] for k in range(len(meta))])
        fy = torch.tensor([int(r[1]) for r in meta], dtype=torch.float32)
        stats = fit_stats(task, 'typed', x)
        xc = featurize(task, 'typed', x, stats)
        xf = featurize(task, 'typed', x[pi].astype(float)+edits.astype(float), stats)
        for seed in (11, 23, 47):
            for regime in ('clean', 'flip', 'f25'):
                dest = OUT / f'{task}_typed_s{seed}' / regime
                dest.mkdir(parents=True, exist_ok=True)
                if (dest / 'model.pt').exists():
                    continue
                torch.manual_seed(seed)
                model = build_arch(task, 'typed')
                init = hashlib.sha256(b''.join(t.numpy().tobytes() for t in model.state_dict().values())).hexdigest()
                opt = torch.optim.Adam(model.parameters(), lr=.01)
                loss_fn = torch.nn.BCEWithLogitsLoss()
                subset = fz['f25'] if regime == 'f25' else np.arange(len(meta))
                xb, yb = xf[subset], fy[subset]
                curve = []; started = time.time()
                for epoch in range(300):
                    opt.zero_grad()
                    loss = loss_fn(model(xc), y)
                    if regime != 'clean':
                        loss = loss + loss_fn(model(xb), yb)
                    loss.backward(); opt.step()
                    curve.append(float(loss.detach()))
                ck = dict(state=model.state_dict(), task=task, arch='typed', seed=seed,
                          regime=regime, mu=stats[0], sd=stats[1], mode=stats[2],
                          n_params=count_params(model), init_sha256=init)
                torch.save(ck, dest / 'model.pt')
                receipt = dict(loss=curve, seconds=time.time()-started, epochs=300, lr=.01,
                               optimizer='Adam', selection='fixed 300 epochs, no eval selection',
                               init_sha256=init, train_sha256=sha(OLD/f'{task}_train/scenes.npz'),
                               flip_sha256=sha(OLD/f'{task}_flips.npz'), n_clean=len(x),
                               n_flip=0 if regime=='clean' else len(subset),
                               model_sha256=sha(dest/'model.pt'))
                (dest/'training.json').write_text(json.dumps(receipt, indent=2))
                print(task, seed, regime, 'loss', curve[-1], 'seconds', receipt['seconds'], flush=True)


def evaluate():
    summary = {}; symmetry = {}
    for task in TASKS:
        bp = ROOT / f'artifacts/p123_upgrade/bank/bank_u10{task}.npz'
        b = np.load(bp); qm = json.loads(str(b['Qmeta']))
        x = np.stack([b['Qx'][::2], b['Qx'][1::2], b['Qx'][::2]+b['Qe'][::2]], axis=1)
        labels = np.array([[m['yA'], m['yB'], m['yAB']] for m in qm[::2]])
        parents = np.array([m['parent'] for m in qm[::2]])
        ev = np.load(OLD/f'{task}_eval/scenes.npz')['positions']
        x0 = ev[parents]
        # Store start, A, B, AB including exact historical sample ids/edits.
        sample = dict(states=x, start=x0, labels=labels, parents=parents,
                      qid=np.array([m['qid'] for m in qm[::2]]), edits=b['Qe'],
                      sample_id=np.array([f'{sha(bp)}:{m["qid"]}' for m in qm[::2]]))
        np.savez_compressed(OUT/f'{task}_samples.npz', **sample)
        arms={}; oks={}; sym={}
        for arch in ('raw','six','typed'):
            for seed in (11,23,47):
                for regime in ('clean','flip','f25'):
                    versions = ('old','fixed') if arch=='typed' else ('old',)
                    for version in versions:
                        base = OUT if version=='fixed' else OLD
                        mp = base/f'{task}_{arch}_s{seed}/{regime}/model.pt'
                        if not mp.exists():
                            continue
                        ck = torch.load(mp, map_location='cpu', weights_only=False)
                        model = build_arch(task,arch); model.load_state_dict(ck['state']); model.eval()
                        stats = ck['mu'],ck['sd'],ck['mode']
                        sc = predict(model,task,arch,x.reshape(-1,TASKS[task]['npts'],2),stats).reshape(-1,3)
                        sc0 = predict(model,task,arch,x0,stats)
                        ok = (sc>0)==labels
                        name=f'{arch}_{regime}_s{seed}_{version}'; oks[name]=ok
                        np.savez_compressed(OUT/f'{task}_{name}_scores.npz',scores=sc,start_score=sc0,correct=ok)
                        arms[name]=dict(J=float(ok.all(1).mean()), successes=int(ok.all(1).sum()), n=len(ok),
                            atomic_pass=float(ok[:,:2].all(1).mean()), atomic_marginals=ok.mean(0).tolist(),
                            checkpoint_sha256=sha(mp), n_params=count_params(model),
                            by_y0={str(y):dict(n=int((labels[:,0]==y).sum()), J=float(ok[labels[:,0]==y].all(1).mean())) for y in (0,1)})
                        if arch=='typed':
                            ref = predict(model,task,arch,ev,stats)
                            diffs=[]; pdiffs=[]
                            for g in groups(task):
                                v=predict(model,task,arch,ev[:,g,:],stats)
                                diffs.append(float(np.max(np.abs(v-ref)))); pdiffs.append(int(((v>0)!=(ref>0)).sum()))
                                if version=='fixed':
                                    np.testing.assert_allclose(v,ref,rtol=1e-5,atol=1e-4)
                                    assert np.array_equal(v>0,ref>0)
                            sym[name]=dict(max_logit_difference=max(diffs), prediction_difference=max(pdiffs), mode=ck['mode'])
        interaction={}
        for seed in (11,23,47):
            names=[f'{a}_{r}_s{seed}_old' for a,r in [('raw','clean'),('raw','flip'),('six','clean'),('six','flip')]]
            j=np.stack([oks[n].all(1) for n in names],1).astype(float)
            iv=j[:,3]-j[:,2]-j[:,1]+j[:,0]
            assert abs(iv.mean()-((j[:,3].mean()-j[:,2].mean())-(j[:,1].mean()-j[:,0].mean())))<1e-12
            interaction[str(seed)] = dict(arms=names, four_J=j.mean(0).tolist(),
                interaction=parent_bootstrap(iv,parents), feature_advantage_flip=parent_bootstrap(j[:,3]-j[:,1],parents),
                by_y0={str(y):parent_bootstrap(iv[labels[:,0]==y],parents[labels[:,0]==y]) for y in (0,1)})
            h=oks[f'raw_clean_s{seed}_old'][:,:2].all(1)&~oks[f'raw_clean_s{seed}_old'][:,2]
            was111=oks[f'raw_clean_s{seed}_old'].all(1)
            for name,ok in oks.items():
                if f'_s{seed}_' not in name: continue
                arms[name].update(CCM_denominator=int(h.sum()), R_full=float(ok[h].all(1).mean()),
                    M=float((ok[h,2]&~ok[h,:2].all(1)).mean()),
                    original111_n=int(was111.sum()), degradation111=float((~ok[was111].all(1)).mean()) if was111.any() else None)
        summary[task]=dict(bank_sha256=sha(bp),sample_sha256=sha(OUT/f'{task}_samples.npz'),n=len(labels),
                           n_parents=len(np.unique(parents)), distance_dimensions=TASKS[task]['in_six'],arms=arms,interaction=interaction)
        symmetry[task]=sym
    (OUT/'U10_CORRECTED.json').write_text(json.dumps(summary,indent=2))
    (OUT/'SYMMETRY_PIPELINE_AUDIT.json').write_text(json.dumps(symmetry,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','eval'],required=True);p.add_argument('--out-dir',type=Path,default=OUT);a=p.parse_args()
    OUT=a.out_dir
    if (OUT/f'{a.stage}_provenance.json').exists():raise FileExistsError('Choose a fresh --out-dir; existing stage is immutable')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{a.stage}_provenance.json').write_text(json.dumps(dict(command=sys.argv,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/f095_campaign/u10_targets.py']},
        analysis='correction/reanalysis on exposed bank; not independent confirmation'),indent=2))
    train() if a.stage=='train' else evaluate()
