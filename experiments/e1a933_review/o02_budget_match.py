"""O02: parameter- and optimization-budget-matched typed vs distance(six) vs raw.

Immutable bank only: reads artifacts/f095_campaign/U10/{T}_train|flips|eval (read
only) and the frozen E-quartet bank artifacts/p123_upgrade/bank/bank_u10{T}.npz.
Everything new is written under artifacts/e1a933_review/data_u10_o02/.

Parameter matching
------------------
The confound to remove is "typed is larger (~9553 params) and therefore maybe
harder to optimise".  raw/six have FEWER parameters (6529-6849) and win, so
enlarging raw/six would only strengthen the winning side and could not explain
typed's deficit.  We therefore SHRINK typed to a matched budget: TypedTriMLP /
TypedDiskMLP with pw=ph=sh=40 -> 6681 params, i.e. within -1.5%/+2.5% of every
raw/six arm of both tasks.  raw/six are left byte-identical to the published
main-line arms so the matched comparison is directly comparable with them.

Optimization budget
-------------------
Equal-recipe comparison ("hist"): Adam lr=1e-2, 300 full-batch steps, BCE,
lambda=1.0 on the clean scenes plus the single-edit (flip) pairs -- exactly the
historical U10 recipe.
Best-recipe comparison ("grid"): lr in {1e-2,3e-3,1e-3} x steps in {300,3000},
run for raw, six and typed_m alike (identical candidate budget).  Selection is
train-only: argmin of the FINAL TRAINING OBJECTIVE (BCE(clean) + BCE(single
edits)) averaged over the three seeds of the flip regime.  The evaluation J is
never used for selection; a descriptive "what J would have picked" line is
reported for transparency only.

Discipline: all three seeds (11/23/47) are kept everywhere, including weak or
collapsed ones; no seed is dropped; no result-driven denominator changes; no
test-J epoch/hyper-parameter selection; every number is traceable to a file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/f095_campaign'))
sys.path.insert(0, str(ROOT / 'experiments/e1a933_review'))
from u10_targets import build_arch, fit_stats, featurize, TASKS, count_params  # noqa: E402
from u10_models import TypedTriMLP, TypedDiskMLP  # noqa: E402
from data_stats import parent_bootstrap  # noqa: E402

torch.set_num_threads(4)

OLD = ROOT / 'artifacts/f095_campaign/U10'
OUT = ROOT / 'artifacts/e1a933_review/data_u10_o02'
SEEDS = (11, 23, 47)
TYPED_M_W = 40
LR_TAG = {'lr01': 1e-2, 'lr003': 3e-3, 'lr001': 1e-3}
GRID = [('lr01', 300), ('lr003', 300), ('lr001', 300),
        ('lr01', 3000), ('lr003', 3000), ('lr001', 3000)]
HIST = ('lr01', 300)
GRID_ARCHS = ('raw', 'six', 'typed_m')
ALL_ARCHS = ('raw', 'six', 'typed_m', 'typed_o')


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def state_sha(model: nn.Module) -> str:
    return hashlib.sha256(
        b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()


def fz_arch(arch: str) -> str:
    """featurize()/fit_stats() input name: typed_m and typed_o share 'typed'."""
    return 'typed' if arch.startswith('typed') else arch


def build(task: str, arch: str) -> nn.Module:
    if arch in ('raw', 'six'):
        return build_arch(task, arch)
    if arch == 'typed_o':
        return build_arch(task, 'typed')          # TypedT*(48,48,48) = 9553 params
    if arch == 'typed_m':
        w = TYPED_M_W
        return TypedTriMLP(w, w, w) if task == 'T1' else TypedDiskMLP(w, w, w)
    raise ValueError(arch)


def recipe_id(recipe) -> str:
    lrt, steps = recipe
    return f'{lrt}_s{steps}'


def arm_name(task, arch, recipe, regime, seed) -> str:
    return f'{task}_{arch}_{recipe_id(recipe)}_{regime}_s{seed}'


# ---------------------------------------------------------------- data loading
def load_bank(task: str) -> dict:
    bp = ROOT / f'artifacts/p123_upgrade/bank/bank_u10{task}.npz'
    b = np.load(bp)
    qm = json.loads(str(b['Qmeta']))
    states = np.stack([b['Qx'][::2], b['Qx'][1::2], b['Qx'][::2] + b['Qe'][::2]], axis=1)
    labels = np.array([[m['yA'], m['yB'], m['yAB']] for m in qm[::2]])
    parents = np.array([m['parent'] for m in qm[::2]])
    y0 = np.array([m['y0'] for m in qm[::2]])
    ev = np.load(OLD / f'{task}_eval/scenes.npz')['positions']
    start = ev[parents]
    ref = np.load(ROOT / f'artifacts/e1a933_review/data_u10/{task}_samples.npz',
                  allow_pickle=True)
    # provenance cross-check: same bank -> same states/labels/parents as R01/R02
    assert np.array_equal(ref['parents'], parents) and np.array_equal(ref['labels'], labels)
    assert np.allclose(ref['states'], states) and np.allclose(ref['start'], start)
    return dict(task=task, bank=str(bp.relative_to(ROOT)), bank_sha256=sha(bp),
                states=states, start=start, labels=labels, y0=y0, parents=parents,
                n=len(labels), n_parents=len(np.unique(parents)))


def load_train(task: str, arch: str) -> dict:
    d = np.load(OLD / f'{task}_train/scenes.npz')
    x, y = d['positions'].astype(float), d['labels'].astype(np.int64)
    fz = np.load(OLD / f'{task}_flips.npz', allow_pickle=True)
    meta = fz['meta']
    pi = np.array([int(r[0]) for r in meta])
    edits = np.stack([fz[f'edit_{k}'] for k in range(len(meta))])
    fy = np.array([int(r[1]) for r in meta], dtype=np.int64)
    f25 = np.asarray(fz['f25'], dtype=np.int64)
    stats = fit_stats(task, fz_arch(arch), x)
    Xc = featurize(task, fz_arch(arch), x, stats)
    Xf = featurize(task, fz_arch(arch), x[pi].astype(float) + edits.astype(float), stats)
    Xq = featurize(task, fz_arch(arch),
                   x[pi[f25]].astype(float) + edits[f25].astype(float), stats)
    return dict(x=x, y=y, pi=pi, edits=edits, fy=fy, f25=f25, stats=stats,
                Xc=Xc, yc=torch.from_numpy(y.astype(np.float32)),
                Xf=Xf, yf=torch.from_numpy(fy.astype(np.float32)),
                Xq=Xq, yq=torch.from_numpy(fy[f25].astype(np.float32)),
                n_flip=len(meta), n_f25=len(f25))


# ------------------------------------------------------------------- training
@torch.no_grad()
def diag(model: nn.Module, d: dict, bce: nn.Module) -> dict:
    model.eval()
    out = {}
    for tag, X, y in (('clean', d['Xc'], d['yc']), ('single', d['Xf'], d['yf']),
                      ('f25', d['Xq'], d['yq'])):
        lg = model(X)
        out[f'{tag}_bce'] = float(bce(lg, y))
        out[f'{tag}_acc'] = float(((lg > 0) == (y > 0.5)).float().mean())
    model.train()
    return out


def train_one(task: str, arch: str, recipe, regime: str, seed: int, d: dict,
              dest: Path) -> dict:
    lr = LR_TAG[recipe[0]]
    steps = recipe[1]
    torch.manual_seed(seed)
    model = build(task, arch)
    init = state_sha(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    dec = max(1, steps // 60)
    curve, t0 = [], time.time()
    for step in range(steps):
        opt.zero_grad()
        loss = bce(model(d['Xc']), d['yc'])
        if regime == 'flip':
            loss = loss + bce(model(d['Xf']), d['yf'])
        elif regime == 'f25':
            loss = loss + bce(model(d['Xq']), d['yq'])
        loss.backward()
        opt.step()
        if step % dec == 0 or step == steps - 1:
            curve.append([step, float(loss.detach())])
    seconds = time.time() - t0
    fit = diag(model, d, bce)
    dest.mkdir(parents=True, exist_ok=True)
    ck = dict(state=model.state_dict(), task=task, arch=arch, recipe=recipe_id(recipe),
              lr=lr, steps=steps, regime=regime, seed=seed,
              mu=d['stats'][0], sd=d['stats'][1], mode=d['stats'][2],
              n_params=count_params(model), init_sha256=init)
    torch.save(ck, dest / 'model.pt')
    rec = dict(arm=dest.name, task=task, arch=arch, recipe=recipe_id(recipe), lr=lr,
               steps=steps, regime=regime, seed=seed, optimizer='Adam',
               objective='BCE(clean) + 1.0*BCE(single-edit)' if regime != 'clean'
               else 'BCE(clean)',
               full_batch=True, n_clean=len(d['y']), n_flip=d['n_flip'],
               n_f25=d['n_f25'], n_params=count_params(model), init_sha256=init,
               final_objective=float(loss.detach()), fit=fit, seconds=seconds,
               loss_curve=curve, checkpoint_sha256=sha(dest / 'model.pt'),
               train_sha256=sha(OLD / f'{task}_train/scenes.npz'),
               flip_sha256=sha(OLD / f'{task}_flips.npz'),
               selection='fixed recipe; no eval-time selection')
    (dest / 'training.json').write_text(json.dumps(rec, indent=1))
    return rec


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def score(model: nn.Module, task: str, arch: str, X: np.ndarray, stats) -> np.ndarray:
    out = []
    for i in range(0, len(X), 1024):
        out.append(model(featurize(task, fz_arch(arch), X[i:i + 1024], stats)).numpy())
    return np.concatenate(out)


def metrics(ok: np.ndarray, ok_start: np.ndarray, labels: np.ndarray,
            y0: np.ndarray, parents: np.ndarray) -> dict:
    j = ok.all(1)
    return dict(
        J=float(j.mean()), successes=int(j.sum()), n=int(len(j)),
        J_with_start=float((j & ok_start).mean()),
        atomic=dict(P=float(ok_start.mean()), A=float(ok[:, 0].mean()),
                    B=float(ok[:, 1].mean()), AB=float(ok[:, 2].mean())),
        by_y0={str(v): dict(n=int((y0 == v).sum()), J=float(j[y0 == v].mean()))
               for v in (0, 1)},
        J_ci=parent_bootstrap(j.astype(float), parents))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['train', 'eval', 'all'], required=True)
    p.add_argument('--out-dir', type=Path, default=OUT)
    p.add_argument('--tasks', default='T1,T2')
    p.add_argument('--grid', choices=['full', 'hist'], default='full',
                   help='full = lr x steps ladder; hist = historical recipe only '
                        '(used for the numerical-execution-order replicate)')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    out = a.out_dir
    tasks = a.tasks.split(',')
    grid = GRID if a.grid == 'full' else [HIST]

    if a.stage in ('train', 'all'):
        if (out / 'train_provenance.json').exists():
            raise FileExistsError('fresh --out-dir required; existing train stage is immutable')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'train_provenance.json').write_text(json.dumps(dict(
            command=sys.argv,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                  text=True).strip(),
            source_sha256={str(q.relative_to(ROOT)): sha(q) for q in (
                Path(__file__), ROOT / 'experiments/f095_campaign/u10_targets.py',
                ROOT / 'experiments/f095_campaign/u10_models.py',
                ROOT / 'experiments/e1a933_review/data_stats.py')},
            bank_sha256={t: sha(ROOT / f'artifacts/p123_upgrade/bank/bank_u10{t}.npz')
                         for t in tasks},
            analysis='reanalysis + new training on the immutable historical bank; '
                     'not independent confirmation',
            param_matching=f'typed shrunk to w={TYPED_M_W} (6681 params)',
            grid=[recipe_id(r) for r in grid], seeds=list(SEEDS),
            torch_threads=a.threads,
            note='thread count changes float reduction order; raw/six hist arms were '
                 'trained at 4 threads and reproduce bit-exactly there'), indent=1))
        log = open(out / 'train.log', 'a')
        for task in tasks:
            for arch in GRID_ARCHS:
                d = load_train(task, arch)
                for recipe in grid:
                    for regime in ('clean', 'flip'):
                        for seed in SEEDS:
                            nm = arm_name(task, arch, recipe, regime, seed)
                            dest = out / 'checkpoints' / nm
                            if (dest / 'model.pt').exists():
                                continue
                            rec = train_one(task, arch, recipe, regime, seed, d, dest)
                            line = (f'{nm} params={rec["n_params"]} '
                                    f'obj={rec["final_objective"]:.5f} '
                                    f'single_bce={rec["fit"]["single_bce"]:.5f} '
                                    f'clean_bce={rec["fit"]["clean_bce"]:.5f} '
                                    f'{rec["seconds"]:.1f}s')
                            print(line, flush=True)
                            log.write(line + '\n')
                            log.flush()
            for arch in ALL_ARCHS:  # historical recipe extras: typed_o + f25 regime
                d = load_train(task, arch)
                for recipe, regime in ((HIST, 'clean'), (HIST, 'flip'), (HIST, 'f25')):
                    for seed in SEEDS:
                        nm = arm_name(task, arch, recipe, regime, seed)
                        dest = out / 'checkpoints' / nm
                        if (dest / 'model.pt').exists():
                            continue
                        rec = train_one(task, arch, recipe, regime, seed, d, dest)
                        line = (f'{nm} params={rec["n_params"]} '
                                f'obj={rec["final_objective"]:.5f} '
                                f'{rec["seconds"]:.1f}s')
                        print(line, flush=True)
                        log.write(line + '\n')
                        log.flush()
        log.close()
    if a.stage == 'train':
        return

    # ------------------------------------------------------------- evaluation
    if (out / 'eval_provenance.json').exists():
        raise FileExistsError('fresh --out-dir required; existing eval stage is immutable')
    (out / 'eval_provenance.json').write_text(json.dumps(dict(
        command=sys.argv,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                              text=True).strip(),
        source_sha256={str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        eval_pool='artifacts/p123_upgrade/bank/bank_u10{T}.npz (frozen E-quartets)',
        torch_threads=a.threads,
        states='start(P) + A + B + AB; labels yA=yB=y0, yAB=1-y0',
        bootstrap='parent-cluster, 2000 resamples, quartet-weighted, seed 933'), indent=1))

    summary = {}
    repro = {}
    for task in tasks:
        bank = load_bank(task)
        parents, labels, y0 = bank['parents'], bank['labels'], bank['y0']
        Xall = np.concatenate([bank['states'].reshape(-1, TASKS[task]['npts'], 2),
                               bank['start']], 0)
        nq = bank['n']
        arms, oks, oks_start, scores_npz = {}, {}, {}, {}
        for ck_dir in sorted((out / 'checkpoints').glob(f'{task}_*')):
            mp = ck_dir / 'model.pt'
            ck = torch.load(mp, map_location='cpu', weights_only=False)
            arch = ck['arch']
            model = build(task, arch)
            model.load_state_dict(ck['state'])
            model.eval()
            stats = (ck['mu'], ck['sd'], ck['mode'])
            sc = score(model, task, arch, Xall, stats)
            sa, st = sc[:3 * nq].reshape(nq, 3), sc[3 * nq:]
            ok, ok_start = (sa > 0) == (labels > 0), (st > 0) == (y0 > 0)
            nm = ck_dir.name
            trec = json.loads((ck_dir / 'training.json').read_text())
            arms[nm] = dict(arch=arch, recipe=ck['recipe'], lr=ck['lr'], steps=ck['steps'],
                            regime=ck['regime'], seed=ck['seed'], n_params=ck['n_params'],
                            init_sha256=ck['init_sha256'],
                            train_fit=dict(final_objective=trec['final_objective'], **trec['fit']),
                            checkpoint_sha256=sha(mp), **metrics(ok, ok_start, labels, y0, parents))
            oks[nm], oks_start[nm] = ok, ok_start
            scores_npz[f'{nm}__scores'] = sa.astype(np.float32)
            scores_npz[f'{nm}__start'] = st.astype(np.float32)
            scores_npz[f'{nm}__correct'] = ok
            scores_npz[f'{nm}__start_correct'] = ok_start
        # CCM denominators: raw historical clean arm, per seed (published definition)
        ccm = {}
        for seed in SEEDS:
            ref = arm_name(task, 'raw', HIST, 'clean', seed)
            h = oks[ref][:, :2].all(1) & ~oks[ref][:, 2]
            was111 = oks[ref].all(1)
            ccm[str(seed)] = dict(reference_arm=ref, H_denominator=int(h.sum()),
                                  original111_n=int(was111.sum()))
            for nm, ok in oks.items():
                if arms[nm]['seed'] != seed:
                    continue
                arms[nm].update(
                    R_full=float(ok[h].all(1).mean()), M=float((ok[h, 2] & ~ok[h, :2].all(1)).mean()),
                    H_denominator=int(h.sum()),
                    original111_n=int(was111.sum()),
                    degradation111=float((~ok[was111].all(1)).mean()) if was111.any() else None)
        # paired contrasts and interactions
        def pair(n1, n2):
            return parent_bootstrap((oks[n1].all(1).astype(float) - oks[n2].all(1).astype(float)),
                                    parents)
        contrasts = {}
        for seed in SEEDS:
            def A(arch, recipe, regime, s=seed):
                return arm_name(task, arch, recipe, regime, s)
            contrasts[f's{seed}'] = dict(
                feature_advantage_flip=pair(A('six', HIST, 'flip'), A('raw', HIST, 'flip')),
                six_flip_minus_typed_m_flip=pair(A('six', HIST, 'flip'), A('typed_m', HIST, 'flip')),
                six_flip_minus_typed_o_flip=pair(A('six', HIST, 'flip'), A('typed_o', HIST, 'flip')),
                typed_m_flip_minus_typed_o_flip=pair(A('typed_m', HIST, 'flip'),
                                                     A('typed_o', HIST, 'flip')),
                typed_m_clean_minus_typed_o_clean=pair(A('typed_m', HIST, 'clean'),
                                                       A('typed_o', HIST, 'clean')),
                hist_four_J=[float(oks[A(a, HIST, r)].all(1).mean())
                             for a, r in (('raw', 'clean'), ('raw', 'flip'),
                                          ('six', 'clean'), ('six', 'flip'))],
                interaction_hist=parent_bootstrap(
                    oks[A('six', HIST, 'flip')].all(1).astype(float)
                    - oks[A('six', HIST, 'clean')].all(1).astype(float)
                    - oks[A('raw', HIST, 'flip')].all(1).astype(float)
                    + oks[A('raw', HIST, 'clean')].all(1).astype(float), parents),
            )
        summary[task] = dict(bank=bank['bank'], bank_sha256=bank['bank_sha256'],
                             n=bank['n'], n_parents=bank['n_parents'],
                             distance_dimensions=TASKS[task]['in_six'],
                             ccm_denominators=ccm, arms=arms, contrasts=contrasts)
        # train-only recipe selection (never uses J): argmin of the mean final
        # training objective of the flip regime over the three seeds.
        sel_table, selected = {}, {}
        have_grid = all((out / 'checkpoints' /
                         arm_name(task, arch, r, reg, s) / 'model.pt').exists()
                        for arch in GRID_ARCHS for r in GRID
                        for reg in ('clean', 'flip') for s in SEEDS)
        for arch in (GRID_ARCHS if have_grid else ()):
            rows = {}
            for recipe in GRID:
                rid = recipe_id(recipe)
                fo = [arms[arm_name(task, arch, recipe, 'flip', s)]['train_fit']['final_objective']
                      for s in SEEDS]
                sb = [arms[arm_name(task, arch, recipe, 'flip', s)]['train_fit']['single_bce']
                      for s in SEEDS]
                cb = [arms[arm_name(task, arch, recipe, 'flip', s)]['train_fit']['clean_bce']
                      for s in SEEDS]
                jj = [arms[arm_name(task, arch, recipe, 'flip', s)]['J'] for s in SEEDS]
                rows[rid] = dict(mean_final_objective=float(np.mean(fo)),
                                 mean_clean_bce=float(np.mean(cb)),
                                 mean_single_bce=float(np.mean(sb)),
                                 per_seed_final_objective=[float(v) for v in fo],
                                 mean_J_flip=float(np.mean(jj)), per_seed_J_flip=jj)
            sel_table[arch] = rows
            selected[arch] = min(rows, key=lambda k: (rows[k]['mean_final_objective'], k))
        best = {}
        for seed in (SEEDS if have_grid else ()):
            def B(arch, regime, s=seed):
                return arm_name(task, arch, (selected[arch].split('_s')[0],
                                             int(selected[arch].split('_s')[1])), regime, s)
            best[f's{seed}'] = dict(
                selected={a: selected[a] for a in GRID_ARCHS},
                four_J=[float(oks[B(a, r)].all(1).mean())
                        for a, r in (('raw', 'clean'), ('raw', 'flip'),
                                     ('six', 'clean'), ('six', 'flip'))],
                feature_advantage_flip=pair(B('six', 'flip'), B('raw', 'flip')),
                six_flip_minus_typed_m_flip=pair(B('six', 'flip'), B('typed_m', 'flip')),
                interaction_best=parent_bootstrap(
                    oks[B('six', 'flip')].all(1).astype(float)
                    - oks[B('six', 'clean')].all(1).astype(float)
                    - oks[B('raw', 'flip')].all(1).astype(float)
                    + oks[B('raw', 'clean')].all(1).astype(float), parents))
        summary[task]['recipe_selection'] = dict(
            rule='argmin over seeds of mean final training objective (BCE(clean)+BCE(single)), '
                 'flip regime; J never used',
            table=sel_table, selected=selected,
            J_argmin_descriptive={a: min(sel_table[a], key=lambda k: -sel_table[a][k]['mean_J_flip'])
                                  for a in sel_table},
            best_recipe_comparison=best) if have_grid else None
        np.savez_compressed(out / f'{task}_arm_scores.npz', **scores_npz)
        # historical-ckpt reproduction check (bit-exact or report J difference)
        for arch, histdir in (('raw', 'raw'), ('six', 'six')):
            for regime in ('clean', 'flip', 'f25'):
                for seed in SEEDS:
                    old = OLD / f'{task}_{histdir}_s{seed}/{regime}/model.pt'
                    if not old.exists():
                        continue
                    oc = torch.load(old, map_location='cpu', weights_only=False)
                    ours = torch.load(out / 'checkpoints' /
                                      arm_name(task, arch, HIST, regime, seed) / 'model.pt',
                                      map_location='cpu', weights_only=False)
                    same = state_sha_dict(oc['state']) == state_sha_dict(ours['state'])
                    repro[f'{task}_{arch}_{regime}_s{seed}'] = dict(
                        historical_ckpt=str(old.relative_to(ROOT)), bit_exact=same)
        for regime in ('clean', 'flip', 'f25'):
            for seed in SEEDS:
                ref = (ROOT / f'artifacts/e1a933_review/data_u10/{task}_typed_s{seed}/'
                       f'{regime}/model.pt')
                ours = out / 'checkpoints' / arm_name(task, 'typed_o', HIST, regime, seed) / 'model.pt'
                if ref.exists() and ours.exists():
                    same = (state_sha_dict(torch.load(ref, map_location='cpu',
                                                      weights_only=False)['state'])
                            == state_sha_dict(torch.load(ours, map_location='cpu',
                                                         weights_only=False)['state']))
                    repro[f'{task}_typed_o_{regime}_s{seed}'] = dict(
                        historical_ckpt=str(ref.relative_to(ROOT)), bit_exact=same)
    (out / 'O02_BUDGET_MATCH.json').write_text(json.dumps(summary, indent=1))
    (out / 'HIST_REPRODUCTION.json').write_text(json.dumps(repro, indent=1))
    print('DONE o02', a.stage, flush=True)


def state_sha_dict(sd: dict) -> str:
    return hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in sd.values())).hexdigest()


if __name__ == '__main__':
    main()
