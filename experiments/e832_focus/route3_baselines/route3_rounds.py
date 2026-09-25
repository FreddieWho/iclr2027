#!/usr/bin/env python3
"""Route 3 three evidence-driven rounds (additive file; does not touch results.json).

R1 optimization diagnostic: same bank/seeds/arms as the frozen baseline, but
  epochs {100, 300} x lr {1e-3, 3e-4}, reporting train/dev BCE and train
  accuracy. Decides whether near-zero J3 is underfitting or structural.
R2 representation diagnostic: same bank/seeds/budget, one fixed MLP family,
  inputs {raw, six, rich_unsorted, rich_sorted, orbit_canonical}. Tests whether
  the failure is the input representation.
R3 fresh-bank confirmation (predeclared before R2 results): new parent seeds,
  analytic parser + rich_sorted + orbit_canonical. Confirms the parser-bounded
  gap persists on independent parents.

Contracts: train/dev only (clean + single flips); selection by dev BCE only;
test J never used for selection; no sealed pools; no old-bank reuse.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments' / 'e832_focus' / 'route1_cross_task'))
sys.path.insert(0, str(ROOT / 'experiments' / 'e832_focus' / 'route3_baselines'))
import common_runner as cr
from strong_baselines import intersects, SegmentSetRel, CapacityRaw

torch.set_num_threads(4)
SEEDS = (11, 23, 47)
G8 = [(0,1,2,3),(1,0,2,3),(0,1,3,2),(1,0,3,2),(2,3,0,1),(3,2,0,1),(2,3,1,0),(3,2,1,0)]

def orbit_canonical(x):
    """Lexicographically minimal 6-distance vector over the legal G8 orbit."""
    p = np.asarray(x, float).reshape(-1, 4, 2)
    ii, jj = np.triu_indices(4, 1)
    cands = []
    for perm in G8:
        q = p[:, list(perm), :]
        cands.append(np.linalg.norm(q[:, ii, :] - q[:, jj, :], axis=-1))
    cands = np.stack(cands, 1)
    best = np.empty(len(p), dtype=np.int64)
    for i in range(len(p)):
        keys = [cands[i, :, k] for k in range(5, -1, -1)]
        best[i] = np.lexsort(keys)[0]
    return cands[np.arange(len(p)), best]

def feat(arm, x):
    x = np.asarray(x, np.float32)
    if arm == 'raw':
        return x.reshape(len(x), -1)
    if arm == 'six':
        return cr.six('source', x)
    if arm == 'orbit_canonical':
        return orbit_canonical(x)
    f = cr.rich('source', x)
    return cr.sort_features('source', f) if arm == 'rich_sorted' else f

class MLP(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).reshape(-1)

def fit_mlp(arm, seed, X, Y, F, dev, epochs, lr):
    mu, sd = feat(arm, X).mean(0), feat(arm, X).std(0) + 1e-8
    def prep(x):
        return torch.from_numpy(((feat(arm, x) - mu) / sd).astype(np.float32))
    xc, yc = prep(X), torch.from_numpy(Y.astype(np.float32))
    xf, yf = prep(F[0]), torch.from_numpy(F[1].astype(np.float32))
    dx = torch.from_numpy(np.r_[prep(dev[0]), prep(dev[2])])
    dy = torch.from_numpy(np.r_[dev[1], dev[3]].astype(np.float32))
    torch.manual_seed(seed)
    m = MLP(xc.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lf = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lf(m(xc), yc) + lf(m(xf), yf)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        devb = float(lf(m(dx), dy))
        train_pred = m(xc).numpy()
    train_bce = float(lf(torch.from_numpy(train_pred), yc))
    train_acc = float(((train_pred > 0) == (Y == 1)).mean())
    return m, mu, sd, {'dev_bce': devb, 'train_bce': train_bce, 'train_acc': train_acc,
                       'seconds': time.time() - t0, 'parameters': sum(p.numel() for p in m.parameters()),
                       'epochs': epochs, 'lr': lr}

def evaluate(task, arm, m, mu, sd, bank):
    def prep(x):
        return torch.from_numpy(((feat(arm, x) - mu) / sd).astype(np.float32))
    m.eval()
    with torch.no_grad():
        zz = np.stack([m(prep(bank['states'][i])).numpy() for i in range(4)])
    lab = np.stack([bank['y0'], bank['y0'], bank['y0'], bank['yab']])
    return cr.metric((zz > 0) == (lab == 1), bank['parent'])

def build_bank(train_seed, dev_seed, test_seed, edit_seed):
    X, Y, _ = cr.gen('source', 256, train_seed)
    F = cr.mine_flips('source', X, Y, train_seed + 1, 4)
    DX, DY, _ = cr.gen('source', 64, dev_seed)
    DS = cr.mine_flips('source', DX, DY, dev_seed + 1, 3)
    T, TY, _ = cr.gen('source', 256, test_seed)
    B = cr.mine_E('source', T, TY, edit_seed, 6, 3)
    assert B is not None
    fp = hashlib.sha256(np.ascontiguousarray(B['states']).tobytes()).hexdigest()[:16]
    return (X, Y, F, (DX, DY, DS[0], DS[1]), B, fp)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts' / 'e832_focus' / 'route3')
    p.add_argument('--rounds', nargs='+', default=['r1', 'r2', 'r3'])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    if 'r1' in a.rounds:
        X, Y, F, dev, B, fp = build_bank(832501, 832502, 832503, 832504)
        rows = []
        for arm in ('capacity', 'relational'):
            for seed in SEEDS:
                for epochs, lr in ((100, 1e-3), (300, 1e-3), (300, 3e-4)):
                    m, mu, sd, rec = fit_relational_or_capacity(arm, seed, X, Y, F, dev, epochs, lr)
                    met = eval_legacy(arm, m, mu, sd, X, Y, F, B)
                    rows.append({'arm': arm, 'seed': seed, **rec, 'metric': met})
        (a.out / 'round1_optim.json').write_text(json.dumps(
            {'bank_fp': fp, 'E': len(B['parent']), 'parents': len(np.unique(B['parent'])), 'rows': rows}, indent=2) + '\n')
        print('r1', len(rows), flush=True)

    if 'r2' in a.rounds:
        X, Y, F, dev, B, fp = build_bank(832501, 832502, 832503, 832504)
        rows = []
        for arm in ('raw', 'six', 'rich_unsorted', 'rich_sorted', 'orbit_canonical'):
            for seed in SEEDS:
                m, mu, sd, rec = fit_mlp(arm, seed, X, Y, F, dev, 300, 1e-3)
                rows.append({'arm': arm, 'seed': seed, **rec, 'metric': evaluate('source', arm, m, mu, sd, B)})
        (a.out / 'round2_repr.json').write_text(json.dumps(
            {'bank_fp': fp, 'E': len(B['parent']), 'parents': len(np.unique(B['parent'])), 'rows': rows}, indent=2) + '\n')
        print('r2', len(rows), flush=True)

    if 'r3' in a.rounds:
        X, Y, F, dev, B, fp = build_bank(833501, 833502, 833503, 833504)
        ap = np.stack([np.array([intersects(s) for s in B['states'][i]]) for i in range(4)])
        lab = np.stack([B['y0'], B['y0'], B['y0'], B['yab']])
        aok = ap == lab
        analytic = {'J3': float((aok[1] & aok[2] & aok[3]).mean()), 'A': float(aok[1].mean()),
                    'B': float(aok[2].mean()), 'AB': float(aok[3].mean())}
        rows = []
        for arm in ('rich_sorted', 'orbit_canonical'):
            for seed in SEEDS:
                m, mu, sd, rec = fit_mlp(arm, seed, X, Y, F, dev, 300, 1e-3)
                rows.append({'arm': arm, 'seed': seed, **rec, 'metric': evaluate('source', arm, m, mu, sd, B)})
        (a.out / 'round3_freshbank.json').write_text(json.dumps(
            {'bank_fp': fp, 'E': len(B['parent']), 'parents': len(np.unique(B['parent'])),
             'analytic_parser': analytic, 'rows': rows}, indent=2) + '\n')
        print('r3', analytic['J3'], len(rows), flush=True)

def fit_relational_or_capacity(arm, seed, X, Y, F, dev, epochs, lr):
    import strong_baselines as b
    mu = X.reshape(-1, 2).mean(0); mu = np.tile(mu, 4)
    sd = X.reshape(-1, 2).std(0) + 1e-8; sd = np.tile(sd, 4)
    def prep(x):
        return torch.from_numpy(((x.reshape(len(x), -1) - mu) / sd).astype(np.float32))
    torch.manual_seed(seed)
    m = b.CapacityRaw() if arm == 'capacity' else b.SegmentSetRel()
    xc, yc = prep(X), torch.from_numpy(Y.astype(np.float32))
    xf, yf = prep(F[0]), torch.from_numpy(F[1].astype(np.float32))
    dx = torch.from_numpy(np.r_[prep(dev[0]), prep(dev[2])])
    dy = torch.from_numpy(np.r_[dev[1], dev[3]].astype(np.float32))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lf = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lf(m(xc), yc) + lf(m(xf), yf)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        devb = float(lf(m(dx), dy)); tp = m(xc).numpy()
    return m, mu, sd, {'dev_bce': devb, 'train_bce': float(lf(torch.from_numpy(tp), yc)),
                       'train_acc': float(((tp > 0) == (Y == 1)).mean()),
                       'seconds': time.time() - t0,
                       'parameters': sum(p.numel() for p in m.parameters()),
                       'epochs': epochs, 'lr': lr}

def eval_legacy(arm, m, mu, sd, X, Y, F, B):
    def prep(x):
        return torch.from_numpy(((x.reshape(len(x), -1) - mu) / sd).astype(np.float32))
    m.eval()
    with torch.no_grad():
        zz = np.stack([m(prep(B['states'][i])).numpy() for i in range(4)])
    lab = np.stack([B['y0'], B['y0'], B['y0'], B['yab']])
    return cr.metric((zz > 0) == (lab == 1), B['parent'])

if __name__ == '__main__':
    main()
