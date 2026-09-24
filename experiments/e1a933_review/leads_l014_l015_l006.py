"""L-015(b) + L-006 on the source crossing task, plus L-014 radius transcription check.

L-015 asks whether injecting the task symmetry (8-element endpoint/segment relabeling)
raises joint correctness and removes labeling dependence.
Two interventions, same budget:
  permaug  : raw MLP trained on all 8 legal relabelings of each state
  setmlp   : segment-set network, exactly invariant under that group
Controls: raw MLP and relation-feature MLP (known positive).

L-006: 10 seeds per arm. Fixed 300 epochs, Adam 1e-2, no eval-J selection.
Thread count is fixed at 4 and recorded; do not compare these weights to a
2-thread rerun.

Eval parents are newly generated and checked against the training bank.
Sealed confirm/holdout pools are not read.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
from coord_mlp import CoordMLP  # noqa: E402
from common import rel_features  # noqa: E402
from core.relations import make_relational_scenes, segment_relation  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402

torch.set_num_threads(4)
THREADS = 4
EPOCHS, LR = 300, 1e-2
SEEDS = (11, 23, 47, 71, 83, 101, 113, 127, 139, 151)
ARMS = ("raw", "permaug", "setmlp", "relfeat")


def group8():
    out = []
    for a, b, c in itertools.product((0, 1), repeat=3):
        idx = [0, 1, 2, 3]
        if a:
            idx[0], idx[1] = idx[1], idx[0]
        if b:
            idx[2], idx[3] = idx[3], idx[2]
        if c:
            idx[0], idx[2] = idx[2], idx[0]
            idx[1], idx[3] = idx[3], idx[1]
        out.append(tuple(idx))
    return out


G8 = group8()


def apply_perm(x, perm):
    x = np.asarray(x, np.float32).reshape(-1, 4, 2)
    return x[:, list(perm), :]


def g8_features(x):
    """10-d features invariant under G8 (within-segment swaps and segment swap), not under S4.

    A full sum-pool over four points would also be invariant under permutations that mix the
    two segments, which the crossing label is not. Sorting *within* each segment and then
    sorting the two segments keeps the pair structure.
    """
    x = np.asarray(x, np.float32).reshape(-1, 4, 2)
    d01 = np.linalg.norm(x[:, 0] - x[:, 1], axis=-1)
    d23 = np.linalg.norm(x[:, 2] - x[:, 3], axis=-1)
    within = np.sort(np.stack([d01, d23], axis=1), axis=1)
    cross = np.stack([
        np.linalg.norm(x[:, 0] - x[:, 2], axis=-1),
        np.linalg.norm(x[:, 0] - x[:, 3], axis=-1),
        np.linalg.norm(x[:, 1] - x[:, 2], axis=-1),
        np.linalg.norm(x[:, 1] - x[:, 3], axis=-1),
    ], axis=1)
    # segment swap and within-segment swap only permute these four; sorting removes that.
    cross = np.sort(cross, axis=1)
    # radii of the two midpoints, sorted, plus absolute direction-cross (parallelism)
    m0 = 0.5 * (x[:, 0] + x[:, 1])
    m1 = 0.5 * (x[:, 2] + x[:, 3])
    mid = np.linalg.norm(m0 - m1, axis=-1, keepdims=True)
    u0 = x[:, 1] - x[:, 0]
    u1 = x[:, 3] - x[:, 2]
    # absolute 2d cross of directions: invariant to reversing either segment and to swapping them
    cross2 = np.abs(u0[:, 0] * u1[:, 1] - u0[:, 1] * u1[:, 0])
    cross2 = cross2 / (np.linalg.norm(u0, axis=-1) * np.linalg.norm(u1, axis=-1) + 1e-8)
    return np.concatenate([within, cross, mid, cross2[:, None]], axis=1).astype(np.float32)


class G8SetMLP(nn.Module):
    """Learned segment-set. Nonlinear pooling per segment, then sum over the two segments.

    Invariant under within-segment swaps and segment swap. Not invariant under mixing
    the two segments, because the nonlinearity is applied before the segment sum.
    """

    def __init__(self, w=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())
        self.psi = nn.Sequential(nn.Linear(w, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())
        self.rho = nn.Linear(w, 1)

    def forward(self, x):
        h = self.phi(x.reshape(-1, 4, 2))
        s0 = self.psi(h[:, 0] + h[:, 1])
        s1 = self.psi(h[:, 2] + h[:, 3])
        return self.rho(s0 + s1).reshape(-1)


class S4SetMLP(nn.Module):
    """Sum-pool over all four points. Invariant under every point permutation, including
    regroupings that change the crossing label. Negative control for too much symmetry."""

    def __init__(self, w=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(2, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())
        self.psi = nn.Sequential(nn.Linear(w, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())
        self.rho = nn.Linear(w, 1)

    def forward(self, x):
        h = self.phi(x.reshape(-1, 4, 2)).sum(dim=-2)
        return self.rho(self.psi(h)).reshape(-1)


def build(arm):
    if arm == "g8set":
        return G8SetMLP(32)
    if arm == "g8wide":
        return G8SetMLP(48)  # one width, chosen to match raw's 6849 parameters (7249)
    if arm == "s4set":
        return S4SetMLP(32)
    if arm == "setmlp":
        return CoordMLP(64, 32, in_dim=8)
    if arm == "relfeat":
        return CoordMLP(64, 32, in_dim=10)
    return CoordMLP(64, 32, in_dim=8)


def featurize(arm, x, stats):
    x = np.asarray(x, np.float32).reshape(-1, 4, 2)
    if arm == "setmlp":
        f = g8_features(x)
        return (f - stats["mu"]) / stats["sd"]
    if arm in ("g8set", "g8wide", "s4set", "raw", "permaug"):
        flat = x.reshape(-1, 2)
        z = (flat - stats["mu"]) / stats["sd"]
        return z.reshape(len(x), 8)
    if arm == "relfeat":
        f = rel_features(x)
        return (f - stats["mu"]) / stats["sd"]
    flat = x.reshape(-1, 2)
    z = (flat - stats["mu"]) / stats["sd"]
    return z.reshape(len(x), 8)


def fit_stats(arm, x):
    x = np.asarray(x, np.float32).reshape(-1, 4, 2)
    if arm == "setmlp":
        f = g8_features(x)
        return {"mu": f.mean(0), "sd": f.std(0) + 1e-8, "kind": "g8_sorted"}
    if arm in ("g8set", "g8wide", "s4set", "raw", "permaug"):
        mu = x.reshape(-1, 2).mean(0)
        sd = x.reshape(-1, 2).std(0) + 1e-8
        return {"mu": mu, "sd": sd, "kind": "shared_xy"}
    if arm == "relfeat":
        f = rel_features(x)
        return {"mu": f.mean(0), "sd": f.std(0) + 1e-8, "kind": "rel10_indexed"}
    mu = x.reshape(-1, 2).mean(0)
    sd = x.reshape(-1, 2).std(0) + 1e-8
    return {"mu": mu, "sd": sd, "kind": "shared_xy"}


def oracle_label(x):
    o = segment_relation(np.asarray(x, float).reshape(4, 2))
    if o.get("ambiguous"):
        return None
    return int(o["label"])


def mine_flips(x, y, seed, cap=8):
    rng = np.random.default_rng(seed)
    edits, parents, labels = [], [], []
    for i in range(len(x)):
        n = 0
        for e, _ in candidates_for_scene(x[i], rng):
            try:
                o = segment_relation(x[i] + e)
            except ValueError:
                continue
            if o.get("ambiguous") or o["margin"] < 0.02:
                continue
            if int(o["label"]) != int(y[i]):
                edits.append(e.astype(np.float32))
                parents.append(i)
                labels.append(int(o["label"]))
                n += 1
                if n >= cap:
                    break
    return (np.stack(edits), np.array(parents), np.array(labels)) if edits else (
        np.zeros((0, 4, 2), np.float32), np.zeros(0, int), np.zeros(0, int))


def mine_eval(x, y, seed, cap_pairs=40):
    """Model-blind E quartets: two keeps and their sum flips."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(x)):
        keeps = []
        for e, _ in candidates_for_scene(x[i], rng):
            try:
                o = segment_relation(x[i] + e)
            except ValueError:
                continue
            if o.get("ambiguous") or o["margin"] < 0.02:
                continue
            if int(o["label"]) == int(y[i]):
                keeps.append(e.astype(np.float32))
                if len(keeps) >= 8:
                    break
        n = 0
        for a in range(len(keeps)):
            for b in range(a + 1, len(keeps)):
                try:
                    o = segment_relation(x[i] + keeps[a] + keeps[b])
                except ValueError:
                    continue
                if o.get("ambiguous"):
                    continue
                if int(o["label"]) != int(y[i]):
                    rows.append((i, keeps[a], keeps[b], int(y[i]), int(o["label"])))
                    n += 1
                    if n >= cap_pairs:
                        break
            if n >= cap_pairs:
                break
    return rows


@torch.no_grad()
def predict(model, arm, x, stats):
    f = torch.tensor(featurize(arm, x, stats), dtype=torch.float32)
    out = []
    for i in range(0, len(f), 2048):
        out.append(model(f[i:i + 2048]).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def j_of(model, arm, stats, parents, base, ea, eb, y0, yab, perm):
    if len(parents) == 0:
        return None
    xb = apply_perm(base, perm)
    eap = apply_perm(ea, perm)
    ebp = apply_perm(eb, perm)
    s0 = predict(model, arm, xb, stats) > 0
    sa = predict(model, arm, xb + eap, stats) > 0
    sb = predict(model, arm, xb + ebp, stats) > 0
    sab = predict(model, arm, xb + eap + ebp, stats) > 0
    ok = (s0 == y0) & (sa == y0) & (sb == y0) & (sab == yab)
    return ok


def train_one(arm, seed, x, y, flips, stats, out_dir):
    dest = out_dir / f"{arm}_s{seed}"
    if (dest / "model.pt").exists():
        ck = torch.load(dest / "model.pt", map_location="cpu", weights_only=False)
        model = build(arm)
        model.load_state_dict(ck["state"])
        return model, ck
    dest.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = build(arm)
    init = hashlib.sha256(b"".join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    xc = torch.tensor(featurize(arm, x, stats), dtype=torch.float32)
    yc = torch.tensor(y, dtype=torch.float32)
    edits, pi, fy = flips
    # Same two-term objective as raw: mean clean BCE + mean flip BCE.
    # Permutation augmentation averages those two terms over G8, so extra
    # labelings do not reweight flips against clean states.
    xc_g = xf_g = None
    if arm == "permaug":
        xc_g = torch.tensor(np.stack([featurize(arm, apply_perm(x, p), stats) for p in G8]), dtype=torch.float32)
        yc_g = yc.repeat(len(G8))
        if len(edits):
            xf_g = torch.tensor(np.stack([featurize(arm, apply_perm(x[pi] + edits, p), stats) for p in G8]), dtype=torch.float32)
            yf_g = torch.tensor(fy, dtype=torch.float32).repeat(len(G8))
    elif len(edits):
        xf = torch.tensor(featurize(arm, x[pi] + edits, stats), dtype=torch.float32)
        yf = torch.tensor(fy, dtype=torch.float32)
    else:
        xf = yf = None
    curve = []
    t0 = time.time()
    for _ in range(EPOCHS):
        opt.zero_grad()
        if arm == "permaug":
            loss = bce(model(xc_g.reshape(-1, xc_g.shape[-1])), yc_g)
            if xf_g is not None:
                loss = loss + bce(model(xf_g.reshape(-1, xf_g.shape[-1])), yf_g)
        else:
            loss = bce(model(xc), yc)
            if xf is not None:
                loss = loss + bce(model(xf), yf)
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
    ck = {
        "state": model.state_dict(), "arm": arm, "seed": seed, "epochs": EPOCHS, "lr": LR,
        "threads": THREADS, "init_sha256": init, "final_loss": curve[-1],
        "seconds": time.time() - t0, "stats_kind": stats["kind"],
    }
    torch.save(ck, dest / "model.pt")
    (dest / "curve.json").write_text(json.dumps({"loss": curve[-1], "seconds": ck["seconds"]}))
    return model, ck


def parent_ci(values, parents, seed=924, n_boot=2000):
    rng = np.random.default_rng(seed)
    parents = np.asarray(parents)
    uniq = np.unique(parents)
    groups = [values[parents == p] for p in uniq]
    means = np.array([g.mean() for g in groups])
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(groups), len(groups))
        boots.append(np.concatenate([groups[i] for i in ix]).mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"estimate": float(values.mean()), "parent_mean": float(means.mean()),
            "ci95": [float(lo), float(hi)], "n": int(len(values)), "n_parents": int(len(uniq))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train = np.load(ROOT / "artifacts/f095_campaign/D01/scenes_N/scenes.npz")
    xtr = train["positions"].astype(np.float32)
    ytr = train["labels"].astype(np.int64)
    # fresh eval parents, split before editing
    xev, yev, mev = make_relational_scenes(256, seed=241015, min_margin=0.02)
    xev, yev = xev.astype(np.float32), yev.astype(np.int64)
    # chunked L-infinity distance; scipy is not importable on this host
    tr = xtr.reshape(-1, 8)
    ev = xev.reshape(-1, 8)
    dist = np.empty(len(ev), np.float32)
    for i in range(len(ev)):
        dist[i] = np.abs(tr - ev[i]).max(axis=1).min()
    assert float(dist.min()) > 1e-4, float(dist.min())
    flips = mine_flips(xtr, ytr, seed=5, cap=8)
    rows = mine_eval(xev, yev, seed=241016, cap_pairs=20)
    parents = np.array([r[0] for r in rows])
    base = xev[parents]
    ea = np.stack([r[1] for r in rows])
    eb = np.stack([r[2] for r in rows])
    y0 = np.array([r[3] for r in rows])
    yab = np.array([r[4] for r in rows])
    np.savez_compressed(a.out / "eval_quartets.npz", parents=parents, base=base, ea=ea, eb=eb, y0=y0, yab=yab,
                        nearest_train_linf=dist)
    meta = {
        "train": "artifacts/f095_campaign/D01/scenes_N/scenes.npz",
        "train_sha256": hashlib.sha256((ROOT / "artifacts/f095_campaign/D01/scenes_N/scenes.npz").read_bytes()).hexdigest(),
        "n_train": int(len(xtr)), "n_eval_parents": int(len(xev)),
        "n_quartets": int(len(rows)), "n_quartet_parents": int(len(np.unique(parents))),
        "min_train_linf": float(dist.min()), "threads": THREADS, "epochs": EPOCHS, "lr": LR,
        "seeds": list(a.seeds), "group": [list(p) for p in G8],
        "sealed_pools": "not read",
    }
    (a.out / "PROTOCOL.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("n_train", "n_eval_parents", "n_quartets", "n_quartet_parents", "min_train_linf")}), flush=True)

    summary = []
    for arm in ARMS:
        stats = fit_stats(arm, xtr)
        for seed in a.seeds:
            model, ck = train_one(arm, seed, xtr, ytr, flips, stats, a.out)
            model.eval()
            ident = G8[0]
            ok = {tuple(p): j_of(model, arm, stats, parents, base, ea, eb, y0, yab, p) for p in G8}
            j_id = float(ok[ident].mean())
            js = [float(v.mean()) for v in ok.values()]
            # quartet prediction-pattern change under relabeling
            base_pat = ok[ident]
            flip_frac = float(np.mean([v != base_pat for p, v in ok.items() if p != ident]))
            # exact invariance of the setmlp: max |logit| difference should be ~0; we use J identity
            row = {
                "arm": arm, "seed": int(seed), "J": j_id,
                "J_min": float(min(js)), "J_max": float(max(js)),
                "J_swing": float(max(js) - min(js)),
                "pred_flip_frac": flip_frac,
                "final_loss": float(ck["final_loss"]), "seconds": float(ck["seconds"]),
                "n": int(len(rows)), "n_parents": int(len(np.unique(parents))),
            }
            # setmlp must be labeling-invariant up to float; record the max J gap
            summary.append(row)
            print(json.dumps(row), flush=True)
            (a.out / f"{arm}_s{seed}" / "eval.json").write_text(json.dumps(row, indent=2))
    (a.out / "SUMMARY.json").write_text(json.dumps({"meta": meta, "rows": summary}, indent=2))

    # seed-level contrasts vs raw, parent-pooled later in the report script
    print("done", len(summary), "arms", flush=True)


if __name__ == "__main__":
    main()
