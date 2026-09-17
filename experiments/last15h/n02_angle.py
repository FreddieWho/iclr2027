#!/usr/bin/env python3
"""N02 round1: angle between semantic normal a and model normal b vs miss."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from common import load_model, preprocess

ART = ROOT / "artifacts" / "discovery_campaign"


def orients_t(x):
    a, b, c, d_ = x[0], x[1], x[2], x[3]
    def o(p, q, r):
        u, v = q - p, r - p
        return u[0] * v[1] - u[1] * v[0]
    return torch.stack([o(a, b, c), o(a, b, d_), o(c, d_, a), o(c, d_, b)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ART / "r04b_s11" / "clean")
    a = p.parse_args()
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    mine = np.load(ART / "r04c_budget" / "mined.npz")
    meta = mine["meta"]
    model, stats = load_model(a.model)
    model.eval()
    rows = []
    n = min(256, len(meta))
    for i in range(n):
        pi = int(meta[i, 0])
        x0 = X[pi].astype(np.float32)
        e = np.stack([mine[f"edit_{i}"]]).astype(np.float32)[0] \
            if f"edit_{i}" in mine else None
        if e is None:
            continue
        x1 = (x0 + e).astype(np.float32)
        xt0 = torch.tensor(x0.reshape(1, -1), requires_grad=True)
        with torch.no_grad():
            l0 = model(preprocess(x0[None], stats)).numpy()[0]
            l1 = model(preprocess(x1[None], stats)).numpy()[0]
        # semantic normal: orient value whose sign flips (must be exactly one)
        with torch.enable_grad():
            o0 = orients_t(torch.tensor(x0))
            o1 = orients_t(torch.tensor(x1))
        s0 = np.sign(o0.detach().numpy())
        s1 = np.sign(o1.detach().numpy())
        changed = np.nonzero(s0 != s1)[0]
        if len(changed) != 1:
            rows.append({"kind": "multi_or_none", "n_changed": len(changed)})
            continue
        k = int(changed[0])
        xx = torch.tensor(x0, requires_grad=True)
        rk = orients_t(xx)[k]
        a_vec = torch.autograd.grad(rk, xx, retain_graph=False)[0].reshape(-1).detach()
        # model normal b = grad(f_b - f_a); new class b_ = sign of oracle new label
        y_new = 1  # flips go 0->1 or 1->0; use direction toward new oracle label below
        from paths import oracle_at
        y_new = oracle_at(x1)[0]
        sgn = 1.0 if y_new == 1 else -1.0
        xx2 = torch.tensor(x0, requires_grad=True)
        # preprocess is linear (standardize) -> chain rule manually
        mu = np.asarray(stats["mu"], dtype=np.float32).reshape(-1)
        sd = np.asarray(stats["sd"], dtype=np.float32).reshape(-1)
        z = (xx2.reshape(-1) - torch.tensor(mu)) / torch.tensor(sd)
        # full forward on standardized input with grad
        inp = ((xx2.reshape(-1) - torch.tensor(mu)) / torch.tensor(sd)).unsqueeze(0)
        h = model.net(inp)
        zz = model.feat_head(h)
        logit = model.cls(zz).squeeze(-1)
        b_vec = torch.autograd.grad(sgn * logit, xx2)[0].reshape(-1).detach().numpy()
        a_np = a_vec.numpy()
        na, nb = np.linalg.norm(a_np), np.linalg.norm(b_vec)
        if na < 1e-12 or nb < 1e-12:
            rows.append({"kind": "zero_grad", "na": float(na), "nb": float(nb)})
            continue
        cos = float(a_np @ b_vec / (na * nb))
        miss = bool((l1 > 0) != bool(y_new))
        rows.append({"kind": "ok", "cos": cos, "miss": miss,
                     "parent_id": pi, "k": k})
    oks = [r for r in rows if r["kind"] == "ok"]
    import collections
    kinds = dict(collections.Counter(r["kind"] for r in rows))
    res = {"n": len(rows), "kinds": kinds}
    if oks:
        cos_m = np.array([r["cos"] for r in oks if r["miss"]])
        cos_c = np.array([r["cos"] for r in oks if not r["miss"]])
        res.update({"n_ok": len(oks), "miss_rate": float(np.mean([r["miss"] for r in oks])),
                    "mean_cos_miss": float(cos_m.mean()) if len(cos_m) else None,
                    "mean_cos_correct": float(cos_c.mean()) if len(cos_c) else None,
                    "rows": oks})
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print(f"SAW: {res['n']} pairs kinds={kinds} " +
          (f"miss={res.get('miss_rate')} cos_miss={res.get('mean_cos_miss')} "
           f"cos_ok={res.get('mean_cos_correct')}" if oks else ""))
    print("NEXT: if low-cos predicts miss -> directional loss trial; else drop N02")
    print("CLAIM: tests direction-vs-magnitude hypothesis for boundary errors")


if __name__ == "__main__":
    main()
