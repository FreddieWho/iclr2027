#!/usr/bin/env python3
"""X03 round1: counterfactual internal variable swap (interchange).

hybrid = donor.AB + base.CD (geometrically valid coord state).
Frozen model hidden z (feat, 32-d): z_patch = z_b + (z_d - z_b) @ B @ A.T,
a rank<=r low-rank linear patch map (B @ A.T has no orthogonal/symmetric/
idempotent constraint, so NOT a mathematical projection); frozen cls reads
out. Target = hybrid TRUE label.
Train pairs on train_101; test on FRESH eval_202 bindings, incl. anti-copy
subset (base/donor same label, hybrid different).
Controls: same-rank RANDOM fixed subspace; full swap (P=I);
ordinary readout MLP on [z_b, z_d] trained on same pairs (decodable vs
swappable). Models: frozen clean (main) + flipmine (reference).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import oracle_at
from common import load_model, preprocess

ART = ROOT / "artifacts" / "discovery_campaign"


def mine_pairs(X, rng, n_want):
    """(base, donor) pairs with valid hybrid; oversample anti-copy cases."""
    lab = []
    for x in X:
        try:
            lab.append(oracle_at(x)[0])
        except ValueError:
            lab.append(-1)
    lab = np.array(lab)
    ok0 = np.nonzero(lab == 0)[0]
    ok1 = np.nonzero(lab == 1)[0]
    pairs, anti = [], []
    for _ in range(n_want * 20):
        if len(pairs) >= n_want and len(anti) >= n_want // 2:
            break
        bi = ok0[int(rng.integers(len(ok0)))] if rng.random() < 0.5 else \
            ok1[int(rng.integers(len(ok1)))]
        di = ok0[int(rng.integers(len(ok0)))] if rng.random() < 0.5 else \
            ok1[int(rng.integers(len(ok1)))]
        if bi == di:
            continue
        b, d = X[bi].astype(float), X[di].astype(float)
        h = np.stack([d[0], d[1], b[2], b[3]])
        try:
            yh, mh, _ = oracle_at(h)
            yb, _, _ = oracle_at(b)
            yd, _, _ = oracle_at(d)
        except ValueError:
            continue
        if mh < 0.005:
            continue
        rec = (b, d, h, yb, yd, yh)
        if yb == yd and yh != yb:
            if len(anti) < n_want // 2:
                anti.append(rec)
        else:
            if len(pairs) < n_want:
                pairs.append(rec)
    return pairs + anti


def feats_of(model, stats, xs):
    with torch.no_grad():
        t = preprocess(xs.astype(np.float32), stats)
        h = model.net(t)
        z = model.feat_head(h)
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--n-train", type=int, default=300, dest="n_train")
    p.add_argument("--n-test", type=int, default=300, dest="n_test")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-filter", choices=["all", "anti"], default="all")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    dtr = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    dte = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    TR = mine_pairs(dtr["positions"].astype(float), rng, a.n_train)
    if a.train_filter == "anti":
        TR = [r for r in TR if r[3] == r[4] and r[5] != r[3]]
    TE = mine_pairs(dte["positions"].astype(float), rng, a.n_test)
    res = {"n_train_pairs": len(TR), "n_test_pairs": len(TE),
           "n_test_anti": sum(1 for r in TE if r[3] == r[4] and r[5] != r[3])}

    for k, mp in (("clean", ART / "r04b_s11" / "clean"),
                  ("flipmine", ART / "r04b_s11" / "flipmine")):
        model, stats = load_model(mp)
        model.eval()
        Zb_tr = feats_of(model, stats, np.stack([r[0] for r in TR]))
        Zd_tr = feats_of(model, stats, np.stack([r[1] for r in TR]))
        yh_tr = torch.tensor([r[5] for r in TR], dtype=torch.float32)
        Zb_te = feats_of(model, stats, np.stack([r[0] for r in TE]))
        Zd_te = feats_of(model, stats, np.stack([r[1] for r in TE]))
        yh_te = np.array([r[5] for r in TE])
        anti = np.array([r[3] == r[4] and r[5] != r[3] for r in TE])

        torch.manual_seed(a.seed)
        A = nn.Parameter(torch.randn(32, a.rank) * 0.1)
        B = nn.Parameter(torch.randn(32, a.rank) * 0.1)
        opt = torch.optim.Adam([A, B], lr=0.02)
        bce = nn.BCEWithLogitsLoss()
        for _ in range(500):
            opt.zero_grad()
            Zp = Zb_tr + (Zd_tr - Zb_tr) @ B @ A.T
            loss = bce(model.cls(Zp).squeeze(-1), yh_tr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            Zp_te = Zb_te + (Zd_te - Zb_te) @ B @ A.T
            pred = (model.cls(Zp_te).squeeze(-1) > 0).numpy().astype(int)
            acc = float((pred == yh_te).mean())
            acc_anti = float((pred[anti] == yh_te[anti]).mean()) if anti.sum() else None
            # controls
            pred_full = (model.cls(Zd_te).squeeze(-1) > 0).numpy().astype(int)
            acc_full = float((pred_full == yh_te).mean())
            Q, _ = torch.linalg.qr(torch.randn(32, 32))
            Pr = Q[:, :a.rank] @ Q[:, :a.rank].T
            Zr_te = Zb_te + (Zd_te - Zb_te) @ Pr
            pred_r = (model.cls(Zr_te).squeeze(-1) > 0).numpy().astype(int)
            acc_rand = float((pred_r == yh_te).mean())
            acc_rand_anti = float((pred_r[anti] == yh_te[anti]).mean()) if anti.sum() else None
        # ordinary readout control: MLP on [z_b, z_d], same pairs
        torch.manual_seed(a.seed)
        ro = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        opt2 = torch.optim.Adam(ro.parameters(), lr=0.01)
        Zin = torch.cat([Zb_tr, Zd_tr], -1)
        for _ in range(500):
            opt2.zero_grad()
            loss = bce(ro(Zin).squeeze(-1), yh_tr)
            loss.backward()
            opt2.step()
        with torch.no_grad():
            pred_o = (ro(torch.cat([Zb_te, Zd_te], -1)).squeeze(-1) > 0).numpy().astype(int)
            acc_read = float((pred_o == yh_te).mean())
            acc_read_anti = float((pred_o[anti] == yh_te[anti]).mean()) if anti.sum() else None
        with torch.no_grad():
            base_pred = (model.cls(Zb_te).squeeze(-1) > 0).numpy().astype(int)
        res[k] = {"interchange_acc": acc, "interchange_anti": acc_anti,
                  "fullswap_acc": acc_full, "randsub_acc": acc_rand,
                  "randsub_anti": acc_rand_anti, "readout_acc": acc_read,
                  "readout_anti": acc_read_anti,
                  "base_acc": float((base_pred
                                     == np.array([r[3] for r in TE])).mean())}
        print(f"SAW {k}: interchange={acc:.3f} (anti={acc_anti}) "
              f"full={acc_full:.3f} randsub={acc_rand:.3f} readout={acc_read:.3f} "
              f"(read_anti={acc_read_anti}) n={len(TE)} anti_n={int(anti.sum())}",
              flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1,
              default=lambda o: (None if o is None else float(o)))
    print("DONE X03")


if __name__ == "__main__":
    main()
