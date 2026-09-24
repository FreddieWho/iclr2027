#!/usr/bin/env python3
"""D08 fair fight: nearest-neighbor repair baselines under one contract.

Train (exact r04b/relflip recipe: train_101, mine-seed 5, full-batch 300ep,
Adam 1e-2, BCE, lam=1.0): raw fliprand s23/s47; rel fliprand x3;
balanced(raw/rel x3, 337pos+337neg seed-fixed); pct-preserve(raw/rel x3,
continuation from same-seed clean ckpt, preserve=train-correct soft targets,
lam_pres=1.0 fixed zero-search).
Eval (dev512): J/S/J*/atomic/AB + R_full/M vs raw_clean full-net B110
(from U1_PERQUARTET, read-only) + train-old-correct regression.
Temp-calib: T fit on train static scenes of flipmine models (output-only).
No sealed-pool reads.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "docs" / "f095dfa_review_pack" / "checks"))
from coord_mlp import CoordMLP  # noqa: E402
from common import load_model, rel_features, preprocess  # noqa: E402
from threshold_certificate import threshold_certificate  # noqa: E402
from u01_eval import load_arm  # noqa: E402

torch.set_num_threads(4)
ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "f095_campaign" / "D08"
EPOCHS, LR, LAM, LAM_PRES = 300, 1e-2, 1.0, 1.0
SEEDS = (11, 23, 47)


def sha_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean_ckpt(enc: str, seed: int) -> Path:
    if enc == "raw":
        return ART / ("r04b_s%d" % seed) / "clean"
    if seed == 11:
        return ART / "r04b_s11" / "relfeat"
    return ART / ("r04b_s%d_relfeat" % seed) / "relfeat"


def train_tensors(enc: str):
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    mine = np.load(ART / "r04b_s11" / "mined.npz", allow_pickle=True)
    meta = mine["meta"]
    K = len(meta)
    ii = np.array([int(r[0]) for r in meta])
    ee = np.stack([mine[f"edit_{k}"] for k in range(K)]).astype(np.float32)
    yf = torch.from_numpy(np.array([int(r[1]) for r in meta], dtype=np.float32))
    if enc == "raw":
        mu, sd = X.reshape(-1, 8).mean(0), X.reshape(-1, 8).std(0) + 1e-8
        fe = lambda A: torch.from_numpy(((A.reshape(-1, 8) - mu) / sd).astype(np.float32))  # noqa: E731
    else:
        Fr = rel_features(X)
        mu, sd = Fr.mean(0), Fr.std(0) + 1e-8
        fe = lambda A: torch.from_numpy(((rel_features(A) - mu) / sd).astype(np.float32))  # noqa: E731
    Xc, yc = fe(X), torch.from_numpy(y)
    Xf = fe((X[ii] + ee))
    return Xc, yc, Xf, yf, mu, sd, K, ii


def train_one(enc, method, seed, epochs):
    torch.manual_seed(seed)
    Xc, yc, Xf, yf, mu, sd, K, _ = train_tensors(enc)
    in_dim = 8 if enc == "raw" else 10
    model = CoordMLP(64, 32, in_dim=in_dim)
    if method == "pct":
        ck, _ = load_model(clean_ckpt(enc, seed))
        model.load_state_dict(ck.state_dict() if hasattr(ck, "state_dict") else ck)
        # teacher: clean baseline train predictions (soft) + correct set
        model.eval()
        with torch.no_grad():
            base_logit = model(Xc).numpy()
        base_pred = (base_logit > 0).astype(int)
        keep = base_pred == yc.numpy().astype(int)
        Tp = torch.from_numpy(1 / (1 + np.exp(-base_logit))).float()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    hist = []
    if method == "fliprand":
        rrand = np.random.default_rng(seed + 777)
        idx = sorted(rrand.choice(K, K // 2, replace=False).tolist())
    elif method == "balanced":
        rbal = np.random.default_rng(seed + 778)
        yfn = yf.numpy().astype(int)
        pos = np.nonzero(yfn == 1)[0]
        neg = np.nonzero(yfn == 0)[0]
        idx = sorted(pos.tolist() + rbal.choice(neg, len(pos), replace=False).tolist())
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        if method == "pct":
            model.train()
            lb = bce(model(Xc), yc)
            lf = bce(model(Xf), yf)
            lp = bce(model(Xc[keep]), Tp[keep])
            loss = lb + LAM * lf + LAM_PRES * lp
        elif method in ("fliprand", "balanced"):
            loss = bce(model(Xc), yc) + LAM * bce(model(Xf[idx]), yf[idx])
        else:
            raise ValueError(method)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    od = OUT / f"{enc}_{method}_s{seed}"
    od.mkdir(parents=True, exist_ok=True)
    info = {"state": model.state_dict(), "hidden": 64, "feat": 32,
            "in_dim": in_dim, "seed": seed, "method": method,
            "epochs": epochs, "lr": LR, "lam": LAM}
    if enc == "rel":
        info.update({"featurize": "rel12", "mu": mu, "sd": sd})
    else:
        info.update({"mu": mu, "sd": sd})
    torch.save(info, od / "model.pt")
    extra = {"n_flips_used": len(idx) if method in ("fliprand", "balanced") else K,
             "lam_pres": LAM_PRES if method == "pct" else None,
             "continued_from": str(clean_ckpt(enc, seed)) if method == "pct" else None,
             "final_loss": hist[-1]}
    json.dump(extra, open(od / "manifest.json", "w"), indent=1)
    print(f"SAW {enc} {method} s{seed} loss={hist[-1]:.4f}", flush=True)
    return od


def eval_bank(model, ck, quads, Qx, Qe):
    fe = ck.get("featurize")
    xs, keys = [], []
    for qid, (i1, m1), (i2, m2) in quads:
        xa = np.asarray(Qx[i1], float)
        xb = np.asarray(Qx[i2], float)
        xab = xa + np.asarray(Qe[i1], float)
        xs += [xa, xb, xab]
        la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
        lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
        keys.append((la, lb, m1["yAB"]))
    with torch.no_grad():
        lgs = []
        for s in range(0, len(xs), 512):
            xx = np.stack(xs[s:s + 512])
            if fe in ("rel12", "sixdist"):
                mu = np.asarray(ck["mu"], dtype=np.float32)
                sd = np.asarray(ck["sd"], dtype=np.float32)
                xx = ((rel_features(xx)[:, :len(mu)] if fe == "sixdist"
                       else rel_features(xx)) - mu) / sd
                xx = xx.astype(np.float32)
                lgs.append(model(torch.from_numpy(xx)).numpy().reshape(-1))
            else:
                mu = np.asarray(ck["mu"], dtype=np.float32)
                sd = np.asarray(ck["sd"], dtype=np.float32)
                xx = ((xx.reshape(-1, len(mu)) - mu) / sd).astype(np.float32)
                lgs.append(model(torch.from_numpy(xx)).numpy().reshape(-1))
    lg = np.concatenate(lgs).reshape(-1, 3)
    return lg, np.array(keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["train", "eval"], required=True)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    a = p.parse_args()
    if a.stage == "train":
        plan = ([("raw", "fliprand", s) for s in (23, 47)] +
                [("rel", "fliprand", s) for s in SEEDS] +
                [(e, m, s) for e in ("raw", "rel") for m in ("balanced", "pct")
                 for s in SEEDS])
        for enc, method, seed in plan:
            od = OUT / f"{enc}_{method}_s{seed}"
            if (od / "model.pt").exists():
                print(f"SKIP {enc} {method} s{seed} (exists)", flush=True)
                continue
            train_one(enc, method, seed, a.epochs)
        return
    # ---- eval ----
    b = np.load(BANK / "bank_dev512.npz", allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    Q = sorted(qid for qid, _, _ in quads)
    # B110 from raw_clean full net (U1 per-quartet, read-only)
    b110 = {}
    with open(ROOT / "artifacts" / "next_novelty" / "u1_factorial" / "U1_PERQUARTET.csv") as f:
        rd = csv.DictReader(f)
        rows = [r for r in rd if r["arm"] == "raw_clean"]
    qid_order = sorted({int(r["qid"]) for r in rows})
    for s in ("11", "23", "47"):
        r = {int(x["qid"]): x for x in rows if x["seed"] == s}
        b110[s] = np.array([(r[q]["correct_A"] == "1" and r[q]["correct_B"] == "1" and
                             r[q]["correct_AB"] == "0") for q in Q])
    # model dirs: reuse + new
    def reuse_raw(method, seed):
        return ART / ("r04b_s%d" % seed) / method
    def reuse_rel(method, seed):
        if method == "relfeat":
            return ART / "r04b_s11" / "relfeat" if seed == 11 else \
                ART / ("r04b_s%d_relfeat" % seed) / "relfeat"
        return ROOT / "artifacts" / "next_novelty" / "relflip" / ("s%d" % seed)
    spec = {}
    for seed in SEEDS:
        spec[f"raw_flipmine_s{seed}"] = reuse_raw("flipmine", seed)
        spec[f"raw_fliprand_s{seed}"] = reuse_raw("fliprand", seed) if seed == 11 else \
            OUT / f"raw_fliprand_s{seed}"
        spec[f"rel_flipmine_s{seed}"] = reuse_rel("relflip", seed)
        if seed == 11:
            spec[f"rel_fliprand_s{seed}"] = None  # no rel fliprand archive; trained new
        for m in ("balanced", "pct"):
            spec[f"raw_{m}_s{seed}"] = OUT / f"raw_{m}_s{seed}"
            spec[f"rel_{m}_s{seed}"] = OUT / f"rel_{m}_s{seed}"
        spec[f"rel_fliprand_s{seed}"] = OUT / f"rel_fliprand_s{seed}"
    # train tensors for temp-fit + old-correct regression
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    Xtr, ytr = d["positions"].astype(np.float32), d["labels"].astype(int)
    res = {"bank": "dev512", "bank_sha256": sha_of(BANK / "bank_dev512.npz"),
           "B110_source": "U1_PERQUARTET raw_clean (read-only)", "arms": {}}
    for name, mp in spec.items():
        if mp is None:
            continue
        model, ck = load_arm(Path(mp))
        model.eval()
        lg, lab = eval_bank(model, ck, quads, Qx, Qe)
        pred = (lg > 0).astype(int)
        ok = (pred == lab)
        cert = threshold_certificate(lg, lab, 0.0)
        H = b110[name.rsplit("_s", 1)[1]]
        eAB, eAEB = ok[:, 2], ok.all(axis=1)
        eAE = ok[:, :2].all(axis=1)
        # temp fit on train static
        fe = ck.get("featurize")
        with torch.no_grad():
            if fe == "rel12":
                Xt = torch.from_numpy(((rel_features(Xtr) - ck["mu"]) / ck["sd"]).astype(np.float32))
            else:
                mu = np.asarray(ck["mu"], dtype=np.float32)
                sd = np.asarray(ck["sd"], dtype=np.float32)
                Xt = torch.from_numpy(((Xtr.reshape(-1, 8) - mu) / sd).astype(np.float32))
            tr_logit = model(Xt).numpy()
        base = load_arm(clean_ckpt("raw" if name.startswith("raw_") else "rel",
                                   int(name.rsplit("_s", 1)[1])))[0]
        base.eval()
        with torch.no_grad():
            old_ok = ((base(Xt).numpy() > 0).astype(int) == ytr)
            new_ok = ((tr_logit > 0).astype(int) == ytr)
        reg = float((old_ok & (~new_ok)).sum() / max(old_ok.sum(), 1))
        # temperature (scalar, train static NLL)
        T = torch.nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.LBFGS([T], max_iter=50)
        yt = torch.from_numpy(ytr.astype(np.float32))
        tl = torch.from_numpy(tr_logit)
        def closure():
            opt.zero_grad()
            loss = nn.BCEWithLogitsLoss()(tl / torch.clamp(T, 0.05, 20), yt)
            loss.backward()
            return loss
        opt.step(closure)
        Tv = float(torch.clamp(T, 0.05, 20).detach())
        tJ = float((((lg / Tv) > 0).astype(int) == lab).all(axis=1).mean())
        res["arms"][name] = {
            "model_sha256": sha_of(Path(mp) / "model.pt"),
            "J": round(float(ok.all(axis=1).mean()), 4),
            "atomic_pass": round(float(ok[:, :2].all(axis=1).mean()), 4),
            "AB_correct": round(float(ok[:, 2].mean()), 4),
            "S": round(cert.S_local_separable, 4),
            "J_star": round(cert.J_star_global_oracle, 4),
            "R_endpoint": round(float(eAB[H].mean()), 4) if H.sum() else None,
            "R_full": round(float(eAEB[H].mean()), 4) if H.sum() else None,
            "M": round(float(((eAB == 1) & (eAE == 0))[H].mean()), 4) if H.sum() else None,
            "n_H": int(H.sum()),
            "train_oldcorrect_regression": round(reg, 4),
            "temp_T": round(Tv, 4),
            "temp_J": round(tJ, 4),
        }
        print(name, {k: v for k, v in res["arms"][name].items()
                     if k in ("J", "R_full", "M", "train_oldcorrect_regression", "temp_T", "temp_J")},
              flush=True)
    with open(OUT / "D08_SUMMARY.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT / "D08_SUMMARY.json")


if __name__ == "__main__":
    main()
