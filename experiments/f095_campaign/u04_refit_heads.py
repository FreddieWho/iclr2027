#!/usr/bin/env python3
"""U04: frozen-encoder cross-refit. Freeze each four-arm encoder (32-dim z),
retrain same-structure Linear(32,1) readouts under static-only vs flip-covered
supervision (identical train budget to the original recipe), plus a small-MLP
readout (flip regime) for linear-reachability. Evaluate S/J*/J + R_full/M on
dev512. No sealed-pool reads; no backbone modification (sha verified).
"""
import argparse
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
sys.path.insert(0, str(ROOT / "experiments" / "ccm_audit"))
sys.path.insert(0, str(ROOT / "docs" / "f095dfa_review_pack" / "checks"))
from coord_mlp import CoordMLP  # noqa: E402
from common import load_model, preprocess  # noqa: E402
from threshold_certificate import threshold_certificate  # noqa: E402

torch.set_num_threads(4)
ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "f095_campaign" / "U04"
ARMS = ("raw_clean", "raw_flipmine", "relfeat", "relflip")
SEEDS = (11, 23, 47)
EPOCHS, LR, LAM = 300, 1e-2, 1.0
INIT_SEED_BASE = 1000


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backbone_sha(model) -> str:
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        if k.startswith("net.") or k.startswith("feat_head."):
            h.update(model.state_dict()[k].cpu().numpy().tobytes())
    return h.hexdigest()


def arm_checkpoint(arm: str, seed: int) -> Path:
    if arm == "relflip":
        return ROOT / "artifacts" / "next_novelty" / "relflip" / ("s%d" % seed)
    if arm == "raw_clean":
        return ART / ("r04b_s%d" % seed) / "clean"
    if arm == "raw_flipmine":
        return ART / ("r04b_s%d" % seed) / "flipmine"
    if arm == "relfeat":
        if seed == 11:
            return ART / "r04b_s11" / "relfeat"
        return ART / ("r04b_s%d_relfeat" % seed) / "relfeat"
    raise ValueError(arm)


@torch.no_grad()
def frozen_features(model, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    zs = []
    for s in range(0, len(X), 1024):
        _, z = model(X[s:s + 1024], return_feat=True)
        zs.append(z)
    return torch.cat(zs)


def train_head(Ztr, ytr, Zfl, yfl, regime, init_seed, epochs, mlp=False):
    torch.manual_seed(init_seed)
    if mlp:
        head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    else:
        head = nn.Linear(32, 1)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    hist = []
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        lb = bce(head(Ztr).reshape(-1), ytr)
        if regime == "flip":
            loss = lb + LAM * bce(head(Zfl).reshape(-1), yfl)
        else:
            loss = lb
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    return head, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--bank", default="dev512")
    a = p.parse_args()
    epochs = a.epochs
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X, y = d["positions"].astype(np.float32), d["labels"].astype(np.float32)
    mine = np.load(ART / "r04b_s11" / "mined.npz", allow_pickle=True)
    meta = mine["meta"]
    # flip tensors in RAW coords; per-encoder preprocessing applied later
    f_ii = np.array([int(r[0]) for r in meta])
    f_ee = np.stack([mine[f"edit_{k}"] for k in range(len(meta))]).astype(np.float32)
    f_y = torch.from_numpy(np.array([int(r[1]) for r in meta], dtype=np.float32))
    print(f"train clean {len(X)}, flips {len(meta)}", flush=True)
    b = np.load(BANK / f"bank_{a.bank}.npz", allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    Q = sorted(qid for qid, _, _ in quads)

    summary = {"bank": a.bank,
               "bank_sha256": sha_of(BANK / f"bank_{a.bank}.npz"),
               "recipe": {"epochs": epochs, "lr": LR, "lam": LAM, "wd": 0,
                          "init_seed_rule": "1000+seed_idx", "full_batch": True},
               "mergeable_linear_fact": "feat_head+cls have no nonlinearity between; "
               "a fresh Linear(32,1) on frozen z has exactly the original head capacity",
               "arms": {}}
    for seed in a.seeds:
        for arm in ARMS:
            mp = arm_checkpoint(arm, seed)
            model, stats = load_model(mp)
            for p_ in model.parameters():
                p_.requires_grad = False
            sha_before = backbone_sha(model)
            # train tensors under this encoder's own preprocessing
            Xc = preprocess(X.reshape(-1, 8), stats)
            yc = torch.from_numpy(y)
            Xf = preprocess((X[f_ii] + f_ee).reshape(-1, 8), stats)
            Zc = frozen_features(model, Xc).detach()
            Zf = frozen_features(model, Xf).detach()
            assert backbone_sha(model) == sha_before
            tag = f"{arm}_s{seed}"
            od = OUT / tag
            od.mkdir(parents=True, exist_ok=True)
            arms = {}
            init_seed = INIT_SEED_BASE + SEEDS.index(seed)
            # eval features (per state) once
            xs_all, keys = [], []
            for qid, (i1, m1), (i2, m2) in quads:
                xa = np.asarray(Qx[i1], float)
                xb = np.asarray(Qx[i2], float)
                xab = xa + np.asarray(Qe[i1], float)
                xs_all += [xa, xb, xab]
                la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
                lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
                keys.append((la, lb, m1["yAB"]))
            Xe = preprocess(np.stack(xs_all), stats)
            Ze = frozen_features(model, Xe).detach().reshape(len(Q), 3, 32)
            lab = np.array(keys)
            assert backbone_sha(model) == sha_before  # backbone untouched
            for regime in ("static", "flip"):
                head, hist = train_head(Zc, yc, Zf, f_y, regime, init_seed, epochs)
                torch.save({"head": head.state_dict(), "mlp": False,
                            "encoder": str(mp), "encoder_sha": sha_before,
                            "regime": regime, "init_seed": init_seed},
                           od / f"head_{regime}.pt")
                with torch.no_grad():
                    lg = head(Ze).numpy().reshape(len(Q), 3)
                pred = (lg > 0).astype(int)
                ok = (pred == lab)
                cert = threshold_certificate(lg, lab, 0.0)
                arms[regime] = {
                    "J": round(float(ok.all(axis=1).mean()), 4),
                    "atomic_pass": round(float(ok[:, :2].all(axis=1).mean()), 4),
                    "AB_correct": round(float(ok[:, 2].mean()), 4),
                    "S": round(cert.S_local_separable, 4),
                    "J_star": round(cert.J_star_global_oracle, 4),
                    "train_loss_final": round(hist[-1], 4),
                    "okAE_E": [int(v) for v in (ok[:, :2].all(axis=1))],
                    "okAB_E": [int(v) for v in ok[:, 2]],
                    "okAEB_E": [int(v) for v in ok.all(axis=1)],
                }
                print(tag, regime, {k: v for k, v in arms[regime].items()
                                    if not k.startswith("ok")}, flush=True)
            # small-MLP reachability, flip regime only
            mhead, mhist = train_head(Zc, yc, Zf, f_y, "flip", init_seed, epochs, mlp=True)
            torch.save({"head": mhead.state_dict(), "mlp": True,
                        "encoder": str(mp), "encoder_sha": sha_before,
                        "regime": "flip", "init_seed": init_seed},
                       od / "head_mlp_flip.pt")
            with torch.no_grad():
                mlg = mhead(Ze).numpy().reshape(len(Q), 3)
            mpred = (mlg > 0).astype(int)
            mok = (mpred == lab)
            mcert = threshold_certificate(mlg, lab, 0.0)
            mlp_flags = {"okAE_E": (mok[:, :2].all(axis=1)),
                         "okAB_E": mok[:, 2],
                         "okAEB_E": mok.all(axis=1)}
            arms["mlp_flip"] = {
                "J": round(float(mok.all(axis=1).mean()), 4),
                "atomic_pass": round(float(mok[:, :2].all(axis=1).mean()), 4),
                "AB_correct": round(float(mok[:, 2].mean()), 4),
                "S": round(mcert.S_local_separable, 4),
                "J_star": round(mcert.J_star_global_oracle, 4),
                "train_loss_final": round(mhist[-1], 4),
            }
            print(tag, "mlp_flip", arms["mlp_flip"], flush=True)
            # R_full/M of flip-regime refits vs same-encoder static refit B110
            sAE = np.array(arms["static"]["okAE_E"])
            sAB = np.array(arms["static"]["okAB_E"])
            H = (sAE == 1) & (sAB == 0)
            rep = {"n_H_staticrefit": int(H.sum())}
            _mlp = {"flip": (np.array(arms["flip"]["okAB_E"]),
                                np.array(arms["flip"]["okAE_E"]),
                                np.array(arms["flip"]["okAEB_E"])),
                     "mlp_flip": (mlp_flags["okAB_E"], mlp_flags["okAE_E"],
                                    mlp_flags["okAEB_E"])}
            for rm in ("flip", "mlp_flip"):
                eAB, eAE, eAEB = _mlp[rm]
                rep[rm] = {
                    "R_endpoint": round(float(eAB[H].mean()), 4) if H.sum() else None,
                    "R_full": round(float(eAEB[H].mean()), 4) if H.sum() else None,
                    "M": round(float(((eAB == 1) & (eAE == 0))[H].mean()), 4) if H.sum() else None,
                }
            arms["repair_vs_staticrefit"] = rep
            for k in ("static", "flip"):
                for fk in ("okAE_E", "okAB_E", "okAEB_E"):
                    del arms[k][fk]
            summary["arms"][tag] = dict(arms, encoder_sha=sha_before,
                                        encoder_model_sha=sha_of(mp / "model.pt"))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"U04_SUMMARY_{a.bank}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", OUT / f"U04_SUMMARY_{a.bank}.json")


if __name__ == "__main__":
    main()
