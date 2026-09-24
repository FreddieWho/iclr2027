#!/usr/bin/env python3
"""Evaluate new f095 arms (typed/sixdist/concat/width cells) on a bank.

Same quartet order and correctness contract as u1_factorial_eval; adds
U03 certificate (S/J*/J) per arm/seed. Model class is dispatched by
checkpoint featurize flag; arch/width/seed carried in the output schema.
No training, no sealed-pool reads.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "docs" / "f095dfa_review_pack" / "checks"))
from coord_mlp import CoordMLP  # noqa: E402
from common import rel_features  # noqa: E402
from u01_models import TypedPairMLP, ConcatMLP  # noqa: E402
from threshold_certificate import threshold_certificate  # noqa: E402

torch.set_num_threads(4)
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"


def load_arm(model_dir: Path):
    ck = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    fe = ck.get("featurize")
    if fe == "scalar8":
        n = ck["state"]
        if any(k.startswith("phi.") for k in n):
            m = TypedPairMLP(ck["hidden"])
        else:
            m = ConcatMLP(ck["hidden"], ck["feat"])
    elif ck.get("in_dim") == 10 or fe == "rel12":
        m = CoordMLP(ck["hidden"], ck["feat"], in_dim=10)
    elif ck.get("in_dim") == 6:
        m = CoordMLP(ck["hidden"], ck["feat"], in_dim=6)
    else:
        m = CoordMLP(ck["hidden"], ck["feat"], in_dim=ck.get("in_dim", 8))
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def featurize_raw(X, ck):
    mu = np.asarray(ck["mu"], dtype=np.float32)
    sd = np.asarray(ck["sd"], dtype=np.float32)
    F = np.asarray(X, dtype=np.float32).reshape(-1, len(mu))
    return torch.from_numpy(((F - mu) / sd).astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", required=True,
                   help="name=path pairs, e.g. typed_clean=artifacts/f095_campaign/U01/typed_w64_s11/clean")
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, required=True, help="new summary json path")
    a = p.parse_args()
    if a.out.exists():
        p.error("--out must be new")
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    Q = sorted(qid for qid, _, _ in quads)
    res = {"bank": a.bank,
           "bank_sha256": hashlib.sha256(
               (BANK / ("bank_" + a.bank + ".npz")).read_bytes()).hexdigest(),
           "arms": {}}
    for spec in a.arms:
        name, mp = spec.split("=", 1)
        model, ck = load_arm(Path(mp))
        msha = hashlib.sha256((Path(mp) / "model.pt").read_bytes()).hexdigest()
        fe = ck.get("featurize")
        xs, keys = [], []
        for qid, (i1, m1), (i2, m2) in quads:
            xa = np.asarray(Qx[i1], float)
            xb = np.asarray(Qx[i2], float)
            xab = xa + np.asarray(Qe[i1], float)
            if fe in ("rel12",):
                xa, xb, xab = rel_features(xa[None])[0], rel_features(xb[None])[0], \
                    rel_features(xab[None])[0]
            elif fe == "sixdist":
                xa, xb, xab = rel_features(xa[None])[0][:6], rel_features(xb[None])[0][:6], \
                    rel_features(xab[None])[0][:6]
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
                    xx = ((xx.reshape(-1, len(mu)) - mu) / sd).astype(np.float32)
                    lgs.append(model(torch.from_numpy(xx)).numpy().reshape(-1))
                else:
                    lgs.append(model(featurize_raw(xx, ck)).numpy().reshape(-1))
        lg = np.concatenate(lgs).reshape(-1, 3)
        pred = (lg > 0).astype(int)
        lab = np.array(keys)
        ok = (pred == lab)
        sc = lg
        cert = threshold_certificate(sc, lab, 0.0)
        res["arms"][name] = {
            "model_sha256": msha, "n": len(Q),
            "J": round(float(ok.all(axis=1).mean()), 4),
            "atomic_pass": round(float(ok[:, :2].all(axis=1).mean()), 4),
            "AB_correct": round(float(ok[:, 2].mean()), 4),
            "S": round(cert.S_local_separable, 4),
            "J_star": round(cert.J_star_global_oracle, 4),
        }
        print(name, res["arms"][name], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
