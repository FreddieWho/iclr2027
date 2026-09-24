#!/usr/bin/env python3
"""U10 orbit averages + sixdist dev512 group average (item 7, eval-only).

T1 group = S3 vertex permutations (Q fixed, 6 elements).
T2 group = {id, A<->B} (center fixed, 2 elements).
Orbit-averaged logits -> J. Legacy typed checkpoints use indexwise moments
and are NOT strictly invariant. Corrected end-to-end assertions and retraining
are in experiments/e1a933_review/data_u10.py; do not infer invariance from J.
Also: U02 sixdist clean/flipmine on bank_dev512 with the G8 group (D02
protocol), completing the sixdist orbit story.
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "ccm_audit"))
sys.path.insert(0, str(ROOT / "docs" / "f095dfa_review_pack" / "checks"))
from common import rel_features  # noqa: E402
from symmetry import GROUP, apply  # noqa: E402
from threshold_certificate import threshold_certificate  # noqa: E402
from u01_eval import load_arm  # noqa: E402
from u10_targets import TASKS, build_arch, featurize  # noqa: E402

torch.set_num_threads(4)
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
U10 = ROOT / "artifacts" / "f095_campaign" / "U10"
SEEDS = (11, 23, 47)
REGIMES = ("clean", "flip", "f25")
T1_PERMS = list(itertools.permutations(range(3)))
T2_PERMS = [(0, 1, 2), (1, 0, 2)]


def orbit_j(task, arch, seed, regime, Qx, Qe, Qm, perms):
    mp = U10 / f"{task}_{arch}_s{seed}" / regime / "model.pt"
    ck = torch.load(mp, map_location="cpu", weights_only=False)
    model = build_arch(task, arch)
    model.load_state_dict(ck["state"])
    model.eval()
    stats = (ck["mu"], ck["sd"], ck["mode"])
    nq = len(Qm) // 2
    la = np.array([Qm[2 * q]["yA"] for q in range(nq)])
    lb = np.array([Qm[2 * q + 1]["yB"] for q in range(nq)])
    ly = np.array([Qm[2 * q]["yAB"] for q in range(nq)])
    lab = np.stack([la, lb, ly], axis=1)
    npts = TASKS[task]["npts"]
    acc = np.zeros((nq, 3))
    with torch.no_grad():
        for perm in perms:
            p = list(perm) + list(range(len(perm), npts))
            for col, arr in ((0, Qx[0::2]), (1, Qx[1::2]),
                             (2, Qx[0::2] + Qe[0::2])):
                Xp = np.asarray(arr, np.float32)[:, p, :]
                lg = []
                for s in range(0, nq, 512):
                    lg.append(model(featurize(task, arch, Xp[s:s + 512], stats))
                              .numpy().reshape(-1))
                acc[:, col] += np.concatenate(lg)
    acc /= len(perms)
    J = float(((acc > 0).astype(int) == lab).all(axis=1).mean())
    return round(J, 4), hashlib.sha256(mp.read_bytes()).hexdigest()


def main():
    res = {"orbit_rule": "average logits over task group, threshold 0",
           "T1_perms": len(T1_PERMS), "T2_perms": len(T2_PERMS), "tasks": {}}
    for task, perms in (("T1", T1_PERMS), ("T2", T2_PERMS)):
        b = np.load(BANK / f"bank_u10{task}.npz", allow_pickle=True)
        Qx, Qe = np.asarray(b["Qx"], float), np.asarray(b["Qe"], float)
        Qm = json.loads(str(b["Qmeta"]))
        tr = {}
        for arch in ("raw", "six", "typed"):
            for seed in SEEDS:
                for regime in REGIMES:
                    J, sha = orbit_j(task, arch, seed, regime, Qx, Qe, Qm, perms)
                    tr[f"{arch}_{regime}_s{seed}"] = {"J_orbit": J, "model_sha": sha}
                    print(f"{task} {arch}/{regime}/s{seed} J_orbit={J}", flush=True)
        res["tasks"][task] = tr
    # sixdist on dev512 (D02 protocol)
    b = np.load(BANK / "bank_dev512.npz", allow_pickle=True)
    Qx, Qe = np.asarray(b["Qx"], float), np.asarray(b["Qe"], float)
    Qm = json.loads(str(b["Qmeta"]))
    by_q = {}
    for i, m in enumerate(Qm):
        by_q.setdefault(m["qid"], []).append(i)
    sixres = {}
    for seed in SEEDS:
        for regime in ("clean", "flipmine"):
            mp = ROOT / "artifacts" / "f095_campaign" / "U02" / \
                f"sixdist_w64_s{seed}" / regime
            model, ck = load_arm(mp)
            model.eval()
            mu = np.asarray(ck["mu"], np.float32)
            sd = np.asarray(ck["sd"], np.float32)
            oks = []
            with torch.no_grad():
                for q, idxs in sorted(by_q.items()):
                    iA = next(i for i in idxs if Qm[i]["ptype"] == "A")
                    iB = next(i for i in idxs if Qm[i]["ptype"] == "B")
                    xa, xb, xab = Qx[iA], Qx[iB], Qx[iA] + Qe[iA]
                    labv = np.array([Qm[iA]["yA"], Qm[iA]["yB"], Qm[iA]["yAB"]])
                    lg = np.zeros(3)
                    for g in GROUP:
                        Xg = np.stack([apply(xa, g), apply(xb, g), apply(xab, g)])
                        F = rel_features(Xg.astype(np.float32))[:, :6]
                        F = torch.from_numpy(((F - mu) / sd).astype(np.float32))
                        lg += model(F).numpy().reshape(-1)
                    oks.append(bool((((lg / len(GROUP)) > 0).astype(int) == labv).all()))
            sixres[f"s{seed}_{regime}"] = round(float(np.mean(oks)), 4)
            print(f"dev512 six {regime} s{seed} J_groupavg={sixres[f's{seed}_{regime}']}",
                  flush=True)
    res["sixdist_dev512_g8"] = sixres
    out = U10 / "U10_ORBIT_EVAL.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
