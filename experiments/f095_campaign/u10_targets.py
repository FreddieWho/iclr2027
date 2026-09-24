#!/usr/bin/env python3
"""U10 targets: gen/mine/train/eval for T1 point-in-triangle + T2 segment-disk.

Stages: --stage gen | mine | train | eval (run in order; train before eval).
Mirrors source recipe (full-batch 300ep, Adam 1e-2, BCE, lam=1.0, cap-12
candidate-order flips, nested f25). Training uses single edits only, never
combos. Eval banks: E-quartets (cap 12/parent, recorded) in bank_dev512
format (Qx/Qe/Qmeta). No sealed-pool reads.
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
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
from coord_mlp import CoordMLP  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402
from u10_oracles import tri_oracle, disk_oracle, T2_RADIUS  # noqa: E402
from u10_models import TypedTriMLP, TypedDiskMLP, count_params  # noqa: E402
sys.path.insert(0, str(ROOT / "docs" / "f095dfa_review_pack" / "checks"))
from threshold_certificate import threshold_certificate  # noqa: E402

torch.set_num_threads(4)
BANKDIR = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUTD = ROOT / "artifacts" / "f095_campaign" / "U10"
EPOCHS, LR, LAM, CAP = 300, 1e-2, 1.0, 12
MARGIN_FLOOR = 0.02
SEEDS = (11, 23, 47)
TASKS = {
    "T1": {"npts": 4, "oracle": tri_oracle, "train_seed": 10101, "eval_seed": 20202,
           "in_raw": 8, "in_six": 6},
    "T2": {"npts": 3, "oracle": disk_oracle, "train_seed": 30303, "eval_seed": 40404,
           "in_raw": 6, "in_six": 3},
}


def gen_scenes(task, n, seed, min_margin=0.02):
    cfg, O = TASKS[task], TASKS[task]["oracle"]
    rng = np.random.default_rng(seed)
    X, Y, M, I = [], [], [], []
    targets = {0: (n + 1) // 2, 1: n // 2}
    counts = {0: 0, 1: 0}
    guard = 0
    while (counts[0] < targets[0] or counts[1] < targets[1]) and guard < n * 500:
        guard += 1
        x = rng.uniform(-0.8, 0.8, size=(cfg["npts"], 2))
        if task == "T1" and abs(float(np.cross(x[1] - x[0], x[2] - x[0]))) < 0.05:
            continue
        try:
            o = O(x, min_margin)
        except ValueError:
            continue
        if o["ambiguous"]:
            continue
        if counts[o["label"]] >= targets[o["label"]]:
            continue
        counts[o["label"]] += 1
        X.append(x); Y.append(o["label"]); M.append(o["margin"]); I.append(len(X) - 1)
    X = np.array(X)
    return {"positions": X, "labels": np.array(Y), "margins": np.array(M),
            "counts": counts, "guard": guard}


def candidates_3pt(x, rng, n_random=64):
    cands = []
    for node in range(3):
        for r in (0.03, 0.06, 0.10, 0.16):
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                e = np.zeros((3, 2))
                e[node] = r * np.array([np.cos(a), np.sin(a)])
                cands.append((e, "single"))
    for r in (0.03, 0.06, 0.10, 0.16):
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            e = np.zeros((3, 2))
            e[[0, 1]] = r * np.array([np.cos(a), np.sin(a)]) / np.sqrt(2)
            cands.append((e, "pair"))
    cen = x.mean(axis=0)
    for r in (0.03, 0.06, 0.10, 0.16):
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            cands.append((np.zeros((3, 2)) + r * np.array([np.cos(a), np.sin(a)]),
                          "global_trans"))
    for ang in (0.05, 0.1, 0.2, -0.05, -0.1, -0.2):
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        cands.append(((x - cen) @ R.T - (x - cen), "global_rot"))
    for s in (0.95, 1.05, 0.9, 1.1):
        cands.append(((s - 1) * (x - cen), "global_scale"))
    for _ in range(n_random):
        e = rng.normal(size=(3, 2))
        e *= rng.choice((0.03, 0.06, 0.10, 0.16)) * 2 / np.linalg.norm(e)
        cands.append((e, "random"))
    return cands


def six_features(X, npts):
    X = np.asarray(X, float).reshape(-1, npts, 2)
    i, j = np.triu_indices(npts, k=1)
    return np.linalg.norm(X[:, i] - X[:, j], axis=-1)


def build_arch(task, arch):
    cfg = TASKS[task]
    if arch == "raw":
        return CoordMLP(64, 32, in_dim=cfg["in_raw"])
    if arch == "six":
        return CoordMLP(64, 32, in_dim=cfg["in_six"])
    if arch == "typed":
        return TypedTriMLP() if task == "T1" else TypedDiskMLP()
    raise ValueError(arch)


def featurize(task, arch, Xarr, stats):
    mu, sd, mode = stats
    if arch == "typed":
        F = np.asarray(Xarr, np.float32).reshape(len(Xarr), -1)
    elif arch == "six":
        F = six_features(Xarr, TASKS[task]["npts"])
    else:
        F = np.asarray(Xarr, np.float32).reshape(len(Xarr), -1)
    return torch.from_numpy(((F - mu) / sd).astype(np.float32))


def fit_stats(task, arch, Xtr):
    if arch == "typed":
        F = np.asarray(Xtr, np.float32).reshape(len(Xtr), -1)
        return F.mean(0), F.std(0) + 1e-8, "scalar8"
    if arch == "six":
        F = six_features(Xtr, TASKS[task]["npts"])
        return F.mean(0), F.std(0) + 1e-8, "six"
    F = np.asarray(Xtr, np.float32).reshape(len(Xtr), -1)
    return F.mean(0), F.std(0) + 1e-8, "raw"


def sha_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["gen", "mine", "train", "eval"], required=True)
    p.add_argument("--task", choices=["T1", "T2", "both"], default="both")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    a = p.parse_args()
    tasks = ["T1", "T2"] if a.task == "both" else [a.task]
    OUTD.mkdir(parents=True, exist_ok=True)

    if a.stage == "gen":
        for task in tasks:
            cfg = TASKS[task]
            for split, n, seed in (("train", 512, cfg["train_seed"]),
                                   ("eval", 512, cfg["eval_seed"])):
                g = gen_scenes(task, n, seed)
                assert len(g["positions"]) == n, (task, split, len(g["positions"]))
                od = OUTD / f"{task}_{split}"
                od.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    od / "scenes.npz", positions=g["positions"].astype(np.float32),
                    labels=g["labels"].astype(np.int64),
                    margins=g["margins"].astype(np.float32))
                json.dump({"task": task, "split": split, "n": n, "seed": seed,
                           "counts": g["counts"], "guard": g["guard"]},
                          open(od / "manifest.json", "w"), indent=1)
                print(f"SAW {task}/{split} n={n} bal={g['counts']}", flush=True)

    if a.stage == "mine":
        for task in tasks:
            cfg, O = TASKS[task], TASKS[task]["oracle"]
            d = np.load(OUTD / f"{task}_train" / "scenes.npz")
            X, y = d["positions"].astype(float), d["labels"].astype(int)
            frng = np.random.default_rng(999)
            flips, f25pool = [], []
            for i in range(len(X)):
                prng = np.random.default_rng(5000 + i)
                cands = (list(candidates_for_scene(X[i], prng)) if cfg["npts"] == 4
                         else candidates_3pt(X[i], prng))
                found = []
                for e, fam in cands:
                    try:
                        o = O(X[i] + np.asarray(e, float), MARGIN_FLOOR)
                    except ValueError:
                        continue
                    if o["ambiguous"]:
                        continue
                    if o["label"] != int(y[i]):
                        found.append({"scene": i, "edit": np.asarray(e, np.float32),
                                      "new": int(o["label"]), "fam": fam})
                        if len(found) >= CAP:
                            break
                flips.extend(found)
            perm = frng.permutation(len(flips)).tolist()
            f25 = sorted(perm[:max(1, len(flips) // 4)])
            np.savez_compressed(
                OUTD / f"{task}_flips.npz",
                **{f"edit_{k}": m["edit"] for k, m in enumerate(flips)},
                meta=np.array([(m["scene"], m["new"], m["fam"]) for m in flips], dtype=object),
                f25=np.array(f25, dtype=np.int64))
            print(f"SAW {task} flips={len(flips)} f25={len(f25)}", flush=True)

    if a.stage == "train":
        for task in tasks:
            d = np.load(OUTD / f"{task}_train" / "scenes.npz")
            X, y = d["positions"].astype(float), d["labels"].astype(int)
            fz = np.load(OUTD / f"{task}_flips.npz", allow_pickle=True)
            meta = fz["meta"]
            F = np.stack([fz[f"edit_{k}"] for k in range(len(meta))]).astype(float)
            fii = np.array([int(r[0]) for r in meta])
            fyf = torch.from_numpy(np.array([int(r[1]) for r in meta], dtype=np.float32))
            f25 = np.asarray(fz["f25"], dtype=int)
            for arch in ("raw", "six", "typed"):
                stats = fit_stats(task, arch, X)
                Xc = featurize(task, arch, X, stats)
                yc = torch.from_numpy(y.astype(np.float32))
                Xf = featurize(task, arch, X[fii] + F, stats)
                Xq = featurize(task, arch, X[fii[f25]] + F[f25], stats)
                yq = fyf[f25]
                for seed in SEEDS:
                    for regime, Xb, yb in (("clean", Xc, yc), ("flip", Xf, fyf),
                                           ("f25", Xq, yq)):
                        torch.manual_seed(seed)
                        model = build_arch(task, arch)
                        opt = torch.optim.Adam(model.parameters(), lr=LR)
                        bce = nn.BCEWithLogitsLoss()
                        model.train()
                        for ep in range(a.epochs):
                            opt.zero_grad()
                            if regime == "clean":
                                loss = bce(model(Xc), yc)
                            else:
                                loss = bce(model(Xc), yc) + LAM * bce(model(Xb), yb)
                            loss.backward()
                            opt.step()
                        od = OUTD / f"{task}_{arch}_s{seed}" / regime
                        od.mkdir(parents=True, exist_ok=True)
                        torch.save({"state": model.state_dict(), "arch": arch,
                                    "task": task, "seed": seed, "regime": regime,
                                    "mu": stats[0], "sd": stats[1], "mode": stats[2],
                                    "n_params": count_params(model)}, od / "model.pt")
                        print(f"SAW {task}/{arch}/s{seed}/{regime} loss={float(loss):.4f} "
                              f"params={count_params(model)}", flush=True)

    if a.stage == "eval":
        res = {}
        for task in tasks:
            cfg, O = TASKS[task], TASKS[task]["oracle"]
            d = np.load(OUTD / f"{task}_eval" / "scenes.npz")
            Xe, ye = d["positions"].astype(float), d["labels"].astype(int)
            Qx, Qe, Qm, qid = [], [], [], 0
            estat = {"E": 0, "Eparents": 0, "unpaired": 0}
            for i in range(len(Xe)):
                prng = np.random.default_rng(6000 + i)
                cands = (list(candidates_for_scene(Xe[i], prng)) if cfg["npts"] == 4
                         else candidates_3pt(Xe[i], prng))
                keeps = []
                for e, fam in cands:
                    try:
                        o = O(Xe[i] + np.asarray(e, float), MARGIN_FLOOR)
                    except ValueError:
                        continue
                    if o["ambiguous"]:
                        continue
                    if o["label"] == int(ye[i]):
                        keeps.append((np.asarray(e, float), fam))
                pairs = [(ka, kb) for ia, ka in enumerate(keeps)
                         for kb in keeps[ia + 1:]]
                if len(pairs) > 4000:
                    sub = sorted(prng.choice(len(pairs), 4000, replace=False).tolist())
                    pairs = [pairs[j] for j in sub]
                made = 0
                for (ea, fa), (eb, fb) in pairs:
                    try:
                        o = O(Xe[i] + ea + eb, MARGIN_FLOOR)
                    except ValueError:
                        continue
                    if o["ambiguous"] or o["label"] == int(ye[i]):
                        continue
                    for ptype, xs, vs in (("A", Xe[i] + ea, eb), ("B", Xe[i] + eb, ea)):
                        Qx.append(xs); Qe.append(vs)
                        Qm.append({"qid": qid, "parent": i, "ptype": ptype,
                                   "fam_A": fa, "fam_B": fb, "y0": int(ye[i]),
                                   "yA": int(ye[i]), "yB": int(ye[i]),
                                   "yAB": int(o["label"]),
                                   "m0": 0.0, "mA": 0.0, "mB": 0.0,
                                   "mAB": round(float(o["margin"]), 4),
                                   "start_lab": int(ye[i]), "end_lab": int(o["label"])})
                    qid += 1
                    made += 1
                    if made >= CAP:
                        break
                estat["E"] += made
                estat["Eparents"] += made > 0
                if not keeps:
                    estat["unpaired"] += 1
            Qx = np.stack(Qx).astype(np.float32)
            Qe = np.stack(Qe).astype(np.float32)
            np.savez_compressed(BANKDIR / f"bank_u10{task}.npz", Qx=Qx, Qe=Qe,
                                Qmeta=json.dumps(Qm),
                                Sx=np.zeros((0, Xe.shape[1], 2), np.float32),
                                Se=np.zeros((0, Xe.shape[1], 2), np.float32),
                                Smeta=json.dumps([]))
            (BANKDIR / f"bank_u10{task}.manifest.json").write_text(json.dumps(
                {"task": task, "n_eval_parents": len(Xe), "E": estat["E"],
                 "Eparents": estat["Eparents"], "unpaired": estat["unpaired"],
                 "sha256": sha_of(BANKDIR / f"bank_u10{task}.npz")}, indent=1))
            print(f"SAW {task} E={estat['E']} Eparents={estat['Eparents']}/{len(Xe)}",
                  flush=True)
            # evaluate all 27 arms
            arms, oks = {}, {}
            b110 = {}
            for arch in ("raw", "six", "typed"):
                for seed in SEEDS:
                    for regime in ("clean", "flip", "f25"):
                        mp = OUTD / f"{task}_{arch}_s{seed}" / regime / "model.pt"
                        ck = torch.load(mp, map_location="cpu", weights_only=False)
                        model = build_arch(task, arch)
                        model.load_state_dict(ck["state"])
                        model.eval()
                        stats = (ck["mu"], ck["sd"], ck["mode"])
                        nq = len(Qm) // 2
                        la = np.array([Qm[2 * q]["yA"] for q in range(nq)])
                        lb = np.array([Qm[2 * q + 1]["yB"] for q in range(nq)])
                        ly = np.array([Qm[2 * q]["yAB"] for q in range(nq)])
                        xab = np.stack([Qx[2 * q] + Qe[2 * q] for q in range(nq)])
                        lab = np.stack([la, lb, ly], axis=1)
                        with torch.no_grad():
                            lA, lB, lAB = [], [], []
                            for s in range(0, nq, 512):
                                sl = slice(s, s + 512)
                                lA.append(model(featurize(task, arch, Qx[2 * s:2 * (s + 512):2],
                                                          stats)).numpy())
                                lB.append(model(featurize(task, arch, Qx[2 * s + 1:2 * (s + 512):2],
                                                          stats)).numpy())
                                lAB.append(model(featurize(task, arch, xab[sl], stats)).numpy())
                        sc = np.stack([np.concatenate(lA), np.concatenate(lB),
                                       np.concatenate(lAB)], axis=1)
                        pred = (sc > 0).astype(int)
                        ok = (pred == lab)
                        cert = threshold_certificate(sc, lab, 0.0)
                        name = f"{arch}_{regime}_s{seed}"
                        arms[name] = {
                            "J": round(float(ok.all(axis=1).mean()), 4),
                            "atomic_pass": round(float(ok[:, :2].all(axis=1).mean()), 4),
                            "AB_correct": round(float(ok[:, 2].mean()), 4),
                            "S": round(cert.S_local_separable, 4),
                            "J_star": round(cert.J_star_global_oracle, 4),
                            "n": nq}
                        oks[name] = ok
                        if arch == "raw" and regime == "clean":
                            b110[seed] = (ok[:, :2].all(axis=1)) & (ok[:, 2] == 0)
                        print(f"  {arch}/{regime}/s{seed} J={arms[name]['J']}", flush=True)
            for name, ok in oks.items():
                seed = int(name.rsplit("_s", 1)[1])
                H = b110[seed]
                eAB, eAEB = ok[:, 2], ok.all(axis=1)
                eAE = ok[:, :2].all(axis=1)
                arms[name]["R_endpoint"] = round(float(eAB[H].mean()), 4) if H.sum() else None
                arms[name]["R_full"] = round(float(eAEB[H].mean()), 4) if H.sum() else None
                arms[name]["M"] = round(float(((eAB == 1) & (eAE == 0))[H].mean()), 4) \
                    if H.sum() else None
            res[task] = {"E": estat["E"], "Eparents": estat["Eparents"], "arms": arms,
                         "b110n": {str(s): int(b110[s].sum()) for s in b110}}
            res[task] = {"E": estat["E"], "Eparents": estat["Eparents"], "arms": arms,
                         "b110n": {str(s): int(b110[s].sum()) for s in b110}}
            # recompute R flags cheaply from stored arms? store minimal: recompute below
            with open(OUTD / f"U10_{task}_EVAL.json", "w") as f:
                json.dump(res[task], f, indent=1)
    print("DONE u10", a.stage)


if __name__ == "__main__":
    main()
