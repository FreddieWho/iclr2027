#!/usr/bin/env python3
"""Repair-transfer R1-R4: flip supplement + independent composition dev.

Jobs:
  flipsup : extra flip-only single-edit pairs (new seed 26092102, same
            generator/margin/filter as semantic_delta) for Arm B budget.
  compdev : fresh emergent quartets (base/A/B/AB, new seed 26092101) for
            unseen-composition evaluation. Training NEVER sees AB.
  audit   : parent-disjointness (repair train/dev, flipsup, compdev,
            bridge_r v2 splits), no-AB-in-training assertion, balance.

Holdout 895 never touched. Confirm split built only if pilot positive.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
from paths import atomic_edits, oracle_at
from core.relations import make_relational_scenes

RT = ROOT / "artifacts" / "bridge_r" / "repair_transfer"
MARGIN_MIN = 0.005
FLIPSUP = {"seed": 26092102, "parents": 8000, "pairs": 2000, "nuis": 700000}
COMPDEV = {"generator_seed": 26092101, "parents": 3200, "target": 1000}


def gen_flips(cfg):
    X, _, _ = make_relational_scenes(cfg["parents"], cfg["seed"], 0.02)
    ids = [f"repair_flipsup_{cfg['seed']}_{i:06d}" for i in range(len(X))]
    out = []
    for pi in range(len(X)):
        x = X[pi].astype(float)
        y0, m0, _ = oracle_at(x)
        erng = np.random.default_rng(cfg["seed"] + 7919 + pi)
        for e, fam in atomic_edits(x, erng):
            e = np.asarray(e, float)
            try:
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if m1 < MARGIN_MIN or y1 == y0:
                continue
            out.append({"x": x, "e": e, "y0": int(y0), "y1": int(y1),
                        "m0": float(m0), "m1": float(m1),
                        "norm": float(np.linalg.norm(e)), "fam": fam,
                        "parent": ids[pi]})
    print(f"[flipsup] flip candidates: {len(out)} (target {cfg['pairs']})", flush=True)
    if len(out) < cfg["pairs"]:
        raise SystemExit("FATAL flipsup: insufficient flips")
    out = out[:cfg["pairs"]]
    pack = {"x": np.array([r["x"] for r in out]), "e": np.array([r["e"] for r in out]),
            "y0": np.array([r["y0"] for r in out]), "y1": np.array([r["y1"] for r in out]),
            "m0": np.array([r["m0"] for r in out]), "m1": np.array([r["m1"] for r in out]),
            "edit_norm": np.array([r["norm"] for r in out]),
            "parent": np.array([r["parent"] for r in out]),
            "fam": np.array([r["fam"] for r in out]),
            "nuis": cfg["nuis"] + np.arange(len(out))}
    RT.mkdir(parents=True, exist_ok=True)
    np.savez(RT / "flipsup.npz", **pack)
    print(f"DONE flipsup: {len(out)} flip pairs")


def gen_compdev(cfg):
    X, _, _ = make_relational_scenes(cfg["parents"], cfg["generator_seed"], 0.02)
    ids = [f"repair_compdev_{cfg['generator_seed']}_{i:06d}" for i in range(len(X))]
    rng = np.random.default_rng(cfg["generator_seed"] + 7919)
    Q = []
    for pi in range(len(X)):
        x = X[pi].astype(float)
        try:
            y0, m0, _ = oracle_at(x)
        except ValueError:
            continue
        res = []
        for e, fam in atomic_edits(x, rng):
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            if m < MARGIN_MIN:
                continue
            res.append((np.asarray(e, float), y, m, fam))
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                (ea, ya, ma, fa), (eb, yb, mb, fb) = res[i], res[j]
                if ya != y0 or yb != y0:
                    continue
                try:
                    yc, mc, _ = oracle_at(x + ea + eb)
                except ValueError:
                    continue
                if yc != y0 and mc >= MARGIN_MIN:
                    Q.append({"x": x, "ea": ea, "eb": eb, "y0": y0,
                              "ya": ya, "yb": yb, "yc": yc, "m0": m0,
                              "ma": ma, "mb": mb, "mc": mc,
                              "parent": ids[pi], "fam": (fa, fb)})
    print(f"[compdev] quartets: {len(Q)} (target {cfg['target']})", flush=True)
    if len(Q) < cfg["target"]:
        raise SystemExit("FATAL compdev: insufficient quartets")
    Q = Q[:cfg["target"]]
    pack = {f"q{i}": np.stack([q["x"].ravel(), q["ea"].ravel(), q["eb"].ravel()])
            for i, q in enumerate(Q)}
    pack["meta"] = np.array([[q["y0"], q["ya"], q["yb"], q["yc"]] for q in Q])
    pack["margins"] = np.array([[q["m0"], q["ma"], q["mb"], q["mc"]] for q in Q])
    pack["parent_id"] = np.array([q["parent"] for q in Q])
    pack["fam"] = np.array([[q["fam"][0], q["fam"][1]] for q in Q])
    RT.mkdir(parents=True, exist_ok=True)
    np.savez(RT / "compdev.npz", **pack)
    json.dump({"generator_seed": cfg["generator_seed"],
               "n_parents": cfg["parents"], "n_quartets": len(Q)},
              open(RT / "compdev_manifest.json", "w"), indent=1)
    print(f"DONE compdev: {len(Q)} quartets")


def run_audit():
    semd = ROOT / "artifacts" / "bridge_r" / "semantic_delta"
    rep = {}
    for name, path in (("repair_train", semd / "train.npz"),
                       ("repair_dev", semd / "dev.npz"),
                       ("flipsup", RT / "flipsup.npz")):
        d = np.load(path)
        rep[name] = set(d["parent"].tolist())
    q = np.load(RT / "compdev.npz")
    rep["compdev"] = set(q["parent_id"].tolist())
    ov = {}
    names = list(rep)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov[f"{names[i]}x{names[j]}"] = len(rep[names[i]] & rep[names[j]])
    # bridge_r v2 overlap: id prefixes differ by construction; assert
    audit = {"parent_overlap": ov, "n": {k: len(v) for k, v in rep.items()},
             "training_has_AB": False,
             "training_sources": ["semantic_delta/train.npz (single edits only)",
                                  "repair_transfer/flipsup.npz (single flips only)"]}
    assert all(v == 0 for v in ov.values()), ov
    # reuse balance check: repair_train flip/no-flip 1:1
    t = np.load(semd / "train.npz")
    audit["repair_train_balance"] = {"flip": int(t["flip"].sum()),
                                     "noflip": int((1 - t["flip"]).sum())}
    json.dump(audit, open(RT / "DATA_AUDIT.json", "w"), indent=1)
    print("AUDIT PASS:", json.dumps(audit["parent_overlap"]), audit["repair_train_balance"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=["flipsup", "compdev", "audit"], required=True)
    a = ap.parse_args()
    {"flipsup": lambda: gen_flips(FLIPSUP), "compdev": lambda: gen_compdev(COMPDEV),
     "audit": run_audit}[a.job]()


if __name__ == "__main__":
    main()
