#!/usr/bin/env python3
"""D03 bank builder: same-origin populations + candidate-order contrast.

Parents: scenes from 16N idx 512+; unseen only for N models, not all models.
D09 metadata ids are local to d09_fresh, so canonical coordinates resolve exclusions.
Per parent: FULL candidate labeling with shared oracle cache (equal oracle
budget by construction; order becomes pure selection). Flip/keep sets under
3 orders (first-12 / random-12-fixed-seed / stratified-12 round-robin).
E-bank: order-free, all valid E over full keep sets (pair subsample <=3000
uniform, seed-fixed, recorded). Singles bank: all valid edits.
Writes bank_d03O.npz / bank_d03E.npz (+manifests, sampling_flow.csv).
No model calls anywhere in this script.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
from r02_search import candidates_for_scene  # noqa: E402
from core.relations import segment_relation  # noqa: E402

BANKDIR = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUTD = ROOT / "artifacts" / "f095_campaign" / "D03"
SAMPLE_SEED, N_PARENTS, CAP = 3003, 600, 12
MARGIN_FLOOR = 0.03
ORDER_SEED = 3004
PAIR_CAP = 3000
FAM_ORDER = ["single", "pair", "global_trans", "global_rot", "global_scale", "random"]


def oracle(x):
    try:
        o = segment_relation(np.asarray(x, float))
    except ValueError:
        return None
    if o["ambiguous"] or o["margin"] < MARGIN_FLOOR:
        return None
    return int(o["label"]), float(o["margin"])


def select_first(items):
    return items[:CAP]


def select_random(items, rng):
    return sorted(rng.choice(items, size=min(CAP, len(items)), replace=False).tolist())


def select_stratified(items, fams):
    byfam = {}
    for i in items:
        byfam.setdefault(fams[i], []).append(i)
    out, k = [], 0
    while len(out) < min(CAP, len(items)):
        added = False
        for f in FAM_ORDER:
            if f in byfam and k < len(byfam[f]) and byfam[f][k] not in out:
                out.append(byfam[f][k])
                added = True
                if len(out) >= min(CAP, len(items)):
                    break
        if not added:
            break
        k += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-parents", type=int, default=N_PARENTS)
    a = p.parse_args()
    sc = np.load(ROOT / "artifacts" / "f095_campaign" / "D01" / "scenes_16N" / "scenes.npz",
                 allow_pickle=True)
    X16, y16 = sc["positions"].astype(float), sc["labels"].astype(int)
    d09 = np.load(BANKDIR / "bank_d09fresh665.npz", allow_pickle=True)
    source09 = np.load(ROOT / "artifacts/discovery_campaign/scenes/d09_fresh/scenes.npz")["positions"]
    assert np.array_equal(source09, X16[512:]), "D09 namespace changed: resolve coordinates again"
    used = set()
    for m in list(json.loads(str(d09["Qmeta"]))) + list(json.loads(str(d09["Smeta"]))):
        # d09_fresh is X16[512:], established by original coordinate hashes.
        used.add(int(m["parent"]) + 512)
    elig = [i for i in range(512, len(X16)) if i not in used]
    rng = np.random.default_rng(SAMPLE_SEED)
    parents = sorted(rng.choice(elig, size=min(a.n_parents, len(elig)), replace=False).tolist())
    print(f"eligible N-unseen {len(elig)}, sampled {len(parents)}, D09-excluded {len(used)}", flush=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    flow = open(OUTD / "sampling_flow.csv", "w", newline="")
    fw = csv.writer(flow)
    fw.writerow(["parent", "fam", "norm", "y0", "y1", "m0", "m1", "verdict", "reason"])
    Sx, Se, Sm = [], [], []
    Qx, Qe, Qm = [], [], []
    qid, eid = 0, 0
    cprop = {"parents": len(parents), "short_flip": 0, "short_keep": 0,
             "unpaired": 0, "E_total": 0, "oracle_evals": 0, "pair_capped": 0}
    norm_hist = {r: [] for r in ("first", "random", "stratified")}
    orng = np.random.default_rng(ORDER_SEED)
    for pi in parents:
        x = X16[pi]
        y0 = int(y16[pi])
        try:
            m0 = float(segment_relation(x)["margin"])
        except ValueError:
            continue
        prng = np.random.default_rng(SAMPLE_SEED * 7919 + pi)
        cands = [(np.asarray(e, float), fam) for e, fam in candidates_for_scene(x, prng)]
        lab, fams = [], {}
        for k, (e, fam) in enumerate(cands):
            fams[k] = fam
            r = oracle(x + e)
            cprop["oracle_evals"] += 1
            nrm = float(np.linalg.norm(e))
            if r is None:
                fw.writerow([pi, fam, round(nrm, 4), y0, "", "", "", "invalid",
                             "ambiguous/margin/ValueError"])
                lab.append(None)
            else:
                y1, m1 = r
                fw.writerow([pi, fam, round(nrm, 4), y0, y1, round(m0, 4), round(m1, 4),
                             "flip" if y1 != y0 else "keep", ""])
                lab.append((y1, m1))
        flips = [k for k, v in enumerate(lab) if v is not None and v[0] != y0]
        keeps = [k for k, v in enumerate(lab) if v is not None and v[0] == y0]
        fset = {"first": select_first(flips),
                "random": select_random(flips, orng) if flips else [],
                "stratified": select_stratified(flips, fams) if flips else []}
        kset = {"first": select_first(keeps),
                "random": select_random(keeps, orng) if keeps else [],
                "stratified": select_stratified(keeps, fams) if keeps else []}
        for rule in fset:
            if len(fset[rule]) < CAP:
                cprop["short_flip"] += 1
            if len(kset[rule]) < CAP:
                cprop["short_keep"] += 1
            norm_hist[rule].extend(
                [round(float(np.linalg.norm(cands[k][0])), 4) for k in fset[rule]])
        if not flips or not keeps:
            cprop["unpaired"] += 1
        for k, v in enumerate(lab):
            if v is None:
                continue
            y1, m1 = v
            Sx.append(x + cands[k][0]); Se.append(cands[k][0])
            Sm.append({"eid": eid, "parent": pi, "fam": fams[k], "y0": y0, "y1": y1,
                       "m0": round(m0, 4), "m1": round(m1, 4),
                       "flip": int(y1 != y0),
                       "sel_flip": sorted([r for r in fset if k in fset[r]]),
                       "sel_keep": sorted([r for r in kset if k in kset[r]])})
            eid += 1
        pairs = [(ka, kb) for ia, ka in enumerate(keeps) for kb in keeps[ia + 1:]]
        capped = False
        if len(pairs) > PAIR_CAP:
            sub = sorted(orng.choice(len(pairs), size=PAIR_CAP, replace=False).tolist())
            pairs = [pairs[i] for i in sub]
            capped, cprop["pair_capped"] = True, cprop["pair_capped"] + 1
        for ka, kb in pairs:
            ea, eb = cands[ka][0], cands[kb][0]
            r = oracle(x + ea + eb)
            cprop["oracle_evals"] += 1
            if r is None:
                continue
            yc, mc = r
            if yc == y0:
                continue
            ya, ma = lab[ka][0], lab[ka][1]
            yb, mb = lab[kb][0], lab[kb][1]
            for ptype, xs, vs, slo in (("A", x + ea, eb, ya), ("B", x + eb, ea, yb)):
                Qx.append(xs); Qe.append(vs)
                Qm.append({"qid": qid, "parent": pi, "ptype": ptype,
                           "fam_A": fams[ka], "fam_B": fams[kb], "y0": y0,
                           "yA": ya, "yB": yb, "yAB": yc,
                           "m0": round(m0, 4), "mA": round(ma, 4),
                           "mB": round(mb, 4), "mAB": round(mc, 4),
                           "start_lab": slo, "end_lab": yc})
            qid += 1
            cprop["E_total"] += 1
    flow.close()
    Sx = np.stack(Sx).astype(np.float32)
    Se = np.stack(Se).astype(np.float32)
    Qx = np.stack(Qx).astype(np.float32)
    Qe = np.stack(Qe).astype(np.float32)
    np.savez_compressed(BANKDIR / "bank_d03O.npz", Sx=Sx, Se=Se, Smeta=json.dumps(Sm),
                        Qx=np.zeros((0, 4, 2), np.float32),
                        Qe=np.zeros((0, 4, 2), np.float32), Qmeta=json.dumps([]))
    np.savez_compressed(BANKDIR / "bank_d03E.npz", Qx=Qx, Qe=Qe, Qmeta=json.dumps(Qm),
                        Sx=np.zeros((0, 4, 2), np.float32),
                        Se=np.zeros((0, 4, 2), np.float32), Smeta=json.dumps([]))
    man = {"parents": parents, "n_parents": len(parents),
           "sample_seed": SAMPLE_SEED, "order_seed": ORDER_SEED, "cap": CAP,
           "margin_floor": MARGIN_FLOOR, "pair_cap": PAIR_CAP,
           "shared_oracle_cache": True, "n_E_quartets": cprop["E_total"],
           "n_singles": len(Sm), "stat": cprop,
           "norm_hist_flip": {r: [round(float(np.mean(v)), 4), len(v)] for r, v in norm_hist.items()},
           "d09_excluded_parents": len(used)}
    for tag in ("bank_d03O", "bank_d03E"):
        pth = BANKDIR / (tag + ".npz")
        (BANKDIR / (tag + ".manifest.json")).write_text(json.dumps(
            dict(man, sha256=hashlib.sha256(pth.read_bytes()).hexdigest()), indent=1))
    print(f"E={cprop['E_total']} singles={len(Sm)} short_flip={cprop['short_flip']} "
          f"short_keep={cprop['short_keep']} unpaired={cprop['unpaired']} "
          f"pair_capped={cprop['pair_capped']} oracle_evals={cprop['oracle_evals']}", flush=True)
    print("wrote bank_d03O.npz bank_d03E.npz", flush=True)


if __name__ == "__main__":
    main()
