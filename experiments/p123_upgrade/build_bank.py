#!/usr/bin/env python3
"""Frozen sample bank: quartets + single edits with x, e stored (see §2.1)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
from paths import oracle_at, atomic_edits  # noqa: E402
from r02_search import candidates_for_scene  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n_parents", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--out", type=Path,
                   default=ROOT / "artifacts" / "p123_upgrade" / "bank")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.load(ART / "scenes" / a.pool / "scenes.npz")
    X = d["positions"].astype(np.float32)
    Q, S = [], []
    qid = eid = 0
    order = rng.permutation(len(X))[:a.n_parents]
    for pi in order:
        x = X[pi].astype(float)
        try:
            y0, m0, _ = oracle_at(x)
        except ValueError:
            continue
        atomic = []
        for e, fam in atomic_edits(x, rng):
            try:
                y, m, _ = oracle_at(x + e)
            except ValueError:
                continue
            atomic.append((np.asarray(e, dtype=float), fam, y, m))
        for i in range(len(atomic)):
            for j in range(i + 1, len(atomic)):
                (ea, fa, ya, ma), (eb, fb, yb, mb) = atomic[i], atomic[j]
                if ya != y0 or yb != y0:
                    continue
                try:
                    yc, mc, _ = oracle_at(x + ea + eb)
                except ValueError:
                    continue
                if yc == y0:
                    continue
                for ptype, (xs0, v) in (("A", (x + ea, eb)),
                                        ("B", (x + eb, ea))):
                    try:
                        slo = oracle_at(xs0)[0]
                        eln = oracle_at(xs0 + v)[0]
                    except ValueError:
                        continue
                    Q.append({"qid": qid, "parent": int(pi), "ptype": ptype,
                              "fam_A": fa, "fam_B": fb,
                              "x": xs0.astype(np.float32),
                              "e": v.astype(np.float32),
                              "y0": y0, "yA": ya, "yB": yb, "yAB": yc,
                              "m0": m0, "mA": ma, "mB": mb, "mAB": mc,
                              "start_lab": slo, "end_lab": eln})
                qid += 1
        for item in candidates_for_scene(x, rng, n_random=16):
            e = np.asarray(item[0], dtype=float)
            try:
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if m1 < 0.005:
                continue
            S.append({"eid": eid, "parent": int(pi), "fam": item[1],
                      "x": x.astype(np.float32), "e": e.astype(np.float32),
                      "y0": y0, "y1": y1, "m0": m0, "m1": m1,
                      "flip": int(y1 != y0)})
            eid += 1
    Qx = np.stack([q["x"] for q in Q]) if Q else np.zeros((0, 4, 2), np.float32)
    Qe = np.stack([q["e"] for q in Q]) if Q else np.zeros((0, 4, 2), np.float32)
    Sx = np.stack([s["x"] for s in S]) if S else np.zeros((0, 4, 2), np.float32)
    Se = np.stack([s["e"] for s in S]) if S else np.zeros((0, 4, 2), np.float32)
    a.out.mkdir(parents=True, exist_ok=True)

    def meta(rows):
        return [{k: v for k, v in r.items() if k not in ("x", "e")}
                for r in rows]

    np.savez(a.out / ("bank_" + a.tag + ".npz"), Qx=Qx, Qe=Qe, Sx=Sx, Se=Se,
             Qmeta=json.dumps(meta(Q)), Smeta=json.dumps(meta(S)))
    h = hashlib.sha256(open(a.out / ("bank_" + a.tag + ".npz"), "rb").read()).hexdigest()
    json.dump({"tag": a.tag, "pool": a.pool, "seed": a.seed,
               "n_parents": int(a.n_parents), "n_paths": len(Q),
               "n_singles": len(S), "sha256": h},
              open(a.out / ("bank_" + a.tag + ".manifest.json"), "w"), indent=1)
    print("SAW bank_%s: paths=%d singles=%d sha=%s" % (a.tag, len(Q), len(S), h[:12]))


if __name__ == "__main__":
    main()
