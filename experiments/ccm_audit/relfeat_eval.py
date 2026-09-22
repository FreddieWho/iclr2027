#!/usr/bin/env python3
"""RelFeat closure (§13): raw clean vs relfeat clean, s11/s23/s47.
Metrics: static, atomic A/B, atomic-pass, AB endpoint, conditional comp
miss, J, single-flip, preserve FA. Dev bank. load_model (featurize-aware).

R3 parameterized entry: --bank {dev512,confirm1007} selects the sample
bank; --model-map overrides checkpoint dirs as JSON
{"raw": ".../r04b_s{seed}/clean", "relfeat": "..."} with {seed}
placeholder; defaults reproduce the historical dev-bank run exactly.
Outputs record bank tag + bank sha256 + per-arm model sha256 so every
number is traceable to frozen inputs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from common import load_model, preprocess  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "next_novelty" / "relfeat"
CKPTS = {"raw": ART / "r04b_s{seed}" / "clean",
         "relfeat_s11": ART / "r04b_s11" / "relfeat",
         "relfeat_s23": ART / "r04b_s23_relfeat" / "relfeat",
         "relfeat_s47": ART / "r04b_s47_relfeat" / "relfeat"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--extra", default="")
    p.add_argument("--bank", default="dev512", choices=["dev512", "confirm1007"])
    p.add_argument("--model-map", default="",
                   help='JSON {"raw": ".../r04b_s{seed}/clean", "relfeat": "..."}'
                        ' path templates with {seed} placeholder; override defaults')
    a = p.parse_args()

    def resolve(tag, seed):
        if tag == "relflip":
            return ROOT / "artifacts" / "next_novelty" / "relflip" / ("s%d" % seed)
        if a.model_map:
            overrides = json.loads(a.model_map)
            if tag in overrides:
                return Path(overrides[tag].format(seed=seed))
        if tag == "relfeat":
            return CKPTS["relfeat_s%d" % seed]
        return ART / ("r04b_s%d" % seed) / "clean"

    import hashlib
    bank_path = BANK / ("bank_%s.npz" % a.bank)
    bank_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    b = np.load(bank_path, allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    Sx, Se = b["Sx"], b["Se"]
    Smeta = json.loads(str(b["Smeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    res = {}
    tags = ("raw", "relfeat") if not a.extra else ("relflip",)
    for seed in (11, 23, 47):
        for tag in tags:
            mp = resolve(tag, seed)
            model, stats = load_model(mp)
            model.eval()

            def pfwd(xs):
                xs = np.asarray(xs, np.float32)
                with torch.no_grad():
                    out = []
                    for s in range(0, len(xs), 512):
                        out.append(model(preprocess(xs[s:s + 512], stats)).numpy().reshape(-1))
                lg = np.concatenate(out)
                return lg, (lg > 0).astype(int)
            R = {}
            for qid, (i1, m1), (i2, m2) in quads:
                xa = np.asarray(Qx[i1], float)
                xb = np.asarray(Qx[i2], float)
                xab = xa + np.asarray(Qe[i1], float)
                la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
                lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
                _, p = pfwd(np.stack([xa, xb, xab]))
                R[qid] = (int(p[0] == la), int(p[1] == lb), int(p[2] == m1["yAB"]))
            n = len(R)
            aA = sum(v[0] for v in R.values()) / n
            aB = sum(v[1] for v in R.values()) / n
            apass = sum(1 for v in R.values() if v[0] and v[1]) / n
            ab = sum(v[2] for v in R.values()) / n
            own = [q for q in R if R[q][0] and R[q][1]]
            cond = 1 - sum(R[q][2] for q in own) / len(own) if own else None
            J = sum(1 for v in R.values() if v == (1, 1, 1)) / n
            Xs = np.asarray(Sx, np.float32)
            Xe = np.asarray(Se, np.float32)
            _, p0 = pfwd(Xs)
            _, p1 = pfwd((Xs.reshape(len(Xs), -1) + Xe.reshape(len(Xe), -1)).reshape(-1, 4, 2))
            y0 = np.array([m["y0"] for m in Smeta])
            y1 = np.array([m["y1"] for m in Smeta])
            fl = np.array([m["flip"] == 1 for m in Smeta])
            static_err = float((p1 != y1).mean())
            single_err = float((p1[fl] != y1[fl]).mean())
            st = (~fl) & (p0 == y0)
            fa = float((p1[st] != p0[st]).mean()) if st.sum() else None
            entry = {
                "n_q": n, "static": round(static_err, 4),
                "atomicA": round(aA, 4), "atomicB": round(aB, 4),
                "atomic_pass": round(apass, 4), "AB_end": round(ab, 4),
                "cond_comp_miss": round(cond, 4) if cond is not None else None,
                "cond_n": len(own), "J": round(J, 4),
                "single": round(single_err, 4),
                "preserve_FA": round(fa, 4) if fa is not None else None,
                "model_dir": str(mp.relative_to(ROOT)),
                "model_sha256": hashlib.sha256((mp / "model.pt").read_bytes()).hexdigest()}
            res[f"s{seed}_{tag}"] = entry
            print("s%d %s: %s" % (seed, tag, res[f"s{seed}_{tag}"]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    out_name = "RELFLIP.json" if a.extra else "RELFEAT.json"
    res["_provenance"] = {
        "bank": a.bank, "bank_sha256": bank_sha,
        "model_map": a.model_map or None,
        "feature_boundary": "relfeat arm uses hand-designed label-free relational "
                            "features (six ordered pair distances + four centroid "
                            "radii); node-pair identity is part of the input; "
                            "task-aware deterministic representation, not a new "
                            "foundation model and not oracle-free."}
    json.dump(res, open(a.out / out_name, "w"), indent=1)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
