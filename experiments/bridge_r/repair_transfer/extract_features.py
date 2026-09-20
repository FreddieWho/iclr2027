#!/usr/bin/env python3
"""Repair-transfer R6: frozen DINOv3-L final R0, extracted ONCE.

Jobs: repair_train / flipsup / repair_dev (single edits, base+endpoint
share stored nuisance seed) / compdev (4 states per quartet, nuisance
base 800000 + qi, fresh RNG per state). Manifest + sha committed;
feature binaries stay local (gitignored).
"""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r"))
from render_v1 import render_canonical, RENDER_CONFIG
from extract_r import load_pinned, vision_forward

ART = ROOT / "artifacts" / "bridge_r"
RT = ART / "repair_transfer"
SEMD = ART / "semantic_delta"
FEAT = RT / "feats"
BB = "dinov3l"
COMPDEV_NUIS = 800000


def r0_batch(proc, net, imgs, batch):
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), batch):
            b = proc(images=imgs[s:s + batch], return_tensors="pt")
            h = vision_forward(net, BB, b)
            last = h.last_hidden_state
            pool = h.pooler_output
            r0 = (pool if pool is not None else last[:, 0]).detach().cpu().numpy()
            out.append(r0)
            print(f"  feat {s + len(r0)}/{len(imgs)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def single_job(proc, net, name, src, batch):
    d = np.load(src)
    imgs, labels, parents, kinds = [], [], [], []
    for i in range(len(d["y0"])):
        seed = int(d["nuis"][i])
        imgs.append(render_canonical(d["x"][i], np.random.default_rng(seed)))
        imgs.append(render_canonical(d["x"][i] + d["e"][i], np.random.default_rng(seed)))
        labels += [int(d["y0"][i]), int(d["y1"][i])]
        parents += [str(d["parent"][i])] * 2
        kinds += ["base", "endpoint"]
    Z = r0_batch(proc, net, imgs, batch)
    return {"z": Z, "y": np.array(labels), "parent": np.array(parents),
            "kind": np.array(kinds)}


def compdev_job(proc, net, batch, shard):
    si, sn = (int(x) for x in shard.split("/"))
    q = np.load(RT / "compdev.npz")
    nq = len(q["meta"])
    lo = (nq * si) // sn
    hi = (nq * (si + 1)) // sn
    imgs, labels, parents, kinds = [], [], [], []
    for qi in range(lo, hi):
        x = q[f"q{qi}"][0].reshape(4, 2)
        ea = q[f"q{qi}"][1].reshape(4, 2)
        eb = q[f"q{qi}"][2].reshape(4, 2)
        states = [x, x + ea, x + eb, x + ea + eb]
        for k, st in enumerate(states):
            imgs.append(render_canonical(st, np.random.default_rng(COMPDEV_NUIS + qi)))
            labels.append(int(q["meta"][qi][k]))
            parents.append(str(q["parent_id"][qi]))
            kinds.append(["base", "A", "B", "AB"][k])
    Z = r0_batch(proc, net, imgs, batch)
    return {"z": Z, "y": np.array(labels), "parent": np.array(parents),
            "kind": np.array(kinds)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", choices=["repair_train", "flipsup", "repair_dev",
                                     "compdev"], required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--shard", default="0/1")
    a = p.parse_args()
    lock = json.load(open(ART / "BRIDGE_R_PROTOCOL_LOCK.json"))
    torch.set_num_threads(8)
    proc, net, cfg = load_pinned(BB, lock)
    si, sn = (int(x) for x in a.shard.split("/"))
    if a.job == "compdev":
        blob = compdev_job(proc, net, a.batch, a.shard)
    else:
        src = {"repair_train": SEMD / "train.npz", "flipsup": RT / "flipsup.npz",
               "repair_dev": SEMD / "dev.npz"}[a.job]
        d = np.load(src)
        n = len(d["y0"])
        lo = (n * si) // sn
        hi = (n * (si + 1)) // sn
        sub = {k: d[k][lo:hi] for k in
               ("x", "e", "y0", "y1", "parent", "nuis")}
        tmp = RT / f"_tmp_{a.job}_{si}.npz"
        np.savez(tmp, **sub)
        blob = single_job(proc, net, a.job, tmp, a.batch)
        tmp.unlink()
    import transformers
    tag = f"shard{si}of{sn}" if sn > 1 else "full"
    FEAT.mkdir(parents=True, exist_ok=True)
    out = FEAT / f"{a.job}.{tag}.npz"
    np.savez(out, **blob)
    buf = io.BytesIO()
    np.savez(buf, **blob)
    meta = {"backbone": BB, "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "transformers_version": transformers.__version__,
            "job": a.job, "shard": tag, "n_states": len(blob["y"]),
            "feature": "final-layer R0", "renderer": RENDER_CONFIG,
            "feature_sha256": hashlib.sha256(buf.getvalue()).hexdigest()}
    json.dump(meta, open(FEAT / f"{a.job}.{tag}.manifest.json", "w"), indent=1,
              default=str)
    print(f"DONE {a.job}[{tag}] n={len(blob['y'])} sha={meta['feature_sha256'][:12]}")


if __name__ == "__main__":
    main()
