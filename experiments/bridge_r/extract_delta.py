#!/usr/bin/env python3
"""Semantic-delta M5: DINOv3-L final-layer R0 for base+endpoint.

Same renderer v1a / processor / revision / frozen backbone as Bridge-R.
Base+endpoint share one nuisance seed (fresh RNG per state, extract_r
convention -> identical backgrounds). Also stores pixel-summary stats
from the same renders (mean/max/var absdiff, changed fraction>0.02).

Splits train/dev only. Confirm features are extracted ONLY when the
one-shot confirm opens (discipline). Holdout never touched.
"""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r"))
from render_v1 import render_canonical, RENDER_CONFIG
from extract_r import load_pinned, vision_forward

ART = ROOT / "artifacts" / "bridge_r"
SEMD = ART / "semantic_delta"
FEAT = SEMD / "feats"
BB = "dinov3l"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "dev"], required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--shard", default="0/1")
    a = p.parse_args()
    lock = json.load(open(ART / "BRIDGE_R_PROTOCOL_LOCK.json"))
    si, sn = (int(x) for x in a.shard.split("/"))
    torch.set_num_threads(8)
    proc, net, cfg = load_pinned(BB, lock)
    d = np.load(SEMD / f"{a.split}.npz")
    n_all = len(d["flip"])
    lo = (n_all * si) // sn
    hi = (n_all * (si + 1)) // sn
    tag = f"shard{si}of{sn}" if sn > 1 else "full"
    Z0, Z1, PX = [], [], []
    with torch.no_grad():
        for s in range(lo, hi, a.batch):
            idx = range(s, min(s + a.batch, hi))
            imgs = []
            for i in idx:
                seed = int(d["nuis"][i])
                imgs.append(render_canonical(d["x"][i], np.random.default_rng(seed)))
                imgs.append(render_canonical(d["x"][i] + d["e"][i],
                                             np.random.default_rng(seed)))
            batch = proc(images=imgs, return_tensors="pt")
            h = vision_forward(net, BB, batch)
            last = h.last_hidden_state
            pool = h.pooler_output
            r0 = (pool if pool is not None else last[:, 0]).detach().cpu().numpy()
            Z0.append(r0[0::2])
            Z1.append(r0[1::2])
            arr = np.stack(imgs)
            D = np.abs(arr[1::2].astype(np.float64) - arr[0::2].astype(np.float64)).mean(axis=3)
            PX.append(np.stack([D.mean(axis=(1, 2)), D.max(axis=(1, 2)),
                                D.var(axis=(1, 2)), (D > 0.02).mean(axis=(1, 2))], axis=1))
            print(f"DELTAFEAT {a.split}[{tag}] {min(s + a.batch, hi) - lo}/{hi - lo}", flush=True)
    import transformers
    blobs = {"z0": np.concatenate(Z0).astype(np.float32),
             "z1": np.concatenate(Z1).astype(np.float32),
             "pix": np.concatenate(PX).astype(np.float32)}
    FEAT.mkdir(parents=True, exist_ok=True)
    np.savez(FEAT / f"delta_{a.split}.{tag}.npz", **blobs,
             flip=d["flip"][lo:hi], y0=d["y0"][lo:hi], y1=d["y1"][lo:hi],
             parent=d["parent"][lo:hi])
    buf = io.BytesIO()
    np.savez(buf, **blobs)
    meta = {"backbone": BB, "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "processor_class": type(proc).__name__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "split": a.split, "n_pairs": hi - lo, "shard": tag,
            "feature": "final-layer R0 (pooler/CLS per R0 def)",
            "renderer": RENDER_CONFIG,
            "feature_sha256": hashlib.sha256(buf.getvalue()).hexdigest()}
    json.dump(meta, open(FEAT / f"delta_{a.split}.{tag}.manifest.json", "w"),
              indent=1, default=str)
    print(f"DONE deltafeat {a.split}[{tag}] sha={meta['feature_sha256'][:12]}")


if __name__ == "__main__":
    main()
