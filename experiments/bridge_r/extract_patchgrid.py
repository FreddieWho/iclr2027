#!/usr/bin/env python3
"""Bridge-R Task 2 (Am03): save full patch grids for spatial readouts.

Re-forwards the frozen backbone on the IDENTICAL v2 train/dev renders
(same renderer v1a, same split nuisance seeds, same pinned model/processor
revisions). Saves per-image patch tokens [N,196,C] (CLS + registers
excluded) as float32 shards + manifest (sha/shape/dtype/revision/split/
renderer). Holdout is REFUSED outright.

Patchgrid npz files are local cache (gitignored); only manifests commit.
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
from extract_r import load_pinned, vision_forward, NUISANCE_BASE

ART = ROOT / "artifacts" / "bridge_r"
OUT = ART / "patchgrid"
ALLOW = ("dinov3s", "dinov3b", "dinov3l")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "dev"], required=True)
    p.add_argument("--backbone", choices=list(ALLOW), required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--shard", default="0/1")
    a = p.parse_args()
    lock = json.load(open(ART / "BRIDGE_R_PROTOCOL_LOCK.json"))
    si, sn = (int(x) for x in a.shard.split("/"))
    torch.set_num_threads(8)
    proc, net, cfg = load_pinned(a.backbone, lock)
    n_register = int(getattr(getattr(net, "config", None), "num_register_tokens", 0) or 0)
    d = np.load(ART / f"quartets_{a.split}_v2.npz")
    nq_all = len(d["meta"])
    lo = (nq_all * si) // sn
    hi = (nq_all * (si + 1)) // sn
    nq = hi - lo
    tag = f"shard{si}of{sn}" if sn > 1 else "full"
    grids = []
    with torch.no_grad():
        for s in range(lo, hi, a.batch):
            idx = range(s, min(s + a.batch, hi))
            imgs = []
            for qi in idx:
                qq = d[f"q{qi}"].reshape(3, 4, 2)
                x, ea, eb = qq[0], qq[1], qq[2]
                seed = NUISANCE_BASE[a.split] + int(qi)
                for st in (x, x + ea, x + eb, x + ea + eb):
                    imgs.append(render_canonical(
                        st, np.random.default_rng(seed)))
            batch = proc(images=imgs, return_tensors="pt")
            h = vision_forward(net, a.backbone, batch)
            last = h.last_hidden_state  # [B, 1+reg+196, C]
            patches = last[:, 1 + n_register:, :]
            assert patches.shape[1] == 196, f"grid changed: {patches.shape}"
            grids.append(patches.detach().cpu().numpy().astype(np.float32))
            print(f"PATCHGRID {a.split}[{tag}] {min(s + a.batch, hi) - lo}/{nq}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    import transformers
    blobs = {"patches": np.concatenate(grids)}
    np.savez(OUT / f"patchgrid_{a.backbone}_{a.split}.{tag}.npz",
             **blobs,
             kinds=np.tile(["base", "A", "B", "AB"], nq),
             labels=np.repeat(d["meta"][lo:hi], 1, axis=0).reshape(-1, 4)[
                 :, [0, 1, 2, 3]].reshape(-1),
             qid=np.repeat(np.arange(lo, hi), 4))
    buf = io.BytesIO()
    np.savez(buf, **blobs)
    meta = {"backbone": a.backbone, "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "processor_revision": cfg.get("processor_revision", cfg["model_revision"]),
            "processor_class": type(proc).__name__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "split": a.split, "n_quartets": nq, "shard": tag,
            "qid_range": [lo, hi],
            "grid": "14x14 (patch tokens only; CLS+registers excluded)",
            "shape": list(blobs["patches"].shape), "dtype": "float32",
            "renderer": RENDER_CONFIG,
            "nuisance_base": NUISANCE_BASE[a.split],
            "feature_sha256": hashlib.sha256(buf.getvalue()).hexdigest()}
    json.dump(meta, open(
        OUT / f"patchgrid_{a.backbone}_{a.split}.{tag}.manifest.json", "w"),
        indent=1, default=str)
    print(f"DONE patchgrid {a.backbone}/{a.split}[{tag}] sha={meta['feature_sha256'][:12]}")


if __name__ == "__main__":
    main()
