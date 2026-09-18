#!/usr/bin/env python3
"""Bridge-R R3/R4: frozen feature extraction with official processors.

Backbones (pinned in BRIDGE_R_PROTOCOL_LOCK.json):
  - dinov2  : facebook/dinov2-base (Dinov2Model)
  - siglip2 : google/siglip2-large-patch16-384 (SiglipModel vision tower)

Readouts per backbone: R0 pooled/CLS, R1 mean patch tokens, R2 2x2-spatial
concat. All backbones fully frozen (eval, no_grad). Canonical 512 renders
with split nuisance seeds (train 100000+qi / dev 200000+qi / holdout
300000+qi); quartet four-states share one seed.

Holdout discipline: --split holdout requires --lock-sha matching
BRIDGE_R_PROTOCOL_LOCK.json `lock_sha` and refuses if HOLDOUT_USED.json
exists. The one-shot runner (run_holdout_once.py) owns holdout reads.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "bridge_r"))
from render_v1 import render_canonical, RENDER_CONFIG

ART = ROOT / "artifacts" / "bridge_r"
NUISANCE_BASE = {"train": 100000, "dev": 200000, "holdout": 300000}


def load_pinned(backbone, lock):
    from transformers import AutoImageProcessor, AutoModel
    cfg = lock["models"][backbone]
    kw = {"revision": cfg["model_revision"], "local_files_only": False}
    proc = AutoImageProcessor.from_pretrained(cfg["model_id"], **kw)
    net = AutoModel.from_pretrained(cfg["model_id"], **kw).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return proc, net, cfg


def vision_forward(net, backbone, batch):
    h = net.vision_model(**batch) if backbone == "siglip2" else net(**batch)
    return h


def readouts_from_hidden(h, backbone):
    """Return dict R0/R1/R2 from a model output. No grad.
    dinov2: CLS at position 0. siglip vision tower: no CLS token."""
    last = h.last_hidden_state  # [B, T, C]
    pool = h.pooler_output
    has_cls = (backbone == "dinov2")
    R0 = pool if pool is not None else last[:, 0]
    patches = last[:, 1:, :] if has_cls else last
    R1 = patches.mean(dim=1)
    # 2x2 spatial: infer grid (square); DINOv2 37x37 -> 18/19 split
    n = patches.shape[1]
    g = int(round(n ** 0.5))
    assert g * g == n, f"non-square patch grid: {n}"
    C = patches.shape[2]
    grid = patches.view(-1, g, g, C)
    h1 = g // 2
    blocks = [grid[:, :h1, :h1, :].mean(dim=(1, 2)),
              grid[:, :h1, h1:, :].mean(dim=(1, 2)),
              grid[:, h1:, :h1, :].mean(dim=(1, 2)),
              grid[:, h1:, h1:, :].mean(dim=(1, 2))]
    R2 = torch.cat(blocks, dim=1)
    names = {"R0_pooled": R0, "R1_mean_patch": R1, "R2_spatial2x2": R2}
    return {k: v.detach().cpu().numpy() for k, v in names.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "dev", "holdout"], required=True)
    p.add_argument("--backbone", choices=["dinov2", "siglip2"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lock-sha", default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--shard", default="0/1",
                   help="i/n contiguous quartet shard; merged post-hoc")
    p.add_argument("--limit", type=int, default=None,
                   help="dev/test only: max quartets (refused for holdout)")
    a = p.parse_args()
    lock = json.load(open(ART / "BRIDGE_R_PROTOCOL_LOCK.json"))
    if a.split == "holdout":
        if (ART / "HOLDOUT_USED.json").exists():
            raise SystemExit("FATAL: holdout already used (fail-closed).")
        if a.lock_sha != lock["lock_sha"]:
            raise SystemExit("FATAL: --lock-sha does not match protocol lock.")
        if a.limit is not None:
            raise SystemExit("FATAL: --limit refused for holdout.")
    si, sn = (int(x) for x in a.shard.split("/"))
    torch.set_num_threads(8)
    proc, net, cfg = load_pinned(a.backbone, lock)
    d = np.load(ART / f"quartets_{a.split}_v2.npz")
    nq_all = len(d["meta"])
    lo = (nq_all * si) // sn
    hi = (nq_all * (si + 1)) // sn
    if a.limit is not None:
        hi = min(hi, lo + a.limit)
    nq = hi - lo
    tag = f"shard{si}of{sn}" if sn > 1 else "full"
    feats = {}
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
            r = readouts_from_hidden(h, a.backbone)
            for k, v in r.items():
                feats.setdefault(k, []).append(v)
            print(f"FEAT {a.split}[{tag}] {min(s + a.batch, hi) - lo}/{nq}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    import transformers
    meta = {"backbone": a.backbone, "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "processor_revision": cfg.get("processor_revision", cfg["model_revision"]),
            "processor_class": type(proc).__name__,
            "processor_config": {k: getattr(proc, k, None) for k in
                                 ("size", "crop_size", "image_mean", "image_std",
                                  "do_resize", "do_center_crop", "do_rescale",
                                  "do_normalize", "resample")},
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "split": a.split, "n_quartets": nq, "shard": tag,
            "qid_range": [lo, hi],
            "renderer": RENDER_CONFIG,
            "nuisance_base": NUISANCE_BASE[a.split]}
    blobs = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    np.savez(a.out / f"feats_{a.backbone}_{a.split}.{tag}.npz",
             **blobs, kinds=np.tile(["base", "A", "B", "AB"], nq),
             labels=np.repeat(d["meta"][lo:hi], 1, axis=0).reshape(-1, 4)[
                 :, [0, 1, 2, 3]].reshape(-1),
             qid=np.repeat(np.arange(lo, hi), 4))
    # sha over feature bytes for the manifest
    import io
    buf = io.BytesIO()
    np.savez(buf, **blobs)
    meta["feature_sha256"] = hashlib.sha256(buf.getvalue()).hexdigest()
    json.dump(meta, open(a.out / f"feats_{a.backbone}_{a.split}.{tag}.manifest.json",
                         "w"), indent=1, default=str)
    print(f"DONE extract {a.backbone}/{a.split}[{tag}] sha={meta['feature_sha256'][:12]}")


if __name__ == "__main__":
    main()
