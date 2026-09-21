#!/usr/bin/env python3
"""Competence-first C0/C2: patch-grid extraction for repair-domain data.

Frozen DINOv3-L final-layer 14x14 patch tokens (CLS+registers excluded).
Renders IDENTICAL to repair-transfer R6 (same renderer v1a, same stored
nuisance seeds). Jobs: repair_train / flipsup / repair_dev (base+endpoint)
and compdev_abb (base/A/B ONLY; AB extracted only if C1 passes).
Grid binaries are local cache (gitignored); manifests commit.
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
FEAT = ART / "competence_first" / "feats"
BB = "dinov3l"
COMPDEV_NUIS = 800000


def grid_batch(proc, net, imgs, batch, n_register):
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), batch):
            b = proc(images=imgs[s:s + batch], return_tensors="pt")
            h = vision_forward(net, BB, b)
            last = h.last_hidden_state
            p = last[:, 1 + n_register:, :]
            assert p.shape[1] == 196, f"grid changed: {p.shape}"
            out.append(p.detach().cpu().numpy().astype(np.float32))
            print(f"  grid {s + len(p)}/{len(imgs)}", flush=True)
    return np.concatenate(out)


def save(tag, grids, labels, parents, kinds, meta_extra):
    FEAT.mkdir(parents=True, exist_ok=True)
    blob = {"patches": grids}
    np.savez(FEAT / f"grid_{tag}.npz", **blob,
             labels=np.array(labels), parent=np.array(parents),
             kind=np.array(kinds))
    buf = io.BytesIO()
    np.savez(buf, **blob)
    import transformers
    meta = {"backbone": BB, "tag": tag,
            "grid": "14x14 patch tokens only", "dtype": "float32",
            "renderer": RENDER_CONFIG,
            "transformers_version": transformers.__version__,
            "feature_sha256": hashlib.sha256(buf.getvalue()).hexdigest()}
    meta.update(meta_extra)
    json.dump(meta, open(FEAT / f"grid_{tag}.manifest.json", "w"), indent=1,
              default=str)
    print(f"DONE grid_{tag} shape={grids.shape} sha={meta['feature_sha256'][:12]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", choices=["repair_train", "flipsup", "repair_dev",
                                     "compdev_abb"], required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--shard", default="0/1")
    a = p.parse_args()
    lock = json.load(open(ART / "BRIDGE_R_PROTOCOL_LOCK.json"))
    torch.set_num_threads(8)
    proc, net, cfg = load_pinned(BB, lock)
    n_register = int(getattr(getattr(net, "config", None), "num_register_tokens", 0) or 0)
    si, sn = (int(x) for x in a.shard.split("/"))
    tagsh = f"shard{si}of{sn}" if sn > 1 else "full"
    if a.job == "compdev_abb":
        q = np.load(RT / "compdev.npz")
        nq = len(q["meta"])
        lo = (nq * si) // sn
        hi = (nq * (si + 1)) // sn
        imgs, labels, parents, kinds = [], [], [], []
        for qi in range(lo, hi):
            x = q[f"q{qi}"][0].reshape(4, 2)
            ea = q[f"q{qi}"][1].reshape(4, 2)
            eb = q[f"q{qi}"][2].reshape(4, 2)
            for k, st in enumerate((x, x + ea, x + eb)):
                imgs.append(render_canonical(
                    st, np.random.default_rng(COMPDEV_NUIS + qi)))
                labels.append(int(q["meta"][qi][k]))
                parents.append(str(q["parent_id"][qi]))
                kinds.append(["base", "A", "B"][k])
        G = grid_batch(proc, net, imgs, a.batch, n_register)
        save(f"{a.job}.{tagsh}", G, labels, parents, kinds,
             {"model_revision": cfg["model_revision"], "n_states": len(labels)})
    else:
        src = {"repair_train": SEMD / "train.npz", "flipsup": RT / "flipsup.npz",
               "repair_dev": SEMD / "dev.npz"}[a.job]
        d = np.load(src)
        n = len(d["y0"])
        lo = (n * si) // sn
        hi = (n * (si + 1)) // sn
        imgs, labels, parents, kinds = [], [], [], []
        for i in range(lo, hi):
            seed = int(d["nuis"][i])
            imgs.append(render_canonical(d["x"][i], np.random.default_rng(seed)))
            imgs.append(render_canonical(d["x"][i] + d["e"][i],
                                         np.random.default_rng(seed)))
            labels += [int(d["y0"][i]), int(d["y1"][i])]
            parents += [str(d["parent"][i])] * 2
            kinds += ["base", "endpoint"]
        G = grid_batch(proc, net, imgs, a.batch, n_register)
        save(f"{a.job}.{tagsh}", G, labels, parents, kinds,
             {"model_revision": cfg["model_revision"], "n_states": len(labels)})


if __name__ == "__main__":
    main()
