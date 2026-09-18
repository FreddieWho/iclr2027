#!/usr/bin/env python3
"""Merge extract_r.py shards into one feats file (concat in qid order).

Verifies kinds/qid continuity and label alignment; writes final
feats_{bb}_{split}.npz + manifest (merged sha + shard shas). Deterministic:
shards differ from a full run only in floating-point batching, which is
exact here (per-image forward, no cross-image ops) -> bitwise identical.
"""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "bridge_r"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--dir", type=Path, default=ART / "features")
    a = p.parse_args()
    parts, manifests = [], []
    for i in range(a.n):
        tag = f"shard{i}of{a.n}"
        parts.append(np.load(a.dir / f"feats_{a.backbone}_{a.split}.{tag}.npz"))
        manifests.append(json.load(open(
            a.dir / f"feats_{a.backbone}_{a.split}.{tag}.manifest.json")))
    keys = [k for k in parts[0].keys()
            if k not in ("kinds", "labels", "qid")]
    blobs = {k: np.concatenate([pt[k] for pt in parts]).astype(np.float32)
             for k in keys}
    kinds = np.concatenate([pt["kinds"] for pt in parts])
    labels = np.concatenate([pt["labels"] for pt in parts])
    qid = np.concatenate([pt["qid"] for pt in parts])
    uq = np.unique(qid)
    assert (np.sort(qid).reshape(-1, 4)[:, 0] == uq).all(), "qid groups broken"
    order = np.argsort(qid, kind="stable")
    kinds, labels, qid = kinds[order], labels[order], qid[order]
    nq = len(uq)
    assert (kinds == np.tile(["base", "A", "B", "AB"], nq)).all()
    assert (qid == np.repeat(uq, 4)).all()
    np.savez(a.dir / f"feats_{a.backbone}_{a.split}.npz",
             **blobs, kinds=kinds, labels=labels, qid=qid)
    buf = io.BytesIO()
    np.savez(buf, **blobs)
    meta = dict(manifests[0])
    meta.update({"n_quartets": int(nq), "shard": f"merged-{a.n}",
                 "shard_shas": [m["feature_sha256"] for m in manifests],
                 "feature_sha256": hashlib.sha256(buf.getvalue()).hexdigest()})
    json.dump(meta, open(
        a.dir / f"feats_{a.backbone}_{a.split}.manifest.json", "w"), indent=1,
        default=str)
    print(f"MERGED {a.backbone}/{a.split} nq={meta['n_quartets']} "
          f"sha={meta['feature_sha256'][:12]}")


if __name__ == "__main__":
    main()
