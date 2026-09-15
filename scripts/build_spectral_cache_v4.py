#!/usr/bin/env python3
"""Precompute exact spectral factors (evals + evecs of the normalized
Laplacian) for the IDSSE dev splits used by AMR v4.

Adjacency is the frozen knn rule on raw positions (same function as all
T5R work); the cache is aligned to split order so training batches index
it with the same snap_idx. Eval paths compute the same factors on the fly
(both exact, hence consistent); the cache exists only for training speed.

Outputs: artifacts/phase4_amr/spectral_cache_{train,valid}.npz with
  U (N,20,20) float32, evals (N,20) float32, split_lock_sha256, n.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
P4 = ROOT / "artifacts" / "phase4_amr"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    T5R3 = load_module(ROOT / "scripts" / "run_t5r3_sanity.py", "t5r3_for_speccache")
    split_lock = json.loads((T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    P4.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        data = T5R3.load_split(split, split_lock)
        adj = torch.from_numpy(T5R3.knn_adjacency_batch(data.raw)).float()
        deg = adj.sum(-1).clamp_min(1.0)
        inv = deg.pow(-0.5)
        lap = torch.eye(20) - adj * inv.unsqueeze(-1) * inv.unsqueeze(-2)
        evals_list, evecs_list = [], []
        with torch.no_grad():
            for start in range(0, len(adj), 512):
                ev, eu = torch.linalg.eigh(lap[start:start + 512])
                evals_list.append(ev.numpy().astype(np.float32))
                evecs_list.append(eu.numpy().astype(np.float32))
        evals = np.concatenate(evals_list)
        evecs = np.concatenate(evecs_list)
        out = P4 / f"spectral_cache_{split}.npz"
        np.savez_compressed(out, U=evecs, evals=evals,
                            split_lock_sha256=hashlib.sha256(
                                (T5R_ROOT / "split_lock.json").read_bytes()).hexdigest(),
                            n=len(adj))
        print(f"{split}: N={len(adj)} max_eig={evals.max():.4f} "
              f"sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
