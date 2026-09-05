#!/usr/bin/env python3
"""T5R5-0 closure audit: per-match visible-valid geometry for locked models.

Reads ONLY already-visible train/valid artifacts. Refuses to run if the
reserved output directory already exists (ordering guard) and never opens
J03WQQ-derived task content.

For J03WN1 and J03WOY, seeds 11/23/47, and models {T5R3 fixed dual, T5R4
selected 2:1}, computes per-match natural geometry-latent Spearman, pair
ranking accuracy, field-zone F1 and centroid MAE, plus pooled-pair and
match-macro Spearman with pair counts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
T5R3_PATH = ROOT / "scripts" / "run_t5r3_sanity.py"
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
DEFAULT_OUTPUT = T5R_ROOT / "t5r5_prelock_audit_v1"

MODELS = {
    "fixed_dual_channel_shared_phase_gat": T5R_ROOT / "t5r3_sanity_v3" / "models" / "fixed_dual_channel_shared_phase_gat_seed{seed}.pt",
    "update_ratio_2to1": T5R_ROOT / "t5r4_round2_v1" / "models" / "update_ratio_2to1_seed{seed}.pt",
}
SEEDS = (11, 23, 47)
RESERVED_DIR = T5R_ROOT / "t5r5_reserved_j03wqq_v1"


def load_t5r3() -> Any:
    spec = importlib.util.spec_from_file_location("run_t5r3_sanity_for_prelock", T5R3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load T5R3 module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if RESERVED_DIR.exists():
        raise RuntimeError(f"refusing prelock audit: reserved output already exists at {RESERVED_DIR}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    T5R3 = load_t5r3()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    split_lock, _ = T5R3.load_lock()
    valid = T5R3.load_split("valid", split_lock)
    adjacency = T5R3.knn_adjacency_batch(valid.raw)
    matches = sorted(set(valid.match_ids.tolist()))
    started = time.perf_counter()
    output.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    with torch.inference_mode():
        for model_id, pattern in MODELS.items():
            for seed in SEEDS:
                checkpoint = pattern.with_name(pattern.name.format(seed=seed))
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing checkpoint {checkpoint}")
                model = T5R3.TaskModel()
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                model.eval()
                z_ctx = T5R3.encode_split(model, valid, adjacency, "raw", "team_mean")
                z_mode = T5R3.encode_split(model, valid, adjacency, "centered", "team_mean")
                phase_pred = model.context(torch.from_numpy(z_ctx))[0].argmax(-1).cpu().numpy()
                zone_pred = model.context(torch.from_numpy(z_ctx))[1].argmax(-1).cpu().numpy()
                centroid_pred = model.context(torch.from_numpy(z_ctx))[2].cpu().numpy()
                centroid_err = np.abs(centroid_pred - valid.centroids).mean(axis=1)
                margins = T5R3.pair_scores(valid, model, z_mode)
                correct = (margins > 0).astype(np.float32)
                latent_pos = np.linalg.norm(z_mode[valid.pair_query] - z_mode[valid.pair_positive], axis=1)
                latent_neg = np.linalg.norm(z_mode[valid.pair_query] - z_mode[valid.pair_negative], axis=1)
                per_match: dict[str, Any] = {}
                spearman_values: list[float] = []
                for match in matches:
                    snap_mask = valid.match_ids == match
                    pair_mask = valid.pair_matches == match
                    n_pairs = int(pair_mask.sum())
                    if n_pairs >= 2:
                        geo = np.concatenate([
                            valid.pair_positive_geometry[pair_mask],
                            valid.pair_negative_geometry[pair_mask],
                        ])
                        lat = np.concatenate([latent_pos[pair_mask], latent_neg[pair_mask]])
                        sp = T5R3.spearman(geo, lat)
                    else:
                        sp = float("nan")
                    if np.isfinite(sp):
                        spearman_values.append(float(sp))
                    per_match[match] = {
                        "n_snapshots": int(snap_mask.sum()),
                        "n_pairs": n_pairs,
                        "pair_accuracy": float(correct[pair_mask].mean()) if n_pairs else float("nan"),
                        "field_zone_f1": float(T5R3.f1_for_labels(valid.zone[snap_mask], zone_pred[snap_mask], 3)),
                        "match_half_f1_auxiliary": float(T5R3.f1_for_labels(valid.phase[snap_mask], phase_pred[snap_mask], 2)),
                        "centroid_mae": float(centroid_err[snap_mask].mean()),
                        "geometry_latent_spearman": float(sp),
                        "spearman_stable": bool(n_pairs >= 30),
                    }
                geo_all = np.concatenate([valid.pair_positive_geometry, valid.pair_negative_geometry])
                lat_all = np.concatenate([latent_pos, latent_neg])
                entries.append({
                    "model_id": model_id,
                    "seed": seed,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "per_match": per_match,
                    "pooled_pair_spearman": float(T5R3.spearman(geo_all, lat_all)),
                    "match_macro_spearman": float(np.mean(spearman_values)) if spearman_values else float("nan"),
                    "total_pairs": int(len(margins)),
                })
                print(f"prelock {model_id} seed={seed}: " + ", ".join(
                    f"{m} sp={per_match[m]['geometry_latent_spearman']:.4f} (n={per_match[m]['n_pairs']})"
                    for m in matches
                ), flush=True)
    write_json(output / "per_match_valid_geometry.json", entries)
    manifest = {
        "status": "T5R5_PRELOCK_AUDIT_COMPLETE",
        "phase": "P3_CAUSAL_MECHANISM",
        "task_lane": "P3-T5R5",
        "scope": "visible valid only; J03WQQ never opened",
        "models": sorted(MODELS),
        "seeds": list(SEEDS),
        "matches": matches,
        "reserved_output_existed": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
