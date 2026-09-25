#!/usr/bin/env python3
"""Validate attached fresh data and refresh the checkpoint-free GPU manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_generator import DEFAULT_OUT, file_sha256, validate

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT / "artifacts/e832_focus/route2/gpu_bundle"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    archive = args.data if args.data.is_file() else args.data / "data.npz"
    data_manifest_path = archive.with_name("data_manifest.json")
    data_manifest = validate(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    data_path_for_command = args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data
    resume = (
        "CUDA_VISIBLE_DEVICES=0 python experiments/e832_focus/route2_visual/gpu_run.py "
        f"--manifest {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}/manifest.json "
        f"--data {data_path_for_command} "
        "--out artifacts/e832_focus/route2/gpu_run"
    )
    manifest = {
        "status": "DATA_READY_BLOCKED_GPU" if not torch.cuda.is_available() else "DATA_READY_GPU_AVAILABLE",
        "scientific_result": "NOT_RUN",
        "formal_claim": "none",
        "checkpoint_files": [],
        "data": {
            "data_path": str(data_path_for_command),
            "archive_path": str((archive.relative_to(ROOT) if archive.is_relative_to(ROOT) else archive)),
            "archive_sha256": data_manifest["archive"]["sha256"],
            "data_manifest_path": str(data_manifest_path.relative_to(ROOT) if data_manifest_path.is_relative_to(ROOT) else data_manifest_path),
            "data_manifest_sha256": file_sha256(data_manifest_path),
            "split_parent_disjoint": data_manifest["split_disjointness"]["parent_sets_pairwise_disjoint"],
        },
        "encoder": "torchvision.models.resnet18(weights=IMAGENET1K_V1); one shared spatial backbone per arm/run",
        "renderer": data_manifest["observation_recipe"],
        "fresh_bank": {
            "parent_seeds": {s: data_manifest["frozen_seeds"][f"{s}_parent"] for s in ("train", "dev", "test")},
            "old_exposed_159_bank_used": False,
            "loaded_existing_scene_or_image_artifacts": False,
        },
        "arms": ["direct", "additive", "representation", "interaction"],
        "arm_contracts": {
            "direct": "global mean-pooled visible image feature; linear head",
            "additive": "red/blue masked pools passed separately through one shared linear head; outputs summed",
            "representation": "concatenated red/blue masked pools; linear head",
            "interaction": "concatenated red/blue masked pools; nonlinear 1024->128->1 head",
        },
        "seeds": [803, 805, 806],
        "train_contract": "singleton train images only (clean + single-edit); no AB/quartet labels; identical exposure for all arms",
        "selection_contract": "singleton dev BCE only",
        "evaluation_contract": "new test quartets; P,A,B,AB plus J3/J4 and paired full-repair/migration/111 regression",
        "metrics": ["P", "A", "B", "AB", "J3", "J4", "atomic_joint", "full_repair", "migration", "111_regression"],
        "counts": data_manifest["counts"],
        "exposure_contract": {
            "train_indices": f"all 0..{data_manifest['counts']['train_singleton_images'] - 1}; saved per run",
            "dev_indices": f"all 0..{data_manifest['counts']['dev_singleton_images'] - 1}; selection only",
            "test_indices": f"all 0..{data_manifest['counts']['test_singleton_images'] - 1}; never used for training/selection",
            "quartet_indices": f"all 0..{data_manifest['counts']['quartet_images'] - 1}; test only",
        },
        "parameter_counts": {
            "direct_additive_head": 513,
            "representation_head": 1025,
            "interaction_head": 131329,
            "backbone": "computed and recorded per completed run",
        },
        "macs": {
            "status": "MISSING_REASON",
            "reason": "no validated MAC profiler is pinned; completed runs record train/inference runtime, exact parameter counts, and the fixed shared-backbone architecture",
        },
        "batch_gpu": 32,
        "resolution": 224,
        "epochs": 20,
        "local_torch_cuda": bool(torch.cuda.is_available()),
        "resume_command": resume,
        "current_statement": "the image-only route is executable/data-ready and GPU-blocked; no visual effect is claimed",
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    status = {
        "status": manifest["status"],
        "cuda_available": bool(torch.cuda.is_available()),
        "manifest": str(manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT) else manifest_path),
        "data_archive_sha256": manifest["data"]["archive_sha256"],
        "data_manifest_sha256": manifest["data"]["data_manifest_sha256"],
        "resume_command": resume,
        "formal_claim": "none",
    }
    (args.out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
