#!/usr/bin/env python3
"""Data-bound formal CUDA runner for the four Route-2 image-only arms.

The runner validates the attached frozen bank before checking CUDA. Without a
GPU it writes ``DATA_READY_BLOCKED_GPU`` and an exact unrun list; it never
fabricates checkpoints, metrics or a scientific claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_generator import digest_array, file_sha256, validate
from visual_mechanism import Mechanism, prep

ROOT = Path(__file__).resolve().parents[3]
ARMS = ["direct", "additive", "representation", "interaction"]
SEEDS = [803, 805, 806]


def metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    correct = (logits > 0) == (labels > 0.5)
    atoms = correct[:, 1] & correct[:, 2]
    joint111 = atoms & correct[:, 3]  # A,B,AB states; P is reported separately
    return {
        "n_quartets": int(len(labels)),
        "P": float(correct[:, 0].mean()),
        "A": float(correct[:, 1].mean()),
        "B": float(correct[:, 2].mean()),
        "AB": float(correct[:, 3].mean()),
        "J3_atomic_joint": float(atoms.mean()),
        "atomic_joint": float(atoms.mean()),
        "J4_full_111": float(joint111.mean()),
        "atomic_denominator": int(atoms.sum()),
        "J3": float(joint111.mean()),
        "J4": float(joint111.mean()),
    }


def _parameter_counts(model: Mechanism) -> dict:
    head = sum(p.numel() for p in model.head.parameters())
    backbone = sum(p.numel() for p in model.backbone.parameters())
    return {"backbone": int(backbone), "head": int(head), "total": int(backbone + head)}


def _backbone_hash(model: Mechanism) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith("head."):
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _raw_image_batch(images, device):
    # Keep native RGB values for the model's visible-color masks. Normalization
    # belongs inside Mechanism.prep; passing a normalized tensor here corrupts
    # the red/blue mask contract.
    raw = torch.as_tensor(np.asarray(images, dtype=np.float32))
    return raw.to(device, non_blocking=True)


def _logits(model, images, batch_size, device):
    model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = _raw_image_batch(images[start:start + batch_size], device)
            output.append(model(x).detach().float().cpu().numpy())
    return np.concatenate(output) if output else np.empty((0,), np.float32)


def _train(data, arm: str, seed: int, config: dict, out: Path) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = Mechanism(arm, input_size=config["resolution"], pretrained=True).to(device)
    initial_backbone_sha256 = _backbone_hash(model)
    exposure = {
        "train_indices": np.arange(len(data["train_images"]), dtype=np.int64),
        "dev_indices": np.arange(len(data["dev_images"]), dtype=np.int64),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed + 100000)
    best_bce = float("inf"); best_epoch = 0; best_state = None; history = []
    train_labels = torch.from_numpy(data["train_labels"])
    dev_labels = torch.from_numpy(data["dev_labels"])
    started = time.monotonic()
    for epoch in range(config["epochs"]):
        model.train(); loss_sum = 0.0; seen = 0
        order = torch.randperm(len(train_labels), generator=generator).numpy()
        for start in range(0, len(order), config["batch_gpu"]):
            indices = order[start:start + config["batch_gpu"]]
            x = _raw_image_batch(data["train_images"][indices], device)
            y = train_labels[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(indices); seen += len(indices)
        dev_logits = _logits(model, data["dev_images"], config["batch_gpu"], device)
        dev_bce = float(F.binary_cross_entropy_with_logits(
            torch.from_numpy(dev_logits), dev_labels
        ))
        row = {"epoch": epoch + 1, "train_BCE": loss_sum / seen, "dev_BCE": dev_bce}
        history.append(row)
        if dev_bce < best_bce:
            best_bce = dev_bce; best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint_path = out / "model.pt"
    torch.save({
        "state_dict": best_state, "arm": arm, "seed": seed, "best_epoch": best_epoch,
        "input_size": config["resolution"], "selection": "singleton dev BCE only",
    }, checkpoint_path)

    quartet_images = data["quartet_images"].reshape(-1, *data["quartet_images"].shape[2:])
    quartet_logits = _logits(model, quartet_images, config["batch_gpu"], device).reshape(-1, 4)
    result_metrics = metrics(quartet_logits, data["quartet_labels"])
    result = {
        "arm": arm, "seed": seed, "best_epoch": best_epoch, "best_dev_BCE": best_bce,
        "metrics": result_metrics,
        "contrast_fields": {
            "full_repair": "MISSING_REASON: defined only in paired direct-versus-arm analysis",
            "migration": "MISSING_REASON: defined only in paired direct-versus-arm analysis",
            "111_regression": "MISSING_REASON: defined only in paired direct-versus-arm analysis",
        },
        "parameter_counts": _parameter_counts(model),
        "initial_shared_backbone_sha256": initial_backbone_sha256,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "train_seconds": time.monotonic() - started,
        "macs": {
            "train": "MISSING_REASON: no validated MAC profiler is pinned; runtime and exact parameter counts are recorded",
            "inference": "MISSING_REASON: no validated MAC profiler is pinned; one shared ResNet18 forward plus the declared head is fixed by architecture",
        },
        "exposure_indices": {
            "train_singleton_indices": exposure["train_indices"].tolist(),
            "train_singleton_indices_sha256": digest_array(exposure["train_indices"]),
            "dev_singleton_indices": exposure["dev_indices"].tolist(),
            "dev_singleton_indices_sha256": digest_array(exposure["dev_indices"]),
            "test_singleton_indices": list(range(len(data["test_images"]))),
            "test_singleton_indices_sha256": digest_array(np.arange(len(data["test_images"]), dtype=np.int64)),
            "test_quartet_flat_indices": list(range(len(quartet_images))),
            "test_quartet_flat_indices_sha256": digest_array(np.arange(len(quartet_images), dtype=np.int64)),
        },
        "input_contract": "identical native64 singleton/quartet images resized to 224 for all arms",
        "formal_claim": "results generated; scientific interpretation pending Route-2 review",
    }
    np.savez_compressed(out / "predictions.npz", logits=quartet_logits,
                        labels=data["quartet_labels"], parents=data["quartet_parents"])
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "exposure_indices.json").write_text(json.dumps(result["exposure_indices"], indent=2) + "\n")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arm": arm, "seed": seed, "best_epoch": best_epoch, "best_dev_BCE": best_bce}, indent=2), flush=True)
    return result


def _paired_contrasts(results: list[dict]) -> dict:
    by_key = {(result["arm"], result["seed"]): result for result in results}
    comparisons = {
        "direct": {
            "full_repair": "MISSING_REASON: direct is the reference and has no paired baseline",
            "migration": "MISSING_REASON: direct is the reference and has no paired baseline",
            "111_regression": "MISSING_REASON: direct is the reference and has no paired baseline",
        }
    }
    for arm in ARMS[1:]:
        rows = []
        for seed in SEEDS:
            baseline_path = Path(by_key[("direct", seed)]["prediction_path"])
            candidate_path = Path(by_key[(arm, seed)]["prediction_path"])
            with np.load(baseline_path) as baseline, np.load(candidate_path) as candidate:
                if not np.array_equal(baseline["labels"], candidate["labels"]):
                    raise ValueError("paired labels differ")
                if not np.array_equal(baseline["parents"], candidate["parents"]):
                    raise ValueError("paired parents differ")
                labels = candidate["labels"]
                base_correct = (baseline["logits"] > 0) == (labels > 0.5)
                cand_correct = (candidate["logits"] > 0) == (labels > 0.5)
            base_atoms = base_correct[:, 1] & base_correct[:, 2]
            cand_atoms = cand_correct[:, 1] & cand_correct[:, 2]
            base_joint = base_atoms & base_correct[:, 3]
            cand_joint = cand_atoms & cand_correct[:, 3]
            base110 = base_atoms & ~base_correct[:, 3]
            rows.append({
                "seed": seed,
                "baseline110_n": int(base110.sum()),
                "full_repair_n": int((base110 & cand_joint).sum()),
                "migration_n": int((base110 & cand_correct[:, 3] & ~cand_atoms).sum()),
                "direct_111_n": int(base_joint.sum()),
                "candidate_111_n": int(cand_joint.sum()),
                "111_regression_n": int((base_joint & ~cand_joint).sum()),
                "A_regression_n": int((base_correct[:, 1] & ~cand_correct[:, 1]).sum()),
                "B_regression_n": int((base_correct[:, 2] & ~cand_correct[:, 2]).sum()),
            })
        comparisons[f"{arm}_vs_direct"] = rows
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="explicit attached data directory or data.npz")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bundle = json.loads(args.manifest.read_text())
    if bundle.get("arms") != ARMS or bundle.get("seeds") != SEEDS:
        raise ValueError("manifest arms/seeds do not match the frozen runner contract")
    data_manifest = validate(args.data, args.manifest)
    archive = args.data if args.data.is_file() else args.data / "data.npz"
    args.out.mkdir(parents=True, exist_ok=True)
    exposure = {
        "train_singleton_indices": list(range(data_manifest["counts"]["train_singleton_images"])),
        "dev_singleton_indices": list(range(data_manifest["counts"]["dev_singleton_images"])),
        "test_singleton_indices": list(range(data_manifest["counts"]["test_singleton_images"])),
        "test_quartet_flat_indices": list(range(data_manifest["counts"]["quartet_images"])),
        "excluded_from_training": ["all dev labels except checkpoint selection", "all test labels", "all quartet labels/AB states"],
    }
    (args.out / "exposure_indices.json").write_text(json.dumps(exposure, indent=2) + "\n")

    unrun = [
        f"train/select/evaluate arm={arm} seed={seed}"
        for arm in ARMS for seed in SEEDS
    ] + ["aggregate J3/J4/A/B/AB/atomic/full-repair/migration/111-regression report"]
    if not torch.cuda.is_available():
        status = {
            "status": "DATA_READY_BLOCKED_GPU", "scientific_result": "NOT_RUN",
            "formal_claim": "none", "cuda_available": False,
            "data_path": str(args.data), "data_archive_sha256": data_manifest["archive"]["sha256"],
            "data_manifest_sha256": file_sha256(archive.with_name("data_manifest.json")),
            "unrun": unrun,
            "statement": "the image-only route is executable/data-ready and GPU-blocked; no visual effect is claimed",
        }
        (args.out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2)); return

    with np.load(archive, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    config = {key: bundle[key] for key in ("resolution", "batch_gpu", "epochs")}
    results = []
    for arm in ARMS:
        for seed in SEEDS:
            arm_out = args.out / f"{arm}_s{seed}"
            if arm_out.exists():
                if not args.overwrite:
                    raise FileExistsError(f"{arm_out} exists; use a new --out or --overwrite")
                shutil.rmtree(arm_out)
            arm_out.mkdir(parents=True)
            result = _train(data, arm, seed, config, arm_out)
            result["prediction_path"] = str((arm_out / "predictions.npz").relative_to(ROOT)) if arm_out.is_relative_to(ROOT) else str(arm_out / "predictions.npz")
            results.append(result)
    analysis = _paired_contrasts(results)
    status = {
        "status": "GPU_RUN_COMPLETE", "scientific_result": "COMPUTED_NOT_YET_INTERPRETED",
        "formal_claim": "none_pending_review", "cuda_available": True,
        "data_archive_sha256": data_manifest["archive"]["sha256"],
        "arms": ARMS, "seeds": SEEDS, "results": results, "regression_analysis": analysis,
    }
    (args.out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": status["status"], "results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
