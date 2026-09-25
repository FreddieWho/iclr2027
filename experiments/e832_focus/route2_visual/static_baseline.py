#!/usr/bin/env python3
"""Clean-only direct ResNet18 baseline for valid Route-2 flow accounting.

This is a contract-correction round, not a hyperparameter search. It uses the
same fresh data, seeds, pretrained initialization, 224 input, batch, and 20
epochs as the formal direct arm, but trains/selects on clean singleton images
only. Existing four-arm predictions are read separately for paired flow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_generator import file_sha256, validate
from visual_mechanism import Mechanism, prep

SEEDS = (803, 805, 806)


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def metrics(logits, labels):
    correct = (logits > 0) == (labels > 0.5)
    atoms = correct[:, 1] & correct[:, 2]
    joint = atoms & correct[:, 3]
    return {
        "n_quartets": int(len(labels)),
        "P": float(correct[:, 0].mean()),
        "A": float(correct[:, 1].mean()),
        "B": float(correct[:, 2].mean()),
        "AB": float(correct[:, 3].mean()),
        "atomic_joint": float(atoms.mean()),
        "J3": float(joint.mean()),
        "J4": float(joint.mean()),
    }


def backbone_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith("head."):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _raw_image_batch(images, device):
    raw = torch.as_tensor(np.asarray(images, dtype=np.float32))
    return raw.to(device, non_blocking=True)


def logits(model, images, batch, device):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            x = _raw_image_batch(images[start:start + batch], device)
            out.append(model(x).detach().float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,), np.float32)


def train_one(data, seed, resolution, batch, epochs, out, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_idx = np.flatnonzero(data["train_clean"])
    dev_idx = np.flatnonzero(data["dev_clean"])
    model = Mechanism("direct", input_size=resolution, pretrained=True).to(device)
    initial_sha = backbone_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed + 100000)
    best = float("inf"); best_epoch = 0; best_state = None; history = []
    started = time.monotonic()
    for epoch in range(epochs):
        model.train(); order = np.random.default_rng(seed + epoch).permutation(train_idx)
        total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            x = _raw_image_batch(data["train_images"][idx], device)
            y = torch.from_numpy(data["train_labels"][idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model(x)
            loss = F.binary_cross_entropy_with_logits(z, y)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(idx)
        dev_z = logits(model, data["dev_images"][dev_idx], batch, device)
        dev_bce = float(F.binary_cross_entropy_with_logits(
            torch.from_numpy(dev_z), torch.from_numpy(data["dev_labels"][dev_idx])))
        row = {"epoch": epoch + 1, "train_BCE": total / len(order), "dev_BCE": dev_bce}
        history.append(row)
        if dev_bce < best:
            best = dev_bce; best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no clean baseline checkpoint")
    model.load_state_dict(best_state)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = out / "model.pt"
    torch.save({"state_dict": best_state, "arm": "direct_static", "seed": seed,
                "best_epoch": best_epoch, "selection": "clean singleton dev BCE only",
                "input_size": resolution}, checkpoint)
    qimages = data["quartet_images"].reshape(-1, *data["quartet_images"].shape[2:])
    qlogits = logits(model, qimages, batch, device).reshape(-1, 4)
    result = {
        "arm": "direct_static", "seed": seed, "best_epoch": best_epoch,
        "best_dev_BCE": best, "metrics": metrics(qlogits, data["quartet_labels"]),
        "train_clean_indices": train_idx.tolist(),
        "train_clean_indices_sha256": digest_array(train_idx),
        "dev_clean_indices": dev_idx.tolist(),
        "dev_clean_indices_sha256": digest_array(dev_idx),
        "initial_shared_backbone_sha256": initial_sha,
        "checkpoint_sha256": file_sha256(checkpoint),
        "train_seconds": time.monotonic() - started,
        "input_contract": "same fresh images/resolution/seed; clean singleton train/dev only",
        "formal_claim": "clean-only baseline; flow interpretation pending review",
    }
    np.savez_compressed(out / "predictions.npz", logits=qlogits,
                        labels=data["quartet_labels"], parents=data["quartet_parents"])
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"arm": "direct_static", "seed": seed,
                      "best_epoch": best_epoch, "best_dev_BCE": best}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUIRED_FOR_STATIC_BASELINE")
    data_manifest = validate(args.data)
    archive = args.data if args.data.is_file() else args.data / "data.npz"
    with np.load(archive, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        result = train_one(data, seed, manifest["resolution"], manifest["batch_gpu"],
                           manifest["epochs"], args.out / f"direct_static_s{seed}", device)
        results.append(result)
    status = {"status": "STATIC_BASELINE_COMPLETE", "scientific_result": "COMPUTED_NOT_YET_INTERPRETED",
              "formal_claim": "none_pending_review", "data_archive_sha256": data_manifest["archive"]["sha256"],
              "seeds": list(SEEDS), "results": results}
    (args.out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": status["status"], "results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
