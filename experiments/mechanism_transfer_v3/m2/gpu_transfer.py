#!/usr/bin/env python3
"""M2 GPU transfer runner for the expanded visual bank.

Cells (one GPU queue, sequential):
  direct_full   ResNet backbone + linear head, BCE on clean+single labels.
  direct_clean  Same, trained on clean images only (valid clean baseline).
  true_geometry True orbit features -> frozen M1 head. No training.
  random_head   True orbit features -> random-init head of the same class.
  geom_front    ResNet backbone (frozen) + linear probe -> 6-dim orbit
                features with MSE geometry loss ONLY -> frozen M1 head.

The frozen head is the M1 orbit_distance CoordMLP. Its checkpoint file is
hash-verified against the recorded SHA before loading; orbit features are
standardized with the checkpoint's own source-train statistics (never refit).

Selection uses singleton dev only (BCE for direct, MSE for geom_front).
Test quartets are never read during training or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _v3common import bank as bank_mod
from _v3common import geom_features, metrics

SEEDS = (803, 805, 806)
EPOCHS = 20
BATCH = 32
LR = 3e-4

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

EXPECTED_HEAD_SHA256 = (
    "8b9c43f805f1d48acd8a2a4f1644d1ce41ff2804c4e8fd0b91865db189e3dde3"
)
HEAD_STATS_MU = np.array(
    [0.7137967, 0.43309657, 1.07712493, 0.84714421, 0.95921337, 1.15118548],
    dtype=np.float64,
)
HEAD_STATS_SD = np.array(
    [0.31143878, 0.23700958, 0.30871978, 0.30637289, 0.3694552, 0.32518385],
    dtype=np.float64,
)
ORBIT_COLUMNS = ("d01", "d02", "d03", "d12", "d13", "d23")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoordMLP(nn.Module):
    """Rebuild of experiments/discovery_campaign/coord_mlp.py::CoordMLP."""

    def __init__(self, hidden: int = 64, feat: int = 32, in_dim: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.feat_head = nn.Linear(hidden, feat)
        self.cls = nn.Linear(feat, 1)

    def forward(self, x):
        return self.cls(self.feat_head(self.net(x))).squeeze(-1)


class ImagePrep(nn.Module):
    """Native RGB in, resize + normalize inside the model (v2 contract)."""

    def __init__(self, size: int = 224):
        super().__init__()
        self.size = size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(torch.float32)
        if x.max() > 1.5:
            x = x / 255.0
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        mean = IMAGENET_MEAN.to(x.device, x.dtype)
        std = IMAGENET_STD.to(x.device, x.dtype)
        return (x - mean) / std


class DirectNet(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.prep = ImagePrep()
        self.backbone = backbone
        self.head = nn.Linear(512, 1)

    def forward(self, images):
        z = self.backbone(self.prep(images))
        pooled = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.head(pooled).squeeze(-1)


class GeomFront(nn.Module):
    """Frozen backbone + linear probe to orbit features; head stays frozen."""

    def __init__(self, backbone: nn.Module, freeze_backbone: bool = True):
        super().__init__()
        self.prep = ImagePrep()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone
        self.probe = nn.Linear(512, 6)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            # A frozen feature extractor includes its BatchNorm running state.
            self.backbone.eval()
        return self

    def forward(self, images):
        if self.freeze_backbone:
            with torch.no_grad():
                z = self.backbone(self.prep(images))
        else:
            z = self.backbone(self.prep(images))
        pooled = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.probe(pooled)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _backbone_hashes(backbone: nn.Module) -> dict:
    parameters = dict(backbone.named_parameters())
    buffers = dict(backbone.named_buffers())
    bn_buffers = {
        name: value
        for name, value in buffers.items()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    return {
        "parameters": {name: _tensor_sha256(value) for name, value in parameters.items()},
        "trainable_parameters": {
            name: _tensor_sha256(value)
            for name, value in parameters.items()
            if value.requires_grad
        },
        "buffers": {name: _tensor_sha256(value) for name, value in buffers.items()},
        "batchnorm_buffers": {
            name: _tensor_sha256(value) for name, value in bn_buffers.items()
        },
    }


def load_backbone(device: torch.device, pretrained: bool = True) -> nn.Module:
    import torchvision.models as models

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    trunk = nn.Sequential(*list(net.children())[:-2])
    return trunk.to(device)


def load_frozen_head(checkpoint: Path, expected_sha256: str, device: torch.device) -> CoordMLP:
    actual = sha256_file(checkpoint)
    if actual != expected_sha256:
        raise ValueError(f"frozen head SHA mismatch: {actual} != {expected_sha256}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head = CoordMLP(64, 32, 6)
    head.load_state_dict(payload["state"])
    stats_mu = np.asarray(payload["stats"]["mu"], dtype=np.float64)
    stats_sd = np.asarray(payload["stats"]["sd"], dtype=np.float64)
    if not (np.allclose(stats_mu, HEAD_STATS_MU) and np.allclose(stats_sd, HEAD_STATS_SD)):
        raise ValueError("checkpoint stats differ from the recorded head contract")
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head.to(device)


def standardize_orbit(raw: np.ndarray) -> np.ndarray:
    return ((raw - HEAD_STATS_MU) / HEAD_STATS_SD).astype(np.float32)


def orbit_targets(coords: np.ndarray) -> np.ndarray:
    """Raw 6-dim whole-orbit features for (n,4,2) scene coordinates."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 4, 2)
    out = np.empty((len(coords), 6), dtype=np.float64)
    for i in range(len(coords)):
        out[i] = geom_features.source_orbit_representative(coords[i])
    return out


def load_bank(bank_dir: Path) -> dict:
    manifest = json.loads((bank_dir / "data_manifest.json").read_text())
    archive = bank_dir / "data.npz"
    if sha256_file(archive) != manifest["archive"]["sha256"]:
        raise ValueError("bank archive SHA mismatch")
    data = dict(np.load(archive))
    return {"manifest": manifest, "data": data}


def singleton_split(data: dict, split: str, clean_only: bool = False):
    images = data[f"{split}_images"].astype(np.float32) / 255.0
    labels = data[f"{split}_labels"].astype(np.float32)
    parents = data[f"{split}_parents"].astype(np.int64)
    coords = data[f"{split}_parent_coords"][parents] + data[f"{split}_edit_vectors"]
    if clean_only:
        keep = data[f"{split}_clean"].astype(bool)
        images, labels, parents, coords = images[keep], labels[keep], parents[keep], coords[keep]
    return images, labels, parents, coords


def quartet_split(data: dict):
    images = data["quartet_images"].astype(np.float32) / 255.0
    labels = data["quartet_labels"].astype(np.float32)
    parents = data["quartet_parents"].astype(np.int64)
    base = data["test_parent_coords"][parents]
    states = np.stack(
        [
            base,
            base + data["quartet_edits_a"],
            base + data["quartet_edits_b"],
            base + data["quartet_edits_a"] + data["quartet_edits_b"],
        ],
        axis=1,
    )
    return images, labels, parents, states.reshape(-1, 4, 2)


def batched_predict(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), BATCH * 4):
            chunk = torch.from_numpy(images[i : i + BATCH * 4]).to(device)
            out.append(model(chunk).detach().cpu().numpy().ravel())
    return np.concatenate(out)


def train_loop(
    model: nn.Module,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    dev_images: np.ndarray,
    dev_labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    loss_kind: str,
    train_targets: np.ndarray | None = None,
    dev_targets: np.ndarray | None = None,
    epoch_size: int | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=LR)
    backbone_before = _backbone_hashes(model.backbone) if isinstance(model, GeomFront) else None
    probe_before = (
        {name: _tensor_sha256(value) for name, value in model.probe.named_parameters()}
        if isinstance(model, GeomFront)
        else None
    )
    backbone_first_batch = None
    best, best_state, history = float("inf"), None, []
    train_t = torch.from_numpy(train_labels).to(device)
    dev_t = torch.from_numpy(dev_labels).to(device)
    if train_targets is not None:
        train_targets = torch.from_numpy(train_targets).to(device)
        dev_targets = torch.from_numpy(dev_targets).to(device)
    n = len(train_images)
    exposure_size = n if epoch_size is None else int(epoch_size)
    if n < 1 or exposure_size < 1:
        raise ValueError("training requires at least one row and one row exposure per epoch")
    for epoch in range(epochs):
        model.train()
        if epoch == 0 and isinstance(model, GeomFront):
            backbone_training_mode = bool(model.backbone.training)
        if exposure_size <= n:
            perm = np.random.permutation(n)[:exposure_size]
        else:
            perm = np.random.randint(0, n, size=exposure_size)
        total, count = 0.0, 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i : i + BATCH]
            batch = torch.from_numpy(train_images[idx]).to(device)
            opt.zero_grad()
            pred = model(batch)
            if loss_kind == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, train_t[idx])
            else:
                loss = F.mse_loss(pred, train_targets[idx])
            loss.backward()
            if backbone_first_batch is None and isinstance(model, GeomFront):
                grads = [
                    param.grad.detach()
                    for param in model.backbone.parameters()
                    if param.grad is not None
                ]
                grad_abs_sum = sum(float(grad.abs().sum().cpu()) for grad in grads)
                grad_sq_sum = sum(float(grad.square().sum().cpu()) for grad in grads)
                backbone_first_batch = {
                    "gradient_tensor_count": len(grads),
                    "nonzero_gradient_tensor_count": sum(
                        int(bool(torch.count_nonzero(grad).item())) for grad in grads
                    ),
                    "gradient_abs_sum": grad_abs_sum,
                    "gradient_l2_norm": float(np.sqrt(grad_sq_sum)),
                }
                if not model.freeze_backbone and backbone_first_batch["gradient_l2_norm"] <= 0:
                    raise RuntimeError("unfrozen geometry frontend received no backbone gradient")
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        model.eval()
        with torch.no_grad():
            dev_pred = batched_predict(model, dev_images, device)
            if loss_kind == "bce":
                dev_loss = float(
                    F.binary_cross_entropy_with_logits(
                        torch.from_numpy(dev_pred).to(device), dev_t
                    )
                )
            else:
                dev_loss = float(np.mean((dev_pred.reshape(len(dev_images), -1) - dev_targets.cpu().numpy()) ** 2))
        history.append({"epoch": epoch, "train_loss": total / count, "dev_loss": dev_loss})
        if dev_loss < best:
            best = dev_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    result = {
        "best_dev_loss": best,
        "history": history,
        "train_unique_rows": int(n),
        "train_row_exposures_per_epoch": int(exposure_size),
        "train_row_exposures_total": int(exposure_size * epochs),
        "train_sampling": "with replacement" if exposure_size > n else "without replacement",
    }
    if isinstance(model, GeomFront):
        backbone_after = _backbone_hashes(model.backbone)
        result["backbone_update_audit"] = {
            "schema": "mechanism_transfer_v3.m2.backbone_update_audit/1",
            "freeze_backbone": model.freeze_backbone,
            "probe_parameters_before_training": probe_before,
            "training_mode_at_first_batch": backbone_training_mode,
            "first_batch_gradient": backbone_first_batch,
            "before_training": backbone_before,
            "after_selected_dev_checkpoint": backbone_after,
        }
    return result


def evaluate_states(
    logits: np.ndarray,
    labels: np.ndarray,
    parents: np.ndarray,
    n_quartets: int,
    seed: int,
) -> dict:
    logits = np.asarray(logits, dtype=np.float64).reshape(n_quartets, 4)
    labels = np.asarray(labels, dtype=np.float64).reshape(n_quartets, 4)
    record = metrics.joint_metrics(logits, labels, p_available=True)
    j3_correct = ((logits[:, 1:] > 0) == (labels[:, 1:] > 0.5)).all(axis=1).astype(float)
    row_ci = metrics.row_weighted_cluster_ci(j3_correct, parents, seed=seed)
    parent_ci = metrics.parent_cluster_ci(j3_correct, parents, seed=seed)
    record["J3_row_weighted_cluster_ci"] = row_ci
    record["J3_parent_equal_weight_ci"] = parent_ci
    # Retain the prior key for readers of old schemas; its estimand is explicit.
    record["J3_parent_cluster_ci"] = parent_ci
    record["n_quartets"] = n_quartets
    record["n_parents"] = len(np.unique(parents))
    return {
        **metrics.validate_table_record(
            {
                "n_quartets": n_quartets,
                "n_eligible_parents": len(np.unique(parents)),
                "n_sampled_parents": len(np.unique(parents)),
                "row_mean": record["J3"],
                "parent_mean": parent_ci["estimate"],
            }
        ),
        **record,
    }


def run_cell(
    *,
    mode: str,
    bank_dir: Path,
    head_checkpoint: Path | None,
    head_sha256: str,
    out_dir: Path,
    seed: int,
    epochs: int,
    device: torch.device,
    pretrained_backbone: bool = True,
    freeze_backbone: bool = True,
) -> dict:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    t0 = time.time()
    loaded = load_bank(bank_dir)
    data = loaded["data"]

    # Seed model construction as well as the epoch sampler; in particular the
    # geometry probe is initialized before train_loop resets its RNG.
    torch.manual_seed(seed)
    np.random.seed(seed)
    backbone = load_backbone(device, pretrained=pretrained_backbone)
    head = None
    if mode in ("true_geometry", "random_head", "geom_front"):
        if mode == "random_head":
            torch.manual_seed(seed)
            head = CoordMLP(64, 32, 6).to(device).eval()
        else:
            head = load_frozen_head(head_checkpoint, head_sha256, device)

    train_images, train_labels, _, train_coords = singleton_split(data, "train")
    dev_images, dev_labels, _, dev_coords = singleton_split(data, "dev")
    quartet_images, quartet_labels, quartet_parents, quartet_coords = quartet_split(data)

    result = {"mode": mode, "seed": seed, "epochs": epochs}
    if mode in ("direct_full", "direct_clean"):
        clean_only = mode == "direct_clean"
        full_train_rows = len(train_images)
        if clean_only:
            train_images, train_labels, _, _ = singleton_split(data, "train", clean_only=True)
        model = DirectNet(backbone).to(device)
        train_info = train_loop(
            model, train_images, train_labels, dev_images, dev_labels,
            seed=seed, epochs=epochs, device=device, loss_kind="bce",
            epoch_size=full_train_rows if clean_only else None,
        )
        result["train"] = {k: v for k, v in train_info.items() if k != "history"}
        result["history"] = train_info["history"]
        torch.save(model.state_dict(), out_dir / "model.pt")
        quartet_logits = batched_predict(model, quartet_images.reshape(-1, 3, 64, 64), device)
    elif mode in ("true_geometry", "random_head"):
        feats = standardize_orbit(orbit_targets(quartet_coords))
        with torch.no_grad():
            quartet_logits = head(torch.from_numpy(feats).to(device)).cpu().numpy()
        result["train"] = {"loss": "none: no training in this cell"}
    elif mode == "geom_front":
        train_orbit = standardize_orbit(orbit_targets(train_coords))
        dev_orbit = standardize_orbit(orbit_targets(dev_coords))
        model = GeomFront(backbone, freeze_backbone=freeze_backbone).to(device)
        train_info = train_loop(
            model, train_images, train_labels, dev_images, dev_labels,
            seed=seed, epochs=epochs, device=device, loss_kind="mse",
            train_targets=train_orbit, dev_targets=dev_orbit,
        )
        result["train"] = {
            k: v for k, v in train_info.items() if k not in ("history", "backbone_update_audit")
        }
        result["history"] = train_info["history"]
        torch.save(model.state_dict(), out_dir / "model_probe.pt")
        front_pred = batched_predict(model, quartet_images.reshape(-1, 3, 64, 64), device)
        front_pred = front_pred.reshape(-1, 6)
        with torch.no_grad():
            quartet_logits = head(torch.from_numpy(front_pred).to(device)).cpu().numpy()
        result["dev_geometry_mse"] = train_info["best_dev_loss"]
        result["freeze_backbone"] = freeze_backbone
        result["backbone_update_audit"] = train_info["backbone_update_audit"]
    else:
        raise ValueError(f"unknown mode {mode}")

    table = evaluate_states(quartet_logits, quartet_labels, quartet_parents, len(quartet_images), seed)
    result["quartet_table"] = table
    np.savez(
        out_dir / "predictions.npz",
        logits=quartet_logits,
        labels=quartet_labels,
        parents=quartet_parents,
    )
    result["seconds"] = time.time() - t0
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", required=True,
                        choices=["direct_full", "direct_clean", "true_geometry", "random_head", "geom_front"])
    parser.add_argument("--head-checkpoint", type=Path, default=None)
    parser.add_argument("--head-sha256", type=str, default=EXPECTED_HEAD_SHA256)
    parser.add_argument("--seed", type=int, default=803)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    args = parser.parse_args()
    if args.mode in ("true_geometry", "geom_front") and args.head_checkpoint is None:
        raise SystemExit("frozen modes require --head-checkpoint")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    device = torch.device("cuda:0")
    result = run_cell(
        mode=args.mode, bank_dir=args.bank, head_checkpoint=args.head_checkpoint,
        head_sha256=args.head_sha256, out_dir=args.out, seed=args.seed,
        epochs=args.epochs, device=device, pretrained_backbone=not args.no_pretrained,
        freeze_backbone=not args.unfreeze_backbone,
    )
    print(json.dumps(result["quartet_table"], indent=2))
    return 0


if __name__ == "__main__":
    main()
