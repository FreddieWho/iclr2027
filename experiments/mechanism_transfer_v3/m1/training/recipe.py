"""Frozen recipe and data loading for the M1.2 source matrix.

Every constant here is copied from the frozen contract in
``reports/e832_focus/STRUCTURE_DECISION.md`` (300 full-batch Adam steps; lr
0.01/0.003/0.001; pooled static+single dev BCE selection; 4 threads; clean or
clean+single-flip supervision from the frozen O01 mining artifacts). The bank,
the training scenes, the dev splits, and the mined flips are read-only inputs
identified by SHA-256 in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from features import ARMS, featurize, feature_schema, fit_stats  # noqa: E402
from models import build_model, macs_per_example, model_card, state_digest  # noqa: E402

ROOT = HERE.parents[3]

TRAIN_PATH = ROOT / "artifacts" / "f095_campaign" / "D01" / "scenes_N" / "scenes.npz"
DEV_PATH = ROOT / "artifacts" / "e832_focus" / "structure" / "dev_single.npz"
FLIPS_PATH = ROOT / "artifacts" / "e832_focus" / "structure" / "train_flips.npz"
BANK_PATH = ROOT / "artifacts" / "e832_focus" / "structure" / "bank_quartets.npz"
BANK_SHA256 = "15f5bf181b37dc9b4309582490f7fc11dfca8b03491251c3724fcb2e1bdd804c"

ART = ROOT / "artifacts" / "mechanism_transfer_v3" / "m1" / "training"
RUNS = ART / "runs"

EPOCHS = 300
LRS = (0.01, 0.003, 0.001)
SEEDS = (11, 23, 47)
SUPERVISIONS = ("clean", "flip")
THREADS = 4
DEV_EVAL_EVERY = 10
BOOT_SEED = 924
BOOT_N = 2000
MARGIN_SLICE = 0.02

# Role-preserving references are pre-stated; the lossy controls are retained so
# the matrix can separate "role information destroyed" from "harder to fit".
ROLE_PRESERVING_REFERENCES = ("orbit_distance", "segment_moment")
LOSSY_CONTROLS = ("rich8_sorted", "rich8_unsorted")

NON_CLAIMS = (
    "This matrix is the source task only. Nothing here licenses cross-task "
    "transfer; the T1/T2 comparison of any of these representations is a "
    "separate experiment that has not been run in v3.",
    "The group-mean reference in artifacts/mechanism_transfer_v3/m1/reference_eval.json "
    "is a zero-train control. It is not trained here and its exact invariance "
    "is constructed.",
    "Seeds are training replicates, not independent datasets; per-seed parent "
    "intervals are reported and never pooled into one interval.",
    "The frozen bank is a development bank already used by earlier reports; "
    "this is a representation comparison on it, not a fresh confirmation.",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values)).tobytes()).hexdigest()


def load_data() -> dict:
    """Load the frozen train scenes, dev split, mined flips, and claim bank."""
    train = np.load(TRAIN_PATH)
    dev = np.load(DEV_PATH)
    flips = np.load(FLIPS_PATH)
    bank = np.load(BANK_PATH, allow_pickle=True)
    bank_sha = sha256_file(BANK_PATH)
    if bank_sha != BANK_SHA256:
        raise RuntimeError(f"bank hash mismatch: {bank_sha} != {BANK_SHA256}")
    data = {
        "train_positions": train["positions"].astype(np.float32),
        "train_labels": train["labels"].astype(np.int64),
        "train_oracle_margin": train["oracle_margin"].astype(np.float64),
        "dev_positions": dev["positions"].astype(np.float32),
        "dev_labels": dev["labels"].astype(np.int64),
        "dev_single_states": dev["single_states"].astype(np.float32),
        "dev_single_labels": dev["single_labels"].astype(np.int64),
        "dev_single_parents": dev["single_parents"].astype(np.int64),
        "flip_states": flips["states"].astype(np.float32),
        "flip_labels": flips["labels"].astype(np.int64),
        "flip_parents": flips["parents"].astype(np.int64),
        "bank": bank,
        "bank_sha256": bank_sha,
        "sources": {
            "train_scenes": {
                "path": str(TRAIN_PATH.relative_to(ROOT)),
                "sha256": sha256_file(TRAIN_PATH),
                "n_scenes": int(len(train["positions"])),
            },
            "dev_single": {
                "path": str(DEV_PATH.relative_to(ROOT)),
                "sha256": sha256_file(DEV_PATH),
                "n_clean": int(len(dev["positions"])),
                "n_single_flip": int(len(dev["single_states"])),
            },
            "train_flips": {
                "path": str(FLIPS_PATH.relative_to(ROOT)),
                "sha256": sha256_file(FLIPS_PATH),
                "n_flips": int(len(flips["states"])),
            },
            "bank_quartets": {
                "path": str(BANK_PATH.relative_to(ROOT)),
                "sha256": bank_sha,
                "n_quartets": int(len(bank["parent_index"])),
                "n_eligible_parents": int(len(np.unique(bank["parent_index"]))),
            },
        },
    }
    return data


def _dev_tensors(arm: str, data: dict, stats: dict) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    clean = featurize(arm, data["dev_positions"], stats)
    flip = featurize(arm, data["dev_single_states"], stats)
    labels = np.concatenate([data["dev_labels"], data["dev_single_labels"]]).astype(np.float32)
    x = torch.from_numpy(np.concatenate([clean, flip], axis=0))
    return x, torch.from_numpy(labels), labels


def batch_contract(data: dict) -> dict:
    """Record the batch regime; full-batch training has no shuffled batch order."""
    return {
        "regime": "full_batch",
        "shuffle": False,
        "optimizer_steps_per_epoch": 1,
        "train_row_order_sha256": sha256_array(
            np.concatenate(
                [
                    data["train_positions"].reshape(len(data["train_positions"]), -1),
                    data["train_labels"][:, None].astype(np.float32),
                ],
                axis=1,
            )
        ),
        "flip_row_order_sha256": sha256_array(
            np.concatenate(
                [
                    data["flip_states"].reshape(len(data["flip_states"]), -1),
                    data["flip_labels"][:, None].astype(np.float32),
                ],
                axis=1,
            )
        ),
    }


def cell_dir(arm: str, seed: int, supervision: str, lr: float) -> Path:
    return RUNS / f"{arm}_s{seed}_{supervision}_lr{lr}"


def train_cell(
    arm: str,
    seed: int,
    supervision: str,
    lr: float,
    data: dict,
    force: bool = False,
) -> dict:
    """One frozen recipe cell: 300 full-batch Adam steps, dev-BCE selection only."""
    if arm not in ARMS:
        raise KeyError(arm)
    if supervision not in SUPERVISIONS:
        raise KeyError(supervision)
    dest = cell_dir(arm, seed, supervision, lr)
    receipt_path = dest / "receipt.json"
    if receipt_path.exists() and not force:
        return json.loads(receipt_path.read_text())
    dest.mkdir(parents=True, exist_ok=True)

    stats = fit_stats(arm, data["train_positions"])
    x_clean = torch.from_numpy(featurize(arm, data["train_positions"], stats))
    y_clean = torch.from_numpy(data["train_labels"].astype(np.float32))
    if supervision == "flip":
        x_flip = torch.from_numpy(featurize(arm, data["flip_states"], stats))
        y_flip = torch.from_numpy(data["flip_labels"].astype(np.float32))
    else:
        x_flip = y_flip = None

    dev_x, dev_y, dev_labels = _dev_tensors(arm, data, stats)

    torch.manual_seed(int(seed))
    model, input_dim = build_model(arm)
    init_digest = state_digest(model.state_dict())
    card = model_card(arm, model, input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_curve: list[float] = []
    dev_curve: list[dict] = []
    started = time.time()
    state_forwards = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_clean), y_clean)
        state_forwards += len(x_clean)
        if x_flip is not None:
            loss = loss + criterion(model(x_flip), y_flip)
            state_forwards += len(x_flip)
        loss.backward()
        optimizer.step()
        train_curve.append(float(loss.detach()))
        if epoch % DEV_EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                logits = model(dev_x)
                dev_curve.append(
                    {
                        "epoch": epoch,
                        "dev_bce": float(criterion(logits, dev_y)),
                        "dev_clean_bce": float(criterion(logits[: len(data["dev_labels"])], dev_y[: len(data["dev_labels"])])),
                        "dev_flip_bce": float(criterion(logits[len(data["dev_labels"]) :], dev_y[len(data["dev_labels"]) :])),
                    }
                )
    seconds = time.time() - started

    model.eval()
    with torch.no_grad():
        dev_logits = model(dev_x)
        dev_bce = float(criterion(dev_logits, dev_y))
        n_clean = len(data["dev_labels"])
        dev_clean_bce = float(criterion(dev_logits[:n_clean], dev_y[:n_clean]))
        dev_flip_bce = float(criterion(dev_logits[n_clean:], dev_y[n_clean:]))
        train_logits = model(x_clean)
        train_error = float(((train_logits > 0) != y_clean.bool()).float().mean())

    selected_by = "final pooled static+single dev BCE; smaller lr wins ties; test J is never read"
    ckpt_path = dest / "model.pt"
    torch.save(
        {
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "arm": arm,
            "seed": int(seed),
            "supervision": supervision,
            "lr": lr,
            "epochs": EPOCHS,
            "in_dim": input_dim,
            "stats": stats,
            "feature_schema": feature_schema(arm),
            "batch_contract": batch_contract(data),
            "init_state_digest": init_digest,
            "model_card": card,
            "train_curve": train_curve,
            "dev_curve": dev_curve,
            "selection": selected_by,
        },
        ckpt_path,
    )
    receipt = {
        "arm": arm,
        "seed": int(seed),
        "supervision": supervision,
        "lr": float(lr),
        "epochs": EPOCHS,
        "threads": THREADS,
        "dev_bce": dev_bce,
        "dev_bce_clean": dev_clean_bce,
        "dev_bce_flip": dev_flip_bce,
        "final_train_loss": train_curve[-1],
        "mean_train_loss": float(np.mean(train_curve)),
        "first_train_loss": train_curve[0],
        "final_train_error": train_error,
        "n_parameters": card["n_parameters"],
        "macs_per_example": card["macs_per_example"],
        "train_state_forwards": state_forwards,
        "train_forward_macs": card["macs_per_example"] * state_forwards,
        "seconds": seconds,
        "init_state_digest": init_digest,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "model_card": card,
        "feature_schema": feature_schema(arm),
        "stats_kind": stats["kind"],
        "batch_contract": batch_contract(data),
        "n_train_clean": int(len(x_clean)),
        "n_train_flip": int(len(x_flip)) if x_flip is not None else 0,
        "n_dev_clean": int(n_clean),
        "n_dev_flip": int(len(dev_y) - n_clean),
        "selection": selected_by,
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
    }
    (dest / "train_curve.json").write_text(json.dumps(train_curve) + "\n")
    (dest / "dev_curve.json").write_text(json.dumps(dev_curve, indent=2) + "\n")
    receipt_path.write_text(json.dumps(receipt, indent=2, default=float) + "\n")
    return receipt


def select(data: dict, arms=ARMS, seeds=SEEDS) -> list[dict]:
    """Pick the dev-BCE-minimal lr per arm x seed x supervision; never test J."""
    chosen = []
    for arm in arms:
        for seed in seeds:
            for supervision in SUPERVISIONS:
                candidates = []
                for lr in LRS:
                    path = cell_dir(arm, seed, supervision, lr) / "receipt.json"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    candidates.append(json.loads(path.read_text()))
                candidates.sort(key=lambda row: (row["dev_bce"], row["lr"]))
                pick = dict(candidates[0])
                pick["selection"] = (
                    "min final pooled static+single dev BCE; smaller lr wins ties; test J not read"
                )
                pick["lr_grid_dev_bce"] = {
                    str(row["lr"]): row["dev_bce"] for row in candidates
                }
                chosen.append(pick)
    (ART / "selection.json").write_text(json.dumps(chosen, indent=2, default=float) + "\n")
    return chosen


def load_selected_model(receipt: dict):
    checkpoint = torch.load(
        ROOT / receipt["checkpoint"], map_location="cpu", weights_only=False
    )
    model, input_dim = build_model(receipt["arm"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint["stats"], input_dim, checkpoint
