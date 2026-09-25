"""Bank evaluation and repair-flow diagnostics for the M1.2 source matrix.

Readouts follow the frozen contract: ``J3`` requires A, B, and AB to be correct
on the same quartet; ``J4`` additionally requires P. Parents are the resampling
unit and each seed gets its own interval. The repair flow is always measured
against a fixed baseline's 110/111 cells, never against the candidate's own
clean arm, so ``R_endpoint = R_full + M`` is an identity rather than a
re-labelling.

Nothing here selects a checkpoint: evaluation consumes ``selection.json``,
which was written from singleton dev BCE only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _v3common import geom_features, metrics  # noqa: E402

from features import featurize  # noqa: E402
from recipe import (  # noqa: E402
    BANK_PATH,
    BOOT_N,
    BOOT_SEED,
    MARGIN_SLICE,
    ROOT,
    load_selected_model,
    sha256_file,
    sha256_array,
)


def rate_interval(values, parents, seed: int = BOOT_SEED, n_boot: int = BOOT_N) -> dict:
    """Row mean plus parent-cluster bootstrap interval for one indicator vector."""
    v = np.asarray(values, dtype=float).ravel()
    p = np.asarray(parents).ravel()
    if len(v) != len(p):
        raise ValueError("values and parents must have the same length")
    if len(v) == 0:
        return {
            "row_mean": None,
            "parent_mean": None,
            "ci95": None,
            "n": 0,
            "n_parents": 0,
        }
    if len(np.unique(p)) < 2:
        return {
            "row_mean": float(v.mean()),
            "parent_mean": None,
            "ci95": None,
            "n": int(len(v)),
            "n_parents": int(len(np.unique(p))),
            "note": "single-parent subset; no cluster interval",
        }
    interval = metrics.parent_cluster_ci(v, p, seed=seed, n_boot=n_boot)
    return {
        "row_mean": float(v.mean()),
        "parent_mean": interval["estimate"],
        "ci95": interval["ci95"],
        "n": interval["n"],
        "n_parents": interval["parents"],
        "unit": interval["unit"],
    }


def predict(model, arm: str, scenes, stats, batch: int = 2048) -> np.ndarray:
    features = torch.from_numpy(featurize(arm, scenes, stats))
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch):
            chunks.append(model(features[start : start + batch]).numpy())
    return np.concatenate(chunks) if chunks else np.zeros(0)


def score_quartets(model, arm: str, stats, bank) -> tuple[np.ndarray, np.ndarray]:
    """Logits and correctness for the four states of every frozen quartet."""
    states = bank["states"]  # (4, N, 4, 2) in P, A, B, AB order
    logits = np.stack([predict(model, arm, states[i], stats) for i in range(4)], axis=1)
    labels = np.stack(
        [bank["y0"], bank["y0"], bank["y0"], bank["yab"]], axis=1
    ).astype(np.float64)
    correct = (logits > 0) == (labels > 0.5)
    return logits, correct


def g8_swing(model, arm: str, stats, bank, n_states: int = 64) -> dict:
    """Structure-invariance diagnostic on real A states; not eight replicates."""
    sample = bank["states"][1, :n_states]
    base = predict(model, arm, sample, stats)
    swings = []
    for perm in geom_features.SOURCE_PERMS[1:]:
        moved = np.asarray(sample, dtype=np.float32)[:, list(perm), :]
        swings.append(float(np.max(np.abs(predict(model, arm, moved, stats) - base))))
    return {
        "max_abs_logit_swing": float(max(swings)) if swings else 0.0,
        "n_states": int(len(sample)),
        "n_group_elements_applied": len(swings),
        "note": (
            "diagnostic on real A states; the group elements are not replicates "
            "and zero swing is by construction for a strictly invariant arm"
        ),
    }


def arm_metrics(correct: np.ndarray, parents, margins) -> dict:
    """J3/J4 and marginals with parent-cluster intervals."""
    ok_p, ok_a, ok_b, ok_ab = (correct[:, i] for i in range(4))
    j3 = ok_a & ok_b & ok_ab
    j4 = ok_p & j3
    atomic = ok_a & ok_b
    high = np.asarray(margins)[:, 1:].min(axis=1) >= MARGIN_SLICE
    return {
        "J3": rate_interval(j3, parents),
        "J4": rate_interval(j4, parents),
        "P_accuracy": rate_interval(ok_p, parents),
        "A_accuracy": rate_interval(ok_a, parents),
        "B_accuracy": rate_interval(ok_b, parents),
        "AB_accuracy": rate_interval(ok_ab, parents),
        "atomic_joint": rate_interval(atomic, parents),
        "CCM_denominator": int(atomic.sum()),
        "CCM": float(j3[atomic].mean()) if atomic.any() else None,
        "J3_margin_ge_0.02": rate_interval(j3[high], np.asarray(parents)[high])
        if high.any()
        else None,
        "margin_slice_n": int(high.sum()),
        "margin_slice_parents": int(len(np.unique(np.asarray(parents)[high])))
        if high.any()
        else 0,
        "margin_quantiles_ma_mb_mab": [
            [float(v) for v in np.quantile(np.asarray(margins)[:, k], [0.1, 0.5, 0.9])]
            for k in (1, 2, 3)
        ],
    }, j3


def flow_from(base_correct: np.ndarray, candidate_correct: np.ndarray, parents) -> dict:
    """Repair flow conditional on the baseline's 110 and 111 cells.

    ``base_correct`` and ``candidate_correct`` are the A, B, AB columns.
    """
    b110 = base_correct[:, 0] & base_correct[:, 1] & ~base_correct[:, 2]
    b111 = base_correct.all(axis=1)
    if b110.any():
        endpoint = rate_interval(candidate_correct[b110, 2], np.asarray(parents)[b110])
        full = rate_interval(candidate_correct[b110].all(axis=1), np.asarray(parents)[b110])
        migration = rate_interval(
            candidate_correct[b110, 2] & ~candidate_correct[b110, :2].all(axis=1),
            np.asarray(parents)[b110],
        )
        residual = float(
            endpoint["row_mean"] - full["row_mean"] - migration["row_mean"]
        )
    else:
        endpoint = full = migration = {
            "row_mean": None,
            "parent_mean": None,
            "ci95": None,
            "n": 0,
            "n_parents": 0,
        }
        residual = None
    regression = (
        rate_interval(~candidate_correct[b111].all(axis=1), np.asarray(parents)[b111])
        if b111.any()
        else {"row_mean": None, "parent_mean": None, "ci95": None, "n": 0, "n_parents": 0}
    )
    return {
        "baseline110_n": int(b110.sum()),
        "baseline111_n": int(b111.sum()),
        "R_endpoint": endpoint,
        "R_full": full,
        "M": migration,
        "regression111": regression,
        "identity_residual": residual,
    }


CONTRASTS = (
    ("sorted_minus_unsorted", "rich8_sorted", "rich8_unsorted"),
    ("orbit_minus_sorted", "orbit_distance", "rich8_sorted"),
    ("orbit_minus_unsorted", "orbit_distance", "rich8_unsorted"),
    ("segment_moment_minus_sorted", "segment_moment", "rich8_sorted"),
    ("segment_moment_minus_orbit", "segment_moment", "orbit_distance"),
    ("repaired_rho_minus_sorted", "repaired_segment_rho", "rich8_sorted"),
    ("rich8_unsorted_minus_raw", "rich8_unsorted", "raw"),
    ("orbit_minus_raw", "orbit_distance", "raw"),
    ("segment_moment_minus_raw", "segment_moment", "raw"),
    ("repaired_rho_minus_raw", "repaired_segment_rho", "raw"),
)


def interval_sign(row: dict) -> int:
    if not row or row.get("ci95") is None:
        return 0
    low, high = row["ci95"]
    if low > 0:
        return 1
    if high < 0:
        return -1
    return 0


def evaluate(seeds, arms, tag: str = "3seed") -> dict:
    selection = json.loads((ROOT / "artifacts/mechanism_transfer_v3/m1/training/selection.json").read_text())
    selected = [row for row in selection if row["seed"] in seeds and row["arm"] in arms]
    if not selected:
        raise RuntimeError("no selected rows for the requested arms/seeds")
    bank = np.load(BANK_PATH, allow_pickle=True)
    parents = bank["parent_index"]
    margins = bank["margins"]
    steps = int(np.prod(bank["states"].shape[:2]))
    started = time.time()
    correct_map: dict[tuple, np.ndarray] = {}
    arms_table = []
    for row in selected:
        model, stats, input_dim, checkpoint = load_selected_model(row)
        infer_started = time.time()
        logits, correct = score_quartets(model, row["arm"], stats, bank)
        inference_seconds = time.time() - infer_started
        key = (row["arm"], row["seed"], row["supervision"])
        correct_map[key] = correct
        arm_record, j3 = arm_metrics(correct, parents, margins)
        arm_record.update(
            {
                "arm": row["arm"],
                "seed": int(row["seed"]),
                "supervision": row["supervision"],
                "lr": row["lr"],
                "n_parameters": row["n_parameters"],
                "macs_per_example": row["macs_per_example"],
                "train_state_forwards": row["train_state_forwards"],
                "train_forward_macs": row["train_forward_macs"],
                "train_seconds": row["seconds"],
                "final_train_loss": row["final_train_loss"],
                "dev_bce": row["dev_bce"],
                "in_dim": input_dim,
                "inference_states": steps,
                "inference_seconds": inference_seconds,
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "feature_schema": row["feature_schema"],
                "lr_grid_dev_bce": row.get("lr_grid_dev_bce"),
                "J3_denominator_quartets": int(len(parents)),
                "J3_denominator_parents": int(len(np.unique(parents))),
                "g8": g8_swing(model, row["arm"], stats, bank),
            }
        )
        arms_table.append(arm_record)
        np.savez_compressed(
            Path(row["checkpoint"]).parent / "quartet_correct.npz",
            logits=logits.astype(np.float32),
            correct=correct,
        )
    contrasts = []
    flows = []
    for seed in seeds:
        for supervision in ("clean", "flip"):
            available = {
                key[0]: correct_map[key]
                for key in correct_map
                if key[1] == seed and key[2] == supervision
            }
            for name, left, right in CONTRASTS:
                if left not in available or right not in available:
                    continue
                left_j3 = available[left][:, 1] & available[left][:, 2] & available[left][:, 3]
                right_j3 = available[right][:, 1] & available[right][:, 2] & available[right][:, 3]
                delta = rate_interval(left_j3.astype(float) - right_j3.astype(float), parents)
                delta.update(
                    {
                        "contrast": name,
                        "left": left,
                        "right": right,
                        "seed": int(seed),
                        "supervision": supervision,
                        "direction": interval_sign(delta),
                    }
                )
                contrasts.append(delta)
            # fixed baseline: raw clean, same seed
            if "raw" in available:
                base = np.stack([available["raw"][:, i] for i in (1, 2, 3)], axis=1)
                for arm in arms:
                    if arm not in available:
                        continue
                    candidate = np.stack([available[arm][:, i] for i in (1, 2, 3)], axis=1)
                    flow = flow_from(base, candidate, parents)
                    flow.update(
                        {
                            "source": "raw_clean_same_seed",
                            "arm": arm,
                            "seed": int(seed),
                            "supervision": supervision,
                        }
                    )
                    flows.append(flow)
            if supervision == "flip":
                for arm in arms:
                    own_clean = correct_map.get((arm, seed, "clean"))
                    if own_clean is None or arm not in available:
                        continue
                    own_base = np.stack([own_clean[:, i] for i in (1, 2, 3)], axis=1)
                    candidate = np.stack([available[arm][:, i] for i in (1, 2, 3)], axis=1)
                    flow = flow_from(own_base, candidate, parents)
                    flow.update(
                        {
                            "source": "within_arm_clean",
                            "arm": arm,
                            "seed": int(seed),
                            "supervision": "flip",
                        }
                    )
                    flows.append(flow)
    interactions = []
    for seed in seeds:
        for name, left, right in CONTRASTS:
            keys = {(left, seed, s) for s in ("clean", "flip")} | {
                (right, seed, s) for s in ("clean", "flip")
            }
            if any(key not in correct_map for key in keys):
                continue

            def j3_of(arm: str, supervision: str) -> np.ndarray:
                ok = correct_map[(arm, seed, supervision)]
                return (ok[:, 1] & ok[:, 2] & ok[:, 3]).astype(float)

            delta = (j3_of(left, "flip") - j3_of(left, "clean")) - (
                j3_of(right, "flip") - j3_of(right, "clean")
            )
            row = rate_interval(delta, parents)
            row.update(
                {
                    "name": f"I_{name}",
                    "left": left,
                    "right": right,
                    "seed": int(seed),
                    "direction": interval_sign(row),
                    "note": (
                        "paired percentage-point interaction on the same E rows; "
                        "a representation x supervision readout, not a new threshold"
                    ),
                }
            )
            interactions.append(row)
    payload = {
        "schema": "mechanism_transfer_v3.m1.training_metrics/1",
        "tag": tag,
        "bank": str(BANK_PATH.relative_to(ROOT)),
        "bank_sha256": sha256_file(BANK_PATH),
        "n_quartets": int(len(parents)),
        "n_parents": int(len(np.unique(parents))),
        "seeds": [int(s) for s in seeds],
        "arms": arms_table,
        "contrasts": contrasts,
        "flows": flows,
        "supervision_interactions": interactions,
        "selection_rule": "final pooled static+single dev BCE; smaller lr wins ties; test J not read",
        "decision_rule": "logit > 0",
        "eval_seconds": time.time() - started,
        "group_elements_are_not_replicates": True,
        "seed_rule": "seeds are training replicates; per-seed intervals are not pooled",
    }
    (ROOT / "artifacts/mechanism_transfer_v3/m1/training" / f"metrics_{tag}.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n"
    )
    return payload
