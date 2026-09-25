"""Parent-aware bank accounting: dedup, sampling, coverage, and provenance.

These are coordinate-bank utilities only. They do not generate scenes, call
task oracles, train models, or read sealed pools.
"""
from __future__ import annotations

import hashlib

import numpy as np

SCHEMA = "mechanism_transfer_v3.bank/1"


def sha256_array(values: np.ndarray) -> str:
    blob = np.ascontiguousarray(np.asarray(values)).tobytes()
    return hashlib.sha256(blob).hexdigest()


def scene_linf(a: np.ndarray, b: np.ndarray) -> float:
    """Whole-scene L-inf distance: max over points and xy axes."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"scene shapes differ: {x.shape} != {y.shape}")
    return float(np.abs(x - y).max(axis=(-2, -1)).max())


def min_linf_to_reference(query, ref, q_block: int = 128, r_block: int = 512) -> np.ndarray:
    """Minimum whole-scene L-inf distance from each query to retained scenes."""
    q = np.asarray(query, dtype=np.float64).reshape(len(query), -1)
    r = np.asarray(ref, dtype=np.float64).reshape(len(ref), -1)
    if q_block < 1 or r_block < 1:
        raise ValueError("block sizes must be positive")
    out = np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), q_block):
        block = q[start : start + q_block]
        best = np.full(len(block), np.inf)
        for ref_start in range(0, len(r), r_block):
            chunk = r[ref_start : ref_start + r_block]
            diff = np.abs(block[:, None, :] - chunk[None, :, :]).max(axis=-1)
            best = np.minimum(best, diff.min(axis=1))
        out[start : start + q_block] = best
    return out


def greedy_dedup(scenes, threshold: float) -> tuple[np.ndarray, int]:
    """Keep later scenes only when whole-scene L-inf reaches the threshold."""
    flat = np.asarray(scenes, dtype=np.float64).reshape(len(scenes), -1)
    if not np.all(np.isfinite(flat)):
        raise ValueError("expected finite scenes")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    keep: list[int] = []
    kept = np.zeros((0, flat.shape[1]), dtype=np.float64)
    dropped = 0
    for i, scene in enumerate(flat):
        if len(kept):
            if float(np.abs(kept - scene).max(axis=1).min()) < threshold:
                dropped += 1
                continue
        keep.append(i)
        kept = flat[keep]
    return np.asarray(keep, dtype=np.int64), dropped


def split_parents_disjoint(parent_ids, seed: int, train_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Split whole parents into disjoint train/test index sets."""
    parents = np.asarray(parent_ids)
    unique = np.unique(parents)
    if len(unique) < 2:
        raise ValueError("need at least two parents for a disjoint split")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train fraction must be in (0,1)")
    order = np.random.default_rng(int(seed)).permutation(unique)
    cut = max(1, min(len(order) - 1, int(round(len(order) * train_fraction))))
    train = set(order[:cut].tolist())
    train_idx = np.flatnonzero([p in train for p in parents])
    test_idx = np.flatnonzero([p not in train for p in parents])
    return train_idx, test_idx


def stratified_sample_by_parent(
    parent_ids, seed: int, per_parent_cap: int | None = None, total_cap: int | None = None
) -> np.ndarray:
    """Sample from the full candidate list with an independent per-parent RNG."""
    parents = np.asarray(parent_ids)
    if per_parent_cap is not None and per_parent_cap < 1:
        raise ValueError("per-parent cap must be positive")
    if total_cap is not None and total_cap < 1:
        raise ValueError("total cap must be positive")
    chosen: list[int] = []
    for parent in np.unique(parents):
        group = np.flatnonzero(parents == parent)
        child = np.random.default_rng([int(seed), int(parent)])
        order = child.permutation(group)
        capped = order if per_parent_cap is None else order[:per_parent_cap]
        chosen.extend(capped.tolist())
    chosen_idx = np.asarray(chosen, dtype=np.int64)
    if total_cap is not None and len(chosen_idx) > total_cap:
        main = np.random.default_rng(int(seed))
        chosen_idx = main.permutation(chosen_idx)[:total_cap]
    return chosen_idx


def coverage_report(
    scenes, parent_ids, families=None, radii=None, margins=None
) -> dict:
    """Report realized coverage, not requested draw counts."""
    scenes = np.asarray(scenes)
    parents = np.asarray(parent_ids)
    if len(scenes) != len(parents):
        raise ValueError("scenes and parents must have the same length")
    unique, counts = np.unique(parents, return_counts=True)
    report = {
        "schema": SCHEMA,
        "n_scenes": int(len(scenes)),
        "n_parents": int(len(unique)),
        "scenes_sha256": sha256_array(scenes),
        "parents_sha256": sha256_array(parents),
        "min_scenes_per_parent": int(counts.min()),
        "max_scenes_per_parent": int(counts.max()),
    }
    if families is not None:
        fams = np.asarray(families)
        if len(fams) != len(scenes):
            raise ValueError("families must match scenes")
        fam_unique, fam_counts = np.unique(fams, return_counts=True)
        report["families"] = {
            str(name): int(count) for name, count in zip(fam_unique, fam_counts)
        }
    else:
        report["families"] = None
    for name, values in (("radii", radii), ("margins", margins)):
        if values is None:
            report[name] = None
            continue
        v = np.asarray(values, dtype=np.float64).ravel()
        if len(v) != len(scenes):
            raise ValueError(f"{name} must match scenes")
        report[name] = {
            "min": float(v.min()),
            "mean": float(v.mean()),
            "max": float(v.max()),
        }
    return report


def provenance_record(
    *,
    name: str,
    seed: int,
    inputs: dict,
    parameters: dict,
    outputs: dict,
) -> dict:
    """Fail-closed provenance: every sampling claim carries inputs and outputs."""
    if not name:
        raise ValueError("provenance needs a name")
    return {
        "schema": SCHEMA,
        "name": name,
        "seed": int(seed),
        "inputs": inputs,
        "parameters": parameters,
        "outputs": outputs,
        "code_module": __name__,
    }
