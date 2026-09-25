"""M1.0 source-G8 collision / near-collision search (CPU-only, no training).

Searches the frozen source bank for opposite-label scenes merged by the
existing G8 sort, with an independent crossing-oracle confirmation and an
equal-label near-neighbor control. A bounded local random search then tries to
reduce the G8 feature distance while keeping opposite labels and positive
geometric margins. Finding a merge is a capability lower bound; not finding an
exact merge does not prove representation completeness.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "mechanism_transfer_v3"))
sys.path.insert(0, str(ROOT / "docs" / "2c3e2d9_mechanism_transfer_pack" / "checks"))

from common import bank as bank_utils  # noqa: E402
from contracts import crossing  # noqa: E402

ART = ROOT / "artifacts" / "mechanism_transfer_v3" / "m1"
BANK = ROOT / "artifacts" / "e832_focus" / "structure" / "bank_quartets.npz"
EXPECTED_BANK_SHA = "15f5bf181b37dc9b4309582490f7fc11dfca8b03491251c3724fcb2e1bdd804c"
SEED = 832302
QUERY_CHUNK = 512
LOCAL_STEPS = 3000
LOCAL_START_SCALE = 0.05
LOCAL_END_SCALE = 0.002
COORD_BOUND = 1.0


def g8_sorted_features(x: np.ndarray) -> np.ndarray:
    """Existing G8 sort of the same eight source quantities (float32)."""
    p = np.asarray(x, dtype=np.float32).reshape(-1, 4, 2)
    d01 = np.linalg.norm(p[:, 0] - p[:, 1], axis=-1)
    d23 = np.linalg.norm(p[:, 2] - p[:, 3], axis=-1)
    within = np.sort(np.stack([d01, d23], axis=1), axis=1)
    cross = np.stack(
        [
            np.linalg.norm(p[:, 0] - p[:, 2], axis=-1),
            np.linalg.norm(p[:, 0] - p[:, 3], axis=-1),
            np.linalg.norm(p[:, 1] - p[:, 2], axis=-1),
            np.linalg.norm(p[:, 1] - p[:, 3], axis=-1),
        ],
        axis=1,
    )
    cross = np.sort(cross, axis=1)
    mid = np.linalg.norm(
        0.5 * (p[:, 0] + p[:, 1]) - 0.5 * (p[:, 2] + p[:, 3]), axis=-1, keepdims=True
    )
    u0 = p[:, 1] - p[:, 0]
    u1 = p[:, 3] - p[:, 2]
    direction = np.abs(u0[:, 0] * u1[:, 1] - u0[:, 1] * u1[:, 0]) / (
        np.linalg.norm(u0, axis=-1) * np.linalg.norm(u1, axis=-1) + 1e-8
    )
    return np.concatenate([within, cross, mid, direction[:, None]], axis=1).astype(
        np.float32
    )


def point_segment_distance(p, a, b) -> float:
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def crossing_margin(scene) -> float:
    s = np.asarray(scene, dtype=np.float64).reshape(4, 2)
    return min(
        point_segment_distance(s[2], s[0], s[1]),
        point_segment_distance(s[3], s[0], s[1]),
        point_segment_distance(s[0], s[2], s[3]),
        point_segment_distance(s[1], s[2], s[3]),
    )


def nearest_pairs(features: np.ndarray, labels: np.ndarray, same_label: bool):
    """Chunked nearest pair under the requested label relation."""
    n = len(features)
    best = (np.inf, -1, -1)
    f = features.astype(np.float32)
    for start in range(0, n, QUERY_CHUNK):
        block = f[start : start + QUERY_CHUNK]
        diff = block[:, None, :] - f[None, :, :]
        dist = np.abs(diff).max(axis=-1)
        rows = np.arange(start, min(start + QUERY_CHUNK, n))[:, None]
        cols = np.arange(n)[None, :]
        if same_label:
            keep = (labels[rows] == labels[cols]) & (cols > rows)
        else:
            keep = (labels[rows] != labels[cols]) & (cols > rows)
        dist = np.where(keep, dist, np.inf)
        flat = int(np.argmin(dist))
        i, j = divmod(flat, n)
        value = float(dist[i, j])
        if value < best[0]:
            best = (value, start + i, j)
    return {"distance": best[0], "index_a": int(best[1]), "index_b": int(best[2])}


def main() -> dict:
    blob = BANK.read_bytes()
    bank_sha = hashlib.sha256(blob).hexdigest()
    if bank_sha != EXPECTED_BANK_SHA:
        raise RuntimeError(f"source bank hash mismatch: {bank_sha}")
    data = np.load(BANK, allow_pickle=True)
    scenes = np.asarray(data["parent_x"], dtype=np.float32)
    labels = np.asarray(data["parent_y"]).astype(int)
    margins = np.asarray(data["parent_margin"], dtype=np.float64)
    usable = np.flatnonzero(np.isfinite(margins) & (margins > 0))
    if len(usable) < 2:
        raise RuntimeError("no positive-margin source scenes to search")
    sub = usable
    features = g8_sorted_features(scenes[sub])
    opposite = nearest_pairs(features, labels[sub], same_label=False)
    control = nearest_pairs(features, labels[sub], same_label=True)
    opp_a = int(sub[opposite["index_a"]])
    opp_b = int(sub[opposite["index_b"]])
    anchor = scenes[opp_a].astype(np.float64)
    anchor_label = bool(crossing(anchor))
    anchor_margin = crossing_margin(anchor)
    ref = scenes[opp_b].astype(np.float64)
    ref_label = bool(crossing(ref))
    ref_margin = crossing_margin(ref)
    start_distance = float(
        np.max(np.abs(features[opposite["index_a"]] - features[opposite["index_b"]]))
    )

    rng = np.random.default_rng(SEED)
    best = {
        "distance": start_distance,
        "scene": anchor,
        "label": anchor_label,
        "margin": anchor_margin,
        "movement_linf": 0.0,
        "step": -1,
    }
    ref_f = g8_sorted_features(ref[None])[0].astype(np.float64)
    for step in range(LOCAL_STEPS):
        frac = step / max(LOCAL_STEPS - 1, 1)
        scale = LOCAL_START_SCALE + frac * (LOCAL_END_SCALE - LOCAL_START_SCALE)
        candidate = anchor + rng.normal(0.0, scale, size=anchor.shape)
        candidate = np.clip(candidate, -COORD_BOUND, COORD_BOUND)
        label = bool(crossing(candidate))
        if label == ref_label:
            continue
        margin = crossing_margin(candidate)
        if not np.isfinite(margin) or margin <= 0.0:
            continue
        feat = g8_sorted_features(candidate[None])[0].astype(np.float64)
        distance = float(np.max(np.abs(feat - ref_f)))
        if distance < best["distance"]:
            best = {
                "distance": distance,
                "scene": candidate,
                "label": label,
                "margin": margin,
                "movement_linf": float(np.abs(candidate - anchor).max()),
                "step": int(step),
            }
    record = {
        "schema": "mechanism_transfer_v3.m1.source_search/1",
        "seed": SEED,
        "bank": "artifacts/e832_focus/structure/bank_quartets.npz",
        "bank_sha256": bank_sha,
        "n_bank_scenes": int(len(scenes)),
        "n_positive_margin_scenes": int(len(sub)),
        "nearest_opposite_label": {
            "g8_feature_distance": float(opposite["distance"]),
            "scene_a": opp_a,
            "scene_b": opp_b,
            "bank_label_a": int(labels[opp_a]),
            "bank_label_b": int(labels[opp_b]),
            "bank_margin_a": float(margins[opp_a]),
            "bank_margin_b": float(margins[opp_b]),
            "crossing_label_a": anchor_label,
            "crossing_label_b": ref_label,
            "crossing_margin_a": anchor_margin,
            "crossing_margin_b": ref_margin,
        },
        "nearest_equal_label_control": {
            "g8_feature_distance": float(control["distance"]),
            "scene_a": int(sub[control["index_a"]]),
            "scene_b": int(sub[control["index_b"]]),
        },
        "bounded_local_search": {
            "steps": LOCAL_STEPS,
            "start_scale": LOCAL_START_SCALE,
            "end_scale": LOCAL_END_SCALE,
            "coordinate_bound": COORD_BOUND,
            "start_distance": start_distance,
            "best_distance": float(best["distance"]),
            "best_margin": float(best["margin"]),
            "best_movement_linf": float(best["movement_linf"]),
            "best_step": int(best["step"]),
            "exact_merge_found": bool(best["distance"] == 0.0),
        },
        "provenance": bank_utils.provenance_record(
            name="source_g8_search",
            seed=SEED,
            inputs={"bank": "artifacts/e832_focus/structure/bank_quartets.npz"},
            parameters={
                "query_chunk": QUERY_CHUNK,
                "local_steps": LOCAL_STEPS,
                "coordinate_bound": COORD_BOUND,
            },
            outputs={"n_positive_margin_scenes": int(len(sub))},
        ),
        "non_claim": (
            "A zero local-search distance would be one constructed opposite-label "
            "merge. A positive best-observed distance is not a lower bound: this "
            "bounded random search only records the closest sampled candidate and "
            "does not rule out an exact or closer merge. Small near-neighbor distance "
            "alone is not indistinguishability."
        ),
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "source_search.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "nearest_opposite_label",
                    "nearest_equal_label_control",
                    "bounded_local_search",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return record


if __name__ == "__main__":
    main()
