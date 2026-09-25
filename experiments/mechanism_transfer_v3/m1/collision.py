"""M1.0 T1 collision formalization (CPU-only, no training).

Formalizes the opposite-label T1 witness: legacy independently-sorted edge and
query-distance bags merge the pair, while unsorted and whole-orbit references
spare it. Production labels/margins come from the repository T1 oracle module
(pure NumPy, not a training entrypoint).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "mechanism_transfer_v3"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))

from common import geom_features  # noqa: E402
from u10_oracles import tri_oracle  # noqa: E402

ART = ROOT / "artifacts" / "mechanism_transfer_v3" / "m1"
SEED = 832301


def feature_gaps(a: np.ndarray, b: np.ndarray, dtype) -> dict:
    fa_sorted = geom_features.legacy_t1_features(
        np.asarray(a, dtype=dtype), sorted_features=True
    )
    fb_sorted = geom_features.legacy_t1_features(
        np.asarray(b, dtype=dtype), sorted_features=True
    )
    fa_raw = geom_features.legacy_t1_features(
        np.asarray(a, dtype=dtype), sorted_features=False
    )
    fb_raw = geom_features.legacy_t1_features(
        np.asarray(b, dtype=dtype), sorted_features=False
    )
    return {
        "sorted_gap": float(np.max(np.abs(fa_sorted - fb_sorted))),
        "sorted_bitwise_equal": bool(np.array_equal(fa_sorted, fb_sorted)),
        "unsorted_gap": float(np.max(np.abs(fa_raw - fb_raw))),
    }


def main() -> dict:
    a, b = geom_features.t1_collision_pair()
    gaps64 = feature_gaps(a, b, np.float64)
    gaps32 = feature_gaps(a, b, np.float32)
    oracle_a = tri_oracle(np.asarray(a, dtype=float), min_margin=0.0)
    oracle_b = tri_oracle(np.asarray(b, dtype=float), min_margin=0.0)
    oracle_a002 = tri_oracle(np.asarray(a, dtype=float), min_margin=0.02)
    oracle_b002 = tri_oracle(np.asarray(b, dtype=float), min_margin=0.02)
    rep_a = geom_features.t1_orbit_representative(a)
    rep_b = geom_features.t1_orbit_representative(b)
    record = {
        "schema": "mechanism_transfer_v3.m1.collision/1",
        "seed": SEED,
        "inside_coordinates": np.asarray(a, dtype=float).tolist(),
        "outside_coordinates": np.asarray(b, dtype=float).tolist(),
        "oracle": {
            "inside": {
                "label": int(oracle_a["label"]),
                "margin": float(oracle_a["margin"]),
                "ambiguous_at_0": bool(oracle_a["ambiguous"]),
                "ambiguous_at_0.02": bool(oracle_a002["ambiguous"]),
            },
            "outside": {
                "label": int(oracle_b["label"]),
                "margin": float(oracle_b["margin"]),
                "ambiguous_at_0": bool(oracle_b["ambiguous"]),
                "ambiguous_at_0.02": bool(oracle_b002["ambiguous"]),
            },
        },
        "legacy_sorted_gap_float64": gaps64["sorted_gap"],
        "legacy_sorted_gap_float32": gaps32["sorted_gap"],
        "legacy_sorted_bitwise_equal_float32": gaps32["sorted_bitwise_equal"],
        "legacy_unsorted_gap_float64": gaps64["unsorted_gap"],
        "legacy_unsorted_gap_float32": gaps32["unsorted_gap"],
        "whole_orbit_gap": float(np.max(np.abs(rep_a - rep_b))),
        "standardization_note": (
            "Any fixed per-feature affine standardization preserves exact "
            "equality: the sorted float32/float64 gap is 0 before scaling, so "
            "it remains 0 after scaling."
        ),
        "non_claim": (
            "This is one constructed opposite-label merge under the legacy T1 "
            "sorted bags. It does not estimate collision prevalence in any "
            "random bank and does not transfer to source G8 by itself."
        ),
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "collision_pair.npz").unlink(missing_ok=True)
    np.savez_compressed(
        ART / "collision_pair.npz",
        inside=np.asarray(a, dtype=np.float64),
        outside=np.asarray(b, dtype=np.float64),
    )
    (ART / "FEATURE_CONTRACT.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return record


if __name__ == "__main__":
    main()
