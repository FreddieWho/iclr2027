"""M1.1 zero-train reference evaluations (CPU-only, no training).

Evaluates the candidate task-sufficient references without fitting anything:
full-orbit lexicographic representatives on the T1 pair and the source
regroup witness, plus a finite-group-mean budget control with fixed random
weights. Group-mean invariance here is by construction and its forward cost
scales with group size; neither is an empirical discovery.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "mechanism_transfer_v3"))

from common import geom_features, source_typed  # noqa: E402

ART = ROOT / "artifacts" / "mechanism_transfer_v3" / "m1"
SEED = 832303
THREADS = 4


def group_mean_outputs(model, scene: np.ndarray, perms) -> dict:
    scene = np.asarray(scene, dtype=np.float32)
    outputs = []
    model.eval()
    with torch.no_grad():
        for perm in perms:
            x = torch.from_numpy(scene[list(perm)].reshape(-1).astype(np.float32))
            outputs.append(float(model(x).detach().cpu().item()))
    outputs = np.asarray(outputs, dtype=np.float64)
    return {
        "per_perm_outputs": [float(v) for v in outputs],
        "group_mean": float(outputs.mean()),
        "per_perm_std": float(outputs.std()),
        "n_forwards": int(len(perms)),
    }


def main() -> dict:
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    model = source_typed.RepairedSegmentRho(width=64, k=16)
    card = source_typed.model_card(model)
    a, b = geom_features.t1_collision_pair()
    rep_a = geom_features.t1_orbit_representative(a)
    rep_b = geom_features.t1_orbit_representative(b)
    mean_a = group_mean_outputs(model, a, geom_features.T1_PERMS)
    mean_b = group_mean_outputs(model, b, geom_features.T1_PERMS)
    x, xp = geom_features.source_regroup_witness()
    src_rep = geom_features.source_orbit_representative(x)
    src_rep_perm = geom_features.source_orbit_representative(xp)
    record = {
        "schema": "mechanism_transfer_v3.m1.reference_eval/1",
        "seed": SEED,
        "torch_threads": THREADS,
        "model_card": card,
        "t1_pair": {
            "whole_orbit_gap": float(np.max(np.abs(rep_a - rep_b))),
            "group_mean_a": mean_a["group_mean"],
            "group_mean_b": mean_b["group_mean"],
            "group_mean_gap": float(abs(mean_a["group_mean"] - mean_b["group_mean"])),
            "per_perm_std_a": mean_a["per_perm_std"],
            "per_perm_std_b": mean_b["per_perm_std"],
            "forwards_per_eval": mean_a["n_forwards"],
        },
        "source_regroup_witness": {
            "orbit_gap": float(np.max(np.abs(src_rep - src_rep_perm))),
            "separated": bool(
                np.max(np.abs(src_rep - src_rep_perm)) > 0.1
            ),
        },
        "non_claim": (
            "Zero-train group-mean invariance is constructed, and random-weight "
            "output gaps are not task evidence. The informative zero-train fact "
            "is that the whole-orbit references spare the T1 pair and the "
            "source regroup witness before any fitting."
        ),
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "reference_eval.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return record


if __name__ == "__main__":
    main()
