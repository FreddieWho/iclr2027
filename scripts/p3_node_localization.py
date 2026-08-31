"""Node/support localization from the frozen normalized-embedding geometry.

The controlled intervention declares the displaced support.  This script
recomputes per-node Jacobian sensitivity from the selected causal-switch
checkpoints and tests whether the declared support is enriched among the top-k
nodes, where k is the declared support size.  It also retains a support-mass
sanity quantity.  It reads no response column and trains no predictor.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SWITCH_ROOT = ROOT / "artifacts/phase3/selected_causal_switch_v1"
PROSPECTIVE_ROOT = ROOT / "artifacts/phase3/support_geometry_prospective_v1"


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_array(value: Any) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.empty(0, dtype=int)
    parsed = json.loads(value) if isinstance(value, str) else value
    return np.asarray(parsed, dtype=int).reshape(-1)


def parse_delta(value: Any) -> np.ndarray:
    parsed = json.loads(value) if isinstance(value, str) else value
    delta = np.asarray(parsed, dtype=np.float32)
    if delta.shape != (20, 2) or not np.isfinite(delta).all():
        raise ValueError(f"invalid delta shape/value: {delta.shape}")
    return delta


def read_interventions(root: Path) -> pd.DataFrame:
    manifest = json.loads((PROSPECTIVE_ROOT / "intervention_manifest.json").read_text(encoding="utf-8"))
    columns = [
        "intervention_id",
        "valid",
        "matched_set_status",
        "delta",
        "source_support_indices",
        "target_support_indices",
    ]
    frames = []
    for shard in manifest["shards"]:
        frames.append(pd.read_parquet(root / str(shard["arms_path"]), columns=columns))
    interventions = pd.concat(frames, ignore_index=True)
    interventions = interventions[interventions["valid"].astype(bool)].copy()
    interventions = interventions[interventions["matched_set_status"].astype(str) == "COMPLETE"]
    return interventions.drop_duplicates("intervention_id").set_index("intervention_id")


def load_wrapper(root: Path, model_id: str):
    import p3_support_geometry as geometry

    seed = int(str(model_id).rsplit("seed", 1)[1])
    mode = "relational_pairwise" if "relational_pairwise" in model_id else "team_mean"
    base = geometry.load_frozen_model(root, f"phase_gat_seed{seed}")
    checkpoint = SWITCH_ROOT / "checkpoints" / f"{model_id}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    base.model.load_state_dict(state, strict=True)
    return geometry.FrozenGeometryModel(
        model_id=model_id,
        architecture="Phase-GAT",
        model=copy.deepcopy(base.model),
        uses_adjacency=True,
        pooling_mode=mode,
    )


def run(root: Path, config_sha256: str) -> dict[str, object]:
    import sys

    sys.path.insert(0, str(root / "scripts"))
    import p3_support_geometry as geometry

    arrays = geometry.load_arrays(root)
    positions = np.asarray(arrays["positions"], dtype=np.float32)
    teams = np.asarray(arrays["team_slots"], dtype=np.int64)
    adjacency = np.asarray(arrays["adjacency"], dtype=np.float32)
    arm_columns = [
        "model_id",
        "sample_index",
        "matched_set_id",
        "intervention_id",
        "source_match_id",
        "source_sample_id",
        "split",
        "graph_id",
        "coalition_id",
        "team_slot",
        "role_group",
        "epsilon",
        "fracture_draw",
        "arm_kind",
        "control_family",
        "matched_set_status",
        "support_size",
        "q_full",
    ]
    arms = pd.read_parquet(SWITCH_ROOT / "switch_arm_geometry.parquet", columns=arm_columns)
    arms = arms[arms["matched_set_status"].astype(str) == "COMPLETE"].copy()
    interventions = read_interventions(root)
    models = sorted(arms["model_id"].astype(str).unique())
    records: list[dict[str, object]] = []
    parity_errors: list[float] = []
    for model_id in models:
        wrapper = load_wrapper(root, model_id)
        model_arms = arms[arms["model_id"].astype(str) == model_id]
        for sample_index, group in model_arms.groupby("sample_index", sort=True):
            _, jacobian = geometry.jacobian_for_layer(
                wrapper,
                positions[int(sample_index)],
                teams[int(sample_index)],
                adjacency[int(sample_index)],
                "pooled_embedding",
            )
            output_dim = jacobian.shape[0]
            jacobian_blocks = jacobian.reshape(output_dim, 20, 2)
            for row in group.itertuples(index=False):
                intervention = interventions.loc[str(row.intervention_id)]
                delta = parse_delta(intervention["delta"])
                source = json_array(intervention["source_support_indices"])
                target = json_array(intervention["target_support_indices"])
                support = target if len(target) else source
                node_vectors = (jacobian_blocks * delta[None, :, :]).sum(axis=2)
                total = node_vectors.sum(axis=1)
                inside = node_vectors[:, support].sum(axis=1) if len(support) else np.zeros(output_dim)
                q_full = 0.5 * float(np.dot(total, total))
                q_inside = 0.5 * float(np.dot(inside, inside))
                observed_q = float(row.q_full)
                parity_errors.append(abs(q_full - observed_q))
                share = q_inside / q_full if q_full > 1e-15 else np.nan
                uniform = float(len(np.unique(support))) / 20.0
                node_sensitivity = 0.5 * np.sum(jacobian_blocks**2, axis=(0, 2))
                ranking = np.argsort(-node_sensitivity, kind="stable")
                k = min(len(np.unique(support)), 20)
                support_set = set(int(index) for index in np.unique(support))
                top_k = set(int(index) for index in ranking[:k])
                hits = len(top_k & support_set)
                ranks = [int(np.flatnonzero(ranking == index)[0]) + 1 for index in support_set]
                topk_recall = hits / k if k else np.nan
                mean_rank = float(np.mean(ranks)) if ranks else np.nan
                records.append({
                    "model_id": model_id,
                    "pooling_mode": wrapper.pooling_mode,
                    "sample_index": int(row.sample_index),
                    "matched_set_id": str(row.matched_set_id),
                    "intervention_id": str(row.intervention_id),
                    "source_match_id": str(row.source_match_id),
                    "source_sample_id": str(row.source_sample_id),
                    "split": str(row.split),
                    "graph_id": str(row.graph_id),
                    "coalition_id": str(row.coalition_id),
                    "team_slot": int(row.team_slot),
                    "role_group": str(row.role_group),
                    "epsilon": float(row.epsilon),
                    "fracture_draw": int(row.fracture_draw),
                    "arm_kind": str(row.arm_kind),
                    "control_family": str(row.control_family),
                    "support_size": int(row.support_size),
                    "declared_support_indices": json.dumps(np.unique(support).tolist(), separators=(",", ":")),
                    "q_full_recomputed": q_full,
                    "q_support_inside_nodewise": q_inside,
                    "q_support_share_nodewise": share,
                    "uniform_support_fraction": uniform,
                    "support_localization_lift": share - uniform,
                    "support_share_exceeds_uniform": bool(share > uniform),
                    "node_sensitivity_topk_recall": topk_recall,
                    "node_sensitivity_topk_lift": topk_recall - uniform if k else np.nan,
                    "node_sensitivity_mean_support_rank": mean_rank,
                    "node_sensitivity_topk_indices": json.dumps(sorted(top_k), separators=(",", ":")),
                    "q_full_parity_abs_error": abs(q_full - observed_q),
                })

    result = pd.DataFrame.from_records(records)
    group_keys = ["model_id", "pooling_mode", "source_match_id"]
    summary = (
        result.groupby(group_keys, sort=True)
        .agg(
            n_complete_arms=("intervention_id", "size"),
            mean_q_support_share=("q_support_share_nodewise", "mean"),
            median_q_support_share=("q_support_share_nodewise", "median"),
            mean_uniform_support_fraction=("uniform_support_fraction", "mean"),
            mean_support_localization_lift=("support_localization_lift", "mean"),
            median_support_localization_lift=("support_localization_lift", "median"),
            fraction_above_uniform=("support_share_exceeds_uniform", "mean"),
            mean_node_sensitivity_topk_recall=("node_sensitivity_topk_recall", "mean"),
            mean_node_sensitivity_topk_lift=("node_sensitivity_topk_lift", "mean"),
            mean_node_sensitivity_support_rank=("node_sensitivity_mean_support_rank", "mean"),
            mean_q_full_parity_abs_error=("q_full_parity_abs_error", "mean"),
        )
        .reset_index()
    )
    rows_path = SWITCH_ROOT / "node_sensitivity_localization_v2.parquet"
    summary_path = SWITCH_ROOT / "node_sensitivity_localization_v2_summary.parquet"
    summary_json = SWITCH_ROOT / "node_sensitivity_localization_v2_summary.json"
    receipt_path = SWITCH_ROOT / "node_sensitivity_localization_v2_receipt.json"
    if any(path.exists() for path in (rows_path, summary_path, summary_json, receipt_path)):
        raise FileExistsError("refusing to overwrite node localization v2 outputs")
    result.to_parquet(rows_path, index=False)
    summary.to_parquet(summary_path, index=False)
    payload = {
        "status": "T5_NODE_SENSITIVITY_LOCALIZATION_COMPLETE",
        "task": "P3-T5",
        "definition": "top-k node Jacobian sensitivity recall for declared support, k=support_size, versus k/20",
        "response_blind": True,
        "response_columns_read": [],
        "uses_controlled_intervention_support": True,
        "does_not_train_predictor": True,
        "n_models": int(len(models)),
        "n_complete_arms": int(len(result)),
        "n_group_rows": int(len(summary)),
        "grouping_unit": "source_match_id",
        "max_q_full_parity_abs_error": float(max(parity_errors) if parity_errors else np.nan),
        "mean_q_full_parity_abs_error": float(np.mean(parity_errors) if parity_errors else np.nan),
        "rows_path": str(rows_path.relative_to(ROOT)),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "input_path": "artifacts/phase3/selected_causal_switch_v1/switch_arm_geometry.parquet",
        "input_sha256": file_sha256(SWITCH_ROOT / "switch_arm_geometry.parquet"),
        "config_sha256": config_sha256,
    }
    write_json(summary_json, payload)
    receipt = {
        **payload,
        "source_code": "scripts/p3_node_localization.py",
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "output_sha256": {
            rows_path.name: file_sha256(rows_path),
            summary_path.name: file_sha256(summary_path),
            summary_json.name: file_sha256(summary_json),
        },
    }
    write_json(receipt_path, receipt)
    return payload


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(run(ROOT, args.config_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
