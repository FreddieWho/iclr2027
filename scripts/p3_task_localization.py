"""Controlled support-region localization diagnostic for P3-T5.

This post-processes the already frozen local-geometry arm table.  The
intervention declares its source support, so the diagnostic measures how much
of the quadratic geometry mass is assigned to that declared region.  It does
not train a predictor, read response values to select rows, or create a new
scientific stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/phase3/selected_causal_switch_v1/switch_arm_geometry.parquet"
DEFAULT_OUTPUT = ROOT / "artifacts/phase3/selected_causal_switch_v1"


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(input_path: Path, output_dir: Path, config_sha256: str) -> dict[str, object]:
    input_columns = [
        "model_id",
        "source_match_id",
        "matched_set_id",
        "source_sample_id",
        "sample_index",
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
        "source_support_indices",
        "target_support_indices",
        "q_full",
        "q_support_inside",
        "q_support_outside",
    ]
    # Deliberately exclude response/raw_response: this diagnostic is
    # determined by declared support and local geometry only.
    arms = pd.read_parquet(input_path, columns=input_columns)
    complete = arms[arms["matched_set_status"].astype(str) == "COMPLETE"].copy()
    if complete.empty:
        raise ValueError("no complete intervention arms available")

    n_nodes = 20.0
    q_full = complete["q_full"].astype(float).to_numpy()
    q_inside = complete["q_support_inside"].astype(float).to_numpy()
    q_outside = complete["q_support_outside"].astype(float).to_numpy()
    q_share = np.divide(
        q_inside,
        q_full,
        out=np.full(len(complete), np.nan, dtype=float),
        where=np.abs(q_full) > 1e-15,
    )
    support_fraction = complete["support_size"].astype(float).to_numpy() / n_nodes
    result = complete[
        [
            "model_id",
            "source_match_id",
            "matched_set_id",
            "source_sample_id",
            "sample_index",
            "split",
            "graph_id",
            "coalition_id",
            "team_slot",
            "role_group",
            "epsilon",
            "fracture_draw",
            "arm_kind",
            "control_family",
            "support_size",
            "source_support_indices",
            "target_support_indices",
            "q_full",
            "q_support_inside",
            "q_support_outside",
        ]
    ].copy()
    result["q_support_share"] = q_share
    result["uniform_support_fraction"] = support_fraction
    result["support_localization_lift"] = q_share - support_fraction
    result["support_localization_margin"] = q_inside - q_outside
    result["support_share_exceeds_uniform"] = q_share > support_fraction

    group_keys = ["model_id", "pooling_mode", "source_match_id"] if "pooling_mode" in complete else ["model_id", "source_match_id"]
    summary = (
        result.assign(**({"pooling_mode": complete["pooling_mode"].to_numpy()} if "pooling_mode" in complete else {}))
        .groupby(group_keys, sort=True, dropna=False)
        .agg(
            n_complete_arms=("matched_set_id", "size"),
            mean_q_support_share=("q_support_share", "mean"),
            median_q_support_share=("q_support_share", "median"),
            mean_uniform_support_fraction=("uniform_support_fraction", "mean"),
            mean_support_localization_lift=("support_localization_lift", "mean"),
            median_support_localization_lift=("support_localization_lift", "median"),
            fraction_above_uniform=("support_share_exceeds_uniform", "mean"),
            mean_support_localization_margin=("support_localization_margin", "mean"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "support_localization.parquet"
    summary_path = output_dir / "support_localization_summary.parquet"
    result.to_parquet(rows_path, index=False)
    summary.to_parquet(summary_path, index=False)
    payload = {
        "status": "T5_SUPPORT_REGION_LOCALIZATION_COMPLETE",
        "task": "P3-T5",
        "definition": "q_support_inside / q_full versus uniform support_size / 20",
        "response_blind": True,
        "uses_controlled_intervention_support": True,
        "does_not_train_predictor": True,
        "input_path": str(input_path.relative_to(ROOT)),
        "input_sha256": file_sha256(input_path),
        "config_sha256": config_sha256,
        "n_input_rows": int(len(arms)),
        "n_complete_arms": int(len(result)),
        "n_group_rows": int(len(summary)),
        "grouping_unit": "source_match_id",
        "rows_path": str(rows_path.relative_to(ROOT)),
        "summary_path": str(summary_path.relative_to(ROOT)),
    }
    summary_json = output_dir / "support_localization_summary.json"
    write_json(summary_json, payload)
    receipt = {
        **payload,
        "source_code": "scripts/p3_task_localization.py",
        "source_code_sha256": file_sha256(Path(__file__).resolve()),
        "output_sha256": {
            "support_localization.parquet": file_sha256(rows_path),
            "support_localization_summary.parquet": file_sha256(summary_path),
            "support_localization_summary.json": file_sha256(summary_json),
        },
    }
    write_json(output_dir / "support_localization_receipt.json", receipt)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.config_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
