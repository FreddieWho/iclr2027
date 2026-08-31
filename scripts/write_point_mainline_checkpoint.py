#!/usr/bin/env python3
"""Write the point-set P1/C1 mainline view from a completed response run.

The full run may contain an optional minimap/vision branch. This script keeps
that run intact and deterministically derives a point-set-only package for the
mainline report. It performs no new model inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "artifacts" / "phase1"
OUT = PHASE1 / "point_mainline"
REPORT = ROOT / "reports" / "EXPLORATION_CHECKPOINT_1_POINT_MAINLINE.md"
POINT_FAMILIES = {"coordinate", "task_trained"}
POINT_IDS = {
    "deepsets_ae_seed11",
    "gat_ae_seed11",
    "phase_gat_seed11",
    "deepsets_ae_seed23",
    "gat_ae_seed23",
    "phase_gat_seed23",
    "deepsets_ae_seed47",
    "gat_ae_seed47",
    "phase_gat_seed47",
}
VISION_MODEL_IDS = {"dinov2_vits14", "openclip_vit_b32"}


def grouped_bootstrap(
    frame: pd.DataFrame,
    value: str,
    group_columns: list[str],
    n_bootstrap: int = 500,
    seed: int = 20260827,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_match = group.groupby("match_id")[value].mean()
        values = by_match.to_numpy(dtype=float)
        draws = (
            rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
            if len(values)
            else np.array([])
        )
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "mean_response": float(values.mean()) if len(values) else None,
                "ci_low": float(np.quantile(draws, 0.025)) if len(draws) else None,
                "ci_high": float(np.quantile(draws, 0.975)) if len(draws) else None,
                "n_matches": int(len(values)),
                "n_records": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_errr(response: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_id, group in response.groupby("model_id"):
        common = (
            group[group["kind"] == "common"]
            .groupby(["source_sample_id", "seed", "epsilon"])["response_distance"]
            .mean()
            .rename("common")
        )
        semantic = (
            group[group["kind"] == "semantic_coalition"]
            .groupby(["source_sample_id", "seed", "epsilon"])["response_distance"]
            .mean()
            .rename("semantic")
        )
        paired = pd.concat([common, semantic], axis=1).dropna()
        rows.append(
            {
                "model_id": model_id,
                "n_pairs": int(len(paired)),
                "errr": float((paired["common"] > paired["semantic"]).mean()) if len(paired) else None,
                "mean_common_minus_semantic": float((paired["common"] - paired["semantic"]).mean())
                if len(paired)
                else None,
            }
        )
    return rows


def compute_notch(summary: pd.DataFrame, primary_energy: float = 1.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if summary.empty:
        return rows
    for model_id, group in summary[summary["epsilon"].eq(primary_energy)].groupby("model_id"):
        curve = group[group["band_id"].notna()].groupby("band_id")["mean_response"].mean()
        if {0, 2, 3, 5}.issubset(set(curve.index)):
            low_high = 0.5 * (curve.loc[0] + curve.loc[5])
            mid = 0.5 * (curve.loc[2] + curve.loc[3])
            rows.append(
                {
                    "model_id": model_id,
                    "notch_depth_candidate": float(1.0 - mid / max(low_high, 1e-12)),
                    "low_high_reference": float(low_high),
                    "mid_response": float(mid),
                    "interpretation": "candidate only; inspect monotonicity and P2 controls",
                }
            )
    return rows


def make_figures(response: pd.DataFrame, summary: pd.DataFrame, errr: pd.DataFrame) -> list[str]:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model_id, group in summary[summary["epsilon"].eq(1.0)].groupby("model_id"):
            curve = group[group["band_id"].notna()].sort_values("band_id")
            if curve.empty:
                continue
            ax.plot(curve["band_id"], curve["mean_response"], marker="o", label=model_id)
            ax.fill_between(curve["band_id"], curve["ci_low"], curve["ci_high"], alpha=0.12)
        ax.set_xlabel("band id (equal-count Laplacian modes)")
        ax.set_ylabel("mean cosine response / epsilon")
        ax.set_title("P1 point-set structural transfer function")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        path = figure_dir / "transfer_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))
    if not errr.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(errr["model_id"], errr["errr"], color="#3b6ea8")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("preliminary ERRR")
        ax.set_title("Point-set common vs semantic coalition: exploratory only")
        fig.tight_layout()
        path = figure_dir / "preliminary_reversal.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))
    support = response[response["kind"].astype(str).str.startswith("support_sweep_")]
    if not support.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for model_id, group in support.groupby("model_id"):
            means = group.groupby("support_size")["normalized_response"].mean().sort_index()
            ax.plot(means.index, means.values, marker="o", label=model_id)
        ax.set_xlabel("support size")
        ax.set_ylabel("mean cosine response / epsilon")
        ax.set_title("Point-set preliminary support sweep")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        path = figure_dir / "support_sweep.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))
    return paths


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((PHASE1 / "pipeline_summary.json").read_text(encoding="utf-8"))
    source_models = json.loads((PHASE1 / "model_manifest.json").read_text(encoding="utf-8"))
    source_response = pd.read_parquet(PHASE1 / "spectrum_response.parquet")
    source_embedding = pd.read_parquet(PHASE1 / "embedding_manifest.parquet")

    manifest_ids = {
        item["model_id"] for item in source_models if item.get("family") in POINT_FAMILIES
    }
    point_ids = sorted(manifest_ids & POINT_IDS)
    if point_ids != sorted(POINT_IDS):
        raise RuntimeError(f"point model manifest is incomplete: {sorted(POINT_IDS - manifest_ids)}")
    response = source_response[source_response["model_id"].isin(point_ids)].copy()
    embedding = source_embedding[source_embedding["model_id"].isin(point_ids)].copy()
    if response.empty or embedding.empty:
        raise RuntimeError("point-set response or embedding manifest is empty")

    summary = grouped_bootstrap(
        response,
        "normalized_response",
        ["model_id", "kind", "epsilon", "band_id", "mode_rank", "support_size"],
    )
    errr_rows = compute_errr(response)
    errr = pd.DataFrame(errr_rows)
    notch_rows = compute_notch(summary)
    figure_paths = make_figures(response, summary, errr)

    response_path = OUT / "spectrum_response.parquet"
    embedding_path = OUT / "embedding_manifest.parquet"
    summary_path = OUT / "spectrum_summary.parquet"
    response.to_parquet(response_path, index=False)
    embedding.to_parquet(embedding_path, index=False)
    summary.to_parquet(summary_path, index=False)

    model_status = [
        item for item in source_summary["models"] if item["model_id"] in point_ids
    ]
    status = (
        "completed"
        if all(item.get("status") == "completed" for item in model_status)
        and source_summary["intervention_validation"]["status"] == "pass"
        else "partial"
    )
    point_summary = {
        "phase": "P1_PHENOMENON",
        "checkpoint": "C1",
        "status": status,
        "study_mode": "exploratory_discovery",
        "data_domain": "football_tracking_only",
        "mainline_modality": "coordinate_point_set",
        "vision_policy": "deferred_auxiliary",
        "models": model_status,
        "n_matches": source_summary["n_matches"],
        "n_samples": source_summary["n_samples"],
        "n_interventions": source_summary["n_interventions"],
        "n_valid_interventions": source_summary["n_valid_interventions"],
        "n_invalid_interventions": source_summary["n_invalid_interventions"],
        "n_response_rows": int(len(response)),
        "intervention_validation": source_summary["intervention_validation"],
        "max_energy_error": source_summary["max_energy_error"],
        "min_target_mode_purity": source_summary["min_target_mode_purity"],
        "errr": errr_rows,
        "notch": notch_rows,
        "mode_accessibility": [
            item for item in source_summary.get("mode_accessibility", []) if item["model_id"] in point_ids
        ],
        "figures": figure_paths,
        "derived_from": "artifacts/phase1/pipeline_summary.json and response artifacts",
        "new_model_inference": False,
        "auxiliary_visual_run": {
            "status": "completed_preserved_not_mainline",
            "models": sorted(VISION_MODEL_IDS),
            "response_artifact": "artifacts/phase1/spectrum_response.parquet",
        },
        "scientific_boundary": "C1 measures candidate point-set response patterns; it does not establish H1/H2 or the P2 same-frequency organization claim.",
    }
    (OUT / "pipeline_summary.json").write_text(
        json.dumps(point_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    point_contract = {
        "phase": "P1_PHENOMENON",
        "checkpoint": "C1",
        "study_mode": "exploratory_discovery",
        "data_domain": "football_tracking_only",
        "mainline_modality": "coordinate_point_set",
        "model_families": sorted(POINT_FAMILIES),
        "model_ids": point_ids,
        "visual_policy": "deferred_auxiliary",
        "source_run": "artifacts/phase1/execution_contract.json",
        "derivation": "deterministic filter and re-aggregation; no new model inference",
    }
    (OUT / "execution_contract.json").write_text(
        json.dumps(point_contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT / "model_manifest.json").write_text(
        json.dumps(
            [item for item in source_models if item["model_id"] in point_ids],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Exploration Checkpoint 1 — Point-Set Mainline",
        "",
        f"- Status: {status}",
        "- Study mode: exploratory_discovery",
        "- Mainline modality: direct player point set",
        f"- Samples: {point_summary['n_samples']} across {point_summary['n_matches']} SkillCorner matches",
        f"- Interventions: {point_summary['n_interventions']} total; {point_summary['n_valid_interventions']} valid; {point_summary['n_invalid_interventions']} boundary-invalid",
        f"- Maximum energy error: {point_summary['max_energy_error']}",
        f"- Minimum target-mode purity: {point_summary['min_target_mode_purity']}",
        "",
        "## Models",
        "",
    ]
    lines.extend(
        f"- {item['model_id']}: {item['status']}" for item in model_status
    )
    lines.extend(
        [
            "",
            "## Measurements",
            "",
            "- Primary response: normalized cosine distance between baseline and intervened point-set embeddings.",
            "- Transfer curves: six equal-count Laplacian bands across all seeds and energies.",
            "- Preliminary ERRR: common versus role-ordered semantic coalition; not a P2 strict same-frequency result.",
            "- Confidence intervals: match-level bootstrap, not frame-level independent bootstrap.",
            f"- Pairing/validity gate: {point_summary['intervention_validation']['status']}; each intervention has one baseline counterpart.",
            "",
            "## Exploratory findings",
            "",
            "These are candidate response patterns for exploration, not confirmatory claims.",
            "",
            "### Preliminary ERRR",
            "",
            "JSON:",
            json.dumps(errr_rows, indent=2, ensure_ascii=False),
            "",
            "### Notch candidates",
            "",
            "JSON:",
            json.dumps(notch_rows, indent=2, ensure_ascii=False),
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend(f"- {path}" for path in figure_paths)
    lines.extend(
        [
            str(response_path.relative_to(ROOT)),
            str(embedding_path.relative_to(ROOT)),
            str(summary_path.relative_to(ROOT)),
            str((OUT / "pipeline_summary.json").relative_to(ROOT)),
            str((OUT / "execution_contract.json").relative_to(ROOT)),
            "",
            "## Scope boundary",
            "",
            "The direct point-set branch is the P1/C1 mainline because the source data already represent players as points. The minimap/vision run is preserved as an auxiliary rendering-robustness result and is excluded from this mainline checkpoint.",
            "P1 does not establish that common motion is more salient, that a mid-frequency notch exists, or that organization exceeds frequency. Those interpretations require the matched controls and alternative graph constructions planned for P2.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(point_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
