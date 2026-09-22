#!/usr/bin/env python3
"""Statistics for the independent P2 fracture-continuity two-arm design."""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
from typing import Any

import numpy as np
import pandas as pd


PAIR_KEYS = [
    "matched_set_id",
    "source_sample_id",
    "source_match_id",
    "split",
    "graph_id",
    "coalition_id",
    "team_slot",
    "role_group",
    "epsilon",
    "fracture_draw",
    "vector_multiset_id",
    "frequency_residual",
    "topology_residual",
]


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def build_pair_effects(response: pd.DataFrame) -> pd.DataFrame:
    """Create one anchor-control pair per model and matched set."""
    _require(
        response,
        [
            *PAIR_KEYS,
            "model_id",
            "architecture",
            "model_seed",
            "arm_kind",
            "control_family",
            "response_distance",
            "normalized_response",
            "valid",
            "matched_set_status",
        ],
        "response",
    )
    valid = response.loc[response["valid"].astype(bool)].copy()
    valid["model_id"] = valid["model_id"].astype(str)
    valid["matched_set_id"] = valid["matched_set_id"].astype(str)
    valid["arm_kind"] = valid["arm_kind"].astype(str)
    valid["control_family"] = valid["control_family"].astype(str)
    valid = valid.loc[
        valid["arm_kind"].isin(["anchor", "control"])
        & valid["control_family"].isin(
            ["anchor", "topology_frequency_same_vector_reassignment"]
        )
    ]
    pair_keys = ["model_id", "matched_set_id"]
    counts = valid.groupby(pair_keys, sort=True).agg(
        n_rows=("arm_kind", "size"),
        n_arm_kinds=("arm_kind", "nunique"),
        n_control_families=("control_family", "nunique"),
    )
    qualifying = counts.loc[
        (counts["n_rows"] == 2)
        & (counts["n_arm_kinds"] == 2)
        & (counts["n_control_families"] == 2)
    ].reset_index()[pair_keys]
    if qualifying.empty:
        raise ValueError("no valid fracture pairs")
    candidate = valid.merge(qualifying, on=pair_keys, how="inner", validate="many_to_one")
    anchor = (
        candidate.loc[candidate["arm_kind"].eq("anchor")]
        .set_index(pair_keys)
        .sort_index()
    )
    control = (
        candidate.loc[candidate["arm_kind"].eq("control")]
        .set_index(pair_keys)
        .sort_index()
    )
    if len(anchor) != len(control) or not anchor.index.equals(control.index):
        raise ValueError("two-arm set is not unique")
    if not np.array_equal(
        anchor["vector_multiset_id"].astype(str).to_numpy(),
        control["vector_multiset_id"].astype(str).to_numpy(),
    ):
        bad = anchor.index[anchor["vector_multiset_id"].astype(str).to_numpy() != control["vector_multiset_id"].astype(str).to_numpy()]
        raise ValueError(f"vector multiset mismatch: {bad[0][1] if len(bad) else 'unknown'}")
    if not np.array_equal(
        anchor["source_match_id"].astype(str).to_numpy(),
        control["source_match_id"].astype(str).to_numpy(),
    ):
        bad = anchor.index[anchor["source_match_id"].astype(str).to_numpy() != control["source_match_id"].astype(str).to_numpy()]
        raise ValueError(f"match mismatch: {bad[0][1] if len(bad) else 'unknown'}")
    result = anchor[
        [key for key in PAIR_KEYS if key not in pair_keys]
        + ["architecture", "model_seed", "matched_set_status"]
    ].copy()
    result["model_id"] = anchor.index.get_level_values("model_id").to_numpy()
    result["matched_set_id"] = anchor.index.get_level_values("matched_set_id").to_numpy()
    result["frequency_residual"] = control["frequency_residual"].to_numpy(dtype=float)
    result["topology_residual"] = control["topology_residual"].to_numpy(dtype=float)
    result["raw_scg"] = (
        anchor["response_distance"].to_numpy(dtype=float)
        - control["response_distance"].to_numpy(dtype=float)
    )
    result["normalized_scg"] = (
        anchor["normalized_response"].to_numpy(dtype=float)
        - control["normalized_response"].to_numpy(dtype=float)
    )
    result["anchor_response"] = anchor["response_distance"].to_numpy(dtype=float)
    result["control_response"] = control["response_distance"].to_numpy(dtype=float)
    result["complete_two_arm"] = anchor["matched_set_status"].astype(str).eq("COMPLETE").to_numpy()
    result["model_id"] = result["model_id"].astype(str)
    result["architecture"] = result["architecture"].astype(str)
    result["model_seed"] = result["model_seed"].astype(int)
    result = result.reset_index(drop=True)
    if result.empty:
        raise ValueError("no valid fracture pairs")
    numeric = result[["raw_scg", "normalized_scg", "frequency_residual", "topology_residual"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("fracture pair contains non-finite values")
    return result.sort_values(["model_id", "matched_set_id"]).reset_index(drop=True)


def aggregate_match_effects(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate intervention draws to match-level, then model-seed level."""
    _require(
        pairs,
        [
            "model_id",
            "architecture",
            "model_seed",
            "source_sample_id",
            "source_match_id",
            "graph_id",
            "role_group",
            "epsilon",
            "fracture_draw",
            "raw_scg",
            "normalized_scg",
            "frequency_residual",
            "topology_residual",
        ],
        "pairs",
    )
    unit_keys = [
        "model_id",
        "architecture",
        "model_seed",
        "source_match_id",
        "graph_id",
        "role_group",
        "epsilon",
        "source_sample_id",
        "fracture_draw",
    ]
    unit = (
        pairs.groupby(unit_keys, as_index=False, sort=True)
        .agg(
            raw_scg=("raw_scg", "mean"),
            normalized_scg=("normalized_scg", "mean"),
            frequency_residual=("frequency_residual", "mean"),
            topology_residual=("topology_residual", "mean"),
            n_pairs=("matched_set_id", "nunique"),
        )
    )
    match_keys = [
        "model_id",
        "architecture",
        "model_seed",
        "source_match_id",
        "graph_id",
        "role_group",
        "epsilon",
    ]
    model_match = (
        unit.groupby(match_keys, as_index=False, sort=True)
        .agg(
            raw_scg=("raw_scg", "mean"),
            normalized_scg=("normalized_scg", "mean"),
            frequency_residual=("frequency_residual", "mean"),
            topology_residual=("topology_residual", "mean"),
            n_source_units=("source_sample_id", "nunique"),
            n_draws=("fracture_draw", "nunique"),
        )
    )
    return unit, model_match


def aggregate_architecture_matches(model_match: pd.DataFrame) -> pd.DataFrame:
    """Average model seeds within each architecture and match."""
    _require(
        model_match,
        [
            "architecture",
            "model_seed",
            "source_match_id",
            "graph_id",
            "role_group",
            "epsilon",
            "raw_scg",
            "normalized_scg",
            "frequency_residual",
            "topology_residual",
        ],
        "model_match",
    )
    keys = ["architecture", "source_match_id", "graph_id", "role_group", "epsilon"]
    return (
        model_match.groupby(keys, as_index=False, sort=True)
        .agg(
            raw_scg=("raw_scg", "mean"),
            normalized_scg=("normalized_scg", "mean"),
            frequency_residual=("frequency_residual", "mean"),
            topology_residual=("topology_residual", "mean"),
            n_model_seeds=("model_seed", "nunique"),
        )
    )


def _seed_for(base_seed: int, key: tuple[Any, ...]) -> int:
    payload = f"{int(base_seed)}|{key!r}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def bootstrap_interval(
    values: np.ndarray, replicates: int = 10000, seed: int = 2026083102
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(values, size=(int(replicates), values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_architecture_matches(
    architecture_match: pd.DataFrame,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 2026083102,
) -> pd.DataFrame:
    _require(
        architecture_match,
        [
            "architecture",
            "source_match_id",
            "graph_id",
            "role_group",
            "epsilon",
            "raw_scg",
            "normalized_scg",
            "frequency_residual",
            "topology_residual",
        ],
        "architecture_match",
    )
    keys = ["architecture", "graph_id", "role_group", "epsilon"]
    rows: list[dict[str, Any]] = []
    for key_values, group in architecture_match.groupby(keys, sort=True):
        raw = group["raw_scg"].to_numpy(dtype=float)
        normalized = group["normalized_scg"].to_numpy(dtype=float)
        low, high = bootstrap_interval(
            raw,
            replicates=bootstrap_replicates,
            seed=_seed_for(bootstrap_seed, tuple(key_values) + ("raw",)),
        )
        nlow, nhigh = bootstrap_interval(
            normalized,
            replicates=bootstrap_replicates,
            seed=_seed_for(bootstrap_seed, tuple(key_values) + ("normalized",)),
        )
        rows.append(
            {
                "architecture": str(key_values[0]),
                "graph_id": str(key_values[1]),
                "role_group": str(key_values[2]),
                "epsilon": float(key_values[3]),
                "n_matches": int(group["source_match_id"].nunique()),
                "raw_scg_mean": float(np.mean(raw)),
                "raw_scg_median": float(np.median(raw)),
                "raw_scg_ci_low": low,
                "raw_scg_ci_high": high,
                "normalized_scg_mean": float(np.mean(normalized)),
                "normalized_scg_ci_low": nlow,
                "normalized_scg_ci_high": nhigh,
                "positive_match_fraction": float(np.mean(raw > 0)),
                "frequency_residual_mean": float(group["frequency_residual"].mean()),
                "topology_residual_mean": float(group["topology_residual"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def leave_one_match_out(architecture_match: pd.DataFrame) -> pd.DataFrame:
    keys = ["architecture", "graph_id", "role_group", "epsilon"]
    rows: list[dict[str, Any]] = []
    for key_values, group in architecture_match.groupby(keys, sort=True):
        for match_id in sorted(group["source_match_id"].astype(str).unique()):
            remaining = group[group["source_match_id"].astype(str) != match_id]
            rows.append(
                {
                    "architecture": str(key_values[0]),
                    "graph_id": str(key_values[1]),
                    "role_group": str(key_values[2]),
                    "epsilon": float(key_values[3]),
                    "left_out_match_id": match_id,
                    "n_matches": int(remaining["source_match_id"].nunique()),
                    "raw_scg_mean": float(remaining["raw_scg"].mean()) if len(remaining) else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(keys + ["left_out_match_id"]).reset_index(drop=True)


def residual_association(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return response-blind matching residual summaries for interpretation."""
    keys = ["model_id", "architecture", "graph_id", "role_group", "epsilon"]
    rows: list[dict[str, Any]] = []
    for key_values, group in pairs.groupby(keys, sort=True):
        residual = group["frequency_residual"].to_numpy(dtype=float)
        effects = group["raw_scg"].to_numpy(dtype=float)
        correlation = float(np.corrcoef(residual, effects)[0, 1]) if len(group) > 1 and np.std(residual) > 0 and np.std(effects) > 0 else float("nan")
        rows.append(
            {
                "model_id": str(key_values[0]),
                "architecture": str(key_values[1]),
                "graph_id": str(key_values[2]),
                "role_group": str(key_values[3]),
                "epsilon": float(key_values[4]),
                "n_pairs": int(len(group)),
                "frequency_residual_mean": float(np.mean(residual)),
                "frequency_residual_max": float(np.max(residual)),
                "raw_scg_frequency_residual_corr": correlation,
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
