#!/usr/bin/env python3
"""Match-level statistics for the formal P2 rigid-intervention run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONTROLS = (
    "arbitrary_same_team",
    "topology_frequency_matched",
    "spectrum_exact_sign_randomized",
)
PAIR_KEYS = (
    "model_id",
    "architecture",
    "model_seed",
    "source_match_id",
    "split",
    "source_sample_id",
    "frame",
    "graph_id",
    "coalition_id",
    "team_slot",
    "role_group",
    "epsilon",
    "direction",
)
SUMMARY_KEYS = (
    "architecture",
    "control_family",
    "support_mode",
    "graph_id",
    "epsilon",
    "role_stratum",
)


def _required(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _band_l1(first: Any, second: Any) -> float:
    a = np.asarray(json.loads(str(first)), dtype=float)
    b = np.asarray(json.loads(str(second)), dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid band_power values in response rows")
    return float(np.abs(a - b).sum())


def build_paired_effects(
    response: pd.DataFrame,
    strict_band_power_l1: float = 0.20,
) -> pd.DataFrame:
    """Pair each semantic arm with each available control without pooling controls."""
    required = {
        *PAIR_KEYS,
        "matched_set_id",
        "matched_set_status",
        "arm_kind",
        "control_family",
        "response_distance",
        "normalized_response",
        "band_power",
    }
    _required(response, required, "response")
    if strict_band_power_l1 < 0:
        raise ValueError("strict_band_power_l1 must be non-negative")
    identity = ["model_id", "matched_set_id", "arm_kind", "control_family"]
    if response.duplicated(identity).any():
        raise ValueError("duplicate response arm within model and matched set")

    records: list[dict[str, Any]] = []
    group_keys = ["model_id", "matched_set_id"]
    for (_, matched_set_id), group in response.groupby(group_keys, sort=True):
        anchors = group[
            group["arm_kind"].eq("semantic_rigid") & group["control_family"].eq("anchor")
        ]
        if len(anchors) == 0:
            # Valid controls without a valid semantic arm have no paired
            # estimand. Their missing anchor remains visible in matching
            # coverage/failure tables, but they cannot enter SCG.
            continue
        if len(anchors) != 1:
            raise ValueError(f"matched set must have exactly one valid semantic anchor: {matched_set_id}")
        anchor = anchors.iloc[0]
        controls = group[group["arm_kind"].eq("control")]
        families = set(controls["control_family"].astype(str))
        unknown = families - set(EXPECTED_CONTROLS)
        if unknown:
            raise ValueError(f"unknown control families in {matched_set_id}: {sorted(unknown)}")
        complete = (
            str(anchor["matched_set_status"]) == "COMPLETE"
            and families == set(EXPECTED_CONTROLS)
            and len(group) == 4
        )
        for _, control in controls.iterrows():
            family = str(control["control_family"])
            band_l1 = _band_l1(anchor["band_power"], control["band_power"])
            base = {key: anchor[key] for key in PAIR_KEYS}
            base.update(
                {
                    "matched_set_id": str(matched_set_id),
                    "control_family": family,
                    "raw_scg": float(anchor["response_distance"] - control["response_distance"]),
                    "normalized_scg": float(anchor["normalized_response"] - control["normalized_response"]),
                    "band_power_l1": band_l1,
                    "complete_four_arm": bool(complete),
                }
            )
            modes = ["pairwise"]
            if complete:
                modes.append("common_four")
                if band_l1 <= strict_band_power_l1 + 1e-12:
                    modes.append("strict_band_l1_020")
            for mode in modes:
                records.append({**base, "support_mode": mode})
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("response contains no semantic-control pairs")
    values = result[["raw_scg", "normalized_scg", "band_power_l1"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("paired effects contain non-finite values")
    return result.sort_values(
        ["model_id", "matched_set_id", "control_family", "support_mode"]
    ).reset_index(drop=True)


def _mean_levels(pairs: pd.DataFrame) -> pd.DataFrame:
    metrics = {"normalized_scg": "mean", "raw_scg": "mean"}
    direction_keys = [
        "model_id",
        "architecture",
        "model_seed",
        "control_family",
        "support_mode",
        "source_match_id",
        "split",
        "source_sample_id",
        "graph_id",
        "coalition_id",
        "team_slot",
        "role_group",
        "epsilon",
    ]
    direction = pairs.groupby(direction_keys, dropna=False, sort=True).agg(metrics).reset_index()
    team_role_keys = [key for key in direction_keys if key != "coalition_id"]
    team_role = direction.groupby(team_role_keys, dropna=False, sort=True).agg(metrics).reset_index()
    sample_role_keys = [key for key in team_role_keys if key != "team_slot"]
    sample_role = team_role.groupby(sample_role_keys, dropna=False, sort=True).agg(metrics).reset_index()
    sample_role = sample_role.rename(columns={"role_group": "role_stratum"})
    overall_keys = [
        column
        for column in sample_role.columns
        if column not in {"role_stratum", "normalized_scg", "raw_scg"}
    ]
    overall = sample_role.groupby(overall_keys, dropna=False, sort=True).agg(metrics).reset_index()
    overall["role_stratum"] = "__overall__"
    columns = list(sample_role.columns)
    return pd.concat([sample_role[columns], overall[columns]], ignore_index=True)


def aggregate_architecture_matches(model_match: pd.DataFrame) -> pd.DataFrame:
    """Average model seeds inside each architecture without treating seeds as matches."""
    keys = [
        "architecture",
        "control_family",
        "support_mode",
        "source_match_id",
        "split",
        "graph_id",
        "epsilon",
        "role_stratum",
    ]
    grouped = model_match.groupby(keys, dropna=False, sort=True)
    result = grouped.agg(
        normalized_scg=("normalized_scg", "mean"),
        raw_scg=("raw_scg", "mean"),
        seed_spread_normalized=("normalized_scg", lambda values: float(np.std(values, ddof=0))),
        seed_spread_raw=("raw_scg", lambda values: float(np.std(values, ddof=0))),
        n_model_seeds=("model_seed", "nunique"),
    ).reset_index()
    return result


def aggregate_match_effects(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply direction -> coalition/team/role -> sample -> match equal weighting."""
    _required(
        pairs,
        {*PAIR_KEYS, "control_family", "support_mode", "raw_scg", "normalized_scg"},
        "paired effects",
    )
    sample_level = _mean_levels(pairs)
    match_keys = [
        "model_id",
        "architecture",
        "model_seed",
        "control_family",
        "support_mode",
        "source_match_id",
        "split",
        "graph_id",
        "epsilon",
        "role_stratum",
    ]
    model_match = (
        sample_level.groupby(match_keys, dropna=False, sort=True)
        .agg(normalized_scg=("normalized_scg", "mean"), raw_scg=("raw_scg", "mean"), n_samples=("source_sample_id", "nunique"))
        .reset_index()
    )
    architecture_match = aggregate_architecture_matches(model_match)
    return model_match, architecture_match


def _bootstrap_interval(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    if len(values) < 2 or n_bootstrap < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indexes].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _group_seed(base_seed: int, key: tuple[Any, ...]) -> int:
    payload = json.dumps([base_seed, *[str(value) for value in key]], separators=(",", ":"))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def summarize_match_effects(
    architecture_match: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 20260831,
) -> pd.DataFrame:
    """Summarize equally weighted match effects; bootstrap resamples matches only."""
    _required(
        architecture_match,
        {*SUMMARY_KEYS, "source_match_id", "normalized_scg", "raw_scg"},
        "architecture match effects",
    )
    records: list[dict[str, Any]] = []
    for key, group in architecture_match.groupby(list(SUMMARY_KEYS), dropna=False, sort=True):
        if group["source_match_id"].duplicated().any():
            raise ValueError(f"architecture summary has duplicate match rows: {key}")
        normalized = group["normalized_scg"].to_numpy(dtype=float)
        raw = group["raw_scg"].to_numpy(dtype=float)
        group_seed = _group_seed(seed, key)
        norm_low, norm_high = _bootstrap_interval(normalized, n_bootstrap, group_seed)
        raw_low, raw_high = _bootstrap_interval(raw, n_bootstrap, group_seed + 1)
        records.append(
            {
                **dict(zip(SUMMARY_KEYS, key)),
                "n_matches": int(group["source_match_id"].nunique()),
                "mean_normalized_scg": float(normalized.mean()),
                "median_normalized_scg": float(np.median(normalized)),
                "positive_match_fraction": float(np.mean(normalized > 0)),
                "normalized_ci_low": norm_low,
                "normalized_ci_high": norm_high,
                "mean_raw_scg": float(raw.mean()),
                "median_raw_scg": float(np.median(raw)),
                "raw_ci_low": raw_low,
                "raw_ci_high": raw_high,
                "mean_seed_spread_normalized": float(group.get("seed_spread_normalized", pd.Series([np.nan])).mean()),
            }
        )
    return pd.DataFrame(records).sort_values(list(SUMMARY_KEYS)).reset_index(drop=True)


def leave_one_match_out(architecture_match: pd.DataFrame) -> pd.DataFrame:
    _required(
        architecture_match,
        {*SUMMARY_KEYS, "source_match_id", "normalized_scg", "raw_scg"},
        "architecture match effects",
    )
    records: list[dict[str, Any]] = []
    for key, group in architecture_match.groupby(list(SUMMARY_KEYS), dropna=False, sort=True):
        match_ids = sorted(group["source_match_id"].astype(str).unique())
        if len(match_ids) < 2:
            continue
        for omitted in match_ids:
            kept = group[~group["source_match_id"].astype(str).eq(omitted)]
            records.append(
                {
                    **dict(zip(SUMMARY_KEYS, key)),
                    "omitted_match_id": omitted,
                    "n_matches_remaining": int(kept["source_match_id"].nunique()),
                    "mean_normalized_scg": float(kept["normalized_scg"].mean()),
                    "mean_raw_scg": float(kept["raw_scg"].mean()),
                }
            )
    return pd.DataFrame(records)


def split_descriptive(architecture_match: pd.DataFrame) -> pd.DataFrame:
    keys = [*SUMMARY_KEYS, "split"]
    return (
        architecture_match.groupby(keys, dropna=False, sort=True)
        .agg(
            n_matches=("source_match_id", "nunique"),
            mean_normalized_scg=("normalized_scg", "mean"),
            median_normalized_scg=("normalized_scg", "median"),
            positive_match_fraction=("normalized_scg", lambda values: float(np.mean(values > 0))),
            mean_raw_scg=("raw_scg", "mean"),
        )
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--strict-band-power-l1", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    response = pd.read_parquet(args.response)
    pairs = build_paired_effects(response, args.strict_band_power_l1)
    model_match, architecture_match = aggregate_match_effects(pairs)
    summary = summarize_match_effects(architecture_match, args.bootstrap_replicates, args.bootstrap_seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output_dir / "paired_effects.parquet", index=False)
    model_match.to_parquet(output_dir / "model_seed_match_effects.parquet", index=False)
    architecture_match.to_parquet(output_dir / "match_level_effects.parquet", index=False)
    summary.to_parquet(output_dir / "scg_summary.parquet", index=False)
    leave_one_match_out(architecture_match).to_parquet(output_dir / "leave_one_match_out.parquet", index=False)
    split_descriptive(architecture_match).to_parquet(output_dir / "split_descriptive.parquet", index=False)
    print(json.dumps({"output_dir": str(output_dir), "n_pairs": len(pairs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
