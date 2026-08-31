#!/usr/bin/env python3
"""Descriptive P2 heterogeneity diagnosis from the frozen fracture outputs.

This module deliberately performs no model inference.  It reuses the completed
P2 fracture statistics, aggregates at the match level, and reports how the
effect varies with architecture, graph, role, energy, and matching residual.
It does not apply a threshold to promote the result to P3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CONDITION_KEYS = ["architecture", "graph_id", "role_group", "epsilon"]
MATCH_KEYS = ["architecture", "source_match_id", "graph_id", "role_group", "epsilon"]
MODEL_KEYS = ["model_id", "architecture", "graph_id", "role_group", "epsilon"]

FACTOR_BACKGROUNDS = {
    "architecture": ["graph_id", "role_group", "epsilon"],
    "graph_id": ["architecture", "role_group", "epsilon"],
    "role_group": ["architecture", "graph_id", "epsilon"],
    "epsilon": ["architecture", "graph_id", "role_group"],
}

LEVEL_ORDER = {
    "architecture": ["DeepSets-AE", "GAT-small-AE", "Phase-GAT"],
    "graph_id": ["delaunay", "knn4"],
    "role_group": ["defender", "midfielder", "forward"],
    "epsilon": [0.25, 0.5],
}


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _require_finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite numeric values")


def _corr(left: pd.Series, right: pd.Series) -> float:
    x = left.to_numpy(dtype=float)
    y = right.to_numpy(dtype=float)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _direction(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _key_mask(frame: pd.DataFrame, key_columns: list[str], key_values: tuple[Any, ...]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in zip(key_columns, key_values):
        if column == "epsilon":
            mask &= np.isclose(frame[column].to_numpy(dtype=float), float(value))
        else:
            mask &= frame[column].astype(str).eq(str(value))
    return mask


def build_condition_profile(
    architecture_match: pd.DataFrame,
    model_match: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> pd.DataFrame:
    """Create one descriptive row per architecture/graph/role/energy condition.

    Match-level values are already averaged over source units and intervention
    draws in the frozen statistics.  Model seeds are summarized within an
    architecture and are not treated as independent matches.
    """
    _require(
        architecture_match,
        CONDITION_KEYS
        + [
            "source_match_id",
            "raw_scg",
            "normalized_scg",
            "frequency_residual",
            "topology_residual",
            "n_model_seeds",
        ],
        "architecture_match",
    )
    _require(
        model_match,
        MODEL_KEYS + ["source_match_id", "raw_scg", "normalized_scg", "model_seed"],
        "model_match",
    )
    _require(
        leave_one_out,
        CONDITION_KEYS + ["left_out_match_id", "n_matches", "raw_scg_mean"],
        "leave_one_out",
    )
    _require_finite(
        architecture_match,
        ["raw_scg", "normalized_scg", "frequency_residual", "topology_residual"],
        "architecture_match",
    )
    _require_finite(model_match, ["raw_scg", "normalized_scg"], "model_match")
    _require_finite(leave_one_out, ["raw_scg_mean"], "leave_one_out")

    match_duplicate = architecture_match.duplicated(MATCH_KEYS, keep=False)
    if bool(match_duplicate.any()):
        raise ValueError("architecture_match has duplicate match-level rows")

    loo_groups = {
        tuple(values): group.copy()
        for values, group in leave_one_out.groupby(CONDITION_KEYS, sort=True, dropna=False)
    }
    model_groups = {
        tuple(values): group.copy()
        for values, group in model_match.groupby(MODEL_KEYS[1:], sort=True, dropna=False)
    }

    rows: list[dict[str, Any]] = []
    for key_values, group in architecture_match.groupby(CONDITION_KEYS, sort=True, dropna=False):
        key_values = tuple(key_values)
        group = group.sort_values("source_match_id").reset_index(drop=True)
        raw = group["raw_scg"].to_numpy(dtype=float)
        normalized = group["normalized_scg"].to_numpy(dtype=float)
        loo = loo_groups.get(key_values)
        if loo is None:
            raise ValueError(f"missing leave-one-match-out rows for condition {key_values}")
        loo = loo.sort_values("left_out_match_id").reset_index(drop=True)
        if len(loo) != len(group):
            raise ValueError(f"LOMO row count mismatch for condition {key_values}")
        if set(loo["left_out_match_id"].astype(str)) != set(group["source_match_id"].astype(str)):
            raise ValueError(f"LOMO match IDs mismatch for condition {key_values}")

        full_raw = float(raw.mean())
        full_normalized = float(normalized.mean())
        full_sign = np.sign(full_raw)
        loo_raw = loo["raw_scg_mean"].to_numpy(dtype=float)
        lomo_same_sign = (
            float(np.mean(np.sign(loo_raw) == full_sign)) if full_sign != 0 else float("nan")
        )

        model_key = (key_values[0], key_values[1], key_values[2], key_values[3])
        model_group = model_groups.get(model_key)
        if model_group is None or model_group.empty:
            raise ValueError(f"missing model-match rows for condition {key_values}")
        seed_means = model_group.groupby("model_id", sort=True)["normalized_scg"].mean()
        seed_values = seed_means.to_numpy(dtype=float)
        seed_mean = float(seed_values.mean())
        seed_sign = np.sign(seed_values)

        rows.append(
            {
                "architecture": str(key_values[0]),
                "graph_id": str(key_values[1]),
                "role_group": str(key_values[2]),
                "epsilon": float(key_values[3]),
                "n_matches": int(group["source_match_id"].nunique()),
                "n_model_seeds": int(len(seed_values)),
                "full_mean_raw_scg": full_raw,
                "full_mean_normalized_scg": full_normalized,
                "direction": _direction(full_raw),
                "match_positive_fraction": float(np.mean(raw > 0)),
                "match_negative_fraction": float(np.mean(raw < 0)),
                "match_sd_normalized_scg": float(np.std(normalized, ddof=1))
                if len(normalized) > 1
                else 0.0,
                "match_min_normalized_scg": float(normalized.min()),
                "match_max_normalized_scg": float(normalized.max()),
                "lomo_same_sign_fraction": lomo_same_sign,
                "lomo_min_raw_scg": float(loo_raw.min()),
                "lomo_max_raw_scg": float(loo_raw.max()),
                "seed_mean_normalized_scg": seed_mean,
                "seed_range_normalized_scg": float(seed_values.max() - seed_values.min()),
                "seed_positive_fraction": float(np.mean(seed_values > 0)),
                "frequency_residual_mean": float(group["frequency_residual"].mean()),
                "topology_residual_mean": float(group["topology_residual"].mean()),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("condition profile is empty")
    return result.sort_values(CONDITION_KEYS).reset_index(drop=True)


def _ordered_levels(factor: str, values: Iterable[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if not any(str(value) == str(item) for item in unique):
            unique.append(value)
    preferred = LEVEL_ORDER.get(factor, [])
    ordered = [value for value in preferred if any(str(value) == str(item) for item in unique)]
    remaining = [value for value in unique if not any(str(value) == str(item) for item in ordered)]
    return ordered + sorted(remaining, key=lambda value: str(value))


def _background_key(columns: list[str], values: tuple[Any, ...]) -> str:
    return "|".join(f"{column}={value}" for column, value in zip(columns, values))


def build_factor_contrasts(profile: pd.DataFrame) -> pd.DataFrame:
    """Enumerate all pairwise contrasts for each factor, without a decision gate."""
    _require(profile, CONDITION_KEYS + ["full_mean_normalized_scg"], "profile")
    if profile.duplicated(CONDITION_KEYS).any():
        raise ValueError("profile has duplicate conditions")

    rows: list[dict[str, Any]] = []
    for factor, backgrounds in FACTOR_BACKGROUNDS.items():
        for background_values, group in profile.groupby(backgrounds, sort=True, dropna=False):
            background_values = tuple(background_values)
            group = group.set_index(factor)
            levels = _ordered_levels(factor, group.index.tolist())
            background = _background_key(backgrounds, background_values)
            for level_a, level_b in combinations(levels, 2):
                value_a = float(group.loc[level_a, "full_mean_normalized_scg"])
                value_b = float(group.loc[level_b, "full_mean_normalized_scg"])
                rows.append(
                    {
                        "factor": factor,
                        "background_key": background,
                        "level_a": str(level_a),
                        "level_b": str(level_b),
                        "value_a_normalized_scg": value_a,
                        "value_b_normalized_scg": value_b,
                        "contrast_normalized_scg": value_b - value_a,
                        "abs_contrast_normalized_scg": abs(value_b - value_a),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("factor contrast table is empty")
    return result.sort_values(["factor", "background_key", "level_a", "level_b"]).reset_index(drop=True)


def build_factor_ranges(profile: pd.DataFrame) -> pd.DataFrame:
    """Summarize the descriptive range across each factor's levels."""
    _require(profile, CONDITION_KEYS + ["full_mean_normalized_scg"], "profile")
    rows: list[dict[str, Any]] = []
    for factor, backgrounds in FACTOR_BACKGROUNDS.items():
        for background_values, group in profile.groupby(backgrounds, sort=True, dropna=False):
            background_values = tuple(background_values)
            by_level = group.groupby(factor, sort=False)["full_mean_normalized_scg"].mean()
            levels = _ordered_levels(factor, by_level.index.tolist())
            values = np.array([float(by_level.loc[level]) for level in levels], dtype=float)
            min_index = int(np.argmin(values))
            max_index = int(np.argmax(values))
            rows.append(
                {
                    "factor": factor,
                    "background_key": _background_key(backgrounds, background_values),
                    "n_levels": int(len(levels)),
                    "levels": ",".join(str(level) for level in levels),
                    "range_normalized_scg": float(values.max() - values.min()),
                    "min_level": str(levels[min_index]),
                    "max_level": str(levels[max_index]),
                    "min_value_normalized_scg": float(values.min()),
                    "max_value_normalized_scg": float(values.max()),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("factor range table is empty")
    return result.sort_values(["factor", "background_key"]).reset_index(drop=True)


def build_factor_summary(ranges: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    _require(ranges, ["factor", "range_normalized_scg"], "ranges")
    _require(contrasts, ["factor", "abs_contrast_normalized_scg"], "contrasts")
    range_summary = (
        ranges.groupby("factor", sort=True)["range_normalized_scg"]
        .agg(
            n_backgrounds="size",
            mean_range_normalized_scg="mean",
            median_range_normalized_scg="median",
            max_range_normalized_scg="max",
        )
        .reset_index()
    )
    contrast_summary = (
        contrasts.groupby("factor", sort=True)["abs_contrast_normalized_scg"]
        .agg(n_pairwise_contrasts="size", mean_abs_contrast_normalized_scg="mean")
        .reset_index()
    )
    return range_summary.merge(contrast_summary, on="factor", validate="one_to_one").sort_values("factor").reset_index(drop=True)


def build_residual_sensitivity(pairs: pd.DataFrame) -> pd.DataFrame:
    """Correlate response effects with both response-blind residual dimensions."""
    keys = ["model_id", "architecture", "graph_id", "role_group", "epsilon"]
    _require(
        pairs,
        keys + [
            "frequency_residual",
            "topology_residual",
            "raw_scg",
            "normalized_scg",
        ],
        "pairs",
    )
    _require_finite(
        pairs,
        ["frequency_residual", "topology_residual", "raw_scg", "normalized_scg"],
        "pairs",
    )
    rows: list[dict[str, Any]] = []
    for key_values, group in pairs.groupby(keys, sort=True, dropna=False):
        key_values = tuple(key_values)
        row = dict(zip(keys, key_values))
        row.update(
            {
                "model_id": str(row["model_id"]),
                "architecture": str(row["architecture"]),
                "graph_id": str(row["graph_id"]),
                "role_group": str(row["role_group"]),
                "epsilon": float(row["epsilon"]),
                "n_pairs": int(len(group)),
                "frequency_residual_mean": float(group["frequency_residual"].mean()),
                "frequency_residual_max": float(group["frequency_residual"].max()),
                "topology_residual_mean": float(group["topology_residual"].mean()),
                "topology_residual_max": float(group["topology_residual"].max()),
                "raw_scg_frequency_residual_corr": _corr(group["frequency_residual"], group["raw_scg"]),
                "raw_scg_topology_residual_corr": _corr(group["topology_residual"], group["raw_scg"]),
                "normalized_scg_frequency_residual_corr": _corr(
                    group["frequency_residual"], group["normalized_scg"]
                ),
                "normalized_scg_topology_residual_corr": _corr(
                    group["topology_residual"], group["normalized_scg"]
                ),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("residual sensitivity table is empty")
    return result.sort_values(keys).reset_index(drop=True)


def build_residual_adjustment(pairs: pd.DataFrame) -> pd.DataFrame:
    """Fit a small residual-sensitivity adjustment for descriptive checking.

    The fitted values are not a replacement for the paired estimand.  They
    answer only whether the observed effect changes materially under a simple
    linear adjustment for the two response-blind residual dimensions.
    """
    keys = ["model_id", "architecture", "graph_id", "role_group", "epsilon"]
    _require(
        pairs,
        keys + ["frequency_residual", "topology_residual", "raw_scg", "normalized_scg"],
        "pairs",
    )
    _require_finite(
        pairs,
        ["frequency_residual", "topology_residual", "raw_scg", "normalized_scg"],
        "pairs",
    )
    rows: list[dict[str, Any]] = []
    for key_values, group in pairs.groupby(keys, sort=True, dropna=False):
        key_values = tuple(key_values)
        frequency = group["frequency_residual"].to_numpy(dtype=float)
        topology = group["topology_residual"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(group)), frequency, topology])
        rank = int(np.linalg.matrix_rank(design))
        row = dict(zip(keys, key_values))
        row.update(
            {
                "model_id": str(row["model_id"]),
                "architecture": str(row["architecture"]),
                "graph_id": str(row["graph_id"]),
                "role_group": str(row["role_group"]),
                "epsilon": float(row["epsilon"]),
                "n_pairs": int(len(group)),
                "design_rank": rank,
                "raw_mean_scg": float(group["raw_scg"].mean()),
                "normalized_mean_scg": float(group["normalized_scg"].mean()),
                "median_frequency_residual": float(np.median(frequency)),
                "median_topology_residual": float(np.median(topology)),
            }
        )
        if rank < 3:
            for name in [
                "raw_intercept",
                "raw_frequency_slope",
                "raw_topology_slope",
                "raw_adjusted_at_median_residual",
                "raw_r2",
                "normalized_intercept",
                "normalized_frequency_slope",
                "normalized_topology_slope",
                "normalized_adjusted_at_median_residual",
                "normalized_r2",
            ]:
                row[name] = float("nan")
        else:
            median_design = np.array([[1.0, np.median(frequency), np.median(topology)]])
            for prefix, response_name in [("raw", "raw_scg"), ("normalized", "normalized_scg")]:
                response = group[response_name].to_numpy(dtype=float)
                coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
                fitted = design @ coefficients
                residual_sum_squares = float(np.sum((response - fitted) ** 2))
                total_sum_squares = float(np.sum((response - response.mean()) ** 2))
                r2 = (
                    1.0 - residual_sum_squares / total_sum_squares
                    if total_sum_squares > 0.0
                    else float("nan")
                )
                row[f"{prefix}_intercept"] = float(coefficients[0])
                row[f"{prefix}_frequency_slope"] = float(coefficients[1])
                row[f"{prefix}_topology_slope"] = float(coefficients[2])
                row[f"{prefix}_adjusted_at_median_residual"] = float((median_design @ coefficients)[0])
                row[f"{prefix}_r2"] = r2
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("residual adjustment table is empty")
    return result.sort_values(keys).reset_index(drop=True)


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def make_summary(
    profile: pd.DataFrame,
    factor_summary: pd.DataFrame,
    residual: pd.DataFrame,
    adjustment: pd.DataFrame,
    route: str,
) -> dict[str, Any]:
    abs_frequency = residual["raw_scg_frequency_residual_corr"].abs().dropna()
    abs_topology = residual["raw_scg_topology_residual_corr"].abs().dropna()
    lomo = profile["lomo_same_sign_fraction"].dropna()
    seed_positive = profile["seed_positive_fraction"].dropna()
    raw_adjustment = adjustment["raw_adjusted_at_median_residual"].sub(
        adjustment["raw_mean_scg"]
    ).abs().dropna()
    normalized_adjustment = adjustment["normalized_adjusted_at_median_residual"].sub(
        adjustment["normalized_mean_scg"]
    ).abs().dropna()
    return _safe_json(
        {
            "status": "P2_H1_HETEROGENEITY_DIAGNOSIS_COMPLETE",
            "checkpoint": "P2_H1_HETEROGENEITY_DIAGNOSIS_COMPLETE",
            "scientific_route_retained": route,
            "interpretation_scope": "descriptive_reanalysis_of_frozen_P2_fracture_outputs",
            "no_threshold_or_p_value_gate": True,
            "input_rows": {
                "conditions": int(len(profile)),
                "residual_strata": int(len(residual)),
            },
            "condition_direction": {
                "positive": int((profile["direction"] == "positive").sum()),
                "negative": int((profile["direction"] == "negative").sum()),
                "zero": int((profile["direction"] == "zero").sum()),
            },
            "lomo_sign_retention": {
                "min": float(lomo.min()) if len(lomo) else None,
                "median": float(lomo.median()) if len(lomo) else None,
                "max": float(lomo.max()) if len(lomo) else None,
                "fraction_exactly_one": float(np.mean(np.isclose(lomo, 1.0))) if len(lomo) else None,
            },
            "seed_direction": {
                "unanimous_fraction": float(
                    np.mean(np.isclose(seed_positive, 0.0) | np.isclose(seed_positive, 1.0))
                )
                if len(seed_positive)
                else None,
                "range_median": float(profile["seed_range_normalized_scg"].median()),
                "range_max": float(profile["seed_range_normalized_scg"].max()),
            },
            "residual_association": {
                "raw_frequency_abs_corr_median": float(abs_frequency.median()) if len(abs_frequency) else None,
                "raw_frequency_abs_corr_max": float(abs_frequency.max()) if len(abs_frequency) else None,
                "raw_topology_abs_corr_median": float(abs_topology.median()) if len(abs_topology) else None,
                "raw_topology_abs_corr_max": float(abs_topology.max()) if len(abs_topology) else None,
            },
            "residual_adjustment": {
                "n_full_rank_strata": int((adjustment["design_rank"] == 3).sum()),
                "raw_median_abs_shift": float(raw_adjustment.median()) if len(raw_adjustment) else None,
                "raw_max_abs_shift": float(raw_adjustment.max()) if len(raw_adjustment) else None,
                "normalized_median_abs_shift": float(normalized_adjustment.median())
                if len(normalized_adjustment)
                else None,
                "normalized_max_abs_shift": float(normalized_adjustment.max())
                if len(normalized_adjustment)
                else None,
            },
            "factor_summary": factor_summary.to_dict(orient="records"),
        }
    )


def _format_number(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.6g}"


def render_report(
    output_path: Path,
    summary: dict[str, Any],
    profile: pd.DataFrame,
    factor_summary: pd.DataFrame,
    residual: pd.DataFrame,
    adjustment: pd.DataFrame,
    stats_dir: str,
) -> None:
    factor_lines = [
        "| factor | backgrounds | median range (normalized SCG) | max range | pairwise contrasts |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in factor_summary.itertuples(index=False):
        factor_lines.append(
            f"| {row.factor} | {int(row.n_backgrounds)} | "
            f"{_format_number(row.median_range_normalized_scg)} | "
            f"{_format_number(row.max_range_normalized_scg)} | "
            f"{int(row.n_pairwise_contrasts)} |"
        )

    positive = profile.nlargest(3, "full_mean_normalized_scg")
    negative = profile.nsmallest(3, "full_mean_normalized_scg")
    residual_frequency = residual["raw_scg_frequency_residual_corr"].abs().dropna()
    residual_topology = residual["raw_scg_topology_residual_corr"].abs().dropna()
    topology_index = residual["raw_scg_topology_residual_corr"].abs().idxmax()
    topology_row = residual.loc[topology_index]
    lines = [
        "# P2-H1 有界异质性诊断",
        "",
        "- checkpoint: `P2_H1_HETEROGENEITY_DIAGNOSIS_COMPLETE`",
        f"- scientific route retained: `{summary['scientific_route_retained']}`",
        f"- source: `{stats_dir}`",
        "- status: descriptive reanalysis only; no model inference was run",
        "",
        "## 目的与边界",
        "",
        "本诊断复用已经冻结的 P2 fracture continuity 统计产物，在比赛层级检查架构、图、角色和能量的异质性，并分别计算频率/拓扑匹配残差与效应的相关性。它不使用固定数值阈值或 p-value 作为升级到 P3 的门槛，也不把比赛、模型 seed 或 intervention draw 当作独立重复。",
        "",
        "## 总览",
        "",
        f"- 条件数：{summary['input_rows']['conditions']}；residual strata：{summary['input_rows']['residual_strata']}。",
        f"- 条件方向：positive={summary['condition_direction']['positive']}，negative={summary['condition_direction']['negative']}，zero={summary['condition_direction']['zero']}。",
        f"- LOMO 同号比例：median={_format_number(summary['lomo_sign_retention']['median'])}，min={_format_number(summary['lomo_sign_retention']['min'])}；其中精确为 1 的条件占 {_format_number(summary['lomo_sign_retention']['fraction_exactly_one'])}。",
        f"- 模型 seed 方向一致的条件占 {_format_number(summary['seed_direction']['unanimous_fraction'])}；seed range 的 median/max 为 {_format_number(summary['seed_direction']['range_median'])}/{_format_number(summary['seed_direction']['range_max'])}。",
        f"- |frequency residual correlation|：median={_format_number(summary['residual_association']['raw_frequency_abs_corr_median'])}，max={_format_number(summary['residual_association']['raw_frequency_abs_corr_max'])}。",
        f"- |topology residual correlation|：median={_format_number(summary['residual_association']['raw_topology_abs_corr_median'])}，max={_format_number(summary['residual_association']['raw_topology_abs_corr_max'])}。",
        f"- 简单 residual adjustment 在完整秩 strata 中的 normalized SCG 中位绝对偏移为 {_format_number(summary['residual_adjustment']['normalized_median_abs_shift'])}，最大为 {_format_number(summary['residual_adjustment']['normalized_max_abs_shift'])}；该调整仅作敏感性检查。",
        f"- 最大 topology residual 相关性出现在 `{topology_row['model_id']}/{topology_row['graph_id']}/{topology_row['role_group']}/eps={topology_row['epsilon']}`，相关系数={_format_number(topology_row['raw_scg_topology_residual_corr'])}；因此不能把整体中位数较小表述为所有条件均不受残差影响。",
        "",
        "## 因素范围（描述性）",
        "",
        "以下范围是在其他因素固定后，跨该因素水平的 normalized SCG 范围。不同因素的水平数不同，因此只作异质性定位，不宣称因素之间存在严格的方差贡献排序。",
        "",
        *factor_lines,
        "",
        "## 极端条件示例",
        "",
        "最大正向条件：",
        "",
    ]
    for row in positive.itertuples(index=False):
        lines.append(
            f"- `{row.architecture}/{row.graph_id}/{row.role_group}/eps={row.epsilon}`: "
            f"normalized SCG={_format_number(row.full_mean_normalized_scg)}，"
            f"match positive fraction={_format_number(row.match_positive_fraction)}。"
        )
    lines.extend(["", "最大负向条件：", ""])
    for row in negative.itertuples(index=False):
        lines.append(
            f"- `{row.architecture}/{row.graph_id}/{row.role_group}/eps={row.epsilon}`: "
            f"normalized SCG={_format_number(row.full_mean_normalized_scg)}，"
            f"match positive fraction={_format_number(row.match_positive_fraction)}。"
        )
    lines.extend(
        [
            "",
            "## 路由解释",
            "",
            "当前结果继续保留 `mixed_or_graph_specific`。本报告只把异质性结构显式化，不把某一条件的正负方向提升为普遍机制，也不因为 LOMO 同号就消除架构/图/角色依赖。是否进入 P3，应在阅读这些分层结果后作开放式研究决策。",
            "",
            "## 产物",
            "",
            "- `condition_profile.csv`：36 个架构×图×角色×能量条件。",
            "- `factor_contrasts.csv`、`factor_ranges.csv`、`factor_summary.csv`：因素对比与描述性范围。",
            "- `residual_sensitivity.csv`：108 个 model×图×角色×能量 residual strata，同时包含频率和拓扑相关性。",
            "- `residual_adjustment.csv`：每个 residual stratum 的简单线性敏感性调整，仅用于诊断。",
            "- `figures/heterogeneity_heatmap.png`、`figures/factor_ranges.png`：快速查看图。",
            "- `diagnosis_summary.json`、`source_provenance.json`、`MANIFEST_SHA256.txt`：状态与 provenance。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_figures(profile: pd.DataFrame, ranges: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    architectures = _ordered_levels("architecture", profile["architecture"].tolist())
    graphs = _ordered_levels("graph_id", profile["graph_id"].tolist())
    roles = _ordered_levels("role_group", profile["role_group"].tolist())
    energies = _ordered_levels("epsilon", profile["epsilon"].tolist())
    values = profile["full_mean_normalized_scg"].to_numpy(dtype=float)
    limit = float(np.max(np.abs(values))) if len(values) else 1.0
    limit = max(limit, 1e-12)

    fig, axes = plt.subplots(
        len(architectures), len(graphs), figsize=(10, 8), squeeze=False, constrained_layout=True
    )
    image = None
    for row_index, architecture in enumerate(architectures):
        for col_index, graph_id in enumerate(graphs):
            axis = axes[row_index][col_index]
            subset = profile[(profile["architecture"] == architecture) & (profile["graph_id"] == graph_id)]
            matrix = (
                subset.pivot(index="epsilon", columns="role_group", values="full_mean_normalized_scg")
                .reindex(index=energies, columns=roles)
            )
            image = axis.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
            axis.set_xticks(range(len(roles)), labels=roles, rotation=35, ha="right")
            axis.set_yticks(range(len(energies)), labels=[str(value) for value in energies])
            axis.set_title(f"{architecture} / {graph_id}")
            axis.set_xlabel("role")
            axis.set_ylabel("epsilon")
            for y_index in range(len(energies)):
                for x_index in range(len(roles)):
                    value = matrix.iloc[y_index, x_index]
                    if pd.notna(value):
                        axis.text(x_index, y_index, f"{float(value):.1e}", ha="center", va="center", fontsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="normalized SCG")
    fig.suptitle("P2-H1 fracture heterogeneity profile")
    fig.savefig(figures_dir / "heterogeneity_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    order = [factor for factor in FACTOR_BACKGROUNDS if factor in set(ranges["factor"])]
    medians = [float(ranges.loc[ranges["factor"].eq(factor), "range_normalized_scg"].median()) for factor in order]
    axis.bar(order, medians, color="#4472c4")
    axis.set_ylabel("median normalized SCG range")
    axis.set_title("Descriptive factor ranges")
    axis.tick_params(axis="x", rotation=25)
    fig.savefig(figures_dir / "factor_ranges.png", dpi=180)
    plt.close(fig)


def run_diagnosis(stats_dir: Path, output_dir: Path, route: str = "mixed_or_graph_specific") -> dict[str, Any]:
    stats_dir = stats_dir.resolve()
    output_dir = output_dir.resolve()
    sources = {
        "architecture_match": stats_dir / "architecture_match_effects.parquet",
        "model_match": stats_dir / "model_match_effects.parquet",
        "leave_one_out": stats_dir / "leave_one_match_out.parquet",
        "pairs": stats_dir / "fracture_pair_effects.parquet",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing P2 statistics inputs: {missing}")

    architecture_match = pd.read_parquet(sources["architecture_match"])
    model_match = pd.read_parquet(sources["model_match"])
    leave_one_out = pd.read_parquet(sources["leave_one_out"])
    pairs = pd.read_parquet(sources["pairs"])
    profile = build_condition_profile(architecture_match, model_match, leave_one_out)
    contrasts = build_factor_contrasts(profile)
    ranges = build_factor_ranges(profile)
    factor_summary = build_factor_summary(ranges, contrasts)
    residual = build_residual_sensitivity(pairs)
    adjustment = build_residual_adjustment(pairs)
    summary = make_summary(profile, factor_summary, residual, adjustment, route)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_dir / "condition_profile.csv", index=False, float_format="%.12g")
    contrasts.to_csv(output_dir / "factor_contrasts.csv", index=False, float_format="%.12g")
    ranges.to_csv(output_dir / "factor_ranges.csv", index=False, float_format="%.12g")
    factor_summary.to_csv(output_dir / "factor_summary.csv", index=False, float_format="%.12g")
    residual.to_csv(output_dir / "residual_sensitivity.csv", index=False, float_format="%.12g")
    adjustment.to_csv(output_dir / "residual_adjustment.csv", index=False, float_format="%.12g")
    _write_json(output_dir / "diagnosis_summary.json", summary)
    _write_json(
        output_dir / "source_provenance.json",
        {
            "source_run": "p2_fracture_continuity_v1",
            "stats_dir": _project_relative(stats_dir),
            "inputs": {
                name: {
                    "path": _project_relative(path),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in sources.items()
            },
            "rows": {
                "architecture_match": len(architecture_match),
                "model_match": len(model_match),
                "leave_one_out": len(leave_one_out),
                "pairs": len(pairs),
            },
            "bioinformatics_isolation": True,
        },
    )
    _write_figures(profile, ranges, output_dir)
    render_report(
        output_dir / "report.md",
        summary,
        profile,
        factor_summary,
        residual,
        adjustment,
        _project_relative(stats_dir),
    )
    _write_json(
        output_dir / "pipeline_summary.json",
        {
            "status": "P2_H1_HETEROGENEITY_DIAGNOSIS_COMPLETE",
            "checkpoint": "P2_H1_HETEROGENEITY_DIAGNOSIS_COMPLETE",
            "source_run": "p2_fracture_continuity_v1",
            "scientific_route_retained": route,
            "n_conditions": len(profile),
            "n_factor_contrasts": len(contrasts),
            "n_residual_strata": len(residual),
            "n_residual_adjustment_strata": len(adjustment),
            "model_inference_run": False,
            "bioinformatics_isolation": True,
        },
    )

    manifest_lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST_SHA256.txt":
            manifest_lines.append(f"{_sha256(path)}  {path.relative_to(output_dir)}")
    (output_dir / "MANIFEST_SHA256.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("artifacts/phase2/p2_fracture_continuity_v1/statistics"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase2/p2_heterogeneity_diagnosis_v1"),
    )
    parser.add_argument("--route", default="mixed_or_graph_specific")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_diagnosis(args.stats_dir, args.output_dir, route=args.route)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
