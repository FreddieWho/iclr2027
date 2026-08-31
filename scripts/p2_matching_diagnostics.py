#!/usr/bin/env python3
"""Response-blind diagnostics for P2 candidate matching."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


FORBIDDEN_COLUMN_TOKENS = ("model", "embedding", "response", "scg", "prediction", "logit")


def assert_response_blind_schema(frame: pd.DataFrame) -> None:
    """Fail closed if downstream model information enters matcher selection."""
    forbidden = [
        str(column)
        for column in frame.columns
        if any(token in str(column).lower() for token in FORBIDDEN_COLUMN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"response-blind diagnostics reject columns: {sorted(forbidden)}")


def build_coverage_curve(
    candidates: pd.DataFrame,
    expected_set_ids: Iterable[str],
    levels: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Build nested candidate coverage while retaining unmatched sets in the denominator."""
    assert_response_blind_schema(candidates)
    required = {"matched_set_id", "candidate_id", "frequency_ratio", "eligible"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate diagnostics missing columns: {missing}")
    expected = tuple(dict.fromkeys(str(value) for value in expected_set_ids))
    if not expected:
        raise ValueError("expected_set_ids cannot be empty")
    values = candidates["frequency_ratio"].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("frequency_ratio must be finite and non-negative")
    if levels is None:
        levels = sorted(set(float(value) for value in values))
    ordered_levels = sorted(set(float(value) for value in levels))
    if not ordered_levels or any(not np.isfinite(value) or value < 0 for value in ordered_levels):
        raise ValueError("coverage levels must be finite and non-negative")
    eligible = candidates[candidates["eligible"].astype(bool)].copy()
    expected_set = set(expected)
    rows: list[dict[str, float | int]] = []
    for level in ordered_levels:
        qualifying = eligible.loc[eligible["frequency_ratio"].astype(float) <= level].copy()
        qualifying = qualifying[qualifying["matched_set_id"].astype(str).isin(expected_set)]
        sort_columns = ["frequency_ratio"]
        if "topology_match_score" in qualifying.columns:
            sort_columns.append("topology_match_score")
        sort_columns.append("candidate_id")
        chosen = qualifying.sort_values(sort_columns).drop_duplicates("matched_set_id")
        row: dict[str, float | int] = {
            "frequency_ratio_level": level,
            "n_expected_sets": len(expected),
            "n_matched_sets": int(len(chosen)),
            "coverage": len(chosen) / len(expected),
        }
        for column in (
            "frequency_ratio",
            "rq_difference",
            "band_power_l1_difference",
            "density_difference",
            "cut_weight_difference",
            "topology_match_score",
        ):
            if column in chosen.columns:
                numeric = chosen[column].astype(float)
                row[f"mean_{column}"] = float(numeric.mean()) if len(numeric) else np.nan
                row[f"max_{column}"] = float(numeric.max()) if len(numeric) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
