import numpy as np
import pandas as pd
import pytest

from scripts.p2_heterogeneity_diagnosis import (
    build_condition_profile,
    build_factor_contrasts,
    build_factor_ranges,
    build_residual_adjustment,
    build_residual_sensitivity,
)


def _small_tables():
    rows = []
    model_rows = []
    loo_rows = []
    for architecture, offset in (("A", 0.0), ("B", 2.0)):
        match_values = {"m1": 1.0 + offset, "m2": 2.0 + offset, "m3": 3.0 + offset}
        for match_id, value in match_values.items():
            rows.append(
                {
                    "architecture": architecture,
                    "source_match_id": match_id,
                    "graph_id": "g",
                    "role_group": "r",
                    "epsilon": 0.25,
                    "raw_scg": value,
                    "normalized_scg": value / 10.0,
                    "frequency_residual": 0.1,
                    "topology_residual": 0.2,
                    "n_model_seeds": 1,
                }
            )
            model_rows.append(
                {
                    "model_id": f"{architecture.lower()}_seed1",
                    "architecture": architecture,
                    "model_seed": 1,
                    "source_match_id": match_id,
                    "graph_id": "g",
                    "role_group": "r",
                    "epsilon": 0.25,
                    "raw_scg": value,
                    "normalized_scg": value / 10.0,
                }
            )
        for left_out in match_values:
            remaining = [v for k, v in match_values.items() if k != left_out]
            loo_rows.append(
                {
                    "architecture": architecture,
                    "graph_id": "g",
                    "role_group": "r",
                    "epsilon": 0.25,
                    "left_out_match_id": left_out,
                    "n_matches": 2,
                    "raw_scg_mean": np.mean(remaining),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(model_rows), pd.DataFrame(loo_rows)


def test_condition_profile_tracks_match_and_seed_robustness():
    architecture_match, model_match, loo = _small_tables()
    profile = build_condition_profile(architecture_match, model_match, loo)

    assert list(profile["architecture"]) == ["A", "B"]
    assert profile["n_matches"].tolist() == [3, 3]
    assert np.allclose(profile["full_mean_normalized_scg"], [0.2, 0.4])
    assert np.allclose(profile["lomo_same_sign_fraction"], 1.0)
    assert np.allclose(profile["seed_range_normalized_scg"], 0.0)


def test_factor_contrasts_and_ranges_are_explicit_and_deterministic():
    rows = []
    for architecture, a in (("DeepSets-AE", 0.0), ("Phase-GAT", 1.0)):
        for graph_id, g in (("delaunay", 0.0), ("knn4", 0.5)):
            for role_group, r in (("defender", 0.0), ("forward", 1.0), ("midfielder", 2.0)):
                for epsilon, e in ((0.25, 0.0), (0.5, 1.0)):
                    rows.append(
                        {
                            "architecture": architecture,
                            "graph_id": graph_id,
                            "role_group": role_group,
                            "epsilon": epsilon,
                            "full_mean_normalized_scg": a + g + r + e,
                        }
                    )
    profile = pd.DataFrame(rows)
    contrasts = build_factor_contrasts(profile)
    ranges = build_factor_ranges(profile)

    assert set(contrasts["factor"]) == {"architecture", "graph_id", "role_group", "epsilon"}
    assert len(contrasts) == 60
    role_mid = contrasts[
        (contrasts["factor"] == "role_group")
        & (contrasts["level_a"] == "defender")
        & (contrasts["level_b"] == "midfielder")
    ]
    assert np.allclose(role_mid["contrast_normalized_scg"], 2.0)
    assert len(ranges) == 44


def test_residual_sensitivity_reports_frequency_and_topology():
    pairs = pd.DataFrame(
        {
            "model_id": ["m"] * 4,
            "architecture": ["A"] * 4,
            "graph_id": ["g"] * 4,
            "role_group": ["r"] * 4,
            "epsilon": [0.25] * 4,
            "frequency_residual": [0.1, 0.2, 0.3, 0.4],
            "topology_residual": [0.4, 0.3, 0.2, 0.1],
            "raw_scg": [1.0, 2.0, 3.0, 4.0],
            "normalized_scg": [0.1, 0.2, 0.3, 0.4],
        }
    )
    result = build_residual_sensitivity(pairs)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["raw_scg_frequency_residual_corr"] > 0.99
    assert row["raw_scg_topology_residual_corr"] < -0.99
    assert row["normalized_scg_frequency_residual_corr"] > 0.99


def test_residual_adjustment_is_descriptive_and_records_design_rank():
    pairs = pd.DataFrame(
        {
            "model_id": ["m"] * 5,
            "architecture": ["A"] * 5,
            "graph_id": ["g"] * 5,
            "role_group": ["r"] * 5,
            "epsilon": [0.25] * 5,
            "frequency_residual": [0.1, 0.2, 0.3, 0.4, 0.5],
            "topology_residual": [0.4, 0.4, 0.2, 0.1, 0.3],
            "raw_scg": [1.0, 2.0, 3.0, 4.0, 5.0],
            "normalized_scg": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    result = build_residual_adjustment(pairs)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["design_rank"] == 3
    assert np.isfinite(row["raw_adjusted_at_median_residual"])
    assert np.isfinite(row["raw_topology_slope"])


def test_missing_columns_fail_closed():
    with pytest.raises(ValueError, match="missing"):
        build_residual_sensitivity(pd.DataFrame({"raw_scg": [1.0]}))
