"""Tests for T5R5 hidden-confirmation gate math (pure functions, no reserved reads)."""

import importlib.util
import sys
from pathlib import Path

RUNNER = Path(__file__).parents[1] / "scripts" / "run_t5r5_hidden_confirmation.py"
SPEC = importlib.util.spec_from_file_location("run_t5r5_hidden_for_tests", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_summary(pair=0.95, geom=0.70, zone=0.96, cent=0.04, z_mode=1e-8, latcorr=0.15,
                 ref_pair=None, ref_geom=None):
    ref_pair = pair - 0.005 if ref_pair is None else ref_pair
    ref_geom = geom - 0.02 if ref_geom is None else ref_geom
    return [
        {"model_id": "update_ratio_2to1", "field_zone_f1_mean": zone, "centroid_mae_mean": cent,
         "z_mode_translation_mean": z_mode, "latent_correlation_mean": latcorr,
         "pair_accuracy_mean": pair, "geometry_spearman_mean": geom,
         "geometry_spearman_by_seed": [geom, geom, geom]},
        {"model_id": "fixed_dual_channel_shared_phase_gat", "field_zone_f1_mean": zone, "centroid_mae_mean": cent,
         "z_mode_translation_mean": z_mode, "latent_correlation_mean": latcorr,
         "pair_accuracy_mean": ref_pair, "geometry_spearman_mean": ref_geom,
         "geometry_spearman_by_seed": [ref_geom] * 3},
        {"model_id": "raw_single_channel_phase_gat_team_mean", "field_zone_f1_mean": zone, "centroid_mae_mean": cent + 0.01,
         "z_mode_translation_mean": 0.2, "latent_correlation_mean": latcorr,
         "pair_accuracy_mean": 0.80, "geometry_spearman_mean": -0.4,
         "geometry_spearman_by_seed": [-0.4] * 3},
    ]


def make_intervention(full=0.7, base=0.2, diag=0.65, direction=0.8):
    block = {}
    for model in ("update_ratio_2to1", "fixed_dual_channel_shared_phase_gat"):
        block[model] = {s: {"status": "OK", "valid_pairs": 200, "full_spearman": full,
                            "diagonal_spearman": diag, "spectral_baseline_spearman": base,
                            "direction_accuracy": direction, "full_minus_baseline": full - base}
                        for s in ("11", "23", "47")}
    return block


def test_strong_pass_when_all_gates_hold() -> None:
    gates = MODULE.decide_gates(make_summary(), make_intervention(), 400)
    assert gates["task_pass"] is True
    assert gates["verdict"] == "T5R5_PASS_STRONG"


def test_mixed_when_intervention_weaker_but_positive() -> None:
    gates = MODULE.decide_gates(make_summary(), make_intervention(full=0.4, base=0.2, diag=0.35), 400)
    assert gates["verdict"] == "T5R5_PASS_TASK_MECHANISM_MIXED"


def test_mechanism_survives_when_task_fails() -> None:
    summary = make_summary(pair=0.90, ref_pair=0.95)
    gates = MODULE.decide_gates(summary, make_intervention(full=0.4, base=0.2, diag=0.35), 400)
    assert gates["task_pass"] is False
    assert gates["verdict"] == "T5R5_FAIL_SELECTED_CANDIDATE_BUT_MECHANISM_SURVIVES"


def test_mechanism_fails_when_nothing_positive() -> None:
    bad = make_intervention(full=0.1, base=0.3, direction=0.4)
    gates = MODULE.decide_gates(make_summary(pair=0.90), bad, 400)
    assert gates["verdict"] == "T5R5_FAIL_MECHANISM"


def test_underpowered_pairs_skip_pair_checks() -> None:
    summary = make_summary()
    for entry in summary:
        entry.pop("pair_accuracy_mean", None)
        entry.pop("geometry_spearman_mean", None)
        entry.pop("geometry_spearman_by_seed", None)
    gates = MODULE.decide_gates(summary, make_intervention(), 12)
    assert gates["pair_underpowered"] is True
    assert gates["task_checks"]["pair_and_geometry"] == "UNDERpowered_SKIPPED"
