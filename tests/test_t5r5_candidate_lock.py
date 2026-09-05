"""Tests for the T5R5 candidate lock (no reserved reads)."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
LOCK = json.loads((T5R_ROOT / "candidate_lock.json").read_text(encoding="utf-8"))
CONFIG = yaml.safe_load((ROOT / "configs" / "t5r5_hidden_confirmation.yaml").read_text(encoding="utf-8"))
INTERVENTION = json.loads((T5R_ROOT / "t5r5_intervention_lock.json").read_text(encoding="utf-8"))
SHADOW = json.loads((T5R_ROOT / "t5r6_shadow_confirmation_lock.json").read_text(encoding="utf-8"))


def test_lock_selects_exactly_2to1() -> None:
    assert LOCK["status"] == "T5R5_CANDIDATE_LOCKED"
    assert LOCK["identity"]["selected_candidate"] == "update_ratio_2to1"
    assert LOCK["model"]["update_schedule_per_epoch"] == {"context": 280, "intrinsic": 140, "total": 420}
    assert LOCK["model"]["parameter_count"] == 141769
    assert set(LOCK["model"]["checkpoints"]) == {"11", "23", "47"}


def test_lock_comparison_set_has_no_other_candidates() -> None:
    assert set(LOCK["frozen_comparison_set"]) >= {"selected", "reference", "raw_baseline", "analytic_control"}
    assert LOCK["frozen_comparison_set"]["selected"] == "update_ratio_2to1"
    assert LOCK["frozen_comparison_set"]["retraining"] == "forbidden"
    assert "update_ratio_3to1" not in json.dumps(LOCK["frozen_comparison_set"])


def test_hidden_config_matches_lock() -> None:
    import hashlib

    assert CONFIG["selected_candidate"] == "update_ratio_2to1"
    assert CONFIG["reserved_match"] == "J03WQQ"
    assert CONFIG["intervention"]["epsilon"] == 0.25
    assert CONFIG["candidate_lock_sha256"] == hashlib.sha256((T5R_ROOT / "candidate_lock.json").read_bytes()).hexdigest()
    assert CONFIG["intervention_lock_sha256"] == hashlib.sha256((T5R_ROOT / "t5r5_intervention_lock.json").read_bytes()).hexdigest()


def test_hidden_task_rules_equal_t5r2() -> None:
    rules = LOCK["hidden_task_generation"]["natural_pair_thresholds"]
    assert rules == {
        "positive_internal_max": 0.28,
        "positive_centroid_min": 0.35,
        "hard_negative_centroid_max": 0.12,
        "hard_negative_internal_min": 0.35,
        "minimum_time_separation_s": 4.0,
        "pair_anchor_stride": 5,
        "same_match_same_half": True,
        "no_cross_split": True,
    }


def test_intervention_and_shadow_locks_frozen() -> None:
    assert INTERVENTION["status"] == "T5R5_INTERVENTION_LOCKED"
    assert INTERVENTION["perturbation_magnitude"]["epsilon"] == 0.25
    assert INTERVENTION["matched_pair_construction"]["support_size"] == 4
    assert INTERVENTION["snapshot_sampling"]["max_snapshots"] == 250
    assert SHADOW["status"] == "T5R6_SHADOW_LOCKED"


def test_no_round3_or_p4_release() -> None:
    assert LOCK["firewall"]["round_3"] == "forbidden"
    assert LOCK["firewall"]["P4_AMR"] == "blocked"
    assert not (ROOT / "configs" / "t5r4_round3.yaml").exists()
    assert LOCK["firewall"]["reserved_reads_allowed"] == 1
