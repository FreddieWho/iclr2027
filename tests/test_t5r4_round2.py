"""Regression checks for the T5R4 Round 2 update-balance framework."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_t5r4_round2.py"
SPEC = spec_from_file_location("run_t5r4_round2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_round2_has_exactly_two_ratio_candidates() -> None:
    config = MODULE.load_config(MODULE.ROUND2_CONFIG)
    candidates = {candidate["id"]: candidate for candidate in config["candidates"]}
    assert set(candidates) == {"update_ratio_3to1", "update_ratio_2to1"}
    assert all(candidate["axis"] == "shared_encoder_task_update_balance" for candidate in candidates.values())


def test_round2_quotas_match_reference_budget() -> None:
    config = MODULE.load_config(MODULE.ROUND2_CONFIG)
    assert config["axis"]["total_optimizer_steps_per_epoch"] == 420
    for candidate in config["candidates"]:
        assert candidate["context_updates_per_epoch"] + candidate["intrinsic_updates_per_epoch"] == 420
    quotas = {candidate["id"]: (candidate["context_updates_per_epoch"], candidate["intrinsic_updates_per_epoch"]) for candidate in config["candidates"]}
    assert quotas == {"update_ratio_3to1": (315, 105), "update_ratio_2to1": (280, 140)}


def test_round2_freezes_loss_routing_and_normalization() -> None:
    config = MODULE.load_config(MODULE.ROUND2_CONFIG)
    frozen = config["frozen"]
    assert frozen["context_weight"] == 1.0 and frozen["intrinsic_weight"] == 1.0
    assert frozen["gradient_routing"] == "shared"
    assert frozen["layernorm"] == "none"
    assert frozen["seeds"] == [11, 23, 47]
    assert config["data"]["reserved_loaded"] is False


def test_round2_models_keep_t5r3_parameter_count() -> None:
    count = sum(parameter.numel() for parameter in MODULE.T5R3.TaskModel().parameters())
    assert count == 141769


def test_cyclic_stream_covers_pool_without_tail_drop() -> None:
    stream = MODULE.build_cyclic_stream(23052, 11 * 100_000, 315, 80)
    assert stream.shape == (80, 315, 64)
    assert len(np.unique(stream[0])) > 315 * 64 - 23052  # first epoch spans into cycle 2
    assert set(np.unique(stream)) == set(range(23052))
    pair_stream = MODULE.build_cyclic_stream(3750, 11 * 200_000, 140, 80)
    assert pair_stream.shape == (80, 140, 64)
    assert set(np.unique(pair_stream)) == set(range(3750))


def test_cyclic_stream_is_deterministic() -> None:
    first = MODULE.build_cyclic_stream(3750, 23 * 200_000, 105, 80)
    second = MODULE.build_cyclic_stream(3750, 23 * 200_000, 105, 80)
    assert np.array_equal(first, second)
