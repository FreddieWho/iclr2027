"""Minimal regression checks for the T5R4 Round 1 framework."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_t5r4_round1.py"
SPEC = spec_from_file_location("run_t5r4_round1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_round1_candidates_use_one_declared_axis() -> None:
    config = MODULE.load_config(MODULE.ROUND1_CONFIG)
    candidates = config["candidates"]
    assert len(candidates) == 4
    assert len({candidate["id"] for candidate in candidates}) == len(candidates)
    assert all(candidate["axis"] for candidate in candidates)


def test_candidate_models_keep_t5r3_parameter_count() -> None:
    counts = {
        sum(parameter.numel() for parameter in MODULE.CandidateTaskModel(candidate["head_placement"]).parameters())
        for candidate in MODULE.load_config(MODULE.ROUND1_CONFIG)["candidates"]
    }
    assert counts == {141769}


def test_mode_head_only_routing_is_declared() -> None:
    candidate = next(
        item for item in MODULE.load_config(MODULE.ROUND1_CONFIG)["candidates"]
        if item["id"] == "routing_mode_head_only"
    )
    assert candidate["gradient_routing"] == "mode_head_only"
