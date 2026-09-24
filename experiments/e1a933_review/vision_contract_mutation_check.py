"""Mutation check for R05/R06 regression guards.

Restores each pre-fix behaviour in-process and confirms the corresponding contract test
FAILS. This is the evidence for the R10 requirement that "restoring the old error must
make the test fail" - passing tests alone do not show the guards are live.

Finding recorded by this check: the four `normalization_commutes_<p>` checks inherited
from `vision_regression.py` could not fail (see the note in `test_vision_contract.py`).
They were replaced by the two group-membership guards exercised below.

Run:  python experiments/e1a933_review/vision_contract_mutation_check.py [--out DIR]
Writes: mutation_check.json  (and prints a summary)
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/f095_campaign"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

torch.set_num_threads(2)

import d04_vision_train as d04  # noqa: E402
import u06_relation_distill as u06  # noqa: E402

import test_vision_contract as tvc  # noqa: E402


def _fails(fn):
    """Return (failed, detail): True when the contract test raises."""
    try:
        fn()
        return False, "test PASSED under the restored bug (guard is NOT live)"
    except AssertionError as exc:
        return True, f"AssertionError: {exc}"
    except Exception as exc:  # any hard failure also counts as detected
        return True, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/e1a933_review/vision_contract_mutation")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    results = {}

    # ---- R06: restore the pre-fix train_mode (bare net.train(), BN buffers update) ----
    good_train_mode = d04.train_mode

    def buggy_train_mode(net, mode):
        net.train()  # old behaviour: no eval() for the frozen probe

    d04.train_mode = buggy_train_mode
    tvc.train_mode = buggy_train_mode
    try:
        failed, detail = _fails(tvc.test_headonly_probe_freezes_parameters_and_buffers)
    finally:
        d04.train_mode = good_train_mode
        tvc.train_mode = good_train_mode
    results["R06_frozen_probe_guard"] = {
        "restored_bug": "train_mode() falls back to bare net.train() (BN buffers update)",
        "test": "test_headonly_probe_freezes_parameters_and_buffers",
        "guard_fired": failed,
        "detail": detail,
    }
    # control: with the fix in place the same test must pass
    passed_after_restore, _ = _fails(tvc.test_headonly_probe_freezes_parameters_and_buffers)
    results["R06_frozen_probe_guard"]["passes_with_fix"] = not passed_after_restore

    # ---- R05: restore the old ordered (index-fixed) objective as `matched_mse` ----
    good_matched_mse = u06.matched_mse

    def ordered_mse(pred, orbit, reduction="mean"):
        losses = ((pred - orbit[:, 0]) ** 2).mean(dim=-1)  # old: fixed endpoint indexing
        return losses.mean() if reduction == "mean" else losses

    u06.matched_mse = ordered_mse
    tvc.matched_mse = ordered_mse
    try:
        failed, detail = _fails(tvc.test_one_consistent_group_element_gives_zero_matched_mse)
        failed2, detail2 = _fails(tvc.test_matched_mse_is_invariant_to_the_hidden_group_element)
    finally:
        u06.matched_mse = good_matched_mse
        tvc.matched_mse = good_matched_mse
    results["R05_matching_guard"] = {
        "restored_bug": "matched_mse replaced by the ordered index-fixed objective",
        "test": [
            "test_one_consistent_group_element_gives_zero_matched_mse",
            "test_matched_mse_is_invariant_to_the_hidden_group_element",
        ],
        "guard_fired": bool(failed and failed2),
        "detail": detail,
        "detail_invariance": detail2,
    }
    passed_after_restore, _ = _fails(tvc.test_one_consistent_group_element_gives_zero_matched_mse)
    results["R05_matching_guard"]["passes_with_fix"] = not passed_after_restore

    # ---- R05: colour swap must not be accepted as a member of the declared group ----
    # Failure mode this characterises: rel10 becoming colour-blind, so that the colour swap
    # turns into a symmetry of the target and H would silently be a larger group than declared.
    good_rel10 = u06.rel10
    good_h = u06.H_ENDPOINTS

    def colour_blind_rel10(x):
        out = good_rel10(x)
        return np.sort(np.asarray(out, float), axis=-1)

    u06.rel10 = colour_blind_rel10
    try:
        failed, detail = _fails(tvc.test_colour_swap_is_outside_the_declared_group)
    finally:
        u06.rel10 = good_rel10
    results["R05_group_boundary_guard"] = {
        "restored_bug": "rel10 made colour-blind (sorted distances) so the colour swap becomes a symmetry",
        "test": "test_colour_swap_is_outside_the_declared_group",
        "guard_fired": failed,
        "detail": detail,
    }
    passed_after_restore, _ = _fails(tvc.test_colour_swap_is_outside_the_declared_group)
    results["R05_group_boundary_guard"]["passes_with_fix"] = not passed_after_restore

    # ---- R05: equivariance must fail when a non-group permutation is declared ----
    u06.H_ENDPOINTS = np.array([[0, 1, 2, 3], [1, 0, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2]])
    tvc.H_ENDPOINTS = u06.H_ENDPOINTS
    try:
        failed, detail = _fails(tvc.test_orbit_equivariant_under_declared_group)
    finally:
        u06.H_ENDPOINTS = good_h
        tvc.H_ENDPOINTS = good_h
    results["R05_equivariance_guard"] = {
        "restored_bug": "H_ENDPOINTS contains a permutation that is not a symmetry of rel10",
        "test": "test_orbit_equivariant_under_declared_group",
        "guard_fired": failed,
        "detail": detail,
    }
    passed_after_restore, _ = _fails(tvc.test_orbit_equivariant_under_declared_group)
    results["R05_equivariance_guard"]["passes_with_fix"] = not passed_after_restore

    summary = {
        "all_guards_fire": all(v["guard_fired"] for v in results.values()),
        "all_pass_with_fix": all(v["passes_with_fix"] for v in results.values()),
        "mutations": results,
    }
    (a.out / "mutation_check.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert summary["all_guards_fire"], "a mutation guard did not fire"
    assert summary["all_pass_with_fix"], "a test failed with the fix restored"


if __name__ == "__main__":
    main()
