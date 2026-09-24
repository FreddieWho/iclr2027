"""Contract tests for the R03 augmentation-state audit (`data_lineage.py --mode augmentation`).

These pin the audit's threshold semantics, the bank state convention, the mined-flip
reconstruction, and the exact-overlap inventory it produced.  They fail if the
detector stops detecting, if the singles bank is read as edited states when it is
not, or if the recorded contamination inventory changes.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_lineage as dl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "artifacts/e1a933_review/data_lineage"


def test_query_states_exact_near_and_clear_bands():
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(200, 8))
    ids = [f"T{i}" for i in range(len(targets))]
    assert dl.query_states(targets, targets, ids)["n_exact"] == len(targets)
    assert dl.query_states(targets + 5e-5, targets, ids)["n_near"] == len(targets)
    clear = dl.query_states(targets + 5e-4, targets, ids)
    assert clear["n_exact"] == 0 and clear["n_near"] == 0


def test_band_is_linf_not_euclidean():
    # 9e-5 in every coordinate: Linf = 9e-5 (near band), L2 = 2.5e-4 (would be clear).
    targets = np.zeros((1, 8))
    res = dl.query_states(np.full((1, 8), 9e-5), targets, ["T0"])
    assert res["n_near"] == 1 and res["n_exact"] == 0


def test_near_tolerance_is_far_below_the_designed_candidate_spacing():
    # Two distinct designed candidate edits are >=0.015 apart in Linf; the near band
    # must stay orders of magnitude below that.
    assert dl.TOL_NEAR < 0.015 / 100
    assert dl.TOL_EXACT < dl.TOL_NEAR


def test_bank_singles_are_start_plus_edit_not_start_alone():
    """build_bank.py:82 stores Sx = start state, Se = edit; Sx alone is the parent."""
    t = dl.bank_targets("d09fresh665")
    X16 = dl.pool_arrays("16N")[0].astype(np.float64).reshape(-1, 8)
    d_start, _ = cKDTree(X16).query(t["single_start_states"], p=np.inf)
    d_edit, _ = cKDTree(X16).query(t["single_states"], p=np.inf)
    assert (d_start <= dl.TOL_EXACT).all()      # start states are training-pool scenes
    assert (d_edit > dl.TOL_NEAR).all()         # evaluated states are not


def test_mined_flip_rebuild_is_byte_identical_to_the_stored_bank():
    stored = np.load(ROOT / "artifacts/discovery_campaign/r04b_s11/mined.npz")
    ii, ee, states, mined = dl.mined_flips("N")
    assert len(ii) == 1534
    assert np.array_equal(ee, np.stack([stored[f"edit_{k}"] for k in range(len(ii))]))
    assert np.array_equal(ii, stored["meta"][:, 0].astype(np.int64))


@pytest.mark.parametrize("bank,expected", [("dev512", 711), ("d03E", 5514), ("d09fresh665", 708)])
def test_bank_quartet_state_counts(bank, expected):
    assert len(dl.bank_targets(bank)["quartet_states"]) == expected


def test_audit_json_detector_power_and_contamination_inventory():
    audit = json.loads((LINEAGE / "AUGMENTATION_STATE_AUDIT.json").read_text())
    for bank in ("dev512", "d03E", "d09fresh665"):
        power = audit["detector_power"][bank]
        assert power["identical"] == power["n_controls"]
        assert power["perturb_5e_5_near"] == power["n_controls"]
        assert power["perturb_5e_4_flagged"] == 0
    assert audit["detector_power"]["planted_16N_mined_flip_in_d03E"]["n_exact"] == 1
    hits = {k: {b: v["banks"][b]["quartet"]["n_exact"] + v["banks"][b]["single"]["n_exact"]
                for b in v["banks"]} for k, v in audit["sets"].items()}
    assert hits["16N:mined_flip"]["d09fresh665"] == 761
    assert hits["4N:mined_flip"]["d09fresh665"] == 163
    for key, per_bank in hits.items():
        for bank, n in per_bank.items():
            if bank == "d09fresh665" and key.endswith(("mined_flip", "g8_orbit_mined")):
                continue
            assert n == 0, (key, bank, n)


def test_matrix_csv_columns_are_filled_and_every_source_resolved():
    rows = list(csv.DictReader((LINEAGE / "MODEL_BANK_OVERLAP.csv").open()))
    assert len(rows) == 288
    assert not any("NOT_RUN" in r["augmentation_exact_state_audit"] for r in rows)
    assert all(r["train_source_resolution"].startswith("resolved:") for r in rows)
    res = json.loads((LINEAGE / "TRAIN_SOURCE_RESOLUTION.json").read_text())
    assert res["n_checkpoints"] == 96 and res["resolved"] == 96
    assert res["by_pool"] == {"N": 63, "4N": 27, "16N": 6}
