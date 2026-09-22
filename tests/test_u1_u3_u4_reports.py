"""U1/U3/U4 pipeline integrity: recomputable consistency only.

No scientific claim is asserted here; these tests guard that the derived
tables and summaries stay mutually consistent after refactorings.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
U1 = ROOT / "artifacts" / "next_novelty" / "u1_factorial"
U3 = ROOT / "artifacts" / "next_novelty" / "u3_confidence"
U4 = ROOT / "artifacts" / "next_novelty" / "u4_contrast"


def test_u1_perquartet_row_count_and_J_consistency():
    rows = list(csv.DictReader(open(U1 / "U1_PERQUARTET.csv")))
    assert len(rows) == 237 * 3 * 4
    s = json.load(open(U1 / "U1_SUMMARY.json"))
    for seed in (11, 23, 47):
        for arm in ("raw_clean", "raw_flipmine", "relfeat", "relflip"):
            js = [int(r["J"]) for r in rows
                  if int(r["seed"]) == seed and r["arm"] == arm]
            assert len(js) == 237
            mean = round(sum(js) / len(js), 4)
            assert mean == s["seeds"]["s%d" % seed]["arms"][arm]["J"]


def test_u1_identity_holds_everywhere():
    s = json.load(open(U1 / "U1_SUMMARY.json"))
    for seed in (11, 23, 47):
        for arm, v in s["seeds"]["s%d" % seed][
                "repair_vs_rawclean_H"].items():
            assert v["identity_ok"] is True


def test_u3_fixed_set_single_definition_and_sections():
    s = json.load(open(U3 / "U3_SUMMARY.json"))
    for seed in (11, 23, 47):
        d = s["seeds"]["s%d" % seed]
        assert "START" in d["fixed_set_def"]
        assert set(d["Q1_fixed_set"]) == {
            "raw_clean", "raw_flipmine", "relfeat", "relflip"}
        assert set(d["Q2_own_confidence_quartiles"]["relflip"]) == {
            "Q0", "Q1", "Q2", "Q3"}
        assert set(d["preserve"]["relflip"]) >= {
            "static_err", "single_flip_err", "preserve_FA"}


def test_u4_four_arms_three_seeds_with_provenance():
    s = json.load(open(U4 / "U4_SUMMARY.json"))
    for seed in (11, 23, 47):
        arms = s["seeds"]["s%d" % seed]
        assert set(arms) == {"centered_clean", "centered_flip",
                             "sixdist_clean", "sixdist_flip"}
        for v in arms.values():
            assert len(v["model_sha256"]) == 16
            assert v["J"][0] <= v["H"][0]
