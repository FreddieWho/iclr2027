"""L007 seed-variance lane: schema and helper checks only.

No scientific conclusion is asserted here.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDDIR = ROOT / "artifacts" / "next_novelty" / "l007_seed"
SEEDS = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]


def test_manifests_record_recipe_and_flip_sha():
    for s in SEEDS:
        m = json.load(open(SEEDDIR / ("s%d" % s) / "manifest.json"))
        assert m["method"] == "flipmine" and m["mine_seed"] == 5
        assert m["epochs"] == 300 and m["n_flips"] == 1534
        assert m["flip_list_sha256"].startswith("abf78112")


def test_checkpoints_exist_for_all_seeds():
    for s in SEEDS:
        assert (SEEDDIR / ("s%d" % s) / "model.pt").exists()


def test_paths_table_shape():
    rows = list(csv.DictReader(open(SEEDDIR / "L007_PATHS.csv")))
    assert len(rows) == 200
    cols = ["terr_s%d" % s for s in SEEDS] + ["terr_ens"]
    for r in rows:
        for c in cols:
            assert r[c] == "" or 0.0 <= float(r[c]) <= 1.0


def test_summary_keys_and_verdict_present():
    s = json.load(open(SEEDDIR / "L007_SEEDVARIANCE.json"))
    assert s["n_paths"] == 200 and s["seeds"] == SEEDS
    for k in ("mean_pairwise_spearman", "ensemble_improvement_vs_mean",
              "criterion_met", "verdict"):
        assert k in s
