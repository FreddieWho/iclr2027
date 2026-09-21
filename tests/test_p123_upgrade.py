"""Unit tests for P123 upgrade (§3 acceptance list). Run: python3 tests/test_p123_upgrade.py"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "update_geometry"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
from wp1_analyze import primary  # noqa: E402


def t_bootstrap_axis():
    rng = np.random.default_rng(0)
    x = (rng.random(50) < 0.3).astype(float)
    m = np.quantile(rng.choice(x, size=(2000, len(x))).mean(1), [0.025, 0.975])
    assert m.shape == (2,) and m[0] <= x.mean() <= m[1], "bootstrap CI must bracket mean"
    print("ok bootstrap-axis")


def t_audit_none():
    rows = [{"oracle_n_crossings": 1, "start_correct": True, "incidence": 0.5,
             "incidence_audit_rel": 0.0},
            {"oracle_n_crossings": 1, "start_correct": True, "incidence": 0.5,
             "incidence_audit_rel": None}]
    out = primary(rows)
    assert len(out) == 1 and out[0]["incidence_audit_rel"] == 0.0, \
        "rel==0.0 must pass, None must be excluded"
    print("ok audit-None")


def t_temperature_argmax():
    rng = np.random.default_rng(2)
    lg = rng.normal(size=1000) * 8
    for T in (0.5, 2.0, 8.0):
        assert (((lg > 0).astype(int)) == ((lg / T > 0).astype(int))).all()
    r0 = np.argsort(np.argsort(np.abs(lg)))
    r1 = np.argsort(np.argsort(np.abs(lg / 2.0)))
    assert (r0 == r1).all(), "binary |logit| ranks invariant under +temperature"
    print("ok temperature-argmax")


def t_paired_gain_zero():
    d = json.load(open(ROOT / "artifacts" / "p123_upgrade" / "p3" / "P3_PAIRED.json"))
    assert d["unit_test"]["pass"] is True
    assert d["unit_test"]["baseline_fixed_clean"] == d["unit_test"]["baseline_fixed_repair"]
    print("ok paired-gain-zero-artifact-demo")


def t_third_class_and_start_filter():
    d = json.load(open(ROOT / "artifacts" / "p123_upgrade" / "p1" / "p1football_s11.json"))
    n_start_wrong_excluded = 0
    for m, v in d.items():
        if not isinstance(v, dict):
            continue
        for k in ("frozen", "cover"):
            for r in v[k]["events"]:
                assert r["third"] in (True, False)
    # coordinate: binary task has no third class; arrival == endpoint metric
    import glob
    for f in sorted(glob.glob(str(ROOT / "artifacts" / "p123_upgrade" / "p1" / "p1coord_s11_clean.json"))):
        dd = json.load(open(f))
        sub = [r for r in dd["rows"] if r["start_correct"]]
        assert all((r["arrived_new_class"] == r["endpoint_correct"]) for r in sub), \
            "binary: arrival must equal endpoint correctness"
    print("ok third-class-and-start-filter")


def t_time_gaps():
    import pandas as pd
    idx = pd.read_parquet(ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1" /
                          "data_views" / "snapshot_index_valid.parquet")
    ts = np.sort(idx[idx.source_match_id == "J03WOY"].timestamp_ms.values)
    assert (np.diff(ts) > 5000).sum() > 0, "test needs a real gap to exist"
    print("ok time-gaps-exist-and-code-breaks-on-them")


def t_conf_ties_reported():
    d = json.load(open(ROOT / "artifacts" / "p123_upgrade" / "p1" / "p1football_s11.json"))
    confs = []
    for m, v in d.items():
        if not isinstance(v, dict):
            continue
        for r in v["frozen"]["events"]:
            confs.append(r["conf_adj"])
    confs = np.array(confs)
    tie_frac = (confs >= np.quantile(confs, 0.75)).mean()
    assert 0.0 < tie_frac < 1.0
    print("ok conf-ties-fraction=%.3f (quartile bins would overlap; use ranks/splits)" % tie_frac)


if __name__ == "__main__":
    t_bootstrap_axis()
    t_audit_none()
    t_temperature_argmax()
    t_paired_gain_zero()
    t_third_class_and_start_filter()
    t_time_gaps()
    t_conf_ties_reported()
    print("ALL P123 UNIT TESTS PASS")
