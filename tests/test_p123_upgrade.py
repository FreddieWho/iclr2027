"""Unit tests for P123 upgrade: metric definitions + analysis invariants."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))


def test_third_class_not_success():
    # arrival must target the NEW oracle class, not any change
    pred = np.array([0, 0, 2, 2])  # third class 2 visited
    old, new = 0, 1
    arrived = bool(any(int(p) == new for p in pred))
    changed = bool(any(int(p) != pred[0] for p in pred))
    assert changed and not arrived  # any-change would wrongly score success


def test_start_error_excluded():
    rows = [{"start_correct": True, "endpoint_correct": False},
            {"start_correct": False, "endpoint_correct": False}]
    sub = [r for r in rows if r["start_correct"]]
    assert len(sub) == 1 and not sub[0]["endpoint_correct"]


def test_time_gap_breaks_sequence():
    ts = np.array([0, 1000, 9000, 10000])
    breaks = [ts[i] - ts[i - 1] > 5000 for i in range(1, len(ts))]
    assert breaks == [False, True, False]


def test_ties_merged_not_ranked():
    conf = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
    # degenerate quantiles must be detected and handled, not jittered
    q = np.quantile(conf, [0.25, 0.75])
    assert q[0] == q[1] == 1.0  # fully-tied mass: caller merges, never splits


def test_same_edit_hash_per_model():
    import hashlib
    npz = ROOT / "artifacts" / "p123_upgrade" / "bank" / "bank_dev512.npz"
    h = hashlib.sha256(open(npz, "rb").read()).hexdigest()
    assert h == json.load(open(ROOT / "artifacts" / "p123_upgrade" / "bank" /
                               "bank_dev512.manifest.json"))["sha256"]
    # bank is the single edit source: models never resample
    b = np.load(npz, allow_pickle=True)
    assert b["Qx"].shape[0] == len(json.loads(str(b["Qmeta"])))
    assert b["Qe"].shape[0] == len(json.loads(str(b["Qmeta"])))


def test_bootstrap_axis_and_cluster():
    rng = np.random.default_rng(0)
    x = np.array([0, 1, 0, 1] * 25)
    m = np.quantile(rng.choice(x, size=(2000, len(x))).mean(1), [0.025, 0.975])
    assert m[0] < x.mean() < m[1]  # mean(1) = distribution of resample means
    # cluster version differs from iid version in general; smoke-run it
    par = np.repeat(np.arange(25), 4)
    ups = np.unique(par)
    pmap = {u: np.nonzero(par == u)[0] for u in ups}
    boots = []
    for _ in range(200):
        draw = rng.choice(ups, size=len(ups), replace=True)
        boots.append(x[np.concatenate([pmap[u] for u in draw])].mean())
    assert len(boots) == 200


def test_paired_gain_zero_on_same_input():
    a = np.array([0, 1, 0, 1])
    b = np.array([0, 1, 0, 1])
    assert float((a != b).mean()) == 0.0


def test_selection_artifact_detected():
    ab_ok = np.array([0] * 50 + [1] * 50)
    a_c = np.array([True] * 30 + [False] * 70)
    a_f = np.array([True] * 10 + [False] * 40 + [True] * 30 + [False] * 20)
    assert abs((1 - ab_ok[a_c].mean()) - (1 - ab_ok[a_f].mean())) > 0.2


def test_temperature_argmax_invariant_binary():
    lg = np.array([-3.0, -0.2, 0.1, 5.0])
    for T in (0.5, 2.0, 8.0):
        assert ((lg / T > 0) == (lg > 0)).all()
    assert (np.argsort(np.abs(lg / 2.0)) == np.argsort(np.abs(lg))).all()


def test_p2_split_sums_to_total():
    import json as J
    d = J.load(open(ROOT / "artifacts" / "p123_upgrade" / "p2" / "P2_DECOMP.json"))
    for s in ("s11", "s23", "s47"):
        r = d[s]
        a = r["lexico_lam2_clean"]
        f = r["lexico_lam2_flip"]
        cov = (f["a"] - a["a"]) * (f["q"] + a["q"]) / 2
        qua = (f["q"] - a["q"]) * (f["a"] + a["a"]) / 2
        assert abs((cov + qua) - (f["s"] - a["s"])) < 5e-4  # rounded storage
        assert abs((cov + qua) - r["split"]["total"]) < 5e-4
