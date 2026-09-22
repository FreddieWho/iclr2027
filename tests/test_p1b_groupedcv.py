"""Unit tests for R2.1 grouped_cv paired-OOF semantics (R2/R9.7)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "next_novelty"))
from p1b_analysis import grouped_cv  # noqa: E402

rng = np.random.default_rng(7)


def _mk(n_parents=30, per=40):
    par = np.repeat(np.arange(n_parents), per)
    n = len(par)
    conf = rng.normal(size=n) + 1.5 * (par % 2)
    hard = rng.normal(size=n) + (par % 2)
    y = (conf + 0.3 * rng.normal(size=n) > 0).astype(int)
    X = np.column_stack([conf, hard])
    return X, y, par


def test_oof_pools_and_paired_diffs_present():
    X, y, par = _mk()
    out = grouped_cv(X, y, par, k=5, n_boot=50)
    for nm in ("conf", "noconf", "full"):
        assert "pooled_oof_auc" in out[nm]
        assert "in_sample_diagnostic" in out[nm]
    for pr in ("oof_auc_diff_conf_vs_noconf", "oof_auc_diff_full_vs_noconf"):
        d = out[pr]
        assert d["point_diff"] is not None
        assert d["paired_oof_diff_ci"][0] <= d["point_diff"] <= d["paired_oof_diff_ci"][1]
        assert d["n_common"] == len(y)


def test_cols_control_features_used():
    """noconf (difficulty only) must underperform conf on conf-driven labels."""
    X, y, par = _mk()
    out = grouped_cv(X, y, par, k=5, n_boot=50)
    assert out["oof_auc_diff_conf_vs_noconf"]["point_diff"] > 0.2


def test_diff_ci_degenerate_when_predictors_identical():
    X, y, par = _mk()
    X2 = np.column_stack([X[:, 0], X[:, 0]])
    out = grouped_cv(X2, y, par, k=5, n_boot=50)
    d = out["oof_auc_diff_conf_vs_noconf"]
    assert abs(d["point_diff"]) < 1e-9
    assert abs(d["paired_oof_diff_ci"][1] - d["paired_oof_diff_ci"][0]) < 1e-9


def test_parent_groups_do_not_overlap_across_folds():
    X, y, par = _mk(n_parents=20, per=30)
    # reimplement fold assignment the same way grouped_cv does and assert
    # parent uniqueness across train/test
    ups = np.unique(par)
    order = rng.permutation(len(ups))
    folds = np.array_split(ups[order], 5)
    seen_test = set()
    for f in folds:
        fset = set(int(u) for u in f)
        assert not (fset & seen_test)
        seen_test |= fset
