"""L007 diagnostic primitives: rank correlation and grouped ridge CV."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
from l007_diagnostic import ridge_cv_r2, spearman  # noqa: E402


def test_spearman_perfect_monotone():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_spearman_ignores_nonfinite():
    assert spearman([1, 2, np.nan, 4], [1, 2, 3, 4]) == 1.0


def test_spearman_too_few_points():
    assert spearman([1, 2], [1, 2]) is None


def test_ridge_cv_r2_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = 3 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=60)
    groups = np.arange(60) % 12
    r2 = ridge_cv_r2(X, y, groups)
    assert r2 is not None and r2 > 0.9


def test_ridge_cv_r2_null_is_not_positive():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = rng.normal(size=60)
    groups = np.arange(60) % 12
    r2 = ridge_cv_r2(X, y, groups)
    assert r2 is None or r2 < 0.2
