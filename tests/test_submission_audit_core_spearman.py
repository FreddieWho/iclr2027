import numpy as np
import pytest
from scipy.stats import spearmanr

from experiments.update_geometry.wp4_unified import spearman_average_rank


def test_spearman_average_ranks_ties():
    x = np.array([1, 1, 2, 3, 3], dtype=float)
    y = np.array([2, 1, 1, 3, 3], dtype=float)
    assert spearman_average_rank(x, y) == round(float(spearmanr(x, y).statistic), 4)


def test_spearman_uses_pairwise_finite_rows():
    x = np.array([1, 2, np.nan, 3, np.inf], dtype=float)
    y = np.array([1, 2, 0, 2, 3], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    assert spearman_average_rank(x, y) == round(
        float(spearmanr(x[keep], y[keep]).statistic), 4
    )


@pytest.mark.parametrize(
    ("x", "y"),
    [([1, 1], [1, 2]), ([1, 2], [4, 4]), ([np.nan, 1], [0, 1])],
)
def test_spearman_undefined_samples_return_none(x, y):
    assert spearman_average_rank(x, y) is None


def test_spearman_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal-length vectors"):
        spearman_average_rank([1, 2], [1])
