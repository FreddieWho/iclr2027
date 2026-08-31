import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_phase0_pipeline.py"
SPEC = importlib.util.spec_from_file_location("phase0_pipeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_equal_energy_is_exact():
    delta = MODULE.equal_energy(np.ones((20, 2)))
    assert np.isclose(np.linalg.norm(delta), 1.0)


def test_band_ranges_cover_all_modes_once():
    bands = MODULE.band_ranges(20, 6)
    joined = np.concatenate(bands)
    assert np.array_equal(np.sort(joined), np.arange(20))


def test_knn_adjacency_is_symmetric_with_zero_diagonal():
    adjacency = MODULE.knn_adjacency(np.arange(20, dtype=float)[:, None].repeat(2, axis=1))
    assert np.allclose(adjacency, adjacency.T)
    assert np.allclose(np.diag(adjacency), 0.0)


def test_normalized_laplacian_is_symmetric():
    adjacency = MODULE.knn_adjacency(np.random.default_rng(1).normal(size=(20, 2)))
    laplacian = MODULE.normalized_laplacian(adjacency)
    assert np.allclose(laplacian, laplacian.T)
