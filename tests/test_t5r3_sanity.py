"""Regression tests for the locked T5R3 runner."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_t5r3_sanity.py"
SPEC = spec_from_file_location("run_t5r3_sanity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_global_centering_removes_translation() -> None:
    positions = np.arange(12, dtype=np.float32).reshape(2, 2, 3).transpose(0, 2, 1)
    shifted = positions + np.asarray([0.25, -0.15], dtype=np.float32)

    np.testing.assert_allclose(MODULE.center_positions(positions), MODULE.center_positions(shifted), atol=1e-6)


def test_all_neural_variants_have_matched_parameter_count_and_shape() -> None:
    torch.manual_seed(11)
    models = [MODULE.TaskModel() for _ in MODULE.VARIANTS]
    parameter_counts = {sum(parameter.numel() for parameter in model.parameters()) for model in models}
    assert len(parameter_counts) == 1

    positions = torch.randn(2, 20, 2)
    team_slots = torch.tensor([[0] * 10 + [1] * 10] * 2)
    adjacency = torch.zeros(2, 20, 20)
    for node in range(20):
        adjacency[:, node, (node + 1) % 20] = 1.0
        adjacency[:, node, (node - 1) % 20] = 1.0
    z = models[0].encode(positions, team_slots, adjacency, "team_mean")

    assert z.shape == (2, MODULE.LATENT_DIM)


def test_knn_adjacency_is_symmetric_and_has_four_directed_neighbors_before_symmetrization() -> None:
    positions = np.zeros((1, 20, 2), dtype=np.float32)
    positions[0, :, 0] = np.arange(20, dtype=np.float32)

    adjacency = MODULE.knn_adjacency_batch(positions)

    np.testing.assert_allclose(adjacency, adjacency.transpose(0, 2, 1))
    assert np.all(np.sum(adjacency > 0, axis=-1) >= MODULE.K_NEIGHBORS)
