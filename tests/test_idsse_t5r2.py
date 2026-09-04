"""Small regression tests for the IDSSE-backed T5R2 task construction."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_idsse_t5r2.py"
SPEC = spec_from_file_location("prepare_idsse_t5r2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_positions_uses_centered_pitch_origin() -> None:
    group = pd.DataFrame(
        {
            "x": [-52.5, 0.0, 52.5],
            "y": [-34.0, 0.0, 34.0],
        }
    )

    normalized = MODULE.normalize_positions(
        group,
        {"pitch_x": 105.0, "pitch_y": 68.0},
    )

    np.testing.assert_allclose(
        normalized,
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
    )


def test_pairwise_signature_and_descriptor_have_fixed_20_node_shape() -> None:
    positions = np.zeros((2, MODULE.N_PLAYER_NODES, 2), dtype=np.float32)
    positions[0, 1, 0] = 1.0

    signatures = MODULE.pairwise_signature(positions)
    descriptors = MODULE.shape_descriptor(signatures)

    assert signatures.shape == (2, MODULE.N_PLAYER_NODES * 19 // 2)
    assert descriptors.shape == (2, 15)
    assert signatures[0, 0] == 1.0


def test_nearest_indices_is_deterministic_and_sorted_by_distance() -> None:
    values = np.asarray([[2.0], [0.0], [1.0], [3.0]], dtype=np.float32)

    result = MODULE.nearest_indices(np.asarray([0.0], dtype=np.float32), values, 3)

    np.testing.assert_array_equal(result, [1, 2, 0])
