import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "n02_upgrade"))

from upgrade_protocol import metrics  # noqa: E402


def test_ccm_is_conditional_composition_miss():
    labels = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (3, 1))
    logits = np.array(
        [
            [-1.0, -1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0, -1.0],
        ]
    )

    result = metrics(logits, labels)

    assert result["atomic_joint"] == 1.0
    assert result["J"] == 2 / 3
    assert result["J4"] == 2 / 3
    assert result["conditional_joint_success"] == 2 / 3
    assert result["composition_miss"] == result["CCM"] == 1 / 3
    assert result["legacy_CCM_success"] == 2 / 3


def test_conditional_metrics_are_undefined_without_atomic_success():
    labels = np.array([[0.0, 0.0, 0.0, 1.0]])
    logits = np.array([[-1.0, 1.0, -1.0, -1.0]])

    result = metrics(logits, labels)

    assert result["CCM_denominator"] == 0
    assert result["conditional_joint_success"] is None
    assert result["composition_miss"] is None
    assert result["CCM"] is None
