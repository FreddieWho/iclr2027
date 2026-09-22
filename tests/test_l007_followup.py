"""L007 follow-up primitives: crossing width on a logit grid."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
from l007_grazing import crossing_width  # noqa: E402


def test_crossing_width_symmetric():
    t = np.linspace(0, 1, 129)
    lg = (t - 0.5) * 10
    w = crossing_width(t, lg, 0.5, k=1.0)
    assert w is not None and abs(w - 0.2) < 0.02


def test_crossing_width_unreached_band():
    t = np.linspace(0, 1, 129)
    lg = (t - 0.5) * 2
    assert crossing_width(t, lg, 0.5, k=2.0) is None


def test_crossing_width_narrower_band_smaller():
    t = np.linspace(0, 1, 129)
    lg = (t - 0.5) * 10
    w1 = crossing_width(t, lg, 0.5, k=1.0)
    w2 = crossing_width(t, lg, 0.5, k=2.0)
    assert w1 < w2
