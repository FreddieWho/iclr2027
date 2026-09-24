#!/usr/bin/env python3
"""D06 passing-channel geometric oracle (proxy, NOT real pass success).

Channel p->r is OPEN iff no defender disc covers the segment interior:
  margin = min_d dist(d, interior(pr)) - radius_d; blocked iff margin < 0.
Endpoint exclusion: discs within `end_excl` of p or r do not count (a
defender standing on the receiver is a different football event, not a
channel block). radius/end_excl come from training-scene scale, never
from model results. Defender ordering must not matter.
"""
from __future__ import annotations
import numpy as np


def point_segment_dist(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    q, a, b = map(lambda v: np.asarray(v, dtype=float), (q, a, b))
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0:
        return float(np.linalg.norm(q - a))
    t = float(np.clip((q - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(q - (a + t * ab)))


def channel_margin(p, r, defenders, radius: float, end_excl: float) -> float:
    """Geometric margin of the p->r channel. Positive = open."""
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    ds = [np.asarray(d, dtype=float) for d in defenders]
    ds = [d for d in ds
          if np.linalg.norm(d - p) > end_excl and np.linalg.norm(d - r) > end_excl]
    if not ds:
        return float("inf")
    return min(point_segment_dist(d, p, r) for d in ds) - float(radius)


def channel_label(p, r, defenders, radius: float, end_excl: float) -> dict:
    m = channel_margin(p, r, defenders, radius, end_excl)
    return {"label": int(m >= 0), "margin": m, "blocked": bool(m < 0)}
