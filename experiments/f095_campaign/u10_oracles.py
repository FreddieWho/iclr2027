#!/usr/bin/env python3
"""U10 target-task oracles (new tasks, NOT renames of existing experiments).

T1 point-in-triangle: points (V1,V2,V3,Q). Label = Q inside triangle (1/0).
  Symmetry: S3 over vertices with Q FIXED (6 elements). Q-vertex swaps are
  OUTSIDE the group (role information is load-bearing).
T2 segment-disk: points (A,B,C=center), fixed radius R0 (task constant).
  Label = segment AB intersects disk(C,R0) (1/0). Symmetry: {id, A<->B}.
Margins are geometric distances (area/length units where applicable, recorded
as margin, never called distance when area-unit).
"""
from __future__ import annotations
import numpy as np

T2_RADIUS = 0.25
TOL = 1e-12


def _seg_point_dist(p, a, b):
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def tri_oracle(x: np.ndarray, min_margin: float = 0.02) -> dict:
    """x (4,2): V1,V2,V3,Q."""
    x = np.asarray(x, float)
    if x.shape != (4, 2) or not np.all(np.isfinite(x)):
        raise ValueError("Expected (4,2) finite V1,V2,V3,Q")
    v1, v2, v3, q = x
    area2 = abs(float(np.cross(v2 - v1, v3 - v1)))
    if area2 < 1e-9:
        raise ValueError("degenerate triangle")
    def s(a, b, c):
        return float(np.cross(b - a, c - a))
    s1, s2, s3 = s(v1, v2, q), s(v2, v3, q), s(v3, v1, q)
    inside = (s1 > 0 and s2 > 0 and s3 > 0) or (s1 < 0 and s2 < 0 and s3 < 0)
    d_edge = min(_seg_point_dist(q, v1, v2), _seg_point_dist(q, v2, v3),
                 _seg_point_dist(q, v3, v1))
    return {"label": int(inside), "margin": float(d_edge),
            "ambiguous": bool(d_edge <= max(TOL, min_margin))}


def disk_oracle(x: np.ndarray, min_margin: float = 0.02,
                radius: float = T2_RADIUS) -> dict:
    """x (3,2): A,B,C=center. Label = segment AB hits disk(C, radius)."""
    x = np.asarray(x, float)
    if x.shape != (3, 2) or not np.all(np.isfinite(x)):
        raise ValueError("Expected (3,2) finite A,B,C")
    a, b, c = x
    d = _seg_point_dist(c, a, b)
    return {"label": int(d <= radius), "margin": float(abs(d - radius)),
            "ambiguous": bool(abs(d - radius) <= max(TOL, min_margin))}
