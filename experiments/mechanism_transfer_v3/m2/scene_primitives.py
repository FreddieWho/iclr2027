"""Thin, frozen wrappers over the existing scene/oracle/renderer primitives.

Nothing here is reimplemented: the parent generator is
``core.relations.make_relational_scenes``, the oracle is
``paths.oracle_at`` (``core.relations.segment_relation``), the edit library is
``paths.atomic_edits`` and the renderer is ``n08_visual.render``. The wrappers
only fix the *conventions* the M2 bank depends on:

* the renderer input is canonically endpoint-sorted within each color, exactly
  as the archived Route-2 pilot did;
* every render of one parent uses one deterministic render seed, so the
  background noise is identical across that parent's states and a
  render-difference is a geometry difference, not a noise difference;
* images are quantized to uint8 once, at the boundary, with a documented
  round-trip check.

No oracle/coordinate information is exposed to any image-only code path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _path in (
    ROOT / "experiments" / "last15h",
    ROOT / "experiments" / "last15h" / "shared",
    ROOT / "docs" / "iclr2027_discovery_campaign_20260917",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core.relations import make_relational_scenes, segment_relation  # noqa: E402
from n08_visual import IMG as RENDER_SIZE  # noqa: E402
from n08_visual import render as legacy_render  # noqa: E402
from paths import atomic_edits, oracle_at  # noqa: E402

IMAGE_SIZE = int(RENDER_SIZE)
CHANNEL_PIXEL_THRESHOLD = 0.2

# Re-exported so M2 modules have a single import point for the frozen
# generator/oracle/edit-library/renderer surface.
__all__ = [
    "IMAGE_SIZE",
    "CHANNEL_PIXEL_THRESHOLD",
    "atomic_edits",
    "oracle",
    "oracle_at",
    "relation",
    "make_relational_scenes",
    "canonical_scene",
    "parent_key",
    "make_parents",
    "render_seed",
    "render_scene",
    "changed_pixels",
    "to_uint8",
    "to_uint8_checked",
    "images_as_float",
    "quantization_matches_float",
]


def oracle(x) -> tuple[int, float, bool]:
    """(label, margin, ambiguous) for one (4,2) scene."""
    return oracle_at(np.asarray(x, dtype=float).reshape(4, 2))


def relation(x) -> dict:
    return segment_relation(np.asarray(x, dtype=float).reshape(4, 2))


def canonical_scene(x) -> np.ndarray:
    """Color-preserving endpoint order: sort the two AB points and the two CD points."""
    x = np.asarray(x, dtype=float).reshape(4, 2)
    return np.asarray(sorted(map(tuple, x[:2])) + sorted(map(tuple, x[2:])))


def parent_key(x) -> str:
    """Physical-parent key; endpoint/segment order independent, color preserving."""
    import hashlib

    ordered = canonical_scene(x)
    return hashlib.sha256(np.ascontiguousarray(ordered).tobytes()).hexdigest()


def make_parents(n: int, seed: int, min_margin: float) -> np.ndarray:
    positions, labels, margins = make_relational_scenes(n, seed=seed, min_margin=min_margin)
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) != n:
        raise RuntimeError(f"requested {n} parents, generated {len(positions)}")
    return positions, np.asarray(labels, dtype=np.int64), np.asarray(margins, dtype=np.float64)


def render_seed(observation_seed: int, parent_id: int) -> int:
    """Deterministic per-parent render seed (design D2 in DECISION_NOTES.md)."""
    return int(observation_seed) + int(parent_id) * 10007


def render_scene(x, seed: int) -> np.ndarray:
    """Canonical native-64 float32 RGB render, shape (3, 64, 64)."""
    image = legacy_render(canonical_scene(x), np.random.default_rng(int(seed)))
    return np.asarray(image, dtype=np.float32)


def changed_pixels(before: np.ndarray, after: np.ndarray) -> int:
    """Renderer-convention pixel change: channel max abs difference > 0.2."""
    a = np.asarray(before, dtype=np.float32)
    b = np.asarray(after, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("render shapes differ")
    return int((np.abs(b - a).max(axis=0) > CHANNEL_PIXEL_THRESHOLD).sum())


def to_uint8(image: np.ndarray) -> np.ndarray:
    scaled = np.rint(np.asarray(image, dtype=np.float32) * 255.0).clip(0, 255)
    return scaled.astype(np.uint8)


def to_uint8_checked(image: np.ndarray) -> np.ndarray:
    """Quantize to uint8 and fail if any channel moves by more than half a step."""
    quantized = to_uint8(image)
    error = np.abs(quantized.astype(np.float32) / 255.0 - np.asarray(image, dtype=np.float32))
    if float(error.max()) > 0.5 / 255.0 + 1e-6:
        raise ValueError("uint8 quantization moved a channel by more than half a step")
    return quantized


def images_as_float(images: np.ndarray) -> np.ndarray:
    """uint8 bank images -> float32 in [0,1] for the frozen mask contract."""
    array = np.asarray(images)
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    raise ValueError(f"expected uint8 bank images, got {array.dtype}")


def quantization_matches_float(images_uint8: np.ndarray, images_float: np.ndarray) -> bool:
    """Round-trip check: uint8 storage must not move any channel by >1/255."""
    a = np.asarray(images_uint8, dtype=np.float32) / 255.0
    b = np.asarray(images_float, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return bool(np.abs(a - b).max() <= 1.0 / 255.0 + 1e-7)
