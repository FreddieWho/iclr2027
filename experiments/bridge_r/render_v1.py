#!/usr/bin/env python3
"""Bridge-R canonical renderer v1.

Renders (4,2) coordinate scenes to canonical 512x512 RGB images for
foundation-model probing. Key properties:
  - SAFE_SCALE = 0.80: coordinates mapped into central ~80% of canvas so no
    official center-crop (DINOv2 224-crop, SigLIP 384-resize) clips endpoints.
    Pure similarity transform (uniform scale + center); segment relations and
    the oracle are invariant under it.
  - Nuisance contract: caller passes an explicit np.random.Generator; the
    background noise draw is the ONLY stochasticity. Quartet four-states must
    share one seed; train/dev/holdout seed spaces must not overlap.
"""
from __future__ import annotations
import numpy as np

CANON = 512
SAFE_SCALE = 0.88
LINE_WIDTH = 16  # px at 512 (8px @256 post-resize; Am01: strokes must survive downsampling)
BG_MEAN = 0.5
BG_JITTER = 0.05
COLORS = [(0.9, 0.1, 0.1)] * 2 + [(0.1, 0.1, 0.9)] * 2  # AB red, CD blue

RENDERER_VERSION = "bridge_r_canonical_v1a"
RENDER_CONFIG = {
    "renderer_version": RENDERER_VERSION,
    "canonical_resolution": CANON,
    "safe_scale": SAFE_SCALE,
    "line_width_px": LINE_WIDTH,
    "background": f"uniform({BG_MEAN - BG_JITTER}, {BG_MEAN + BG_JITTER})",
    "colors": {"AB": "red", "CD": "blue"},
}


def render_canonical(x, rng, bg_seed_note=""):
    """Render a (4,2) scene in ~[-1,1]^2 to (512,512,3) float32 HWC."""
    img = np.full((CANON, CANON, 3),
                  BG_MEAN + rng.uniform(-BG_JITTER, BG_JITTER), np.float32)
    pts = np.asarray(x, float) * SAFE_SCALE
    px = ((pts + 1) / 2 * (CANON - 1)).astype(int).clip(0, CANON - 1)
    for s, (p, q) in enumerate([(px[0], px[1]), (px[2], px[3])]):
        n = int(np.hypot(*(q - p)) * 2) + 1
        for t in np.linspace(0, 1, max(n, 2)):
            r = int(round(p[0] + t * (q[0] - p[0])))
            c = int(round(p[1] + t * (q[1] - p[1])))
            r0, r1 = max(0, r - LINE_WIDTH), min(CANON, r + LINE_WIDTH + 1)
            c0, c1 = max(0, c - LINE_WIDTH), min(CANON, c + LINE_WIDTH + 1)
            img[c0:c1, r0:r1] = COLORS[2 * s]
    return img
