#!/usr/bin/env python3
"""Image-only visible red/blue mechanism (Route 2).

Inference accepts RGB image batches only. The shared ResNet18 spatial backbone
runs once; visible-color pixel masks select global or color-pooled features.
Coordinates, parent IDs, oracle fields, edit answers and combination IDs are
not model inputs. CPU smoke is execution-only and remains ``PILOT_ONLY``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[3]
THREADS = 4
torch.set_num_threads(THREADS)
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def _image_nchw(images) -> tuple[torch.Tensor, torch.Tensor]:
    # Accept both the runner's numpy batches and the already-preprocessed CUDA
    # tensor path. The latter must not be routed through np.asarray().
    if torch.is_tensor(images):
        raw = images.detach()
    else:
        raw = torch.as_tensor(np.asarray(images, dtype=np.float32))
    if raw.ndim == 3:
        raw = raw.unsqueeze(0)
    if raw.ndim != 4 or raw.shape[1] != 3:
        raise ValueError(f"expected RGB NCHW or NHWC, got {tuple(raw.shape)}")
    if raw.shape[-1] == 3 and raw.shape[1] != 3:
        raw = raw.permute(0, 3, 1, 2).contiguous()
    return raw, raw


def prep(images, input_size: int = 224) -> torch.Tensor:
    """Normalize RGB and bilinearly resize native observations to model input."""
    raw, _ = _image_nchw(images)
    if input_size != raw.shape[-1]:
        raw = F.interpolate(raw, size=(input_size, input_size), mode="bilinear", align_corners=False)
    mean = MEAN.to(device=raw.device, dtype=raw.dtype)
    std = STD.to(device=raw.device, dtype=raw.dtype)
    return (raw - mean) / std


def visible_segment_features(model, images, mask_pool: str = "nearest"):
    """Return global, red and blue features from one shared image forward.

    mask_pool='nearest' preserves the legacy contract exactly. 'area' uses
    area-weighted downsampling so natively visible segments keep nonzero
    probability mass instead of being silently erased to a zero vector.
    """
    raw, _ = _image_nchw(images)
    device = next(model.parameters()).device
    x = prep(images).to(device)
    raw = raw.to(device)
    # fc is never used; all four arms receive the same shared backbone forward.
    zmap = model.backbone(x)
    z = zmap.mean((2, 3))
    masks = {
        "red": (raw[:, 0] > .15) & (raw[:, 1] < .35) & (raw[:, 2] < .35),
        "blue": (raw[:, 2] > .15) & (raw[:, 0] < .35) & (raw[:, 1] < .35),
    }
    if mask_pool not in ("nearest", "area"):
        raise ValueError(f"unknown mask_pool {mask_pool}")
    out = {"global": z}
    for name, mask in masks.items():
        # Missing color is a zero mask, never an oracle/coordinate fallback.
        weights = F.interpolate(mask.to(z.dtype).unsqueeze(1), size=zmap.shape[-2:], mode=mask_pool)
        out[name] = (zmap * weights).sum((2, 3)) / weights.sum((2, 3)).clamp_min(1.0)
    return out


class Mechanism(nn.Module):
    """Predeclared direct/additive/representation/interaction arms."""

    def __init__(self, mode: str = "interaction", input_size: int = 224, pretrained: bool = False,
                 mask_pool: str = "nearest"):
        super().__init__()
        if mode not in {"direct", "additive", "representation", "interaction"}:
            raise ValueError(mode)
        weights = "IMAGENET1K_V1" if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = nn.Identity()
        self.backbone = nn.Sequential(*list(self.encoder.children())[:-2])
        self.mode = mode
        self.input_size = input_size
        self.mask_pool = mask_pool
        if mode in {"direct", "additive"}:
            self.head = nn.Linear(512, 1)
        elif mode == "representation":
            self.head = nn.Linear(1024, 1)
        else:
            self.head = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, images):
        features = visible_segment_features(self, images, self.mask_pool)
        if self.mode == "direct":
            return self.head(features["global"]).reshape(-1)
        if self.mode == "additive":
            # Same shared linear head applied separately to visible color pools.
            return (self.head(features["red"]) + self.head(features["blue"])).reshape(-1)
        concatenated = torch.cat([features["red"], features["blue"]], -1)
        if self.mode == "representation":
            return self.head(concatenated).reshape(-1)
        return self.head(concatenated).reshape(-1)


def _line(im, a, b, color):
    # Renderer retained only for the explicitly non-scientific execution smoke.
    a = np.asarray(a); b = np.asarray(b)
    n = max(2, int(np.linalg.norm(b - a) * 3) + 1)
    for t in np.linspace(0, 1, n):
        im[round((1 - t) * a[1] + t * b[1]), round((1 - t) * a[0] + t * b[0]), color] = 1.0


def render_smoke(seed=832601, n=16):
    rng = np.random.default_rng(seed); ims = []; ys = []
    for _ in range(n):
        x = rng.uniform(-.8, .8, size=(4, 2)); im = np.zeros((64, 64, 3), np.float32)
        xy = np.rint((x + 1) * 28).astype(int)
        _line(im, xy[0], xy[1], 0); _line(im, xy[2], xy[3], 2)
        def cross(a, b): return float(np.cross(b - a, a))
        d1 = cross(x[1] - x[0], x[2] - x[0]); d2 = cross(x[1] - x[0], x[3] - x[0])
        d3 = cross(x[3] - x[2], x[0] - x[2]); d4 = cross(x[3] - x[2], x[1] - x[2])
        ys.append(int(((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))))
        ims.append(im.transpose(2, 0, 1))
    return np.asarray(ims, np.float32), np.asarray(ys, np.float32)


def smoke(out):
    x, y = render_smoke(); t0 = time.time(); records = []
    for mode in ("direct", "additive", "interaction", "representation"):
        torch.manual_seed(7); model = Mechanism(mode); model.encoder.eval()
        # Frozen encoder keeps the four-step CPU execution check bounded.
        for parameter in model.encoder.parameters(): parameter.requires_grad = False
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for _ in range(4):
            optimizer.zero_grad(); logits = model(x[:4])
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[:4]))
            loss.backward(); optimizer.step()
        with torch.no_grad(): logits = model(x)
        records.append({"mode": mode, "loss": float(loss.detach()), "logit_shape": list(logits.shape), "finite": bool(torch.isfinite(logits).all())})
    payload = {
        "status": "PILOT_ONLY",
        "purpose": "CPU execution smoke, not a visual or performance result",
        "threads": THREADS, "encoder": "torchvision resnet18 weights=None", "n": len(y),
        "steps": 4, "records": records, "seconds": time.time() - t0,
        "no_inference_hidden_fields": ["A/B index", "coordinates", "oracle labels", "AB truth", "single-edit answer", "combination ID"],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/e832_focus/route2/cpu_smoke.json")
    args = parser.parse_args()
    if not args.smoke: parser.error("use --smoke; formal GPU execution is deliberately not claimed")
    print(json.dumps(smoke(args.out), indent=2))


if __name__ == "__main__":
    main()
