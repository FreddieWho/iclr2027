"""Mass-preserving visual input contracts.

Legacy nearest-neighbor mask pooling can erase thin visible structure. Fixed
pooling downsamples masks with area averaging, keeps red/blue probability
mass, and raises on an empty pooled mask instead of silently returning zero.
RGB-derived color masks are allowed and disclosed; oracle/coordinate masks at
inference time are not represented here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
RED = {"red_min": 0.15, "green_max": 0.35, "blue_max": 0.35}
BLUE = {"blue_min": 0.15, "red_max": 0.35, "green_max": 0.35}


class EmptyMaskError(ValueError):
    """Raised when a pooled color mask has no mass: never silent-zero."""


def as_nchw(images) -> torch.Tensor:
    """Accept NCHW or NHWC RGB batches; reject ambiguous/non-RGB layouts."""
    raw = images.detach() if torch.is_tensor(images) else torch.as_tensor(
        np.asarray(images, dtype=np.float32)
    )
    if raw.ndim == 3:
        raw = raw.unsqueeze(0)
    if raw.ndim != 4:
        raise ValueError(f"expected a 3- or 4-dimensional RGB batch, got {tuple(raw.shape)}")
    channel_first = raw.shape[1] == 3
    channel_last = raw.shape[-1] == 3
    if channel_first and not channel_last:
        return raw
    if channel_last and not channel_first:
        return raw.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"ambiguous or non-RGB image layout: {tuple(raw.shape)}")


def resize_rgb(nchw: torch.Tensor, input_size: int) -> torch.Tensor:
    if nchw.ndim != 4 or nchw.shape[1] != 3:
        raise ValueError(f"expected NCHW RGB, got {tuple(nchw.shape)}")
    if not isinstance(input_size, int) or input_size < 1:
        raise ValueError("input_size must be a positive integer")
    if nchw.shape[2] == input_size and nchw.shape[3] == input_size:
        return nchw
    return F.interpolate(nchw, size=(input_size, input_size), mode="bilinear", align_corners=False)


def normalize_rgb(
    nchw: torch.Tensor, mean: torch.Tensor = IMAGENET_MEAN, std: torch.Tensor = IMAGENET_STD
) -> torch.Tensor:
    if nchw.ndim != 4 or nchw.shape[1] != 3:
        raise ValueError(f"expected NCHW RGB, got {tuple(nchw.shape)}")
    return (nchw - mean.to(device=nchw.device, dtype=nchw.dtype)) / std.to(
        device=nchw.device, dtype=nchw.dtype
    )


def prepare_for_backbone(images, input_size: int) -> torch.Tensor:
    """Native RGB in, resized/normalized tensor out; size is always threaded."""
    return normalize_rgb(resize_rgb(as_nchw(images), input_size))


def color_masks(nchw: torch.Tensor) -> dict[str, torch.Tensor]:
    """RGB-threshold masks from native values; inference never uses oracle masks."""
    raw = as_nchw(nchw)
    red = (
        (raw[:, 0] > RED["red_min"])
        & (raw[:, 1] < RED["green_max"])
        & (raw[:, 2] < RED["blue_max"])
    )
    blue = (
        (raw[:, 2] > BLUE["blue_min"])
        & (raw[:, 0] < BLUE["red_max"])
        & (raw[:, 1] < BLUE["green_max"])
    )
    return {"red": red, "blue": blue}


def pooled_mask_weights(
    mask_native: torch.Tensor, output_size: tuple[int, int], mode: str = "area"
) -> torch.Tensor:
    """Downsample a native boolean mask to a feature-map grid.

    ``area`` is the fixed mass-preserving choice. ``nearest`` is the legacy
    choice, kept only so audits can measure how much visible mass it erases.
    """
    mask = torch.as_tensor(np.asarray(mask_native, dtype=bool))
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"expected [n,1,h,w] or [n,h,w] mask, got {tuple(mask.shape)}")
    if output_size[0] < 1 or output_size[1] < 1:
        raise ValueError("output_size must be positive")
    if mode not in ("area", "nearest"):
        raise ValueError("mode must be 'area' or 'nearest'")
    return F.interpolate(mask.to(torch.float32), size=output_size, mode=mode)


def mask_mass_stats(
    mask_native: torch.Tensor, output_size: tuple[int, int], mode: str = "area"
) -> dict:
    """Separate missing color from downsampling loss before any pooling mean."""
    native = torch.as_tensor(np.asarray(mask_native, dtype=bool))
    if native.ndim == 2:
        native = native.unsqueeze(0)
    weights = pooled_mask_weights(native, output_size, mode=mode)
    native_pixels = native.reshape(len(native), -1).sum(dim=1)
    pooled_mass = weights.reshape(len(weights), -1).sum(dim=1)
    erased = (native_pixels > 0) & (pooled_mass <= 0)
    return {
        "native_pixels": native_pixels,
        "pooled_mass": pooled_mass,
        "native_visible_but_pooled_zero": erased,
    }


def pooled_means(
    feature_map: torch.Tensor, masks_native: dict[str, torch.Tensor], output_size: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Area-weighted means; an empty pooled mask is an error, not a zero vector."""
    if feature_map.ndim != 4:
        raise ValueError(f"expected [n,c,h,w] features, got {tuple(feature_map.shape)}")
    if (feature_map.shape[2], feature_map.shape[3]) != (output_size[0], output_size[1]):
        raise ValueError("feature_map grid must equal output_size")
    means: dict[str, torch.Tensor] = {}
    for name, native in masks_native.items():
        weights = pooled_mask_weights(native, output_size).to(feature_map.dtype)
        mass = weights.sum(dim=(2, 3), keepdim=True)
        if bool((mass <= 0).any()):
            raise EmptyMaskError(f"pooled mask {name!r} has no mass at {output_size}")
        means[name] = (feature_map * weights).sum(dim=(2, 3)) / mass.reshape(len(mass), 1)
    return means
