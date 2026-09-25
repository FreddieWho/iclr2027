"""Frozen M2 visual-bank contract.

Every field below is fixed before generation. The generator refuses to run
with a config whose declared constants do not match ``FROZEN`` unless a caller
explicitly builds a different ``BankConfig`` (tests do this with tiny sizes).
Nothing here trains a model, renders at model resolution, or reads a sealed
pool.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SCHEMA = "mechanism_transfer_v3.m2.bank/1"

# Fresh v3 seeds. Independent of the archived Route-2 pilot (8327xx) so the
# expanded bank is a genuinely new draw, not a re-render of the pilot.
FROZEN_SEEDS = {
    "train_parent": 832801,
    "dev_parent": 832802,
    "test_parent": 832803,
    "train_observation": 832811,
    "dev_observation": 832812,
    "test_observation": 832813,
    "quartet_mining": 832821,
    "quartet_observation": 832822,
}

# 384 drawn test parents is the pre-registered size: the model-blind yield
# pilot measured 92.97% eligible parents at the fixed margin/radius schedule,
# i.e. about 357 eligible test parents, inside the 200-500 target band.
FROZEN_SPLIT_SIZES = {"train": 1024, "dev": 128, "test": 384}

# One margin and one radius schedule are shared by singletons and quartets.
# The legacy miner used margin 0.005 and radius 0.10 only, which is why the
# archived pilot bank produced 28 quartets over 19 eligible parents.
FROZEN_ORACLE_MARGIN_FLOOR = 0.02
FROZEN_RADII = (0.10, 0.20, 0.35, 0.50)
FROZEN_SINGLETON_DRAWS = 16
FROZEN_QUARTET_DRAWS = 16
FROZEN_MAX_SINGLE_EDITS_PER_PARENT = 2
FROZEN_MAX_QUARTETS_PER_PARENT = 2
# Resource bound only, deliberately non-binding: the substantive frozen
# constraint is 2 quartets per eligible test parent. The realized count is
# reported in the manifest (``yields.quartet.accepted``), not imposed here.
FROZEN_MAX_QUARTETS_TOTAL = 1024
FROZEN_MIN_CHANGED_PIXELS = 20
FROZEN_DEDUP_LINF = 1e-4
FROZEN_IMAGE_SIZE = 64
FROZEN_IMAGE_DTYPE = "uint8"

# Frozen RGB thresholds, identical to the v3 shared visual contract. Recorded
# in the manifest so a mask audit can never silently use different numbers.
FROZEN_RED_THRESHOLD = {"red_min": 0.15, "green_max": 0.35, "blue_max": 0.35}
FROZEN_BLUE_THRESHOLD = {"blue_min": 0.15, "red_max": 0.35, "green_max": 0.35}

# Legacy renderer stamp half-width (``n08_visual.render`` writes a square of
# half-width ``lw`` around each sampled pixel). The analytic image baseline
# uses it to correct the projection extremes back to endpoint centres; it is
# renderer geometry, not oracle information.
FROZEN_STAMP_HALF_WIDTH = 2


@dataclass(frozen=True)
class BankConfig:
    schema: str = SCHEMA
    seeds: dict = field(default_factory=lambda: dict(FROZEN_SEEDS))
    split_sizes: dict = field(default_factory=lambda: dict(FROZEN_SPLIT_SIZES))
    oracle_margin_floor: float = FROZEN_ORACLE_MARGIN_FLOOR
    radii: tuple = FROZEN_RADII
    singleton_draws: int = FROZEN_SINGLETON_DRAWS
    quartet_draws: int = FROZEN_QUARTET_DRAWS
    max_single_edits_per_parent: int = FROZEN_MAX_SINGLE_EDITS_PER_PARENT
    max_quartets_per_parent: int = FROZEN_MAX_QUARTETS_PER_PARENT
    max_quartets_total: int = FROZEN_MAX_QUARTETS_TOTAL
    min_changed_pixels: int = FROZEN_MIN_CHANGED_PIXELS
    dedup_linf: float = FROZEN_DEDUP_LINF
    image_size: int = FROZEN_IMAGE_SIZE
    image_dtype: str = FROZEN_IMAGE_DTYPE
    red_threshold: dict = field(default_factory=lambda: dict(FROZEN_RED_THRESHOLD))
    blue_threshold: dict = field(default_factory=lambda: dict(FROZEN_BLUE_THRESHOLD))
    stamp_half_width: int = FROZEN_STAMP_HALF_WIDTH

    def validate(self) -> None:
        if self.image_size != FROZEN_IMAGE_SIZE:
            raise ValueError("renderer is fixed at 64 native pixels")
        if self.image_dtype != FROZEN_IMAGE_DTYPE:
            raise ValueError("bank images are stored as uint8; see loader contract")
        if self.oracle_margin_floor <= 0:
            raise ValueError("oracle margin floor must be positive")
        if not self.radii or any(r <= 0 for r in self.radii):
            raise ValueError("radii must be positive")
        if self.min_changed_pixels < 1:
            raise ValueError("pixel threshold must be positive")
        if self.max_single_edits_per_parent < 0 or self.max_quartets_per_parent < 0:
            raise ValueError("per-parent caps must be nonnegative")
        if self.max_quartets_total < 1:
            raise ValueError("total quartet cap must be positive")
        if set(self.split_sizes) != {"train", "dev", "test"}:
            raise ValueError("split sizes must name exactly train/dev/test")
        if set(self.seeds) != set(FROZEN_SEEDS):
            raise ValueError("seed table must name exactly the frozen seed keys")
        for split in ("train", "dev", "test"):
            for suffix in ("parent", "observation"):
                key = f"{split}_{suffix}"
                if not isinstance(self.seeds[key], int):
                    raise ValueError(f"seed {key} must be an int")

    def as_dict(self) -> dict:
        return asdict(self)


FROZEN = BankConfig()


def image_scale_note() -> str:
    return (
        "images are stored as uint8 in [0,255]; the model sees float32 in [0,1] obtained as "
        "images.astype(float32)/255 (m2.loader.images_as_float). The frozen color thresholds "
        "are applied to that [0,1] float range."
    )
