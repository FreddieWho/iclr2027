"""Singleton and quartet mining with realized failure-yield accounting.

Two established constructions are used, matching the archived project code
exactly:

* **singleton** supervision: a single atomic edit that *flips* the oracle
  label (``route2_visual.data_generator._single_states``);
* **quartet** supervision: an *emergent* quartet where each single edit
  preserves the label and the combination flips it
  (``next6_ef0f7a3.u01_quartet.mine_quartets``). The legacy miner is not called
  here because it used margin 0.005, radius 0.10 only and no changed-pixel
  filter; this module keeps its label semantics but applies the frozen
  contract: one margin floor, one radius schedule, one pixel threshold.

Every single-edit candidate lands in exactly one stage-1 bucket, so ``drawn``
equals the sum of the stage-1 buckets plus ``single_kept``. Every quartet pair
is accounted in one stage-2 bucket. Coverage is reported from accepted states,
never from requested draw counts (repair R3).
"""
from __future__ import annotations

import numpy as np

from scene_primitives import (
    atomic_edits,
    changed_pixels,
    oracle,
    parent_key,
    render_scene,
    render_seed,
    to_uint8_checked,
)

STAGE1_BUCKETS = (
    "oracle_valueerror",
    "ambiguous",
    "label_not_flipped",
    "label_not_preserved",
    "below_margin",
    "below_pixel_threshold",
    "single_kept",
)
STAGE2_BUCKETS = (
    "pair_oracle_valueerror",
    "pair_combo_preserved",
    "pair_below_margin",
    "pair_below_pixel_threshold",
)
PAIR_TOTAL_KEY = "pair_drawn"


def _new_yield(stage: str) -> dict:
    log = {name: 0 for name in STAGE1_BUCKETS}
    log.update({name: 0 for name in STAGE2_BUCKETS})
    log[PAIR_TOTAL_KEY] = 0
    log.update({"stage": stage, "drawn": 0, "accepted": 0, "per_family": {}, "per_radius": {}})
    return log


def _count(yld: dict, key: str, family: str, radius: float) -> None:
    yld[key] += 1
    fam = yld["per_family"].setdefault(family, {"drawn": 0, "single_kept": 0})
    fam["drawn"] += 1
    if key == "single_kept":
        fam["single_kept"] += 1
    rad = yld["per_radius"].setdefault(
        round(float(radius), 4), {"drawn": 0, **{name: 0 for name in STAGE1_BUCKETS}}
    )
    rad["drawn"] += 1
    rad[key] += 1


def mine_singletons(parents, config, observation_seed: int):
    """Clean state plus up to ``max_single_edits_per_parent`` label-flipping edits."""
    images, labels, parent_ids, clean, families, radii, margins, changed, states, edits = (
        [], [], [], [], [], [], [], [], [], [],
    )
    yield_log = _new_yield("singleton")
    per_parent_edits: dict[int, int] = {}
    for parent_id, scene in enumerate(parents):
        x = np.asarray(scene, dtype=np.float64)
        y0, margin0, ambiguous0 = oracle(x)
        if ambiguous0 or margin0 < config.oracle_margin_floor:
            raise ValueError(f"parent {parent_id} violates the frozen margin contract")
        seed = render_seed(observation_seed, parent_id)
        clean_image = render_scene(x, seed)
        images.append(to_uint8_checked(clean_image))
        labels.append(y0)
        parent_ids.append(parent_id)
        clean.append(True)
        families.append("clean")
        radii.append(0.0)
        margins.append(float(margin0))
        changed.append(0)
        states.append(0)
        edits.append(np.zeros((4, 2), dtype=np.float32))
        count = 0
        for draw in range(config.singleton_draws):
            radius = config.radii[draw % len(config.radii)]
            candidates = atomic_edits(
                x,
                np.random.default_rng(int(observation_seed) + parent_id * 101 + draw),
                radius=radius,
            )
            for edit, family in candidates:
                yield_log["drawn"] += 1
                try:
                    y1, margin1, ambiguous1 = oracle(x + edit)
                except ValueError:
                    _count(yield_log, "oracle_valueerror", family, radius)
                    continue
                if ambiguous1:
                    _count(yield_log, "ambiguous", family, radius)
                    continue
                if y1 == y0:
                    _count(yield_log, "label_not_flipped", family, radius)
                    continue
                if margin1 < config.oracle_margin_floor:
                    _count(yield_log, "below_margin", family, radius)
                    continue
                after = render_scene(x + edit, seed)
                changed_count = changed_pixels(clean_image, after)
                if changed_count < config.min_changed_pixels:
                    _count(yield_log, "below_pixel_threshold", family, radius)
                    continue
                _count(yield_log, "single_kept", family, radius)
                yield_log["accepted"] += 1
                images.append(to_uint8_checked(after))
                labels.append(y1)
                parent_ids.append(parent_id)
                clean.append(False)
                families.append(family)
                radii.append(float(radius))
                margins.append(float(margin1))
                changed.append(int(changed_count))
                states.append(count + 1)
                edits.append(np.asarray(edit, dtype=np.float32))
                count += 1
                if count >= config.max_single_edits_per_parent:
                    break
            if count >= config.max_single_edits_per_parent:
                break
        per_parent_edits[parent_id] = count
    histogram = (
        {str(int(k)): int(v) for k, v in zip(*np.unique(list(per_parent_edits.values()), return_counts=True))}
        if per_parent_edits
        else {}
    )
    yield_log["edit_count_histogram"] = histogram
    yield_log["parents_at_edit_cap"] = int(
        sum(v >= config.max_single_edits_per_parent for v in per_parent_edits.values())
    )
    yield_log["parents_with_zero_edits"] = int(sum(v == 0 for v in per_parent_edits.values()))
    arrays = {
        "images": np.asarray(images, dtype=np.uint8),
        "labels": np.asarray(labels, dtype=np.uint8),
        "parents": np.asarray(parent_ids, dtype=np.int32),
        "clean": np.asarray(clean, dtype=np.bool_),
        "edit_families": np.asarray(families, dtype="U32"),
        "edit_radii": np.asarray(radii, dtype=np.float32),
        "oracle_margins": np.asarray(margins, dtype=np.float32),
        "changed_pixels": np.asarray(changed, dtype=np.int32),
        "state_index": np.asarray(states, dtype=np.int16),
        "edit_vectors": np.asarray(edits, dtype=np.float32),
    }
    return arrays, yield_log


def mine_quartets(parents, config, mining_seed: int, observation_seed: int):
    """Emergent quartets: both single edits preserve the label, the combination flips it.

    Same margin floor, radius schedule and changed-pixel threshold as the
    singleton miner. All four states of a quartet are rendered with that
    parent's single render seed, so the four images share one background and
    an AB-vs-P render difference is a geometry difference.
    """
    keys = [parent_key(x) for x in parents]
    images, labels, margins, parents_out, keys_out = [], [], [], [], []
    edits_a, edits_b, families_a, families_b, radii_a, radii_b, changed_out = (
        [], [], [], [], [], [], [],
    )
    yield_log = _new_yield("quartet")
    yield_log["parents_rejected_margin"] = 0
    eligible_parents = 0
    parents_at_cap = 0
    for parent_id, scene in enumerate(parents):
        x = np.asarray(scene, dtype=np.float64)
        y0, margin0, ambiguous0 = oracle(x)
        if ambiguous0 or margin0 < config.oracle_margin_floor:
            yield_log["parents_rejected_margin"] += 1
            continue
        candidates = []
        for draw in range(config.quartet_draws):
            radius = config.radii[draw % len(config.radii)]
            for edit, family in atomic_edits(
                x,
                np.random.default_rng(int(mining_seed) + parent_id * 101 + draw),
                radius=radius,
            ):
                yield_log["drawn"] += 1
                try:
                    y1, margin1, ambiguous1 = oracle(x + edit)
                except ValueError:
                    _count(yield_log, "oracle_valueerror", family, radius)
                    continue
                if ambiguous1:
                    _count(yield_log, "ambiguous", family, radius)
                    continue
                if y1 != y0:
                    _count(yield_log, "label_not_preserved", family, radius)
                    continue
                if margin1 < config.oracle_margin_floor:
                    _count(yield_log, "below_margin", family, radius)
                    continue
                _count(yield_log, "single_kept", family, radius)
                candidates.append(
                    (np.asarray(edit, dtype=np.float64), family, float(radius), float(margin1))
                )
        if len(candidates) < 2:
            continue
        eligible_parents += 1
        seed = render_seed(observation_seed, parent_id)
        base_image = render_scene(x, seed)
        accepted_here = 0
        for i in range(len(candidates)):
            if accepted_here >= config.max_quartets_per_parent:
                break
            for j in range(i + 1, len(candidates)):
                if accepted_here >= config.max_quartets_per_parent:
                    break
                edit_a, family_a, radius_a, margin_a = candidates[i]
                edit_b, family_b, radius_b, margin_b = candidates[j]
                yield_log[PAIR_TOTAL_KEY] += 1
                try:
                    yc, margin_c, ambiguous_c = oracle(x + edit_a + edit_b)
                except ValueError:
                    yield_log["pair_oracle_valueerror"] += 1
                    continue
                if ambiguous_c or yc == y0:
                    yield_log["pair_combo_preserved"] += 1
                    continue
                if margin_c < config.oracle_margin_floor:
                    yield_log["pair_below_margin"] += 1
                    continue
                image_a = render_scene(x + edit_a, seed)
                image_b = render_scene(x + edit_b, seed)
                image_c = render_scene(x + edit_a + edit_b, seed)
                cp_a = changed_pixels(base_image, image_a)
                cp_b = changed_pixels(base_image, image_b)
                cp_c = changed_pixels(base_image, image_c)
                if min(cp_a, cp_b, cp_c) < config.min_changed_pixels:
                    yield_log["pair_below_pixel_threshold"] += 1
                    continue
                images.append(
                    np.stack(
                        [
                            to_uint8_checked(base_image),
                            to_uint8_checked(image_a),
                            to_uint8_checked(image_b),
                            to_uint8_checked(image_c),
                        ]
                    )
                )
                labels.append([y0, y0, y0, 1 - y0])
                margins.append([float(margin0), margin_a, margin_b, float(margin_c)])
                parents_out.append(int(parent_id))
                keys_out.append(keys[parent_id])
                edits_a.append(edit_a.astype(np.float32))
                edits_b.append(edit_b.astype(np.float32))
                families_a.append(family_a)
                families_b.append(family_b)
                radii_a.append(radius_a)
                radii_b.append(radius_b)
                changed_out.append([cp_a, cp_b, cp_c])
                accepted_here += 1
                yield_log["accepted"] += 1
        parents_at_cap += int(accepted_here >= config.max_quartets_per_parent)
    yield_log["eligible_parents"] = int(eligible_parents)
    yield_log["parents_at_quartet_cap"] = int(parents_at_cap)
    yield_log["quartets_per_eligible_parent"] = (
        float(len(labels) / eligible_parents) if eligible_parents else 0.0
    )
    yield_log["quartet_count"] = int(len(labels))
    yield_log["parents_with_quartets"] = int(len(set(parents_out)))
    family_counts: dict[str, int] = {}
    for name in families_a + families_b:
        family_counts[name] = family_counts.get(name, 0) + 1
    yield_log["quartet_edit_families"] = family_counts
    arrays = {
        "images": np.asarray(images, dtype=np.uint8).reshape(-1, 4, 3, 64, 64),
        "labels": np.asarray(labels, dtype=np.uint8).reshape(-1, 4),
        "margins": np.asarray(margins, dtype=np.float32).reshape(-1, 4),
        "parents": np.asarray(parents_out, dtype=np.int32),
        "keys": np.asarray(keys_out, dtype="U64"),
        "edits_a": np.asarray(edits_a, dtype=np.float32).reshape(-1, 4, 2),
        "edits_b": np.asarray(edits_b, dtype=np.float32).reshape(-1, 4, 2),
        "families_a": np.asarray(families_a, dtype="U32"),
        "families_b": np.asarray(families_b, dtype="U32"),
        "radii_a": np.asarray(radii_a, dtype=np.float32),
        "radii_b": np.asarray(radii_b, dtype=np.float32),
        "changed_pixels": np.asarray(changed_out, dtype=np.int32).reshape(-1, 3),
    }
    return arrays, yield_log
