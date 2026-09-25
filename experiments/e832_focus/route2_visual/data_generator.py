#!/usr/bin/env python3
"""Deterministic fresh-parent visual data for Route 2.

This generator creates a new bank from fixed parent seeds before rendering.
It never loads an existing scene/image artifact. Coordinates and oracle labels
are used only to construct singleton supervision; inference consumes images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "artifacts/e832_focus/route2/data"

# Frozen before generation. These are Route-2-only seeds, not inherited pools.
SEEDS = {
    "train_parent": 832701,
    "dev_parent": 832702,
    "test_parent": 832703,
    "train_observation": 832711,
    "dev_observation": 832712,
    "test_observation": 832713,
    "quartet_mining": 832721,
    "quartet_observation": 832722,
}
SPLIT_SIZES = {"train": 128, "dev": 32, "test": 128}
MAX_SINGLE_EDITS_PER_PARENT = 2
MAX_QUARTETS_PER_PARENT = 2

# The generator functions are code dependencies only; no data path is opened.
for directory in (
    ROOT / "experiments/last15h",
    ROOT / "experiments/last15h/shared",
    ROOT / "experiments/next6_ef0f7a3",
    ROOT / "experiments/f095_campaign",
    ROOT / "docs/iclr2027_discovery_campaign_20260717",
    ROOT / "docs/iclr2027_discovery_campaign_20260917",
):
    if directory.exists():
        sys.path.insert(0, str(directory))

from paths import atomic_edits, oracle_at  # noqa: E402
from u01_quartet import mine_quartets  # noqa: E402
from core.relations import make_relational_scenes  # noqa: E402
from n08_visual import render as legacy_render  # noqa: E402


def digest_array(value: object) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_parent_key(x: object) -> str:
    """Color-preserving endpoint-order key used only for split auditing."""
    x = np.asarray(x, dtype=np.float64).reshape(4, 2)
    ordered = np.asarray(sorted(map(tuple, x[:2])) + sorted(map(tuple, x[2:])))
    return digest_array(ordered)


def render(x: object, rng: np.random.Generator) -> np.ndarray:
    """Corrected canonical native-64 renderer (same-color endpoint sorting)."""
    x = np.asarray(x).reshape(4, 2)
    ordered = np.asarray(sorted(map(tuple, x[:2])) + sorted(map(tuple, x[2:])))
    return np.asarray(legacy_render(ordered, rng), dtype=np.float32)


def _single_states(x: np.ndarray, parent: int, observation_seed: int) -> tuple[list, list, list, list, list]:
    y0, margin0, ambiguous0 = oracle_at(x)
    if ambiguous0 or margin0 < 0.02:
        raise ValueError("generated parent unexpectedly ambiguous")
    images = [render(x, np.random.default_rng(observation_seed + parent * 17))]
    labels = [float(y0)]
    parents = [parent]
    clean = [True]
    families = ["clean"]
    count = 0
    for trial in range(16):
        radius = (0.10, 0.20, 0.35, 0.50)[trial % 4]
        candidates = atomic_edits(
            x, np.random.default_rng(observation_seed + parent * 101 + trial), radius=radius
        )
        for edit, family in candidates:
            try:
                y1, margin1, _ = oracle_at(x + edit)
            except ValueError:
                continue
            if y1 == y0 or margin1 < 0.02:
                continue
            before = render(x, np.random.default_rng(observation_seed + parent * 1009 + count))
            after = render(x + edit, np.random.default_rng(observation_seed + parent * 1009 + count))
            if int((np.abs(after - before).max(0) > 0.2).sum()) < 20:
                continue
            images.append(after)
            labels.append(float(y1))
            parents.append(parent)
            clean.append(False)
            families.append(family)
            count += 1
            if count >= MAX_SINGLE_EDITS_PER_PARENT:
                return images, labels, parents, clean, families
    return images, labels, parents, clean, families


def generate(out: Path = DEFAULT_OUT) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="route2_data_", dir=out.parent))
    arrays: dict[str, np.ndarray] = {}
    parent_keys: dict[str, list[str]] = {}
    test_parents = None
    split_records: dict[str, dict] = {}

    try:
        for split in ("train", "dev", "test"):
            n = SPLIT_SIZES[split]
            parent_seed = SEEDS[f"{split}_parent"]
            obs_seed = SEEDS[f"{split}_observation"]
            xs, _, _ = make_relational_scenes(n, seed=parent_seed, min_margin=0.02)
            xs = np.asarray(xs, dtype=np.float64)
            if len(xs) != n:
                raise RuntimeError(f"{split}: requested {n}, generated {len(xs)}")
            parent_keys[split] = [canonical_parent_key(x) for x in xs]
            if split == "test":
                test_parents = xs.copy()
            if len(set(parent_keys[split])) != n:
                raise RuntimeError(f"{split}: duplicate physical parents")
            images, labels, parents, clean, families = [], [], [], [], []
            for parent, x in enumerate(xs):
                rows = _single_states(x, parent, obs_seed)
                images.extend(rows[0]); labels.extend(rows[1]); parents.extend(rows[2])
                clean.extend(rows[3]); families.extend(rows[4])
            arrays[f"{split}_images"] = np.asarray(images, dtype=np.float32)
            arrays[f"{split}_labels"] = np.asarray(labels, dtype=np.float32)
            arrays[f"{split}_parents"] = np.asarray(parents, dtype=np.int32)
            arrays[f"{split}_clean"] = np.asarray(clean, dtype=np.bool_)
            arrays[f"{split}_edit_families"] = np.asarray(families, dtype="U32")
            arrays[f"{split}_parent_keys"] = np.asarray(parent_keys[split], dtype="U64")
            split_records[split] = {
                "parent_seed": parent_seed,
                "observation_seed": obs_seed,
                "parent_count": n,
                "singleton_image_count": len(labels),
                "clean_singleton_count": int(sum(clean)),
                "edited_singleton_count": int(len(clean) - sum(clean)),
                "parent_ids_equal_contiguous_split_ids": set(parents) == set(range(n)),
            }

        train_keys, dev_keys, test_keys = (set(parent_keys[s]) for s in ("train", "dev", "test"))
        overlaps = {
            "train_dev": len(train_keys & dev_keys),
            "train_test": len(train_keys & test_keys),
            "dev_test": len(dev_keys & test_keys),
        }
        if any(overlaps.values()):
            raise RuntimeError(f"parent split overlap: {overlaps}")

        xs = np.asarray(test_parents, dtype=np.float64)  # test parents only
        mined = mine_quartets(xs, np.random.default_rng(SEEDS["quartet_mining"]))
        lookup = {canonical_parent_key(x): i for i, x in enumerate(xs)}
        per_parent: dict[int, int] = {}
        q_images, q_labels, q_parents = [], [], []
        for x, ea, eb, y0, ya, yb, yab in mined:
            parent = lookup[canonical_parent_key(x)]
            if per_parent.get(parent, 0) >= MAX_QUARTETS_PER_PARENT:
                continue
            states = (x, x + ea, x + eb, x + ea + eb)
            labels = (y0, ya, yb, yab)
            image_index = len(q_images)
            for state_index, state in enumerate(states):
                q_images.append(render(state, np.random.default_rng(
                    SEEDS["quartet_observation"] + parent * 10007 + state_index
                )))
            q_labels.append(labels)
            q_parents.append(parent)
            per_parent[parent] = per_parent.get(parent, 0) + 1
        if not q_labels:
            raise RuntimeError("miner produced no quartets for frozen test parents")
        arrays["quartet_images"] = np.asarray(q_images, dtype=np.float32).reshape(-1, 4, 3, 64, 64)
        arrays["quartet_labels"] = np.asarray(q_labels, dtype=np.float32)
        arrays["quartet_parents"] = np.asarray(q_parents, dtype=np.int32)

        np.savez_compressed(temp / "data.npz", **arrays)
        array_hashes = {name: digest_array(value) for name, value in sorted(arrays.items())}
        image_keys = {
            split: {digest_array(image) for image in arrays[f"{split}_images"]}
            for split in ("train", "dev", "test")
        }
        image_overlaps = {
            "train_dev": len(image_keys["train"] & image_keys["dev"]),
            "train_test": len(image_keys["train"] & image_keys["test"]),
            "dev_test": len(image_keys["dev"] & image_keys["test"]),
        }
        if any(image_overlaps.values()):
            raise RuntimeError(f"singleton image split overlap: {image_overlaps}")
        manifest = {
            "schema_version": 1,
            "status": "DATA_READY",
            "formal_claim": "none",
            "scientific_result": "NOT_RUN",
            "provenance": {
                "fresh_generator_only": True,
                "loaded_existing_scene_or_image_artifacts": False,
                "old_exposed_159_bank_used": False,
                "sealed_or_dev512_pool_read": False,
                "freezing_statement": "all parent, observation, mining, and renderer seeds were fixed in SEEDS before generation",
            },
            "frozen_seeds": SEEDS,
            "split_sizes": SPLIT_SIZES,
            "splits": split_records,
            "counts": {
                **{f"{s}_singleton_images": len(arrays[f"{s}_labels"]) for s in ("train", "dev", "test")},
                "quartets": len(arrays["quartet_labels"]),
                "quartet_images": int(len(arrays["quartet_labels"]) * 4),
                "quartet_parents": len(set(arrays["quartet_parents"].tolist())),
                "ab_training_images": 0,
            },
            "split_disjointness": {
                "parent_overlap_counts": overlaps,
                "parent_sets_pairwise_disjoint": not any(overlaps.values()),
                "train_dev_test_image_hash_overlaps": image_overlaps,
                "image_sets_pairwise_disjoint": not any(image_overlaps.values()),
            },
            "training_contract": "singleton train images only; clean + atomic single-edit states; no AB or quartet training labels",
            "selection_contract": "singleton dev BCE only",
            "evaluation_contract": "new test singletons plus test-only quartets in P,A,B,AB state order",
            "observation_recipe": {
                "native_resolution": [3, 64, 64],
                "model_input": "bilinear resize native64 to configured 224, align_corners=False",
                "renderer": "n08_visual render after lexicographic endpoint sorting within each visible color",
                "background": "per-image uniform gray noise in [0.45,0.55)",
                "colors": {"P_A": "red", "P_B": "blue"},
                "single_edit_recipe": "up to 2 valid atomic single flips/parent from 16 fixed draws; radii .10/.20/.35/.50; oracle margin >= .02; >=20 changed pixels",
                "quartet_recipe": "mine_quartets(test parents only); retain at most 2 quartets/parent in stable miner order",
            },
            "inference_contract": "Mechanism.forward(images) only; no coordinates, parent IDs, oracle labels, edit answers, AB truth, or combination IDs",
            "array_sha256": array_hashes,
            "archive": {"path": "data.npz", "sha256": file_sha256(temp / "data.npz")},
        }
        (temp / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if out.exists():
            shutil.rmtree(out)
        temp.replace(out)
        return manifest
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def validate(data_path: Path, bundle_manifest: Path | None = None) -> dict:
    data_path = Path(data_path)
    archive = data_path if data_path.is_file() else data_path / "data.npz"
    manifest_path = archive.with_name("data_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if file_sha256(archive) != manifest["archive"]["sha256"]:
        raise ValueError("data archive SHA256 mismatch")
    with np.load(archive, allow_pickle=False) as data:
        for name, expected in manifest["array_sha256"].items():
            if name not in data or digest_array(data[name]) != expected:
                raise ValueError(f"array SHA256 mismatch: {name}")
        keys = {s: set(data[f"{s}_parent_keys"].tolist()) for s in ("train", "dev", "test")}
        if any(keys[a] & keys[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
            raise ValueError("parent split overlap")
        image_keys = {
            split: {digest_array(image) for image in data[f"{split}_images"]}
            for split in ("train", "dev", "test")
        }
        if any(image_keys[a] & image_keys[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
            raise ValueError("singleton image split overlap")
        for split in ("train", "dev", "test"):
            if data[f"{split}_images"].shape[0] != data[f"{split}_labels"].shape[0]:
                raise ValueError(f"{split} image/label count mismatch")
        if data["quartet_images"].shape[:2] != (manifest["counts"]["quartets"], 4):
            raise ValueError("quartet shape mismatch")
    if bundle_manifest is not None:
        bundle = json.loads(Path(bundle_manifest).read_text())
        attached = bundle["data"]
        if Path(attached["archive_path"]).name != archive.name:
            raise ValueError("explicit data archive is not the manifest attachment")
        if attached["archive_sha256"] != manifest["archive"]["sha256"]:
            raise ValueError("bundle/data manifest archive mismatch")
        if attached["data_manifest_sha256"] != file_sha256(manifest_path):
            raise ValueError("bundle data-manifest SHA256 mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        payload = validate(args.out, args.bundle_manifest)
    else:
        payload = generate(args.out)
        payload = validate(args.out)
    print(json.dumps({
        "status": payload["status"],
        "data": str(args.out),
        "counts": payload["counts"],
        "parent_split_disjoint": payload["split_disjointness"]["parent_sets_pairwise_disjoint"],
        "archive_sha256": payload["archive"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
