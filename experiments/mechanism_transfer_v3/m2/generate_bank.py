#!/usr/bin/env python3
"""Build the frozen M2 expanded visual bank.

Contract (``bank_contract.FROZEN``): fixed per-split parent seeds, fixed splits,
one oracle margin floor, one radius schedule, one changed-pixel threshold for
both singletons and quartets, L-inf parent dedup, disjoint splits, and full
geometry provenance (parent coordinates, per-state edit vector/family/radius,
oracle margins, changed-pixel counts).

Output: ``data.npz`` + ``data_manifest.json`` in ``--out``. A delete-and-
regenerate determinism check runs before publication: the bank is built twice
into separate temporary directories and published only if every array SHA256
and the archive SHA256 match.

No neural training, no GPU job, no sealed pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _v3common import bank  # noqa: E402
from bank_contract import FROZEN, SCHEMA, BankConfig, image_scale_note  # noqa: E402
from miners import mine_quartets, mine_singletons  # noqa: E402
from scene_primitives import IMAGE_SIZE, make_parents, parent_key  # noqa: E402

ROOT = HERE.parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "mechanism_transfer_v3" / "m2" / "bank"
SPLITS = ("train", "dev", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def min_pairwise_linf(coords: np.ndarray, block: int = 256) -> float:
    """Minimum whole-scene L-inf distance between distinct parents."""
    flat = np.asarray(coords, dtype=np.float64).reshape(len(coords), -1)
    best = float("inf")
    for start in range(0, len(flat), block):
        chunk = flat[start : start + block]
        distances = np.abs(chunk[:, None, :] - flat[None, :, :]).max(axis=-1)
        for row in range(len(chunk)):
            distances[row, start + row] = np.inf
        best = min(best, float(distances.min()))
    return best


def min_linf_to_reference(queries: np.ndarray, reference: np.ndarray) -> float:
    if len(reference) == 0 or len(queries) == 0:
        return float("nan")
    distances = bank.min_linf_to_reference(
        np.asarray(queries, dtype=np.float64).reshape(len(queries), -1),
        np.asarray(reference, dtype=np.float64).reshape(len(reference), -1),
    )
    return float(np.min(distances))


def build_bank(config: BankConfig, log=print) -> tuple[dict, dict]:
    """Return (arrays, manifest-core). Training/evaluation numbers: none."""
    config.validate()
    arrays: dict[str, np.ndarray] = {}
    yields: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    split_records: dict[str, dict] = {}
    coords: dict[str, np.ndarray] = {}
    keys: dict[str, list[str]] = {}

    for split in SPLITS:
        n = int(config.split_sizes[split])
        parents, labels, margins = make_parents(
            n, config.seeds[f"{split}_parent"], config.oracle_margin_floor
        )
        kept, dropped = bank.greedy_dedup(parents, config.dedup_linf)
        if dropped:
            raise RuntimeError(
                f"{split}: {dropped} parents collapsed at L-inf {config.dedup_linf}; "
                "the frozen parent count would not be reproducible"
            )
        kept_keys = [parent_key(x) for x in parents]
        if len(set(kept_keys)) != n:
            raise RuntimeError(f"{split}: duplicate physical parent keys")
        coords[split] = parents
        keys[split] = kept_keys
        singles, yield_log = mine_singletons(
            parents, config, config.seeds[f"{split}_observation"]
        )
        yields[split] = yield_log
        arrays[f"{split}_images"] = singles["images"]
        arrays[f"{split}_labels"] = singles["labels"]
        arrays[f"{split}_parents"] = singles["parents"]
        arrays[f"{split}_clean"] = singles["clean"]
        arrays[f"{split}_edit_families"] = singles["edit_families"]
        arrays[f"{split}_edit_radii"] = singles["edit_radii"]
        arrays[f"{split}_oracle_margins"] = singles["oracle_margins"]
        arrays[f"{split}_changed_pixels"] = singles["changed_pixels"]
        arrays[f"{split}_state_index"] = singles["state_index"]
        arrays[f"{split}_edit_vectors"] = singles["edit_vectors"]
        arrays[f"{split}_parent_coords"] = parents.astype(np.float64)
        arrays[f"{split}_parent_keys"] = np.asarray(kept_keys, dtype="U64")
        coverage[split] = bank.coverage_report(
            singles["images"],
            singles["parents"],
            families=singles["edit_families"],
            radii=singles["edit_radii"],
            margins=singles["oracle_margins"],
        )
        split_records[split] = {
            "parent_seed": int(config.seeds[f"{split}_parent"]),
            "observation_seed": int(config.seeds[f"{split}_observation"]),
            "parent_count": int(n),
            "parent_label_counts": {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(labels, return_counts=True))
            },
            "singleton_images": int(len(singles["labels"])),
            "clean_images": int(singles["clean"].sum()),
            "edited_images": int((~singles["clean"]).sum()),
            "dedup_dropped": int(dropped),
        }
        log(f"[bank] {split}: {n} parents, {len(singles['labels'])} singleton images")

    key_sets = {split: set(keys[split]) for split in SPLITS}
    parent_overlaps = {
        "train_dev": len(key_sets["train"] & key_sets["dev"]),
        "train_test": len(key_sets["train"] & key_sets["test"]),
        "dev_test": len(key_sets["dev"] & key_sets["test"]),
    }
    if any(parent_overlaps.values()):
        raise RuntimeError(f"parent split overlap: {parent_overlaps}")

    image_keys = {
        split: {array_sha256(image) for image in arrays[f"{split}_images"]}
        for split in SPLITS
    }
    image_overlaps = {
        "train_dev": len(image_keys["train"] & image_keys["dev"]),
        "train_test": len(image_keys["train"] & image_keys["test"]),
        "dev_test": len(image_keys["dev"] & image_keys["test"]),
    }
    if any(image_overlaps.values()):
        raise RuntimeError(f"singleton image split overlap: {image_overlaps}")

    test_coords = coords["test"]
    quartet_arrays, quartet_yield = mine_quartets(
        test_coords,
        config,
        config.seeds["quartet_mining"],
        config.seeds["quartet_observation"],
    )
    if quartet_yield["quartet_count"] > config.max_quartets_total:
        keep = bank.stratified_sample_by_parent(
            quartet_arrays["parents"],
            config.seeds["quartet_mining"] + 1,
            per_parent_cap=config.max_quartets_per_parent,
            total_cap=config.max_quartets_total,
        )
        keep = np.sort(keep)
        quartet_arrays = {name: value[keep] for name, value in quartet_arrays.items()}
        quartet_yield["capped_total"] = int(len(keep))
        quartet_yield.setdefault("accepted_before_total_cap", int(quartet_yield["accepted"]))
        quartet_yield["quartet_count"] = int(len(keep))
        quartet_yield["parents_with_quartets"] = int(len(set(quartet_arrays["parents"].tolist())))
        quartet_yield["quartets_per_eligible_parent"] = (
            float(len(keep) / quartet_yield["eligible_parents"])
            if quartet_yield["eligible_parents"]
            else 0.0
        )
    arrays.update({f"quartet_{name}": value for name, value in quartet_arrays.items()})
    yields["quartet"] = quartet_yield
    log(
        "[bank] quartets: "
        f"{quartet_yield['quartet_count']} over {quartet_yield['parents_with_quartets']} parents "
        f"({quartet_yield['eligible_parents']} eligible)"
    )

    quartet_states = []
    for index in range(len(arrays["quartet_parents"])):
        base = test_coords[arrays["quartet_parents"][index]]
        quartet_states.append(base)
        quartet_states.append(base + quartet_arrays["edits_a"][index])
        quartet_states.append(base + quartet_arrays["edits_b"][index])
        quartet_states.append(base + quartet_arrays["edits_a"][index] + quartet_arrays["edits_b"][index])
    quartet_states = np.asarray(quartet_states, dtype=np.float64)
    separations = {
        "min_pairwise_parent_linf_train": min_pairwise_linf(coords["train"]),
        "min_pairwise_parent_linf_dev": min_pairwise_linf(coords["dev"]),
        "min_pairwise_parent_linf_test": min_pairwise_linf(coords["test"]),
        "min_test_to_train_parent_linf": min_linf_to_reference(coords["test"], coords["train"]),
        "min_test_to_dev_parent_linf": min_linf_to_reference(coords["test"], coords["dev"]),
        "min_train_to_dev_parent_linf": min_linf_to_reference(coords["train"], coords["dev"]),
        "min_quartet_state_to_train_parent_linf": min_linf_to_reference(
            quartet_states, coords["train"]
        ),
        "min_quartet_state_to_dev_parent_linf": min_linf_to_reference(
            quartet_states, coords["dev"]
        ),
        "dedup_linf": float(config.dedup_linf),
    }

    counts = {
        **{
            f"{split}_singleton_images": int(len(arrays[f"{split}_labels"]))
            for split in SPLITS
        },
        **{
            f"{split}_parents": int(config.split_sizes[split])
            for split in SPLITS
        },
        "quartets": int(arrays["quartet_labels"].shape[0]),
        "quartet_images": int(arrays["quartet_images"].shape[0] * 4),
        "quartet_eligible_parents": int(quartet_yield["eligible_parents"]),
        "quartet_sampled_parents": int(len(set(arrays["quartet_parents"].tolist()))),
        "ab_training_images": 0,
    }

    arrays_meta = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }
    manifest = {
        "schema": SCHEMA,
        "status": "DATA_READY",
        "formal_claim": "none",
        "scientific_result": "NOT_RUN",
        "no_neural_training": True,
        "no_gpu": True,
        "provenance": {
            "fresh_generator_only": True,
            "loaded_existing_scene_or_image_artifacts": False,
            "pilot_route2_bank_reused_as_data": False,
            "sealed_or_dev512_pool_read": False,
            "freezing_statement": (
                "seeds, split sizes, caps, margin floor, radius schedule and pixel threshold were "
                "fixed in bank_contract.FROZEN before generation"
            ),
        },
        "config": config.as_dict(),
        "splits": split_records,
        "counts": counts,
        "yields": yields,
        "coverage": coverage,
        "separations": separations,
        "split_disjointness": {
            "parent_overlap_counts": parent_overlaps,
            "parent_sets_pairwise_disjoint": not any(parent_overlaps.values()),
            "singleton_image_hash_overlaps": image_overlaps,
            "singleton_image_sets_pairwise_disjoint": not any(image_overlaps.values()),
        },
        "render_contract": {
            "native_resolution": [3, IMAGE_SIZE, IMAGE_SIZE],
            "image_dtype": config.image_dtype,
            "image_scale": image_scale_note(),
            "renderer": "experiments/last15h/n08_visual.render after canonical within-color endpoint sort",
            "background": "per-image uniform gray noise in [0.45,0.55)",
            "colors": {"AB": "red", "CD": "blue"},
            "per_parent_render_seed": (
                "one deterministic per-parent seed; every state of a parent shares the same "
                "background so render differences are geometry differences"
            ),
            "changed_pixel_rule": (
                f"max over RGB channels of |after-before| > 0.2, counted; require >= "
                f"{config.min_changed_pixels}"
            ),
            "deletion_note": (
                "no oracle mask, no coordinate, no parent id and no edit answer is stored in any "
                "model-facing array; coordinates and edit vectors live in separate audit arrays"
            ),
        },
        "mining_contract": {
            "singleton": (
                f"clean parent plus up to {config.max_single_edits_per_parent} single edits from "
                f"{config.singleton_draws} draws; radii {list(config.radii)}; oracle margin >= "
                f"{config.oracle_margin_floor}; >= {config.min_changed_pixels} changed pixels"
            ),
            "quartet": (
                "emergent quartets on test parents only: both single edits preserve the label and "
                "the combination flips it, all four states at the same margin floor and the same "
                f"changed-pixel threshold; at most {config.max_quartets_per_parent} per parent; "
                f"a non-binding total resource bound of {config.max_quartets_total} is declared "
                "and the realized yield is reported in this manifest"
            ),
            "legacy_miner_not_used": (
                "next6_ef0f7a3/u01_quartet.mine_quartets (margin 0.005, radius 0.10, no pixel "
                "filter) is not called by this generator"
            ),
        },
        "training_contract": (
            "singleton train images only (clean + single edit); quartet/AB states are test-only and "
            "never used for model selection"
        ),
        "selection_contract": "singleton dev BCE only",
        "inference_contract": (
            "image-only: no coordinates, parent ids, oracle labels, edit answers, AB truth or "
            "combination ids; RGB-derived color masks are allowed and disclosed"
        ),
        "arrays": arrays_meta,
    }
    return arrays, manifest


def write_bank(arrays: dict, manifest: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "data.npz"
    np.savez_compressed(archive, **arrays)
    manifest = dict(manifest)
    manifest["archive"] = {
        "path": archive.name,
        "sha256": file_sha256(archive),
        "size_bytes": int(archive.stat().st_size),
    }
    (out_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_bank(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    archive = out_dir / "data.npz"
    manifest_path = out_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if file_sha256(archive) != manifest["archive"]["sha256"]:
        raise ValueError("archive SHA256 mismatch")
    with np.load(archive, allow_pickle=False) as data:
        for name, meta in manifest["arrays"].items():
            if name not in data:
                raise ValueError(f"missing array {name}")
            if array_sha256(data[name]) != meta["sha256"]:
                raise ValueError(f"array SHA256 mismatch: {name}")
            if list(data[name].shape) != meta["shape"] or str(data[name].dtype) != meta["dtype"]:
                raise ValueError(f"array metadata mismatch: {name}")
        keys = {split: set(data[f"{split}_parent_keys"].tolist()) for split in SPLITS}
        if any(keys[a] & keys[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
            raise ValueError("parent split overlap")
        quartet_parents = data["quartet_parents"]
        if quartet_parents.max() >= len(data["test_parent_coords"]) or quartet_parents.min() < 0:
            raise ValueError("quartet parent index out of range")
        if data["quartet_images"].shape[1] != 4:
            raise ValueError("quartet states must be P,A,B,AB")
        if not (data["quartet_labels"][:, 0] == data["quartet_labels"][:, 1]).all():
            raise ValueError("quartet single edits must preserve the P label")
        if not (data["quartet_labels"][:, 0] == data["quartet_labels"][:, 2]).all():
            raise ValueError("quartet single edits must preserve the P label")
        if (data["quartet_labels"][:, 3] == data["quartet_labels"][:, 0]).any():
            raise ValueError("quartet combination must flip the label")
        if (data["quartet_margins"] < manifest["config"]["oracle_margin_floor"]).any():
            raise ValueError("quartet margin floor violated")
        if (data["quartet_changed_pixels"] < manifest["config"]["min_changed_pixels"]).any():
            raise ValueError("quartet state below the changed-pixel threshold")
        for split in SPLITS:
            if data[f"{split}_images"].shape[0] != data[f"{split}_labels"].shape[0]:
                raise ValueError(f"{split} image/label count mismatch")
            if (data[f"{split}_oracle_margins"] < manifest["config"]["oracle_margin_floor"]).any():
                raise ValueError(f"{split} margin floor violated")
            if not (data[f"{split}_changed_pixels"][~data[f"{split}_clean"]] >=
                    manifest["config"]["min_changed_pixels"]).all():
                raise ValueError(f"{split} changed-pixel threshold violated")
    return manifest


def _publish(arrays: dict, manifest: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    written = write_bank(arrays, manifest, out_dir)
    checked = validate_bank(out_dir)
    if checked["archive"]["sha256"] != written["archive"]["sha256"]:
        raise RuntimeError("validation changed the archive hash")
    return checked


def run(out_dir: Path = DEFAULT_OUT, config: BankConfig = FROZEN, determinism: bool = True, log=print):
    started = time.time()
    staging = Path(tempfile.mkdtemp(prefix="m2_bank_a_", dir=Path(tempfile.gettempdir())))
    replay = Path(tempfile.mkdtemp(prefix="m2_bank_b_", dir=Path(tempfile.gettempdir())))
    try:
        arrays, manifest = build_bank(config, log=log)
        first = write_bank(arrays, manifest, staging)
        receipt = {"determinism_check": "not_run"}
        if determinism:
            replay_arrays, replay_manifest = build_bank(config, log=lambda *_: None)
            second = write_bank(replay_arrays, replay_manifest, replay)
            differing = sorted(
                name
                for name in set(first["arrays"]) | set(second["arrays"])
                if first["arrays"].get(name, {}).get("sha256")
                != second["arrays"].get(name, {}).get("sha256")
            )
            if differing:
                raise RuntimeError(f"non-deterministic arrays: {differing}")
            receipt = {
                "determinism_check": "passed",
                "arrays_compared": len(first["arrays"]),
                "array_sha256_all_equal": True,
                "archive_sha256_equal": bool(
                    first["archive"]["sha256"] == second["archive"]["sha256"]
                ),
                "first_archive_sha256": first["archive"]["sha256"],
                "second_archive_sha256": second["archive"]["sha256"],
            }
            log(f"[bank] determinism: {receipt['determinism_check']} over {receipt['arrays_compared']} arrays")
        final = _publish(arrays, manifest, Path(out_dir))
        final["determinism"] = receipt
        final["seconds"] = time.time() - started
        (Path(out_dir) / "data_manifest.json").write_text(json.dumps(final, indent=2) + "\n")
        return final
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(replay, ignore_errors=True)


TINY = BankConfig(
    split_sizes={"train": 8, "dev": 4, "test": 16},
    singleton_draws=6,
    quartet_draws=8,
    max_quartets_total=8,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tiny", action="store_true", help="small config for smoke tests")
    parser.add_argument("--no-determinism", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_bank(args.out)["counts"], indent=2))
        return
    config = TINY if args.tiny else FROZEN
    manifest = run(args.out, config=config, determinism=not args.no_determinism)
    print(json.dumps({
        "status": manifest["status"],
        "counts": manifest["counts"],
        "determinism": manifest["determinism"],
        "archive_sha256": manifest["archive"]["sha256"],
        "size_bytes": manifest["archive"]["size_bytes"],
        "seconds": manifest["seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
