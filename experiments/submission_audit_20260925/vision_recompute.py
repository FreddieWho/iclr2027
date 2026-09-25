#!/usr/bin/env python3
"""Deterministically re-score archived visual predictions; performs no inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (803, 805, 806)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def safe_ratio(numerator: int | float, denominator: int | float):
    return float(numerator / denominator) if denominator else None


def resnet18_macs(width: int, resolution: int) -> int:
    """Count Conv/Linear MACs from layer shapes, without a model forward."""
    def out_size(size: int, kernel: int, stride: int, padding: int) -> int:
        return (size + 2 * padding - kernel) // stride + 1

    h = out_size(resolution, 7, 2, 3)
    total = h * h * (64 * width) * (3 * 7 * 7)
    channels_in = 64 * width
    h = out_size(h, 3, 2, 1)
    for base_channels, first_stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
        channels_out = base_channels * width
        for block in range(2):
            stride = first_stride if block == 0 else 1
            h_out = out_size(h, 3, stride, 1)
            total += h_out * h_out * channels_out * channels_in * 9
            total += h_out * h_out * channels_out * channels_out * 9
            if stride != 1 or channels_in != channels_out:
                total += h_out * h_out * channels_out * channels_in
            h, channels_in = h_out, channels_out
    return total + channels_in


def score(logits: np.ndarray, labels: np.ndarray) -> dict:
    correct = (np.asarray(logits) > 0) == (np.asarray(labels) > 0.5)
    atom = correct[:, 1] & correct[:, 2]
    joint = atom & correct[:, 3]
    full = joint & correct[:, 0]
    atom_n = int(atom.sum())
    joint_n = int(joint.sum())
    miss_n = int((atom & ~correct[:, 3]).sum())
    return {
        "n_quartets": int(len(labels)),
        "P": float(correct[:, 0].mean()),
        "A": float(correct[:, 1].mean()),
        "B": float(correct[:, 2].mean()),
        "AB": float(correct[:, 3].mean()),
        "atomic_joint": float(atom.mean()),
        "atomic_joint_n": atom_n,
        "J3": float(joint.mean()),
        "J": float(joint.mean()),
        "J4": float(full.mean()),
        "conditional_joint_success": safe_ratio(joint_n, atom_n),
        "composition_miss": safe_ratio(miss_n, atom_n),
        "CCM": safe_ratio(miss_n, atom_n),
        "composition_miss_n": miss_n,
    }


def parent_means(values: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(parents)
    return ids, np.asarray([values[parents == parent].mean() for parent in ids], dtype=float)


def parent_bootstrap(values: np.ndarray, parents: np.ndarray, seed: int,
                     n_boot: int = 4000) -> list[float]:
    _, means = parent_means(np.asarray(values, dtype=float), parents)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(n_boot, len(means)))
    return np.quantile(means[draws].mean(axis=1), [0.025, 0.975]).tolist()


def row_weighted_parent_bootstrap(values: np.ndarray, parents: np.ndarray, seed: int,
                                  n_boot: int = 2000) -> list[float]:
    """Resample parents, then retain every quartet for each sampled parent."""
    ids = np.unique(parents)
    groups = [np.flatnonzero(parents == parent) for parent in ids]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(ids, len(ids), replace=True)
        ix = np.concatenate([groups[np.searchsorted(ids, parent)] for parent in sampled])
        draws.append(np.asarray(values)[ix].mean())
    return np.quantile(draws, [0.025, 0.975]).tolist()


def paired_summary(a: dict, b: dict, n_boot: int = 4000,
                   seed: int = 962009, row_boot: bool = False) -> dict:
    assert np.array_equal(a["labels"], b["labels"])
    assert np.array_equal(a["parent"], b["parent"])
    labels, parents = a["labels"], a["parent"]
    ok_a = (a["logits"] > 0) == (labels > 0.5)
    ok_b = (b["logits"] > 0) == (labels > 0.5)
    j_a, j_b = ok_a[:, 1:].all(1), ok_b[:, 1:].all(1)
    delta = j_b.astype(float) - j_a.astype(float)
    _, parent_delta = parent_means(delta, parents)
    h = ok_a[:, 1] & ok_a[:, 2] & ~ok_a[:, 3]
    baseline_111 = j_a
    candidate_111 = j_b
    ci = (row_weighted_parent_bootstrap(delta, parents, seed, n_boot)
          if row_boot else parent_bootstrap(delta, parents, seed, n_boot))
    return {
        "n_quartets": int(len(labels)),
        "parent_n": int(len(np.unique(parents))),
        "delta_J_quartet_weighted": float(delta.mean()),
        "delta_J_parent_equal": float(parent_delta.mean()),
        "parent_cluster_bootstrap95": ci,
        "bootstrap_estimand": "quartet_weighted_with_parent_cluster_resampling" if row_boot else "parent_equal",
        "baseline_110_n": int(h.sum()),
        "full_repair_n": int((h & candidate_111).sum()),
        "migration_n": int((h & ok_b[:, 3] & ~(ok_b[:, 1] & ok_b[:, 2])).sum()),
        "baseline_111_n": int(baseline_111.sum()),
        "regression_111_n": int((baseline_111 & ~candidate_111).sum()),
    }


def load_prediction(folder: Path, name: str = "predictions.npz") -> dict:
    return read_npz(folder / name)


def _same_eval_bank(reference: dict | None, current: dict) -> dict:
    if reference is None:
        return current
    assert np.array_equal(reference["labels"], current["labels"])
    assert np.array_equal(reference["parent"], current["parent"])
    return reference


def audit_upgrade() -> dict:
    out = ROOT / "artifacts/n02_upgrade/gpu_out"
    receipt_path = out / "receipt.json"
    receipt = read_json(receipt_path)
    stored = read_json(out / "upgrade_analysis.json")
    matrix = read_json(out / "structure_matrix.json")
    assert receipt["status"] == "MATRIX_COMPLETE"
    completed = receipt["completed"]
    assert len(completed) == 30 and len({row["arm"] for row in completed}) == 10
    assert len({row["seed"] for row in completed}) == 3

    common = None
    arm_records = []
    arrays = {}
    hashes = {"receipt.json": sha256(receipt_path),
              "upgrade_analysis.json": sha256(out / "upgrade_analysis.json"),
              "structure_matrix.json": sha256(out / "structure_matrix.json"),
              "smaledit_manifest.json": sha256(ROOT / "artifacts/n02_upgrade/smaledit_manifest.json"),
              "smaledit_bank.npz": sha256(ROOT / "artifacts/n02_upgrade/smaledit_bank.npz")}
    legacy_checks = []
    for item in completed:
        arm, seed = item["arm"], int(item["seed"])
        folder = out / f"{arm}_s{seed}"
        pred_path = folder / "predictions.npz"
        pred = load_prediction(folder)
        common = _same_eval_bank(common, pred)
        arrays[(arm, seed)] = pred
        hashes[f"{arm}_s{seed}/predictions.npz"] = sha256(pred_path)
        result = read_json(folder / "result.json")
        metric = score(pred["logits"], pred["labels"])
        ids, j_by_parent = parent_means(
            (((pred["logits"] > 0) == (pred["labels"] > 0.5))[:, 1:].all(1)).astype(float),
            pred["parent"],
        )
        metric["parent_n"] = int(len(ids))
        metric["parent_equal_J3"] = float(j_by_parent.mean())
        metric["archived_result_CCM"] = result.get("CCM")
        metric["archived_CCM_matches_legacy_conditional_success"] = bool(
            result.get("CCM") is not None
            and np.isclose(result["CCM"], metric["conditional_joint_success"], atol=1e-12)
        )
        legacy_checks.append(metric["archived_CCM_matches_legacy_conditional_success"])
        arm_records.append({"arm": arm, "seed": seed, **metric})

    assert all(legacy_checks)
    labels, parents = common["labels"], common["parent"]
    old_analysis_path = out / "upgrade_analysis.json"
    p0 = {}
    p0_checks = []
    for init in ("pretrained", "random"):
        rows = []
        for seed in SEEDS:
            base = arrays[(f"A_standard_{init}_flip", seed)]
            cand = arrays[(f"B_standard_{init}_flip", seed)]
            result = paired_summary(base, cand, seed=962009)
            original = stored["P0_B_minus_A_standard_oldbank"][init][SEEDS.index(seed)]
            assert result["n_quartets"] == original["n"]
            assert np.isclose(result["delta_J_parent_equal"], original["delta_J_parent"], atol=1e-12)
            assert np.allclose(result["parent_cluster_bootstrap95"], original["parent_bootstrap95"], atol=1e-12)
            p0_checks.append(True)
            rows.append({"seed": seed, **result})
        p0[init] = rows

    p1_by_seed = []
    p1_vectors = []
    for seed in SEEDS:
        j_vectors = {}
        row_j = {}
        ref = arrays[("A_standard_random_flip", seed)]
        seed_ids = np.unique(ref["parent"])
        for obs in ("A", "W", "B"):
            pred = arrays[(f"{obs}_standard_random_flip", seed)]
            assert np.array_equal(ref["parent"], pred["parent"])
            assert np.array_equal(ref["labels"], pred["labels"])
            j = (((pred["logits"] > 0) == (pred["labels"] > 0.5))[:, 1:].all(1)).astype(float)
            row_j[obs] = float(j.mean())
            _, j_vectors[obs] = parent_means(j, pred["parent"])
        num = float((j_vectors["W"] - j_vectors["A"]).mean())
        den = float((j_vectors["B"] - j_vectors["A"]).mean())
        f = num / den if abs(den) > 1e-12 else None
        rng = np.random.default_rng(962300 + seed)
        draws = rng.integers(0, len(seed_ids), size=(4000, len(seed_ids)))
        f_draws = ((j_vectors["W"][draws].mean(1) - j_vectors["A"][draws].mean(1)) /
                   (j_vectors["B"][draws].mean(1) - j_vectors["A"][draws].mean(1)))
        ci = np.quantile(f_draws[np.isfinite(f_draws)], [0.025, 0.975]).tolist()
        original = stored["P1_compute_fraction_random_standard_oldbank"][SEEDS.index(seed)]
        assert np.isclose(original["f_WA_over_BA"], f, atol=1e-12)
        assert int(original["seed"]) == seed
        p1_vectors.append(j_vectors)
        p1_by_seed.append({
            "seed": seed,
            "J_row_weighted": {f"J_{key}": value for key, value in row_j.items()},
            "J_parent_equal": {f"J_{key}": float(j_vectors[key].mean()) for key in ("A", "W", "B")},
            "delta_W_minus_A_parent_equal": num,
            "delta_B_minus_A_parent_equal": den,
            "f_WA_over_BA": float(f),
            "parent_cluster_bootstrap95": ci,
            "bootstrap_n": 4000,
        })

    rng = np.random.default_rng(962399)
    mean_f_draws, pooled_f_draws = [], []
    seed_count, parent_count = len(SEEDS), len(np.unique(parents))
    for _ in range(4000):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        nums, dens, fs = [], [], []
        for si in sampled_seeds:
            vectors = p1_vectors[int(si)]
            pick = rng.integers(0, parent_count, size=parent_count)
            n = float((vectors["W"][pick] - vectors["A"][pick]).mean())
            d = float((vectors["B"][pick] - vectors["A"][pick]).mean())
            nums.append(n); dens.append(d)
            fs.append(n / d if abs(d) > 1e-12 else np.nan)
        mean_f_draws.append(float(np.nanmean(fs)))
        pooled_f_draws.append(float(np.mean(nums) / np.mean(dens)))
    p1_mean_f = float(np.mean([row["f_WA_over_BA"] for row in p1_by_seed]))
    pooled_num = float(np.mean([row["delta_W_minus_A_parent_equal"] for row in p1_by_seed]))
    pooled_den = float(np.mean([row["delta_B_minus_A_parent_equal"] for row in p1_by_seed]))
    p1 = {
        "per_seed": p1_by_seed,
        "descriptive_arithmetic_mean_of_seed_f": p1_mean_f,
        "seed_range": [min(r["f_WA_over_BA"] for r in p1_by_seed), max(r["f_WA_over_BA"] for r in p1_by_seed)],
        "ratio_of_cross_seed_mean_parent_deltas": pooled_num / pooled_den,
        "hierarchical_bootstrap_mean_of_seed_f95": np.quantile(mean_f_draws, [0.025, 0.975]).tolist(),
        "hierarchical_bootstrap_ratio_of_pooled_deltas95": np.quantile(pooled_f_draws, [0.025, 0.975]).tolist(),
        "hierarchical_bootstrap": "4000 resamples of 3 seeds, then 85 parent clusters within each sampled seed; exploratory because aggregation was not frozen and seeds are not independent tasks",
        "frozen_aggregation": "NOT_SPECIFIED; the report's majority-by-arithmetic-mean label is descriptive",
    }

    def interaction(init: str, seed: int, sample_name: str, file_name: str) -> dict:
        files = {}
        for obs in ("A", "B"):
            for stem in ("standard", "lowstride"):
                path = out / f"{obs}_{stem}_{init}_flip_s{seed}" / file_name
                files[(obs, stem)] = read_npz(path)
                hashes[f"{obs}_{stem}_{init}_flip_s{seed}/{file_name}"] = sha256(path)
        first = files[("A", "standard")]
        for value in files.values():
            assert np.array_equal(first["parent"], value["parent"])
            assert np.array_equal(first["labels"], value["labels"])
        j = {key: (((value["logits"] > 0) == (value["labels"] > 0.5))[:, 1:].all(1)).astype(float)
             for key, value in files.items()}
        delta = (j[("B", "lowstride")] - j[("A", "lowstride")]) - \
                (j[("B", "standard")] - j[("A", "standard")])
        ids, parent_delta = parent_means(delta, first["parent"])
        ci = parent_bootstrap(delta, first["parent"], seed=962010)
        return {
            "init": init, "seed": seed, "stratum": sample_name,
            "n_quartets": int(len(delta)), "parent_n": int(len(ids)),
            "interaction_quartet_weighted": float(delta.mean()),
            "interaction_parent_equal": float(parent_delta.mean()),
            "parent_cluster_bootstrap95": ci,
        }

    p2_old, p2_small = [], []
    for init in ("pretrained", "random"):
        for seed in SEEDS:
            p2_old.append(interaction(init, seed, "old_178_quartet_bank", "predictions.npz"))
            p2_small.append(interaction(init, seed, "fresh_small_edit_bank", "smaledit_predictions.npz"))
    small_manifest = read_json(ROOT / "artifacts/n02_upgrade/smaledit_manifest.json")
    p2_gate = {
        "frozen_prediction": "lowstride interaction negative with parent CI excluding zero under both initializations",
        "new_small_edit_bank_sufficient_denominator": int(small_manifest["rows"]) >= 60 and int(small_manifest["parents"]) >= 30,
        "new_small_edit_bank_rows": int(small_manifest["rows"]),
        "new_small_edit_bank_parents": int(small_manifest["parents"]),
        "new_small_edit_max_quartets_per_parent": int(np.unique(
            read_npz(out / "A_standard_pretrained_flip_s803" / "smaledit_predictions.npz")["parent"],
            return_counts=True)[1].max()),
        "new_small_edit_parent_cap": 4,
        "observed_gate": "UNRESOLVED_NOT_SUPPORTED; pretrained signs vary and random-init interactions are positive",
        "per_seed_results": p2_small,
        "old_bank_results": p2_old,
    }
    p2_checks = []
    for key, actuals in (("P2_lowstride_interaction_smaledit", p2_small),
                         ("P2_lowstride_interaction_oldbank", p2_old)):
        for actual in actuals:
            original = next(row for row in stored[key]
                            if row["init"] == actual["init"] and row["seed"] == actual["seed"])
            p2_checks.append(
                actual["n_quartets"] == original["n"]
                and actual["parent_n"] == original["parent_n"]
                and np.isclose(actual["interaction_parent_equal"], original["interaction_parent"], atol=1e-12)
                and np.allclose(actual["parent_cluster_bootstrap95"], original["parent_bootstrap95"], atol=1e-12)
            )
    assert all(p2_checks)

    capacities = {
        key: {"parameters": int(value["parameters"]), "conv_linear_MACs_per_image": int(value["conv_linear_MACs"])}
        for key, value in matrix.items() if "parameters" in value and "conv_linear_MACs" in value
    }
    standard_64_macs = resnet18_macs(1, 64)
    standard_224_macs = resnet18_macs(1, 224)
    wide_standard_64_macs = resnet18_macs(4, 64)
    assert standard_64_macs == int(matrix["standard_64"]["conv_linear_MACs"])
    assert standard_224_macs == int(matrix["standard_224"]["conv_linear_MACs"])
    wide_params = int(receipt["wide_preflight"]["params"])
    standard_params = int(matrix["standard_224"]["parameters"])
    return {
        "source_receipt": {
            "status": receipt["status"], "completed_runs": len(completed),
            "unique_factorial_cells": len({row["arm"] for row in completed}),
            "seeds": sorted({int(row["seed"]) for row in completed}),
            "config": receipt.get("config"), "source_sha256": receipt.get("source_sha256"),
            "bundle_manifest_sha256": receipt.get("bundle_manifest_sha256"),
            "data_sha256": receipt.get("data_sha256"),
        },
        "input_sha256": hashes,
        "old_bank": {"n_quartets": int(len(labels)), "parent_n": int(len(np.unique(parents))),
                     "independent_across_arms": True},
        "arm_metrics": arm_records,
        "archived_CCM_semantics": {
            "definition_in_frozen_outputs": "legacy conditional joint success = J / atomic_joint",
            "definition_in_current_metric_contract": "CCM = conditional composition miss = (atomic_correct AND AB_wrong) / atomic_correct",
            "all_30_archived_CCM_fields_match_legacy_success": bool(all(legacy_checks)),
            "archived_results_modified": False,
        },
        "P0_new_backend_B_minus_A": {"per_seed": p0, "matches_archived_upgrade_analysis": bool(all(p0_checks))},
        "P1_random_standard_compute_or_capacity_fraction": p1,
        "P2_lowstride_interaction": {
            "old_178_quartet_bank_per_seed": p2_old,
            "fresh_small_edit_bank": p2_gate,
            "matches_archived_upgrade_analysis": bool(all(p2_checks)),
        },
        "compute_capacity": {
            "matrix": capacities,
            "W_standard_64": {"parameters": wide_params, "conv_linear_MACs_per_image": wide_standard_64_macs,
                              "mac_count_method": "analytical layer-shape count; no model forward"},
            "B_standard_224": {"parameters": standard_params, "conv_linear_MACs_per_image": standard_224_macs},
            "W_standard_vs_B_standard_parameter_ratio": safe_ratio(wide_params, standard_params),
            "W_standard_vs_B_standard_MAC_ratio": safe_ratio(wide_standard_64_macs, standard_224_macs),
            "base_MAC_counts_match_structure_matrix": True,
            "interpretation": "A-wide exceeds B in MACs but is about 16x larger in parameters; compute and capacity are jointly changed, not separated.",
        },
    }


def audit_old_48_interaction() -> dict:
    root = ROOT / "artifacts/e1a933_review/gpu_finish_20260924/sampling/extracted/sampling_results"
    rows, hashes = [], {}
    means = {}
    for init in ("pretrained", "random"):
        for seed in SEEDS:
            files = {}
            for obs in ("A", "B"):
                for stem in ("standard", "lowstride"):
                    path = root / f"{obs}_{stem}_{init}_flip_s{seed}" / "predictions.npz"
                    files[(obs, stem)] = read_npz(path)
                    hashes[f"{obs}_{stem}_{init}_flip_s{seed}/predictions.npz"] = sha256(path)
            first = files[("A", "standard")]
            for value in files.values():
                assert np.array_equal(first["parent"], value["parent"])
                assert np.array_equal(first["labels"], value["labels"])
            js = {key: (((value["logits"] > 0) == (value["labels"] > 0.5))[:, 1:].all(1)).astype(float)
                  for key, value in files.items()}
            delta = (js[("B", "lowstride")] - js[("A", "lowstride")]) - \
                    (js[("B", "standard")] - js[("A", "standard")])
            ids, by_parent = parent_means(delta, first["parent"])
            ci = parent_bootstrap(delta, first["parent"], seed=962010)
            rows.append({"init": init, "seed": seed, "stratum": "all_178", "n_quartets": int(len(delta)),
                         "parent_n": int(len(ids)), "interaction_parent_equal": float(by_parent.mean()),
                         "parent_cluster_bootstrap95": ci})
            mask = first.get("small_edit", np.zeros(len(delta), dtype=bool)).astype(bool)
            small_ids = np.unique(first["parent"][mask])
            if mask.any():
                small_delta = delta[mask]
                _, small_by_parent = parent_means(small_delta, first["parent"][mask])
                small_ci = parent_bootstrap(small_delta, first["parent"][mask], seed=962010)
                rows.append({"init": init, "seed": seed, "stratum": "historical_small_edit",
                             "n_quartets": int(mask.sum()), "parent_n": int(len(small_ids)),
                             "interaction_parent_equal": float(small_by_parent.mean()),
                             "parent_cluster_bootstrap95": small_ci})
    for init in ("pretrained", "random"):
        means[init] = {
            "all_178_seed_mean_interaction": float(np.mean([
                row["interaction_parent_equal"] for row in rows
                if row["init"] == init and row["stratum"] == "all_178"])),
            "small_edit_seed_mean_interaction": float(np.mean([
                row["interaction_parent_equal"] for row in rows
                if row["init"] == init and row["stratum"] == "historical_small_edit"])),
        }
    return {"result_root": str(root.relative_to(ROOT)), "per_seed": rows,
            "descriptive_seed_means": means, "input_sha256": hashes,
            "warning": "This old 48-arm run differs from the new upgrade in hardware/software and other execution inputs; non-replication cannot identify backend causality."}


def audit_cuda_matrix() -> dict:
    root = ROOT / "artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix"
    receipt_path = root / "receipt.json"
    receipt = read_json(receipt_path)
    stored = read_json(root / "paired_analysis.json")
    assert receipt["status"] == "MATRIX_COMPLETE" and len(receipt["completed"]) == 21
    arrays, records = {}, []
    common = None
    hashes = {"receipt.json": sha256(receipt_path), "paired_analysis.json": sha256(root / "paired_analysis.json"),
              "ARTIFACT_MANIFEST.json": sha256(root / "ARTIFACT_MANIFEST.json")}
    for item in receipt["completed"]:
        arm, seed = item["arm"], int(item["seed"])
        pred_path = root / f"{arm}_s{seed}" / "predictions.npz"
        pred = read_npz(pred_path)
        common = _same_eval_bank(common, pred)
        arrays[(arm, seed)] = pred
        hashes[f"{arm}_s{seed}/predictions.npz"] = sha256(pred_path)
        metrics = score(pred["logits"], pred["labels"])
        metrics["parent_n"] = int(len(np.unique(pred["parent"])))
        metrics["archived_CCM_legacy_success"] = read_json(pred_path.parent / "result.json").get("CCM")
        records.append({"arm": arm, "seed": seed, **metrics})

    o03 = []
    for init in ("pretrained", "random"):
        for seed in SEEDS:
            base_arm, cand_arm = f"{init}_static", f"{init}_flip"
            base, cand = arrays[(base_arm, seed)], arrays[(cand_arm, seed)]
            comparison = paired_summary(base, cand, n_boot=2000, seed=951000 + seed, row_boot=True)
            key = f"{cand_arm} minus {base_arm}"
            frozen = stored[key]["seeds"][SEEDS.index(seed)]
            assert np.isclose(comparison["delta_J_quartet_weighted"], frozen["J_difference"], atol=1e-12)
            assert np.allclose(comparison["parent_cluster_bootstrap95"], frozen["parent_bootstrap95"], atol=1e-12)
            o03.append({"init": init, "seed": seed, **comparison,
                        "matches_archived_paired_analysis": True})

    n01 = []
    for seed in SEEDS:
        baseline = arrays[("pretrained_flip", seed)]
        for cand_arm in ("ordered", "matched", "shuffled_matched"):
            cand = arrays[(cand_arm, seed)]
            n01.append({"candidate": cand_arm, "baseline": "pretrained_flip_BCE_only", "seed": seed,
                        **paired_summary(baseline, cand, n_boot=2000, seed=951000 + seed, row_boot=True)})

    aux_mse, aux_deltas = [], []
    aux_root = root
    aux_arms = ("pretrained_flip", "ordered", "matched", "shuffled_matched")
    for seed in SEEDS:
        errors = {}
        dev_parent = None
        for arm in aux_arms:
            path = aux_root / f"{arm}_s{seed}" / "dev_aux_predictions.npz"
            aux = read_npz(path)
            hashes[f"{arm}_s{seed}/dev_aux_predictions.npz"] = sha256(path)
            prediction, target = aux["prediction"], aux["target_orbit"]
            ordered_error = ((prediction - target[:, 0]) ** 2).mean(-1)
            matched_error = ((prediction[:, None, :] - target) ** 2).mean(-1).min(-1)
            assert np.allclose(ordered_error, aux["ordered_error"], atol=1e-7)
            assert np.allclose(matched_error, aux["matched_error"], atol=1e-7)
            if dev_parent is None:
                dev_parent = aux["parent"]
            else:
                assert np.array_equal(dev_parent, aux["parent"])
            errors[arm] = matched_error
            _, parent_means_err = parent_means(matched_error, aux["parent"])
            aux_mse.append({"arm": arm, "seed": seed, "n_dev_images": int(len(prediction)),
                            "dev_parent_n": int(len(np.unique(aux["parent"]))),
                            "selected_checkpoint_matched_mse_row_weighted": float(matched_error.mean()),
                            "selected_checkpoint_matched_mse_parent_equal": float(parent_means_err.mean())})
        for other in ("ordered", "pretrained_flip", "shuffled_matched"):
            delta = errors["matched"] - errors[other]
            ids, by_parent = parent_means(delta, dev_parent)
            ci = parent_bootstrap(delta, dev_parent, seed=962500 + seed)
            aux_deltas.append({"candidate": "matched", "baseline": other, "seed": seed,
                               "n_dev_images": int(len(delta)), "dev_parent_n": int(len(ids)),
                               "matched_mse_difference_parent_equal": float(by_parent.mean()),
                               "parent_cluster_bootstrap95": ci})

    return {
        "receipt": {"status": receipt["status"], "completed_runs": len(receipt["completed"]),
                    "source_sha256": receipt.get("source_sha256"),
                    "bundle_manifest_sha256": receipt.get("bundle_manifest_sha256"),
                    "data_sha256": receipt.get("data_sha256")},
        "input_sha256": hashes,
        "test_bank": {"n_quartets": int(len(common["labels"])),
                      "parent_n": int(len(np.unique(common["parent"]))),
                      "shared_by_O03_N01": True},
        "arm_metrics": records,
        "O03_static_to_flip": {
            "per_seed": o03,
            "mean_delta_pretrained_quartet_weighted": float(np.mean([r["delta_J_quartet_weighted"] for r in o03 if r["init"] == "pretrained"])),
            "mean_delta_random_quartet_weighted": float(np.mean([r["delta_J_quartet_weighted"] for r in o03 if r["init"] == "random"])),
            "point_estimate_weighting": "quartet_weighted; 95% interval resamples parents retaining all their quartets",
        },
        "N01_J_contrasts": n01,
        "N01_selected_checkpoint_auxiliary_error": {"per_arm_seed": aux_mse, "matched_minus_baselines": aux_deltas,
            "interpretation": "matched target has lower saved dev geometry error; this learnability evidence does not establish additional J repair."},
    }


def audit_n03() -> dict:
    root = ROOT / "artifacts/e1a933_review/gpu_finish_20260924/n03/extracted/n03_results"
    receipt_path = root / "receipt.json"
    receipt = read_json(receipt_path)
    stored = read_json(root / "paired_analysis.json")
    assert receipt["status"] == "MATRIX_COMPLETE" and len(receipt["completed"]) == 6
    arrays, arms, hashes = {}, [], {"receipt.json": sha256(receipt_path),
                                     "paired_analysis.json": sha256(root / "paired_analysis.json"),
                                     "ARTIFACT_MANIFEST.json": sha256(root / "ARTIFACT_MANIFEST.json")}
    common = None
    for item in receipt["completed"]:
        arm, seed = item["arm"], int(item["seed"])
        path = root / f"{arm}_s{seed}" / "predictions.npz"
        pred = read_npz(path); common = _same_eval_bank(common, pred)
        arrays[(arm, seed)] = pred; hashes[f"{arm}_s{seed}/predictions.npz"] = sha256(path)
        arms.append({"arm": arm, "seed": seed, **score(pred["logits"], pred["labels"]),
                     "parent_n": int(len(np.unique(pred["parent"])))})
        oracle_path = path.parent / "oracle_replacement_DIAGNOSTIC.npz"
        oracle = read_npz(oracle_path); hashes[f"{arm}_s{seed}/oracle_replacement_DIAGNOSTIC.npz"] = sha256(oracle_path)
        assert np.array_equal(pred["labels"], oracle["labels"])
    pairwise = []
    for seed in SEEDS:
        geometry = arrays[("geometry_bottleneck", seed)]
        generic = arrays[("generic_bottleneck", seed)]
        pairwise.append({"candidate": "geometry_bottleneck", "baseline": "generic_bottleneck", "seed": seed,
                         **paired_summary(generic, geometry, n_boot=2000, seed=951000 + seed, row_boot=True)})
    analytic_root = ROOT / "artifacts/e1a933_review/vision_n03_analytic"
    analytic_path = analytic_root / "predictions.npz"
    analytic = read_npz(analytic_path)
    hashes["vision_n03_analytic/predictions.npz"] = sha256(analytic_path)
    analytic_score = score(analytic["logits"], analytic["labels"])
    analytic_score.update({"parent_n": int(len(np.unique(analytic["parent"]))),
                           "invalid_state_n": int((~analytic["valid_fit"]).sum()),
                           "scope": "known colors and renderer stroke rule; fixed negative for invalid fits"})
    oracle_scores = []
    for item in receipt["completed"]:
        arm, seed = item["arm"], int(item["seed"])
        path = root / f"{arm}_s{seed}" / "oracle_replacement_DIAGNOSTIC.npz"
        pred = read_npz(path)
        oracle_scores.append({"arm": arm, "seed": seed, **score(pred["logits"], pred["labels"]),
                              "scope": "true geometry through frozen learned head; diagnostic only"})
    return {
        "receipt": {"status": receipt["status"], "completed_runs": len(receipt["completed"]),
                    "source_sha256": receipt.get("source_sha256"), "data_sha256": receipt.get("data_sha256")},
        "input_sha256": hashes,
        "test_bank": {"n_quartets": int(len(common["labels"])), "parent_n": int(len(np.unique(common["parent"]))),
                      "same_as_O03_N01": True},
        "arm_metrics": arms,
        "geometry_minus_generic_parent_cluster_contrasts": pairwise,
        "stored_paired_analysis_keys": list(stored),
        "oracle_replacement_diagnostics": oracle_scores,
        "classical_pixel_baseline": analytic_score,
    }


def audit_vlm() -> dict:
    data_root = ROOT / "artifacts/e1a933_review/vlm_frozen_272"
    result_root = ROOT / "artifacts/e1a933_review/gpu_finish_20260924/vlm/extracted/vlm_results"
    manifest_path = data_root / "manifest.json"
    receipt_path = result_root / "receipt.json"
    run_path = result_root / "run_manifest.json"
    response_path = result_root / "responses.jsonl"
    manifest, receipt, run = read_json(manifest_path), read_json(receipt_path), read_json(run_path)
    responses = [json.loads(line) for line in response_path.read_text().splitlines() if line.strip()]
    assert receipt["status"] == "MATRIX_COMPLETE" and len(receipt["completed"]) == 272
    assert receipt["expected"] == 272 and len(responses) == 272
    assert len({row["id"] for row in responses}) == 272
    ids = {row["id"] for row in responses}
    assert ids == {row["id"] for row in manifest["records"]}
    image_checks = []
    for record in manifest["records"]:
        image_path = data_root / record["image"]
        image_checks.append(sha256(image_path) == record["sha256"])
    assert all(image_checks)
    by_parent, sanity = {}, []
    for row in responses:
        if row["kind"] == "sanity":
            sanity.append(row)
        else:
            by_parent.setdefault(row["parent"], {})[row["state"]] = row
    quartets = [row for row in by_parent.values() if set(row) == {"base", "A", "B", "AB"}]
    assert len(quartets) == 64 and all(len(row) == 4 for row in by_parent.values())
    states = ("base", "A", "B", "AB")
    correct = np.asarray([[q[state]["pred"] == q[state]["label"] for state in states] for q in quartets], bool)
    atom = correct[:, 1] & correct[:, 2]
    joint = atom & correct[:, 3]
    full = correct.all(1)
    parse_failure = sum(row["parse_status"] == "parse_failure" for row in responses)
    refusal = sum(row["parse_status"] == "refusal" for row in responses)
    conditional_model_error = int((atom & ~correct[:, 3] & np.asarray([
        q["AB"]["parse_status"] == "parsed" for q in quartets], bool)).sum())
    conditional_nonresponse = int((atom & ~correct[:, 3]).sum()) - conditional_model_error
    rng = np.random.default_rng(26092406)
    draws = rng.integers(0, len(quartets), (2000, len(quartets)))
    return {
        "receipt": {"status": receipt["status"], "completed_requests": len(responses), "expected": receipt["expected"]},
        "input_sha256": {"manifest.json": sha256(manifest_path), "receipt.json": sha256(receipt_path),
                         "run_manifest.json": sha256(run_path), "responses.jsonl": sha256(response_path)},
        "image_sha256_checks_passed": int(sum(image_checks)),
        "signature": {key: run.get(key) for key in ("model", "revision", "torch", "transformers", "prompt", "max_new_tokens", "image_size", "do_sample", "dtype")},
        "denominators": {"individual_image_requests": len(responses), "quartet_rows": len(quartets),
                         "quartet_parent_n": len(by_parent), "sanity_images": len(sanity)},
        "quartet_metrics": {
            "atomic_A_accuracy": float(correct[:, 1].mean()),
            "atomic_B_accuracy": float(correct[:, 2].mean()),
            "atomic_both_correct_n": int(atom.sum()),
            "J3": float(joint.mean()), "J4": float(full.mean()),
            "conditional_composition_miss": safe_ratio(int((atom & ~correct[:, 3]).sum()), int(atom.sum())),
            "conditional_AB_model_error_n": conditional_model_error,
            "conditional_AB_nonresponse_n": conditional_nonresponse,
            "J_parent_bootstrap95": np.quantile(joint[draws].mean(1), [0.025, 0.975]).tolist(),
            "parse_failure_requests": int(parse_failure), "refusal_requests": int(refusal),
        },
        "sanity": {"n": len(sanity), "accuracy": float(np.mean([row["pred"] == row["label"] for row in sanity])),
                   "parse_failure": sum(row["parse_status"] == "parse_failure" for row in sanity),
                   "refusal": sum(row["parse_status"] == "refusal" for row in sanity)},
        "scope": "one frozen 3B VLM on a reused synthetic bank; 272 means image requests, not independent quartet replications; atomic competence is low, so the conditional miss cannot establish composition-specific failure or an all-VLM claim.",
    }


def build_report(out_path: Path) -> dict:
    result = {
        "schema": "iclr2027_submission_audit_vision_recompute_v1",
        "method": "NumPy-only deterministic rescoring of saved prediction arrays, raw VLM responses, receipts, and manifests. No training, model inference, external download, or sealed holdout access.",
        "script_sha256": sha256(Path(__file__).resolve()),
        "n02_upgrade": audit_upgrade(),
        "historical_48_arm_N02": audit_old_48_interaction(),
        "O03_N01": audit_cuda_matrix(),
        "N03": audit_n03(),
        "O06": audit_vlm(),
        "NOT_RUN": [
            "No new GPU training, model inference, data download, or provider/API call.",
            "No sealed holdout/confirmation data opened.",
            "No fresh-bank replication for O03, N01, N03, or O06.",
            "No transfer of the N03 geometry bottleneck to football/tracking was run.",
            "No pure-compute effect was isolated from capacity in N02 P1.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts/submission_audit_20260925/vision/recomputation.json")
    args = parser.parse_args()
    result = build_report(args.out)
    print(json.dumps({"out": str(args.out), "schema": result["schema"],
                      "P1_mean_f": result["n02_upgrade"]["P1_random_standard_compute_or_capacity_fraction"]["descriptive_arithmetic_mean_of_seed_f"],
                      "O06_conditional_miss": result["O06"]["quartet_metrics"]["conditional_composition_miss"]}, indent=2))


if __name__ == "__main__":
    main()
