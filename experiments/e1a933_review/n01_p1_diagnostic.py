#!/usr/bin/env python3
"""N01 P1-style diagnostic on the frozen 21-arm vision matrix (no retraining, CPU only).

Repairs applied here (R04 contract, see docs/e1a933_review/01_RESULT_CRITICAL_REPAIRS.md):
the retention set ("high confidence") is defined from the *start* state logit
``lg0`` through the parent map, never from the endpoint logit.  Unconditional
endpoint risk and start-correct-conditioned update risk use separate
denominators and are never pooled.  The threshold is fixed on the declared
development reference set (dev single states), and the realised coverage is
reported instead of being labelled "top 25%".

Task geometry (frozen, from experiments/e1a933_review/vision_protocol.generate):
each test quartet is [x, x+ea, x+eb, x+ea+eb] with labels [y0, y0, y0, 1-y0]
(mine_quartets keeps both atomics label-preserving and the combo label-flipping).
So the P1 "keep" states are quartet columns 1 and 2, and the "flip" state is
column 3; the start state is column 0, whose logit is the only legitimate
confidence source.

Inputs are read-only.  The only new computation is a forward pass of the
recovered, hash-verified checkpoints over the dev bank to fix the threshold.
No training, no GPU, no test-bank-dependent threshold.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ARMS = ["pretrained_flip", "ordered", "matched", "shuffled_matched"]
ARM_LABEL = {
    "pretrained_flip": "BCE-only",
    "ordered": "ordered rel10",
    "matched": "matched rel10",
    "shuffled_matched": "shuffled-matched",
}
SEEDS = [803, 805, 806]
PAIRS = [
    ("matched", "ordered"),
    ("matched", "pretrained_flip"),
    ("matched", "shuffled_matched"),
    ("ordered", "pretrained_flip"),
    ("shuffled_matched", "pretrained_flip"),
]
# Threshold quantiles on the declared dev reference set.  top-tertile is the
# convention inherited from the D03/R04 repair; the others are sensitivity.
THRESHOLD_QUANTILES = {"dev_top_tertile": 2 / 3, "dev_median": 1 / 2, "dev_top_quartile": 3 / 4}

KEEP = np.array([False, True, True, False])   # atomics preserve the label
FLIP = ~KEEP                                   # combo flips the label


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parent_bootstrap(delta_rows, parent_of_row, parents, n_boot=2000, seed=971000):
    """Parent-clustered bootstrap CI for a per-row paired delta (NaN = not in subset)."""
    rng = np.random.default_rng(seed)
    idx_by_parent = {p: np.flatnonzero(parent_of_row == p) for p in parents}
    boots = np.empty(n_boot, float)
    for b in range(n_boot):
        sampled = rng.choice(parents, len(parents), replace=True)
        ix = np.concatenate([idx_by_parent[p] for p in sampled])
        vals = delta_rows[ix]
        with np.errstate(invalid="ignore"):
            boots[b] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else np.nan
    return [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))]


def extract_checkpoints(archive: Path, cache: Path, manifest: dict, wanted):
    """Recover model.pt bytes from the collected archive, verified against manifest."""
    cache.mkdir(parents=True, exist_ok=True)
    pending = {}
    for arm, seed in wanted:
        rel = f"{arm}_s{seed}/model.pt"
        dst = cache / f"{arm}_s{seed}" / "model.pt"
        if dst.exists() and sha256_file(dst) == manifest[rel]:
            continue
        pending[f"cuda_matrix/{rel}"] = (dst, manifest[rel])
    if pending:
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf:
                if member.name in pending:
                    dst, expected = pending[member.name]
                    data = tf.extractfile(member).read()
                    assert hashlib.sha256(data).hexdigest() == expected, \
                        f"archive member hash mismatch: {member.name}"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(data)
                    del pending[member.name]
        assert not pending, f"missing archive members: {sorted(pending)}"


def start_confidence_retention(start_logit_by_parent, row_parents, threshold):
    """R04 retention rule: the unedited *start* score decides, shared across a parent's edits.

    Same semantics as ``d03_eval.start_confidence_mask``; endpoint scores never enter.
    """
    return np.array([abs(start_logit_by_parent[p]) >= threshold for p in row_parents])


def p1_metrics(logits, labels, parent, retained):
    """P1 decomposition for vision quartets.

    Quartet columns are [start, editA, editB, editAB] with labels [y0, y0, y0, 1-y0],
    so keep = columns 1-2 and flip = column 3.  ``retained`` is per-quartet and comes
    from the start logit only.  D1 (all states), D2 (retained, unconditional on start
    correctness) and D3 (retained AND start-correct) are reported separately.
    """
    parents = np.unique(parent)
    lg0 = logits[:, 0]
    ok0 = (lg0 > 0) == labels[:, 0]
    err = (logits > 0) != labels
    atomic_ok = ((logits[:, 1] > 0) == labels[:, 1]) & ((logits[:, 2] > 0) == labels[:, 2])
    combo_ok = (logits[:, 3] > 0) == labels[:, 3]
    joint_ok = atomic_ok & combo_ok
    keep_err = err[:, 1:3]
    out = {
        "coverage_quartet": float(retained.mean()),
        "coverage_parent": float(len({p for p, r in zip(parent, retained) if r}) / len(parents)),
        "n_quartets": int(len(parent)), "n_parents": int(len(parents)),
        "start_acc_quartet": float(ok0.mean()),
        "keep_err_all": float(keep_err.mean()),
        "flip_err_all": float(err[:, 3].mean()),
        "atomic_joint_all": float(atomic_ok.mean()),
        "combo_acc_all": float(combo_ok.mean()),
        "J_all": float(joint_ok.mean()),
        "CCM_denominator": int(atomic_ok.sum()),
        "CCM": float(joint_ok.sum() / atomic_ok.sum()) if atomic_ok.sum() else None,
        # denominator D2: retained quartets, NOT conditioned on start correctness
        "keep_err_retained": float(keep_err[retained].mean()) if retained.any() else None,
        "flip_err_retained": float(err[:, 3][retained].mean()) if retained.any() else None,
        "keep_n_retained": int(retained.sum() * 2), "flip_n_retained": int(retained.sum()),
        # denominator D3: retained AND start-correct
        "keep_err_retained_startok": float(keep_err[retained & ok0].mean()) if (retained & ok0).any() else None,
        "flip_err_retained_startok": float(err[:, 3][retained & ok0].mean()) if (retained & ok0).any() else None,
        "keep_n_retained_startok": int((retained & ok0).sum() * 2),
        "flip_n_retained_startok": int((retained & ok0).sum()),
        "n_quartets_retained_startok": int((retained & ok0).sum()),
        # denominator D1 restricted to start-correct (no confidence filter)
        "keep_err_startok_all": float(keep_err[ok0].mean()) if ok0.any() else None,
        "flip_err_startok_all": float(err[:, 3][ok0].mean()) if ok0.any() else None,
        "keep_n_startok_all": int(ok0.sum() * 2), "flip_n_startok_all": int(ok0.sum()),
        "J_retained": float(joint_ok[retained].mean()) if retained.any() else None,
        "baseline110_n_retained": int((atomic_ok & ~combo_ok & retained).sum()),
        "full_repair_n_retained": int((atomic_ok & combo_ok & retained).sum()),
        "baseline111_n_retained": int((joint_ok & retained).sum()),
    }
    return out, retained, ok0, joint_ok, atomic_ok, combo_ok, err


def write_csv(path: Path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path,
                    default=ROOT / "artifacts/e1a933_review/vision_gpu_interpretation/cuda_matrix")
    ap.add_argument("--archive", type=Path,
                    default=ROOT / "artifacts/e1a933_review/remote_vision_20260924_ssh30891/"
                            "collected/cuda_matrix_results.tar.gz")
    ap.add_argument("--bundle", type=Path, default=ROOT / "artifacts/e1a933_review/vision_cuda_bundle")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/e1a933_review/vision_n01_p1")
    ap.add_argument("--ckpt-cache", type=Path, default=Path("/tmp/e1a933_n01p1_ckpt"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.out.exists() and not a.force:
        raise FileExistsError(f"use a fresh --out-dir or pass --force: {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "dev_logits").mkdir(exist_ok=True)

    os.environ.setdefault("TORCH_HOME", str(a.bundle.resolve() / "torch_cache"))
    import torch
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(a.bundle.resolve()))
    import vision_cuda_protocol as protocol

    provenance = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "git_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "command": sys.argv,
        "threads": a.threads,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "inputs": {},
    }

    # ---- input integrity -------------------------------------------------
    bundle_manifest = json.loads((a.bundle / "manifest.json").read_text())
    bundle_hashes = {e["path"]: e["sha256"] for e in bundle_manifest["files"]}
    data_path = a.bundle / "data.npz"
    assert sha256_file(data_path) == bundle_hashes["data.npz"], "bundle data.npz hash mismatch"
    provenance["inputs"]["bundle_data_npz"] = {
        "path": str(data_path.relative_to(ROOT)), "sha256": bundle_hashes["data.npz"]}
    provenance["inputs"]["archive"] = {
        "path": str(a.archive.relative_to(ROOT)), "sha256": sha256_file(a.archive)}
    provenance["inputs"]["protocol_sha256"] = sha256_file(a.bundle / "vision_cuda_protocol.py")

    matrix_manifest = json.loads((a.matrix / "ARTIFACT_MANIFEST.json").read_text())
    manifest = {e["path"]: e["sha256"] for e in matrix_manifest}
    provenance["inputs"]["matrix_manifest_sha256"] = sha256_file(a.matrix / "ARTIFACT_MANIFEST.json")

    wanted = [(arm, seed) for arm in ARMS for seed in SEEDS]
    extract_checkpoints(a.archive, a.ckpt_cache, manifest, wanted)

    data = dict(np.load(data_path, allow_pickle=True))
    dev_clean = data["dev_clean"].astype(bool)
    provenance["dev_reference"] = {
        "split": "dev", "n_states": int(len(dev_clean)), "n_clean_states": int(dev_clean.sum()),
        "n_parents": int(len(np.unique(data["dev_parents"]))),
        "rule": "per-model |logit| quantile over dev clean (unedited) states",
    }

    # ---- per-run artifacts ----------------------------------------------
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            rp = a.matrix / f"{arm}_s{seed}"
            ckpt = a.ckpt_cache / f"{arm}_s{seed}" / "model.pt"
            ck_hash = sha256_file(ckpt)
            result = json.loads((rp / "result.json").read_text())
            assert ck_hash == manifest[f"{arm}_s{seed}/model.pt"] == result["checkpoint_sha256"]
            assert sha256_file(rp / "predictions.npz") == manifest[f"{arm}_s{seed}/predictions.npz"]
            q = np.load(rp / "predictions.npz")
            logits, labels, parent = q["logits"], q["labels"], q["parent"]

            # start-state geometry must hold before any confidence is used
            assert np.array_equal(labels[:, 0], labels[:, 1])
            assert np.array_equal(labels[:, 0], labels[:, 2])
            assert np.all(labels[:, 3] != labels[:, 0]), "column 3 must be the flipping combo"
            lg0 = logits[:, 0]
            parents = np.unique(parent)
            spread = max(float(np.abs(lg0[parent == p] - lg0[parent == p][0]).max()) for p in parents)
            assert spread == 0.0, f"start logit not shared within parent: {spread}"

            # dev threshold, fixed before any test-bank number is touched
            dev_lg_path = a.out / "dev_logits" / f"{arm}_s{seed}.npz"
            if dev_lg_path.exists():
                dev = np.load(dev_lg_path)
                dev_lg = dev["logits"]
                assert dev["checkpoint_sha256"].item() == ck_hash
            else:
                net = protocol.DistillNet(10)
                net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state"])
                t0 = time.monotonic()
                dev_lg = protocol.logits(net, data["dev_images"], res=224, batch=32)
                np.savez_compressed(dev_lg_path, logits=dev_lg, clean=dev_clean,
                                    labels=data["dev_labels"], parents=data["dev_parents"],
                                    checkpoint_sha256=np.array(ck_hash),
                                    seconds=np.array(time.monotonic() - t0))
                del net
            assert np.array_equal(np.load(dev_lg_path)["clean"], dev_clean)
            thresholds = {k: float(np.quantile(np.abs(dev_lg[dev_clean]), qq))
                          for k, qq in THRESHOLD_QUANTILES.items()}

            runs[(arm, seed)] = dict(
                arm=arm, seed=seed, logits=logits, labels=labels, parent=parent, lg0=lg0,
                parents=parents, thresholds=thresholds, checkpoint_sha256=ck_hash,
                result=result, test_single=np.load(rp / "test_single_predictions.npz"),
            )
        print(f"[dev thresholds fixed] {arm}", flush=True)

    # ---- P1 metrics ------------------------------------------------------
    def p1_block(run, thr_key):
        thr = run["thresholds"][thr_key]
        parent, lg0 = run["parent"], run["lg0"]
        logits, labels, parents = run["logits"], run["labels"], run["parents"]
        start_logit_by_parent = {p: float(lg0[parent == p][0]) for p in parents}
        retained = start_confidence_retention(start_logit_by_parent, parent, thr)
        out, retained, ok0, joint_ok, atomic_ok, combo_ok, err = p1_metrics(logits, labels, parent, retained)
        out["threshold"] = thr
        # diagnostic only: retention by each state's own endpoint |logit| (the R04 bug shape)
        for kind, sel in (("keep", KEEP), ("flip", FLIP)):
            e = np.abs(logits[:, sel]) >= thr
            out[f"endpoint_conf_coverage_{kind}"] = float(e.mean())
            out[f"endpoint_conf_err_{kind}"] = float(err[:, sel][e].mean()) if e.any() else None
            out[f"endpoint_conf_n_{kind}"] = int(e.sum())
        return out, retained, ok0, joint_ok, atomic_ok, combo_ok, err

    rows, detail = [], {}
    for (arm, seed), run in runs.items():
        primary, retained, ok0, joint_ok, atomic_ok, combo_ok, err = p1_block(run, "dev_top_tertile")
        row = {"arm": arm, "arm_label": ARM_LABEL[arm], "seed": seed,
               "checkpoint_sha256": run["checkpoint_sha256"], "best_epoch": run["result"]["best_epoch"],
               "J_all": primary["J_all"], "P_start": primary["start_acc_quartet"],
               "atomic_joint_all": primary["atomic_joint_all"], "AB_acc_all": primary["combo_acc_all"],
               "CCM": primary["CCM"], "CCM_denominator": primary["CCM_denominator"],
               "n_quartets": primary["n_quartets"], "n_parents": primary["n_parents"],
               "dev_thr_top_tertile": run["thresholds"]["dev_top_tertile"],
               "dev_thr_median": run["thresholds"]["dev_median"],
               "dev_thr_top_quartile": run["thresholds"]["dev_top_quartile"]}
        for key in ["coverage_quartet", "coverage_parent", "keep_err_all", "flip_err_all",
                    "keep_err_retained", "flip_err_retained", "keep_n_retained", "flip_n_retained",
                    "keep_err_retained_startok", "flip_err_retained_startok",
                    "keep_n_retained_startok", "flip_n_retained_startok", "n_quartets_retained_startok",
                    "keep_err_startok_all", "flip_err_startok_all", "keep_n_startok_all", "flip_n_startok_all",
                    "J_retained", "baseline110_n_retained", "full_repair_n_retained", "baseline111_n_retained",
                    "endpoint_conf_coverage_keep", "endpoint_conf_coverage_flip",
                    "endpoint_conf_err_keep", "endpoint_conf_err_flip"]:
            row[key] = primary[key]
        for tk in ("dev_median", "dev_top_quartile"):
            alt = p1_block(run, tk)[0]
            row[f"cov_{tk}"] = alt["coverage_quartet"]
            row[f"keep_err_retained_{tk}"] = alt["keep_err_retained"]
            row[f"flip_err_retained_{tk}"] = alt["flip_err_retained"]
        rows.append(row)
        detail[(arm, seed)] = {"retained": retained, "ok0": ok0, "joint_ok": joint_ok,
                               "atomic_ok": atomic_ok, "combo_ok": combo_ok, "err": err}

    # ---- paired contrasts ------------------------------------------------
    paired_rows = []
    reference = json.loads((a.matrix / "paired_analysis.json").read_text())
    for cand, base in PAIRS:
        for seed in SEEDS:
            rc, rb = runs[(cand, seed)], runs[(base, seed)]
            assert np.array_equal(rc["labels"], rb["labels"]) and np.array_equal(rc["parent"], rb["parent"])
            dc, db = detail[(cand, seed)], detail[(base, seed)]
            parent, parents = rc["parent"], rc["parents"]
            # regression check: reproduce the archived paired analysis exactly
            ao = (rb["logits"] > 0) == rb["labels"]
            bo = (rc["logits"] > 0) == rc["labels"]
            aj, bj = ao[:, 1:].all(1), bo[:, 1:].all(1)
            h = ao[:, 1] & ao[:, 2] & ~ao[:, 3]
            if f"{cand} minus {base}" in reference:
                arch = [r for r in reference[f"{cand} minus {base}"]["seeds"] if r["seed"] == seed][0]
                assert abs(float((bj.astype(float) - aj.astype(float)).mean()) - arch["J_difference"]) < 1e-12
                assert int(h.sum()) == arch["baseline110_n"] and int((bj & h).sum()) == arch["full_repair_n"]
                assert int((bo[:, 3] & ~(bo[:, 1] & bo[:, 2]) & h).sum()) == arch["migration_n"]
                assert int((aj & ~bj).sum()) == arch["111_regression_n"]
            common = dc["retained"] & db["retained"]
            common_ok = common & dc["ok0"]          # same states for both arms (start is arm-independent)
            d_joint = dc["joint_ok"].astype(float) - db["joint_ok"].astype(float)
            d_keep = (dc["err"][:, 1:3].astype(float) - db["err"][:, 1:3].astype(float)).ravel()
            d_flip = dc["err"][:, 3].astype(float) - db["err"][:, 3].astype(float)
            keep_parent = np.repeat(parent, 2)
            paired_rows.append({
                "candidate": cand, "baseline": base, "seed": seed,
                "J_diff_all": float(d_joint.mean()),
                "J_diff_all_parent_boot95": parent_bootstrap(d_joint, parent, parents, seed=971000 + seed),
                "n_common_retained_quartets": int(common.sum()),
                "n_common_retained_parents": int(len(np.unique(parent[common]))),
                "J_diff_common_retained": float(d_joint[common].mean()) if common.any() else None,
                "J_diff_common_retained_boot95": parent_bootstrap(
                    np.where(common, d_joint, np.nan), parent, parents, seed=972000 + seed),
                "keep_err_diff_common_retained": float(d_keep[np.repeat(common, 2)].mean()) if common.any() else None,
                "keep_err_diff_common_retained_boot95": parent_bootstrap(
                    np.where(np.repeat(common, 2), d_keep, np.nan), keep_parent, parents, seed=973000 + seed),
                "flip_err_diff_common_retained": float(d_flip[common].mean()) if common.any() else None,
                "flip_err_diff_common_retained_boot95": parent_bootstrap(
                    np.where(common, d_flip, np.nan), parent, parents, seed=974000 + seed),
                "keep_err_diff_common_retained_startok":
                    float(d_keep[np.repeat(common_ok, 2)].mean()) if common_ok.any() else None,
                "keep_err_diff_common_retained_startok_boot95": parent_bootstrap(
                    np.where(np.repeat(common_ok, 2), d_keep, np.nan), keep_parent, parents, seed=975000 + seed),
                "flip_err_diff_common_retained_startok":
                    float(d_flip[common_ok].mean()) if common_ok.any() else None,
                "flip_err_diff_common_retained_startok_boot95": parent_bootstrap(
                    np.where(common_ok, d_flip, np.nan), parent, parents, seed=976000 + seed),
                "baseline110_n": int(h.sum()), "full_repair_n": int((bj & h).sum()),
                "migration_n": int((bo[:, 3] & ~(bo[:, 1] & bo[:, 2]) & h).sum()),
                "baseline111_n": int(aj.sum()), "111_regression_n": int((aj & ~bj).sum()),
            })

    # ---- outputs ---------------------------------------------------------
    write_csv(a.out / "n01_p1_per_arm_seed.csv", rows)
    write_csv(a.out / "n01_p1_paired_contrasts.csv", paired_rows)

    # single combined table: per-arm P1 levels and pair-level repair/migration in one file
    combined = []
    for r in rows:
        combined.append({
            "row_type": "arm", "arm": r["arm"], "arm_label": r["arm_label"], "baseline": "",
            "seed": r["seed"], "n_quartets": r["n_quartets"], "n_parents": r["n_parents"],
            "J": r["J_all"], "P_start": r["P_start"], "coverage_quartet": r["coverage_quartet"],
            "keep_err_all": r["keep_err_all"], "flip_err_all": r["flip_err_all"],
            "keep_err_retained": r["keep_err_retained"], "flip_err_retained": r["flip_err_retained"],
            "keep_n_retained": r["keep_n_retained"], "flip_n_retained": r["flip_n_retained"],
            "keep_err_retained_startok": r["keep_err_retained_startok"],
            "flip_err_retained_startok": r["flip_err_retained_startok"],
            "J_retained": r["J_retained"], "baseline110_n_retained": r["baseline110_n_retained"],
            "full_repair_n_retained": r["full_repair_n_retained"],
        })
    for r in paired_rows:
        combined.append({
            "row_type": "pair", "arm": r["candidate"], "arm_label": ARM_LABEL[r["candidate"]],
            "baseline": r["baseline"], "seed": r["seed"],
            "J_diff_all": r["J_diff_all"],
            "J_diff_all_parent_boot95": r["J_diff_all_parent_boot95"],
            "n_common_retained_quartets": r["n_common_retained_quartets"],
            "J_diff_common_retained": r["J_diff_common_retained"],
            "J_diff_common_retained_boot95": r["J_diff_common_retained_boot95"],
            "keep_err_diff_common_retained": r["keep_err_diff_common_retained"],
            "keep_err_diff_common_retained_boot95": r["keep_err_diff_common_retained_boot95"],
            "flip_err_diff_common_retained": r["flip_err_diff_common_retained"],
            "flip_err_diff_common_retained_boot95": r["flip_err_diff_common_retained_boot95"],
            "baseline110_n": r["baseline110_n"], "full_repair_n": r["full_repair_n"],
            "migration_n": r["migration_n"], "baseline111_n": r["baseline111_n"],
            "111_regression_n": r["111_regression_n"],
        })
    write_csv(a.out / "n01_p1_table_combined.csv", combined)

    denom = []
    for (arm, seed), run in runs.items():
        retained, ok0 = detail[(arm, seed)]["retained"], detail[(arm, seed)]["ok0"]
        err = detail[(arm, seed)]["err"]
        for kind, sel in (("keep", KEEP), ("flip", FLIP)):
            for label, mask, note in (
                ("all_states", np.ones_like(retained), "无保留筛选；全部编辑状态"),
                ("retained", retained, "仅起点高置信（不要求 start-correct）"),
                ("retained_startcorrect", retained & ok0, "起点高置信且起点判对（统一 start-correct）"),
            ):
                state_mask = np.repeat(mask, 2).reshape(-1, 2) if kind == "keep" else mask
                n = int(state_mask.sum())
                denom.append({
                    "arm": arm, "arm_label": ARM_LABEL[arm], "seed": seed, "state_kind": kind,
                    "denominator": label, "note": note, "n_states": n, "n_quartets": int(mask.sum()),
                    "error_rate": float(err[:, sel][state_mask].mean()) if n else None,
                    "retention_coverage_states": float(state_mask.mean()),
                })
    write_csv(a.out / "n01_p1_denominator_table.csv", denom)

    stability = []
    for (arm, seed), run in runs.items():
        ts = run["test_single"]
        clean, tp, tlg = ts["clean"].astype(bool), ts["parent"], ts["logits"]
        q_parent, q_lg0 = run["parent"], run["lg0"]
        qv = np.array([float(q_lg0[q_parent == p][0]) for p in run["parents"]])
        sv = np.array([float(tlg[clean & (tp == p)].mean()) for p in run["parents"]])
        stability.append({"arm": arm, "seed": seed, "n_parents": len(qv),
                          "pearson_r_quartet_start_vs_single_clean_mean": float(np.corrcoef(qv, sv)[0, 1]),
                          "sign_agreement": float(((qv > 0) == (sv > 0)).mean())})
    write_csv(a.out / "n01_p1_start_confidence_stability.csv", stability)

    summary = {
        "status": "P1_DIAGNOSTIC_COMPLETE_ZERO_RETRAIN_CPU_ONLY",
        "scientific_question": "matched/ordered/shuffled/BCE 四臂在起点置信条件下的 preserve/flip 风险是否有额外差异",
        "primary_threshold_rule": "per-model |lg0| >= dev-clean top-tertile (2/3 quantile)",
        "provenance": provenance,
        "arms": ARMS, "arm_labels": ARM_LABEL, "seeds": SEEDS,
        "per_arm_seed": rows, "paired_contrasts": paired_rows,
        "denominator_table": denom, "start_confidence_stability": stability,
        "notes": [
            "起点置信取 quartet 第0列（未编辑父场景渲染）的 logit；同一 parent 的多个 quartet 起点 logit 相同（脚本内断言 spread==0）。",
            "keep = quartet 第1/2列（原子编辑保持标签），flip = 第3列（组合编辑翻转标签）。",
            "无条件风险与 start-correct 条件风险分母分开；覆盖率按实际值报告。",
            "dev 阈值来自恢复的 checkpoint 在 dev 单状态上的前向（无训练、无 GPU、无测试集泄漏）。",
            "endpoint_conf_* 仅为诊断列：显示 R04 所修 bug 的口径差异，不作为本 lane 主口径。",
        ],
    }
    (a.out / "n01_p1_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    outputs = sorted(p for p in a.out.rglob("*") if p.is_file())
    (a.out / "PROVENANCE.json").write_text(json.dumps({
        **provenance,
        "outputs": {str(p.relative_to(a.out)): sha256_file(p) for p in outputs},
    }, indent=2, default=str))
    print(json.dumps({"status": summary["status"], "out": str(a.out),
                      "n_per_arm_rows": len(rows), "n_paired_rows": len(paired_rows)}, indent=2))


if __name__ == "__main__":
    main()
