#!/usr/bin/env python3
"""P4 R3: AMR v6 momentum-teacher predictive bundle (JEPA/BYOL/MAE) on SHARED trunk.

Route 1: exact spectral basis (cached eigh factors), 3 density-equalized
  bands [0,1.0)/[1.0,1.3)/[1.3,2.0], Slepian-vector interventions with mu
  reported per update.
Route 2: VICReg variance hinge + within-mode covariance on the FINAL
  mode embedding; stop-gradient asymmetry on the route branch (perturbed
  predicts fixed clean target); frozen pair_loss (already normalizes);
  active-triplet-fraction guard with pre-registered early-stop
  (0.0 for 3 consecutive epochs -> COLLAPSED).
Route 3 (GradNorm controller) NOT implemented; instead init-calibrated
  fixed W_ROUTE (R3 judgment middle; frozen in lock).

Protocol: 3 seeds (parallel workers), 80 epochs, 210 ctx / 140 pair /
70 route = 420/epoch (equal compute), train split only, dev eval only
(frozen T5R metrics + frozen intervention eval + singular spectrum +
mu diagnostics). No reserved/external reads.

Outputs: artifacts/phase4_amr/m1_v6/{record_seed*.json, models/,
summary.json, manifest.json, SHA256SUMS}.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
P4 = ROOT / "artifacts" / "phase4_amr"
OUTPUT = P4 / "m1_v6"
SEEDS = (11, 23, 47)
EPOCHS = 80
CTX_QUOTA, PAIR_QUOTA, ROUTE_QUOTA = 210, 140, 70
ROUTE_BATCH = 8
SUPPORT_SIZES = (2, 3, 4, 5)
EPSILON = 0.25
W_TASK, W_CTX = 1.0, 1.0
W_ROUTE = 1.0  # R3 teacher-routing weight (judgment middle; F#3 0.03-0.10 range is R4 fallback)
W_KOLEO = 0.1  # DINOv2 subordinate spread weight
CORRUPTIONS = os.environ.get("AMR_CORRUPTIONS", "both")  # R1 ablation switch: "both" | "slepian" | "mask" (frozen in lock)
assert CORRUPTIONS in ("both", "slepian", "mask"), f"bad AMR_CORRUPTIONS={CORRUPTIONS}"
LOCKFILE = P4 / ("m1_v6nomask_config_lock.json" if CORRUPTIONS != "both" else "m1_v6_config_lock.json")
WARMUP_EPOCHS = 20  # epochs 1-20: ctx+pair only (frozen-recipe regime), no route updates
RAMP_EPOCHS = 20  # epochs 21-40: W_ROUTE ramps linearly 0 -> 2.0; full after
W_VICVAR, W_VICCOV = 1.0, 0.04  # VICReg-ratio-mirrored (25/25/1 -> 1.0/x/0.04)
VIC_GAMMA = 0.02  # init-calibrated floor on raw final-embedding per-dim std
N_INTERVENTION_SNAPSHOTS = 40
PAIR_MARGIN = 0.20


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_lock() -> dict[str, Any]:
    lock = json.loads(LOCKFILE.read_text(encoding="utf-8"))
    for rel, digest in lock["code_hashes"].items():
        if sha256_file(ROOT / rel) != digest:
            raise RuntimeError(f"REFUSED: code hash mismatch vs lock: {rel}")
    consts = lock["constants"]
    assert consts["W_ROUTE"] == W_ROUTE, "W_ROUTE out of sync with lock"
    assert consts["W_VICVAR"] == W_VICVAR and consts["W_VICCOV"] == W_VICCOV
    assert consts["VIC_GAMMA"] == VIC_GAMMA
    assert consts["W_KOLEO"] == W_KOLEO, "W_KOLEO out of sync with lock"
    assert consts["CORRUPTIONS"] == CORRUPTIONS, "CORRUPTIONS out of sync with lock"
    for split in ("train", "valid"):
        key = f"spectral_cache_{split}_sha256"
        if sha256_file(P4 / f"spectral_cache_{split}.npz") != lock[key]:
            raise RuntimeError(f"REFUSED: spectral cache mismatch: {split}")
    return lock


def route_batch(model: Any, teacher: Any, T5R3: Any, v6: Any, batch: dict[str, torch.Tensor],
                spec_cache: dict[str, np.ndarray], snap_idx: np.ndarray,
                rng: np.random.Generator) -> dict[str, Any]:
    """One momentum-teacher routing update (R3): student sees Slepian-perturbed
    input, predicts the teacher's CLEAN band features through the per-band
    predictor MLP. Teacher is frozen (no grads anywhere); stop-grad belt and
    suspenders via torch.no_grad. Slepian sampling identical to v4/v5 (mu logged)."""
    centered = batch["centered"][snap_idx]
    team = batch["team"][snap_idx]
    adj = batch["adjacency"][snap_idx]
    n = len(snap_idx)
    band = int(rng.integers(0, 3))
    support_size = int(SUPPORT_SIZES[rng.integers(0, len(SUPPORT_SIZES))])
    corr = {"slepian": "slepian", "mask": "mask"}.get(
        CORRUPTIONS, "slepian" if int(rng.integers(0, 2)) == 0 else "mask")
    team_np = team.numpy()
    deltas, mus, mask_sizes = [], [], []
    student_input = centered.clone()
    if corr == "slepian":
        for i in range(n):
            team_id = int(rng.integers(0, 2))
            nodes = np.where(team_np[i] == team_id)[0]
            chosen = rng.choice(nodes, size=support_size, replace=False)
            mask = np.zeros(20, dtype=bool)
            mask[chosen] = True
            delta, mu = v6.slepian_intervention(spec_cache["U"][snap_idx[i]], spec_cache["evals"][snap_idx[i]],
                                                band, mask, EPSILON, rng)
            if mu < 1e-8:
                delta = np.zeros((20, 2))
            deltas.append(delta)
            mus.append(mu)
        student_input = student_input + torch.from_numpy(np.stack(deltas)).float()
    else:
        for i in range(n):
            team_id = int(rng.integers(0, 2))
            nodes = np.where(team_np[i] == team_id)[0]
            size = int(v6.MASK_SIZES[rng.integers(0, len(v6.MASK_SIZES))])
            chosen = rng.choice(nodes, size=size, replace=False)
            student_input[i, chosen] = model.mask_token.detach()
            mask_sizes.append(size)
    spec_t = (torch.from_numpy(spec_cache["evals"][snap_idx]),
              torch.from_numpy(spec_cache["U"][snap_idx]))
    with torch.no_grad():
        teacher.eval()
        tgt = teacher.band_features(centered, team, adj, spec_t)[:, band, :]
    r_after = model.band_features(student_input, team, adj, spec_t)[:, band, :]
    pred = model.predictor_heads[band](r_after)
    l_teach = ((F.normalize(pred, dim=-1) - F.normalize(tgt, dim=-1)).pow(2)
               .sum(dim=-1).mean())
    inv_response = (r_after - model.band_features(centered, team, adj, spec_t)[:, band, :].detach()).pow(2).sum(dim=-1).mean().detach()
    return {"loss": l_teach, "l_eqv": l_teach.detach(), "inv_response": inv_response,
            "band": band, "support_size": support_size, "mu_mean": float(np.mean(mus)) if mus else 0.0,
            "corr": corr, "mask_size_mean": float(np.mean(mask_sizes)) if mask_sizes else 0.0}


def train_v4(model: Any, teacher: Any, T5R3: Any, R2: Any, v6: Any, train: Any,
             adjacency: np.ndarray, spec_cache: dict[str, np.ndarray],
             seed: int) -> dict[str, Any]:
    T5R3.set_seed(seed)
    model.train()
    pred_params = [p for head in model.predictor_heads for p in head.parameters()]
    pred_ids = {id(p) for p in pred_params}
    base_params = [p for p in model.parameters() if id(p) not in pred_ids]
    optimizer = torch.optim.Adam([{"params": base_params, "lr": 1e-3},
                                  {"params": pred_params, "lr": 1e-2}],
                                 weight_decay=0.0)
    TOTAL_STEPS = EPOCHS * (CTX_QUOTA + PAIR_QUOTA) + max(0, EPOCHS - WARMUP_EPOCHS) * ROUTE_QUOTA
    gstep = 0
    batch = T5R3.tensors(train, adjacency)
    ctx_stream = R2.build_cyclic_stream(len(train.raw), seed * 100_000, CTX_QUOTA, EPOCHS)
    pair_stream = R2.build_cyclic_stream(len(train.pair_query), seed * 200_000, PAIR_QUOTA, EPOCHS)
    rng = np.random.default_rng(seed * 300_000)
    started = time.perf_counter()
    last: dict[str, float] = {}
    mu_stats: dict[str, list[float]] = {f"b{b}_s{s}": [] for b in range(3) for s in SUPPORT_SIZES}
    zero_streak, collapsed = 0, False
    for epoch in range(EPOCHS):
        model.train()
        ctx_total = ctx_seen = 0
        for slot in range(CTX_QUOTA):
            index = torch.from_numpy(ctx_stream[epoch, slot])
            z = model.context_channel(batch["raw"][index], batch["team"][index], batch["adjacency"][index])
            loss = T5R3.context_loss(model, z, {"phase": batch["phase"][index],
                                                "zone": batch["zone"][index],
                                                "centroids": batch["centroids"][index]})
            optimizer.zero_grad(set_to_none=True)
            (W_CTX * loss).backward()
            optimizer.step()
            gstep += 1
            v6.update_teacher(teacher, model, v6.tau_schedule(gstep, TOTAL_STEPS))
            ctx_total += float(loss.detach()) * len(index)
            ctx_seen += len(index)
        pair_total = pair_seen = 0
        vicvar_total = viccov_total = koleo_total = 0.0
        active_total = 0.0
        for slot in range(PAIR_QUOTA):
            index = torch.from_numpy(pair_stream[epoch, slot])
            qi, pi, ni = batch["pair_query"][index], batch["pair_positive"][index], batch["pair_negative"][index]
            all_idx = torch.cat([qi, pi, ni], dim=0)
            z_flat = model.band_features(
                batch["centered"][all_idx], batch["team"][all_idx], batch["adjacency"][all_idx],
                (torch.from_numpy(spec_cache["evals"][all_idx.numpy()]),
                 torch.from_numpy(spec_cache["U"][all_idx.numpy()]))).reshape(len(all_idx), -1)
            loss = T5R3.pair_loss(model, z_flat, len(index))
            emb_raw = model.mode_embedding(z_flat)
            var_l, cov_l = v6.vicreg_var_cov(emb_raw, VIC_GAMMA)
            qz, pz, nz = emb_raw.split(len(index), dim=0)
            active = v6.active_triplet_fraction(F.normalize(qz, dim=-1), F.normalize(pz, dim=-1),
                                               F.normalize(nz, dim=-1), PAIR_MARGIN)
            koleo = v6.koleo_spread(F.normalize(emb_raw, dim=-1))
            optimizer.zero_grad(set_to_none=True)
            (W_TASK * loss + W_VICVAR * var_l + W_VICCOV * cov_l + W_KOLEO * koleo).backward()
            optimizer.step()
            gstep += 1
            v6.update_teacher(teacher, model, v6.tau_schedule(gstep, TOTAL_STEPS))
            pair_total += float(loss.detach()) * len(index)
            pair_seen += len(index)
            vicvar_total += float(var_l.detach())
            viccov_total += float(cov_l.detach())
            koleo_total += float(koleo.detach())
            active_total += active
        route_totals = {"l_eqv": 0.0, "inv_response": 0.0, "mu": 0.0}
        mask_trials, mask_size_total = 0, 0.0
        if epoch < WARMUP_EPOCHS:
            w_route_now = 0.0
        elif epoch < WARMUP_EPOCHS + RAMP_EPOCHS:
            w_route_now = W_ROUTE * (epoch + 1 - WARMUP_EPOCHS) / RAMP_EPOCHS
        else:
            w_route_now = W_ROUTE
        n_route_updates = 0
        for _ in range(ROUTE_QUOTA if w_route_now > 0 else 0):
            snap_idx = rng.integers(0, len(train.raw), size=ROUTE_BATCH)
            out = route_batch(model, teacher, T5R3, v6, batch, spec_cache, snap_idx, rng)
            optimizer.zero_grad(set_to_none=True)
            (w_route_now * out["loss"]).backward()
            optimizer.step()
            gstep += 1
            v6.update_teacher(teacher, model, v6.tau_schedule(gstep, TOTAL_STEPS))
            n_route_updates += 1
            if out["corr"] == "mask":
                mask_trials += 1
                mask_size_total += out["mask_size_mean"]
            route_totals["l_eqv"] += float(out["l_eqv"])
            route_totals["inv_response"] += float(out["inv_response"])
            route_totals["mu"] += out["mu_mean"]
            mu_stats[f"b{out['band']}_s{out['support_size']}"].append(out["mu_mean"])
        active_frac = active_total / PAIR_QUOTA
        zero_streak = zero_streak + 1 if active_frac == 0.0 else 0
        last = {"context": ctx_total / ctx_seen, "pair": pair_total / pair_seen,
                "vicvar": vicvar_total / PAIR_QUOTA, "viccov": viccov_total / PAIR_QUOTA,
                "active_triplet_frac": active_frac,
                "l_eqv": route_totals["l_eqv"] / max(n_route_updates, 1),
                "inv_response": route_totals["inv_response"] / max(n_route_updates, 1),
                "mu_mean": route_totals["mu"] / max(n_route_updates, 1),
                "mask_trials": mask_trials,
                "mask_size_mean": (mask_size_total / mask_trials) if mask_trials else 0.0,
                "koleo": koleo_total / PAIR_QUOTA,
                "w_route_now": w_route_now}
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == EPOCHS:
            print(f"seed={seed} epoch={epoch + 1}/{EPOCHS} ctx={last['context']:.4f} "
                  f"pair={last['pair']:.4f} active={active_frac:.3f} eqv={last['l_eqv']:.4f} "
                  f"vicvar={last['vicvar']:.4f} mu={last['mu_mean']:.3f} wr={last['w_route_now']:.2f} "
                  f"elapsed={time.perf_counter() - started:.0f}s", flush=True)
        if any(not np.isfinite(x) for x in last.values()):
            return {"epochs": epoch + 1, "status": "FAILED_NONFINITE", "last_loss": last,
                    "elapsed_seconds": round(time.perf_counter() - started, 1)}
        if zero_streak >= 3:
            collapsed = True
            print(f"seed={seed} COLLAPSED at epoch {epoch + 1} (active-triplet 0 x3)", flush=True)
            break
    return {"epochs": EPOCHS, "quotas": {"context": CTX_QUOTA, "pair": PAIR_QUOTA, "route": ROUTE_QUOTA},
            "status": "COLLAPSED" if collapsed else "TRAINED",
            "weights": {"task": W_TASK, "ctx": W_CTX, "route": W_ROUTE,
                        "vicvar": W_VICVAR, "viccov": W_VICCOV, "vic_gamma": VIC_GAMMA},
            "mu_by_band_support_train_mean": {k: (float(np.mean(v)) if v else None)
                                              for k, v in mu_stats.items()},
            "last_loss": last, "elapsed_seconds": round(time.perf_counter() - started, 1)}


M1BASE: Any = None


def run_one(seed: int, threads: int, output: Path) -> None:
    started = time.perf_counter()
    torch.set_num_threads(threads)
    lock = verify_lock()
    T5R3 = load_module(ROOT / "scripts" / "run_t5r3_sanity.py", f"t5r3_for_v6_{seed}")
    M1BASE.T5R3_REF = T5R3
    R2 = load_module(ROOT / "scripts" / "run_t5r4_round2.py", f"r2_for_v6_{seed}")
    v6 = load_module(ROOT / "scripts" / "amr_model_v6.py", f"v6_for_m1v6_{seed}")
    sweep = load_module(ROOT / "scripts" / "run_p3_epsilon_sweep.py", f"sweep_for_v6_{seed}")
    sweep.EPSILONS = (0.25,)
    sweep.N_SNAPSHOTS = N_INTERVENTION_SNAPSHOTS
    T5R5 = load_module(ROOT / "scripts" / "run_t5r5_hidden_confirmation.py", f"t5r5_for_v6_{seed}")
    FC = load_module(ROOT / "scripts" / "p2_fracture_controls.py", f"fc_for_v6_{seed}")
    split_lock = json.loads((T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    train_adj = T5R3.knn_adjacency_batch(train.raw)
    valid_adj = T5R3.knn_adjacency_batch(valid.raw)
    spec_cache = {k: np.load(P4 / f"spectral_cache_train.npz")[k] for k in ("U", "evals")}

    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== AMR v6 seed={seed} threads={threads} ===", flush=True)
    T5R3.set_seed(seed)
    model = v6.AMRModelV6()
    # Routing is teacher-gated (alpha buffer unused); zeroed for determinism.
    model.set_route_alpha(torch.zeros(model.n_bands))
    teacher = v6.build_teacher(model)
    params = int(sum(p.numel() for p in model.parameters()))
    training = train_v4(model, teacher, T5R3, R2, v6, train, train_adj, spec_cache, seed)
    ckpt = models_dir / f"amr_v6_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt)

    with torch.inference_mode():
        train_ctx = M1BASE.encode_ctx(model, train, train_adj)
        train_bands = M1BASE.encode_bands(model, train, train_adj)
        valid_ctx = M1BASE.encode_ctx(model, valid, valid_adj)
        valid_bands = M1BASE.encode_bands(model, valid, valid_adj)
        tag = {"both": "amr_v6_momentum_teacher", "slepian": "amr_v6nomask_slepian_only",
               "mask": "amr_v6mask_only"}[CORRUPTIONS]
        record = {"candidate_id": tag, "seed": seed, "parameter_count": params,
                  "train": {"context": T5R3.prediction_metrics(train, model, train_ctx),
                            "intrinsic": T5R3.pair_metrics(train, model, train_bands),
                            "translation": M1BASE.translation_response(model, train, train_adj, train_ctx, train_bands)},
                  "valid": {"context": T5R3.prediction_metrics(valid, model, valid_ctx),
                            "intrinsic": T5R3.pair_metrics(valid, model, valid_bands),
                            "translation": M1BASE.translation_response(model, valid, valid_adj, valid_ctx, valid_bands)},
                  "cross_readout": "shared_trunk_by_design_R3",
                  "training": training,
                  "teacher": {"type": "momentum_ema", "tau": "cosine 0.99->1.0 per-step",
                              "mirrored_prefixes": list(v6.TEACHER_PARAM_PREFIXES),
                              "predictor_lr": "10x trunk (Tian et al.)",
                              "corruptions": ["slepian", "masked_player"]},
                  "checkpoint": {"path": str(ckpt.relative_to(ROOT)), "sha256": sha256_file(ckpt)}}
    v = record["valid"]
    print(f"seed={seed} valid zone={v['context']['field_zone_match_grouped_macro_f1']:.4f} "
          f"pair={v['intrinsic']['match_grouped_ranking_accuracy']:.4f}", flush=True)
    with torch.inference_mode():
        zb = torch.from_numpy(valid_bands)
        emb = model.mode_embedding(zb).numpy()
        _, svals, _ = np.linalg.svd(emb - emb.mean(0, keepdims=True), full_matrices=False)
        record["valid_mode_embedding_singular_top10"] = [float(x) for x in svals[:10]]

    raw_all = np.concatenate([train.raw, valid.raw]).astype(np.float64)
    team_all = np.concatenate([train.team_slots, valid.team_slots])
    snap_ids = train.snapshot_ids.tolist() + valid.snapshot_ids.tolist()
    torch.set_num_threads(1)
    sets = sweep.build_sets(T5R3, T5R5, FC, raw_all, team_all, snap_ids)
    model2 = v6.AMRModelV6()
    model2.load_state_dict(torch.load(ROOT / record["checkpoint"]["path"], map_location="cpu", weights_only=True))
    model2.eval()
    record["intervention_dev"] = sweep.evaluate_model(model2, sets, FC, T5R3)
    record["n_intervention_sets"] = len(sets)
    record["lock_sha256_at_train"] = hashlib.sha256(LOCKFILE.read_bytes()).hexdigest()
    record["torch_threads_train"] = threads
    rho = record["intervention_dev"]["0.25"].get("full_spearman")
    print(f"seed={seed} intervention full rho={rho if rho is None else round(rho, 3)} "
          f"total={time.perf_counter() - started:.0f}s", flush=True)
    write_json(output / f"record_seed{seed}.json", record)


def merge(output: Path) -> int:
    records = []
    for seed in SEEDS:
        path = output / f"record_seed{seed}.json"
        if not path.is_file():
            print(f"REFUSED: missing {path}", file=sys.stderr)
            return 2
        records.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {"method": {"both": "amr_v6_momentum_teacher", "slepian": "amr_v6nomask_slepian_only",
                          "mask": "amr_v6mask_only"}[CORRUPTIONS],
               "lock": "artifacts/phase4_amr/m1_v6nomask_config_lock.json" if CORRUPTIONS != "both" else "artifacts/phase4_amr/m1_v6_config_lock.json",
               "evidence_scope": "train/valid dev only", "n_intervention_sets": records[0]["n_intervention_sets"],
               "records": records}
    write_json(output / "summary.json", summary)
    manifest = {"status": ("P4_R3_AMR_V6_DEV_COMPLETE" if CORRUPTIONS == "both"
                           else "GOAL_R1_NOMASK_DEV_COMPLETE"),
                "seeds": list(SEEDS),
                "elapsed_seconds_max_worker": max(r["training"]["elapsed_seconds"] for r in records)}
    write_json(output / "manifest.json", manifest)
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"]}), flush=True)
    return 0


def main() -> int:
    import argparse
    global M1BASE
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    M1BASE = load_module(ROOT / "scripts" / "run_amr_m1.py", "m1base_for_v6")
    output = args.output.resolve()
    if args.merge:
        return merge(output)
    if args.seed is None:
        print("REFUSED: pass --seed N (worker) or --merge", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"record_seed{args.seed}.json").exists():
        print("REFUSED: record exists", file=sys.stderr)
        return 2
    run_one(args.seed, args.threads, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
