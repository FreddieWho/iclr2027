#!/usr/bin/env python3
"""P4 GOAL R2: JGCL-style single-geometry InfoNCE on SHARED trunk.
New loss ontology (Track I P1): ONE InfoNCE whose positives are natural pairs
  + Slepian views of the query (in-batch negatives, tau=0.1). Removes the
  tug-of-war at the loss-ontology level: no triplet, no teacher/predictor, no VICReg/KoLeo/mask.
  Context loss kept (frozen). Warm-start 20 ctx-only + ramp 20 (0->1.0).
  Budget 210 ctx + 210 InfoNCE = 420/epoch (equal compute, recomposed slots).
  Collapse guard: mean mode-embedding std < 1e-4 x3 epochs -> COLLAPSED.
  H1 bar applies; R2 read: margins sharper than triplet系 = big-number hope.
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
OUTPUT = P4 / "m1_jgcl"
SEEDS = (11, 23, 47)
EPOCHS = 80
CTX_QUOTA, INFO_QUOTA = 210, 210  # 420/epoch equal-compute; slot recomposition by design
ROUTE_BATCH = 8
SUPPORT_SIZES = (2, 3, 4, 5)
EPSILON = 0.25
W_CTX, W_INFO = 1.0, 1.0
TAU = 0.1  # SupCon/JGCL range (Track I)
# (route weight removed: single-loss ontology; see W_INFO)
# (KoLeo removed: single-loss ontology)
# (no corruption switch: Slepian views only, always on)
# (switch removed)
LOCKFILE = P4 / "m1_jgcl_config_lock.json"
WARMUP_EPOCHS = 20  # epochs 1-20: ctx only (no InfoNCE updates)
RAMP_EPOCHS = 20  # epochs 21-40: W_INFO ramps linearly 0 -> 1.0; full after
# (VICReg removed: single-loss ontology)
# (gamma removed)
N_INTERVENTION_SNAPSHOTS = 40
# (triplet removed: single-loss ontology)


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
    assert consts["W_CTX"] == W_CTX and consts["W_INFO"] == W_INFO, "weights out of sync"
    assert consts["TAU"] == TAU, "TAU out of sync with lock"
    for split in ("train", "valid"):
        key = f"spectral_cache_{split}_sha256"
        if sha256_file(P4 / f"spectral_cache_{split}.npz") != lock[key]:
            raise RuntimeError(f"REFUSED: spectral cache mismatch: {split}")
    return lock


def infonce_loss(zq: torch.Tensor, zp: torch.Tensor, zv: torch.Tensor,
                   tau: float) -> torch.Tensor:
    """Multi-positive InfoNCE: anchors=queries, candidates=q+p+v (in-batch negs).

    Positives of anchor i: natural positive p_i and Slepian view v_i.
    """
    q = F.normalize(zq, dim=-1)
    pp = F.normalize(zp, dim=-1)
    v = F.normalize(zv, dim=-1)
    B = q.shape[0]
    cand = torch.cat([q, pp, v], dim=0)
    sims = (q @ cand.T) / tau
    idx = torch.arange(B, device=q.device)
    pos = torch.stack([sims[idx, B + idx], sims[idx, 2 * B + idx]], dim=1)
    denom = torch.ones_like(sims, dtype=torch.bool)
    denom[idx, idx] = False
    lse = torch.logsumexp(sims.masked_fill(~denom, float("-inf")), dim=1)
    return (-(torch.logsumexp(pos, dim=1) - lse)).mean()


def sample_slepian_views(jg: object, batch: dict, spec_cache: dict,
                         q_global: np.ndarray, team_q: torch.Tensor,
                         band: int, support_size: int,
                         rng: np.random.Generator) -> tuple:
    """Slepian displacement per query (same-team support); returns delta, mu_mean."""
    n = len(q_global)
    team_np = team_q.numpy()
    U_all, E_all = spec_cache["U"], spec_cache["evals"]
    deltas, mus = [], []
    for i in range(n):
        team_id = int(rng.integers(0, 2))
        nodes = np.where(team_np[i] == team_id)[0]
        chosen = rng.choice(nodes, size=support_size, replace=False)
        mask = np.zeros(20, dtype=bool)
        mask[chosen] = True
        delta, mu = jg.slepian_intervention(U_all[q_global[i]], E_all[q_global[i]],
                                            band, mask, EPSILON, rng)
        if mu < 1e-8:
            delta = np.zeros((20, 2))
        deltas.append(delta)
        mus.append(mu)
    return torch.from_numpy(np.stack(deltas)).float(), float(np.mean(mus))


def train_jgcl(model: Any, T5R3: Any, R2: Any, jg: Any, train: Any,
               adjacency: np.ndarray, spec_cache: dict[str, np.ndarray],
               seed: int) -> dict[str, Any]:
    T5R3.set_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    batch = T5R3.tensors(train, adjacency)
    ctx_stream = R2.build_cyclic_stream(len(train.raw), seed * 100_000, CTX_QUOTA, EPOCHS)
    pair_stream = R2.build_cyclic_stream(len(train.pair_query), seed * 200_000, INFO_QUOTA, EPOCHS)
    rng = np.random.default_rng(seed * 300_000)
    started = time.perf_counter()
    last: dict[str, float] = {}
    mu_stats: dict[str, list[float]] = {f"b{b}_s{s}": [] for b in range(3) for s in SUPPORT_SIZES}
    lowstd_streak, collapsed = 0, False
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
            ctx_total += float(loss.detach()) * len(index)
            ctx_seen += len(index)
        if epoch < WARMUP_EPOCHS:
            w_now = 0.0
        elif epoch < WARMUP_EPOCHS + RAMP_EPOCHS:
            w_now = W_INFO * (epoch + 1 - WARMUP_EPOCHS) / RAMP_EPOCHS
        else:
            w_now = W_INFO
        info_total = mu_total = std_total = 0.0
        n_info = 0
        for _ in range(INFO_QUOTA if w_now > 0 else 0):
            index = torch.from_numpy(pair_stream[epoch, n_info])
            qi = batch["pair_query"][index]
            pi = batch["pair_positive"][index]
            band = int(rng.integers(0, 3))
            support_size = int(SUPPORT_SIZES[rng.integers(0, len(SUPPORT_SIZES))])
            delta, mu = sample_slepian_views(
                jg, batch, spec_cache, qi.numpy(), batch["team"][qi], band, support_size, rng)
            spec_q = (torch.from_numpy(spec_cache["evals"][qi.numpy()]),
                      torch.from_numpy(spec_cache["U"][qi.numpy()]))
            spec_p = (torch.from_numpy(spec_cache["evals"][pi.numpy()]),
                      torch.from_numpy(spec_cache["U"][pi.numpy()]))
            zq = model.mode_embedding(model.band_features(
                batch["centered"][qi], batch["team"][qi], batch["adjacency"][qi], spec_q))
            zp = model.mode_embedding(model.band_features(
                batch["centered"][pi], batch["team"][pi], batch["adjacency"][pi], spec_p))
            zv = model.mode_embedding(model.band_features(
                batch["centered"][qi] + delta, batch["team"][qi], batch["adjacency"][qi], spec_q))
            loss = infonce_loss(zq, zp, zv, TAU)
            optimizer.zero_grad(set_to_none=True)
            (w_now * loss).backward()
            optimizer.step()
            n_info += 1
            info_total += float(loss.detach())
            mu_total += mu
            std_total += float(torch.cat([zq, zp, zv]).detach().std())
            mu_stats[f"b{band}_s{support_size}"].append(mu)
        emb_std = std_total / max(n_info, 1)
        if epoch >= WARMUP_EPOCHS:
            lowstd_streak = lowstd_streak + 1 if emb_std < 1e-4 else 0
        last = {"context": ctx_total / ctx_seen,
                "infonce": info_total / max(n_info, 1),
                "emb_std": emb_std,
                "mu_mean": mu_total / max(n_info, 1),
                "w_now": w_now}
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == EPOCHS:
            print(f"seed={seed} epoch={epoch + 1}/{EPOCHS} ctx={last['context']:.4f} "
                  f"info={last['infonce']:.4f} embstd={emb_std:.5f} mu={last['mu_mean']:.3f} "
                  f"w={w_now:.2f} elapsed={time.perf_counter() - started:.0f}s", flush=True)
        if any(not np.isfinite(x) for x in last.values()):
            return {"epochs": epoch + 1, "status": "FAILED_NONFINITE", "last_loss": last,
                    "elapsed_seconds": round(time.perf_counter() - started, 1)}
        if lowstd_streak >= 3:
            collapsed = True
            print(f"seed={seed} COLLAPSED at epoch {epoch + 1} (emb-std floor x3)", flush=True)
            break
    return {"epochs": EPOCHS, "quotas": {"context": CTX_QUOTA, "infonce": INFO_QUOTA},
            "status": "COLLAPSED" if collapsed else "TRAINED",
            "weights": {"ctx": W_CTX, "info": W_INFO, "tau": TAU},
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
    print(f"=== AMR jgcl seed={seed} threads={threads} ===", flush=True)
    T5R3.set_seed(seed)
    model = v6.AMRModelV6()
    # No teacher in R2 (single-loss ontology); alpha buffer zeroed for determinism.
    model.set_route_alpha(torch.zeros(model.n_bands))
    params = int(sum(p.numel() for p in model.parameters()))
    training = train_jgcl(model, T5R3, R2, v6, train, train_adj, spec_cache, seed)
    ckpt = models_dir / f"amr_jgcl_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt)

    with torch.inference_mode():
        train_ctx = M1BASE.encode_ctx(model, train, train_adj)
        train_bands = M1BASE.encode_bands(model, train, train_adj)
        valid_ctx = M1BASE.encode_ctx(model, valid, valid_adj)
        valid_bands = M1BASE.encode_bands(model, valid, valid_adj)
        tag = "amr_jgcl_single_geometry"
        record = {"candidate_id": tag, "seed": seed, "parameter_count": params,
                  "train": {"context": T5R3.prediction_metrics(train, model, train_ctx),
                            "intrinsic": T5R3.pair_metrics(train, model, train_bands),
                            "translation": M1BASE.translation_response(model, train, train_adj, train_ctx, train_bands)},
                  "valid": {"context": T5R3.prediction_metrics(valid, model, valid_ctx),
                            "intrinsic": T5R3.pair_metrics(valid, model, valid_bands),
                            "translation": M1BASE.translation_response(model, valid, valid_adj, valid_ctx, valid_bands)},
                  "cross_readout": "shared_trunk_single_loss_R2",
                  "training": training,
                  "jgcl": {"positives": ["natural_pair", "slepian_view"],
                           "negatives": "in-batch (3B-3 per anchor)",
                           "tau": TAU, "dropped": ["triplet", "teacher", "predictor",
                                                     "VICReg", "KoLeo", "mask"]},
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
    summary = {"method": "amr_jgcl_single_geometry",
               "lock": "artifacts/phase4_amr/m1_jgcl_config_lock.json",
               "evidence_scope": "train/valid dev only", "n_intervention_sets": records[0]["n_intervention_sets"],
               "records": records}
    write_json(output / "summary.json", summary)
    manifest = {"status": "GOAL_R2_JGCL_DEV_COMPLETE",
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
