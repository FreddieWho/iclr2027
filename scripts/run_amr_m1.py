#!/usr/bin/env python3
"""P4-N3: AMR M1 training + dev evaluation under the frozen m1_config_lock.

Protocol: 3 seeds (parallel workers), 80 epochs, 210 context / 140 pair /
70 route updates per epoch (=420, equal compute to the frozen T5R recipe),
train split only; valid-split evaluation with the frozen T5R metric
functions (context/intrinsic/translation) plus the frozen intervention
evaluation (epsilon-sweep path) and per-(band, support-size) route
diagnostics. No reserved/external reads.

Outputs: artifacts/phase4_amr/m1_v1/{record_seed*.json, models/,
summary.json, manifest.json, SHA256SUMS}.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
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
OUTPUT = P4 / "m1_v1"
SEEDS = (11, 23, 47)
EPOCHS = 80
CTX_QUOTA, PAIR_QUOTA, ROUTE_QUOTA = 210, 140, 70
ROUTE_BATCH = 8
SUPPORT_SIZES = (2, 3, 4, 5)
EPSILON = 0.25
W_TASK, W_CTX, W_ROUTE, W_VAR, W_ORTH = 1.0, 1.0, 1.0, 0.1, 0.01
N_INTERVENTION_SNAPSHOTS = 40


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
    lock = json.loads((P4 / "m1_config_lock.json").read_text(encoding="utf-8"))
    for rel, digest in lock["code_hashes"].items():
        if sha256_file(ROOT / rel) != digest:
            raise RuntimeError(f"REFUSED: code hash mismatch vs lock: {rel}")
    return lock


def route_batch(model: Any, T5R3: Any, amr: Any, batch: dict[str, torch.Tensor],
                snap_idx: np.ndarray, band: int, support_size: int,
                rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """One route update: support-conditioned whitened intervention on the
    mode channel; routing loss + anti-collapse + decoupling (lock spec)."""
    centered = batch["centered"][snap_idx]
    team = batch["team"][snap_idx]
    adj = batch["adjacency"][snap_idx]
    lap = amr.normalized_laplacian(adj)
    xi = torch.from_numpy(rng.standard_normal((len(snap_idx), 20, 2))).float()
    mask = torch.zeros(len(snap_idx), 20, 1)
    for i in range(len(snap_idx)):
        team_id = int(rng.integers(0, 2))
        nodes = np.where(team[i].numpy() == team_id)[0]
        chosen = rng.choice(nodes, size=support_size, replace=False)
        mask[i, chosen, 0] = 1.0
    xi = xi * mask
    coeff = model.band_coeffs[band]
    filtered = amr.apply_chebyshev(lap, xi, coeff)
    norm = filtered.reshape(len(snap_idx), -1).norm(dim=1).clamp_min(1e-12)
    delta = EPSILON * filtered / norm[:, None, None]
    with torch.no_grad():
        target = amr.apply_chebyshev(lap, delta, coeff)
        target_pooled = T5R3.team_pool(target, team)
    feats_before = model.band_features(centered, team, adj)
    feats_after = model.band_features(centered + delta, team, adj)
    d_r = feats_after[:, band, :] - feats_before[:, band, :]
    l_inv = d_r.pow(2).sum(dim=-1).mean()
    l_eqv = (model.eqv_heads[band](d_r) - target_pooled).pow(2).sum(dim=-1).mean()
    alpha = model.route_alpha[band]
    l_route = alpha * l_inv + (1.0 - alpha) * l_eqv
    std = feats_before.std(dim=0).mean(dim=-1)
    l_var = torch.relu(1.0 - std).sum()
    z_ctx = model.context_channel(batch["raw"][snap_idx], team, adj)
    z_mode = model.mode_embedding(feats_before)
    zc = z_ctx - z_ctx.mean(dim=0, keepdim=True)
    zm = z_mode - z_mode.mean(dim=0, keepdim=True)
    l_orth = (zc.T @ zm / max(len(snap_idx) - 1, 1)).pow(2).sum()
    return {"loss": l_route + W_VAR * l_var + W_ORTH * l_orth, "l_route": l_route.detach(),
            "l_inv": l_inv.detach(), "l_eqv": l_eqv.detach(), "l_var": l_var.detach(),
            "l_orth": l_orth.detach(), "band": band, "support_size": support_size}


def train_m1(model: Any, T5R3: Any, R2: Any, amr: Any, train: Any,
             adjacency: np.ndarray, seed: int) -> dict[str, Any]:
    T5R3.set_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    batch = T5R3.tensors(train, adjacency)
    ctx_stream = R2.build_cyclic_stream(len(train.raw), seed * 100_000, CTX_QUOTA, EPOCHS)
    pair_stream = R2.build_cyclic_stream(len(train.pair_query), seed * 200_000, PAIR_QUOTA, EPOCHS)
    rng = np.random.default_rng(seed * 300_000)
    started = time.perf_counter()
    last: dict[str, float] = {}
    band_stats: dict[str, list[float]] = {f"b{b}_s{s}": [] for b in range(6) for s in SUPPORT_SIZES}
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
        pair_total = pair_seen = 0
        for slot in range(PAIR_QUOTA):
            index = torch.from_numpy(pair_stream[epoch, slot])
            qi, pi, ni = batch["pair_query"][index], batch["pair_positive"][index], batch["pair_negative"][index]
            all_idx = torch.cat([qi, pi, ni], dim=0)
            z = model.encode(batch["centered"][all_idx], batch["team"][all_idx], batch["adjacency"][all_idx], "team_mean")
            loss = T5R3.pair_loss(model, z, len(index))
            optimizer.zero_grad(set_to_none=True)
            (W_TASK * loss).backward()
            optimizer.step()
            pair_total += float(loss.detach()) * len(index)
            pair_seen += len(index)
        route_totals = {"l_route": 0.0, "l_inv": 0.0, "l_eqv": 0.0, "l_var": 0.0, "l_orth": 0.0}
        for _ in range(ROUTE_QUOTA):
            snap_idx = rng.integers(0, len(train.raw), size=ROUTE_BATCH)
            band = int(rng.integers(0, 6))
            support_size = int(SUPPORT_SIZES[rng.integers(0, len(SUPPORT_SIZES))])
            out = route_batch(model, T5R3, amr, batch, snap_idx, band, support_size, rng)
            optimizer.zero_grad(set_to_none=True)
            (W_ROUTE * out["loss"]).backward()
            optimizer.step()
            for k in route_totals:
                route_totals[k] += float(out[k])
            band_stats[f"b{band}_s{support_size}"].append(float(out["l_route"]))
        last = {"context": ctx_total / ctx_seen, "pair": pair_total / pair_seen,
                **{k: v / ROUTE_QUOTA for k, v in route_totals.items()}}
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == EPOCHS:
            print(f"seed={seed} epoch={epoch + 1}/{EPOCHS} ctx={last['context']:.4f} "
                  f"pair={last['pair']:.4f} route={last['l_route']:.4f} "
                  f"inv={last['l_inv']:.5f} eqv={last['l_eqv']:.5f} "
                  f"elapsed={time.perf_counter() - started:.0f}s", flush=True)
    band_means = {k: (float(np.mean(v)) if v else None) for k, v in band_stats.items()}
    return {"epochs": EPOCHS, "quotas": {"context": CTX_QUOTA, "pair": PAIR_QUOTA, "route": ROUTE_QUOTA},
            "last_loss": last, "route_loss_by_band_support_train_mean": band_means,
            "elapsed_seconds": round(time.perf_counter() - started, 1)}


def encode_ctx(model: Any, data: Any, adjacency: np.ndarray) -> np.ndarray:
    batch = T5R3_REF.tensors(data, adjacency)
    model.eval()
    outs = []
    with torch.inference_mode():
        for start in range(0, len(data.raw), T5R3_REF.BATCH_SIZE):
            end = min(start + T5R3_REF.BATCH_SIZE, len(data.raw))
            outs.append(model.context_channel(batch["raw"][start:end], batch["team"][start:end],
                                              batch["adjacency"][start:end]).cpu().numpy())
    return np.concatenate(outs, axis=0)


def encode_bands(model: Any, data: Any, adjacency: np.ndarray) -> np.ndarray:
    batch = T5R3_REF.tensors(data, adjacency)
    model.eval()
    outs = []
    with torch.inference_mode():
        for start in range(0, len(data.raw), T5R3_REF.BATCH_SIZE):
            end = min(start + T5R3_REF.BATCH_SIZE, len(data.raw))
            outs.append(model.encode(batch["centered"][start:end], batch["team"][start:end],
                                     batch["adjacency"][start:end], "team_mean").cpu().numpy())
    return np.concatenate(outs, axis=0)


T5R3_REF: Any = None


def translation_response(model: Any, data: Any, adjacency: np.ndarray,
                         z_ctx: np.ndarray, z_bands: np.ndarray) -> dict[str, Any]:
    shifted_raw = data.raw + T5R3_REF.GLOBAL_TRANSLATION[None, None, :]
    shifted_centered = T5R3_REF.center_positions(shifted_raw)
    model.eval()
    outs_ctx, outs_bands = [], []
    with torch.inference_mode():
        for start in range(0, len(data.raw), T5R3_REF.BATCH_SIZE):
            end = min(start + T5R3_REF.BATCH_SIZE, len(data.raw))
            team_t = torch.from_numpy(data.team_slots[start:end])
            adj_t = torch.from_numpy(adjacency[start:end])
            outs_ctx.append(model.context_channel(torch.from_numpy(shifted_raw[start:end]).float(),
                                                  team_t, adj_t).cpu().numpy())
            outs_bands.append(model.encode(torch.from_numpy(shifted_centered[start:end]).float(),
                                           team_t, adj_t, "team_mean").cpu().numpy())
    return {"translation_vector_normalized_pitch": T5R3_REF.GLOBAL_TRANSLATION.tolist(),
            "z_ctx_context_accessibility_response": T5R3_REF.mean_cosine_distance(z_ctx, np.concatenate(outs_ctx)),
            "z_mode_global_translation_response": T5R3_REF.mean_cosine_distance(z_bands, np.concatenate(outs_bands)),
            "centered_input_recomputed_after_translation": True}


def run_one(seed: int, threads: int, output: Path) -> None:
    global T5R3_REF
    started = time.perf_counter()
    torch.set_num_threads(threads)
    lock = verify_lock()
    T5R3 = load_module(ROOT / "scripts" / "run_t5r3_sanity.py", f"t5r3_for_m1_{seed}")
    T5R3_REF = T5R3
    R2 = load_module(ROOT / "scripts" / "run_t5r4_round2.py", f"r2_for_m1_{seed}")
    amr = load_module(ROOT / "scripts" / "amr_model.py", f"amr_for_m1_{seed}")
    sweep = load_module(ROOT / "scripts" / "run_p3_epsilon_sweep.py", f"sweep_for_m1_{seed}")
    sweep.EPSILONS = (0.25,)
    sweep.N_SNAPSHOTS = N_INTERVENTION_SNAPSHOTS
    T5R5 = load_module(ROOT / "scripts" / "run_t5r5_hidden_confirmation.py", f"t5r5_for_m1_{seed}")
    FC = load_module(ROOT / "scripts" / "p2_fracture_controls.py", f"fc_for_m1_{seed}")
    split_lock = json.loads((T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    train_adj = T5R3.knn_adjacency_batch(train.raw)
    valid_adj = T5R3.knn_adjacency_batch(valid.raw)

    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== AMR M1 seed={seed} threads={threads} ===", flush=True)
    T5R3.set_seed(seed)
    model = amr.AMRModel()
    model.set_route_alpha(torch.tensor(lock["routing_assignment"]["alpha"], dtype=torch.float32))
    params = int(sum(p.numel() for p in model.parameters()))
    training = train_m1(model, T5R3, R2, amr, train, train_adj, seed)
    ckpt = models_dir / f"amr_m1_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt)

    with torch.inference_mode():
        train_ctx = encode_ctx(model, train, train_adj)
        train_bands = encode_bands(model, train, train_adj)
        valid_ctx = encode_ctx(model, valid, valid_adj)
        valid_bands = encode_bands(model, valid, valid_adj)
        record = {"candidate_id": "amr_m1_fixed_routing", "seed": seed, "parameter_count": params,
                  "train": {"context": T5R3.prediction_metrics(train, model, train_ctx),
                            "intrinsic": T5R3.pair_metrics(train, model, train_bands),
                            "translation": translation_response(model, train, train_adj, train_ctx, train_bands)},
                  "valid": {"context": T5R3.prediction_metrics(valid, model, valid_ctx),
                            "intrinsic": T5R3.pair_metrics(valid, model, valid_bands),
                            "translation": translation_response(model, valid, valid_adj, valid_ctx, valid_bands)},
                  "cross_readout": "not_applicable_channel_separated_by_construction",
                  "training": training,
                  "checkpoint": {"path": str(ckpt.relative_to(ROOT)), "sha256": sha256_file(ckpt)}}
    v = record["valid"]
    print(f"seed={seed} valid zone={v['context']['field_zone_match_grouped_macro_f1']:.4f} "
          f"pair={v['intrinsic']['match_grouped_ranking_accuracy']:.4f} "
          f"z_mode_translation={v['translation']['z_mode_global_translation_response']:.4f}", flush=True)

    raw_all = np.concatenate([train.raw, valid.raw]).astype(np.float64)
    team_all = np.concatenate([train.team_slots, valid.team_slots])
    snap_ids = train.snapshot_ids.tolist() + valid.snapshot_ids.tolist()
    torch.set_num_threads(1)
    sets = sweep.build_sets(T5R3, T5R5, FC, raw_all, team_all, snap_ids)
    model2 = amr.AMRModel()
    model2.load_state_dict(torch.load(ROOT / record["checkpoint"]["path"], map_location="cpu", weights_only=True))
    model2.eval()
    record["intervention_dev"] = sweep.evaluate_model(model2, sets, FC, T5R3)
    record["n_intervention_sets"] = len(sets)
    record["lock_sha256_at_train"] = hashlib.sha256((P4 / "m1_config_lock.json").read_bytes()).hexdigest()
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
    summary = {"method": "amr_m1_fixed_routing", "lock": "artifacts/phase4_amr/m1_config_lock.json",
               "evidence_scope": "train/valid dev only", "n_intervention_sets": records[0]["n_intervention_sets"],
               "records": records}
    write_json(output / "summary.json", summary)
    manifest = {"status": "P4_N3_AMR_M1_DEV_COMPLETE", "seeds": list(SEEDS),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
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
