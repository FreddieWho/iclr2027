#!/usr/bin/env python3
"""P3 update-ratio sweep (review item 10e; user approved 2026-09-05).

Post-hoc ablation, NOT selection: the selected model remains 2:1. This
sweep characterizes the update-allocation variable by filling the missing
grid points 1:0 (420/0) and 1:1 (210/210) under the frozen recipe
(identical architecture, data, streams, epochs, batch, seeds, contrastive
pair loss). Train on train split only; evaluate on train/valid only; no
reserved/external reads; no lock is modified.

Parallel layout: one worker per (ratio, seed) -> 6 workers x 8 threads.
Outputs: artifacts/phase3/ratio_sweep_v1/{record_*.json, models/,
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

ROOT = Path(__file__).resolve().parents[1]
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
OUTPUT = ROOT / "artifacts" / "phase3" / "ratio_sweep_v1"
SEEDS = (11, 23, 47)
RATIOS = {"update_ratio_1to0": (420, 0), "update_ratio_1to1": (210, 210)}
EPOCHS = 80
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


def parse_args() -> Any:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", choices=sorted(RATIOS), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def train_ratio(model: Any, T5R3: Any, R2: Any, train: Any, adjacency: np.ndarray,
                seed: int, context_quota: int, intrinsic_quota: int) -> dict[str, Any]:
    T5R3.set_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    batch = T5R3.tensors(train, adjacency)
    context_stream = R2.build_cyclic_stream(len(train.raw), seed * 100_000, context_quota, EPOCHS) if context_quota else None
    pair_stream = R2.build_cyclic_stream(len(train.pair_query), seed * 200_000, intrinsic_quota, EPOCHS) if intrinsic_quota else None
    started = time.perf_counter()
    last = {"context": float("nan"), "intrinsic": float("nan")}
    for epoch in range(EPOCHS):
        model.train()
        ctx_total, ctx_seen = 0.0, 0
        for slot in range(context_quota):
            index = torch.from_numpy(context_stream[epoch, slot])
            z = model.encode(batch["raw"][index], batch["team"][index], batch["adjacency"][index], "team_mean")
            loss = T5R3.context_loss(model, z, {"phase": batch["phase"][index],
                                                "zone": batch["zone"][index],
                                                "centroids": batch["centroids"][index]})
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ctx_total += float(loss.detach()) * len(index)
            ctx_seen += len(index)
        pair_total, pair_seen = 0.0, 0
        for slot in range(intrinsic_quota):
            index = torch.from_numpy(pair_stream[epoch, slot])
            qi = batch["pair_query"][index]
            pi = batch["pair_positive"][index]
            ni = batch["pair_negative"][index]
            all_idx = torch.cat([qi, pi, ni], dim=0)
            z = model.encode(batch["centered"][all_idx], batch["team"][all_idx], batch["adjacency"][all_idx], "team_mean")
            loss = T5R3.pair_loss(model, z, len(index))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pair_total += float(loss.detach()) * len(index)
            pair_seen += len(index)
        last = {"context": ctx_total / max(ctx_seen, 1), "intrinsic": pair_total / max(pair_seen, 1)}
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == EPOCHS:
            print(f"seed={seed} epoch={epoch + 1}/{EPOCHS} context={last['context']:.5f} "
                  f"intrinsic={last['intrinsic']:.5f} elapsed={time.perf_counter() - started:.0f}s", flush=True)
    return {"epochs": EPOCHS, "context_updates_per_epoch": context_quota,
            "intrinsic_updates_per_epoch": intrinsic_quota, "objective": "contrastive_triplet",
            "last_loss": last, "elapsed_seconds": round(time.perf_counter() - started, 1)}


def run_one(ratio: str, seed: int, threads: int, output: Path) -> None:
    started = time.perf_counter()
    torch.set_num_threads(threads)
    T5R3 = load_module(ROOT / "scripts" / "run_t5r3_sanity.py", "t5r3_for_ratio_sweep")
    R2 = load_module(ROOT / "scripts" / "run_t5r4_round2.py", "r2_for_ratio_sweep")
    sweep = load_module(ROOT / "scripts" / "run_p3_epsilon_sweep.py", "sweep_for_ratio_sweep")
    sweep.EPSILONS = (0.25,)
    sweep.N_SNAPSHOTS = N_INTERVENTION_SNAPSHOTS
    T5R5 = load_module(ROOT / "scripts" / "run_t5r5_hidden_confirmation.py", "t5r5_for_ratio_sweep")
    FC = load_module(ROOT / "scripts" / "p2_fracture_controls.py", "fc_for_ratio_sweep")
    split_lock = json.loads((T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    train_adj = T5R3.knn_adjacency_batch(train.raw)
    valid_adj = T5R3.knn_adjacency_batch(valid.raw)
    context_quota, intrinsic_quota = RATIOS[ratio]

    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {ratio} ({context_quota}/{intrinsic_quota}) seed={seed} threads={threads} ===", flush=True)
    T5R3.set_seed(seed)
    model = T5R3.TaskModel()
    params = int(sum(p.numel() for p in model.parameters()))
    training = train_ratio(model, T5R3, R2, train, train_adj, seed, context_quota, intrinsic_quota)
    ckpt = models_dir / f"{ratio}_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt)
    spec = {"name": ratio, "context_view": "raw", "intrinsic_view": "centered", "pooling": "team_mean"}
    with torch.inference_mode():
        train_ctx = T5R3.encode_split(model, train, train_adj, "raw", "team_mean")
        train_int = T5R3.encode_split(model, train, train_adj, "centered", "team_mean")
        valid_ctx = T5R3.encode_split(model, valid, valid_adj, "raw", "team_mean")
        valid_int = T5R3.encode_split(model, valid, valid_adj, "centered", "team_mean")
        record = {"candidate_id": ratio, "seed": seed, "parameter_count": params,
                  "train": {"context": T5R3.prediction_metrics(train, model, train_ctx),
                            "intrinsic": T5R3.pair_metrics(train, model, train_int),
                            "translation": T5R3.translation_metrics(model, train, train_adj, spec, train_ctx, train_int)},
                  "valid": {"context": T5R3.prediction_metrics(valid, model, valid_ctx),
                            "intrinsic": T5R3.pair_metrics(valid, model, valid_int),
                            "translation": T5R3.translation_metrics(model, valid, valid_adj, spec, valid_ctx, valid_int),
                            "cross_readout": T5R3.cross_readout_metrics(valid, model, valid_ctx, valid_int)},
                  "training": training,
                  "checkpoint": {"path": str(ckpt.relative_to(ROOT)), "sha256": sha256_file(ckpt)}}
    print(f"{ratio} seed={seed} valid zone={record['valid']['context']['field_zone_match_grouped_macro_f1']:.4f} "
          f"pair={record['valid']['intrinsic']['match_grouped_ranking_accuracy']:.4f}", flush=True)

    raw_all = np.concatenate([train.raw, valid.raw]).astype(np.float64)
    team_all = np.concatenate([train.team_slots, valid.team_slots])
    snap_ids = train.snapshot_ids.tolist() + valid.snapshot_ids.tolist()
    torch.set_num_threads(1)
    sets = sweep.build_sets(T5R3, T5R5, FC, raw_all, team_all, snap_ids)
    model = T5R3.TaskModel()
    model.load_state_dict(torch.load(ROOT / record["checkpoint"]["path"], map_location="cpu", weights_only=True))
    model.eval()
    record["intervention_dev"] = sweep.evaluate_model(model, sets, FC, T5R3)
    record["n_intervention_sets"] = len(sets)
    record["torch_threads_train"] = threads
    rho = record["intervention_dev"]["0.25"].get("full_spearman")
    print(f"{ratio} seed={seed} intervention full rho={rho if rho is None else round(rho, 3)} "
          f"total={time.perf_counter() - started:.0f}s", flush=True)
    write_json(output / f"record_{ratio}_seed{seed}.json", record)


def merge(output: Path) -> int:
    records = []
    for ratio in RATIOS:
        for seed in SEEDS:
            path = output / f"record_{ratio}_seed{seed}.json"
            if not path.is_file():
                print(f"REFUSED: missing {path}", file=sys.stderr)
                return 2
            records.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {"sweep_id": "update_ratio_sweep", "role": "post_hoc_ablation_NOT_selection",
               "selected_model_remains": "update_ratio_2to1",
               "protocol": "train split only; train/valid eval only; no reserved/external reads",
               "n_intervention_sets": records[0]["n_intervention_sets"], "records": records}
    write_json(output / "summary.json", summary)
    manifest = {"status": "P3_RATIO_SWEEP_COMPLETE", "ratios": RATIOS, "seeds": list(SEEDS),
                "parallelism": {"workers": 6, "torch_threads_train_per_worker": records[0]["torch_threads_train"]},
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
    args = parse_args()
    output = args.output.resolve()
    if args.merge:
        return merge(output)
    if args.ratio is None or args.seed is None:
        print("REFUSED: pass --ratio R --seed N (worker) or --merge", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"record_{args.ratio}_seed{args.seed}.json").exists():
        print("REFUSED: record exists", file=sys.stderr)
        return 2
    run_one(args.ratio, args.seed, args.threads, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
