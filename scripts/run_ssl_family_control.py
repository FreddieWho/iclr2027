#!/usr/bin/env python3
"""P3 SSL objective-family control (review Part 4, item D2).

Question: is the support-sensitive response pathology specific to the
contrastive (triplet-margin) intrinsic objective, or does it emerge under
a different self-supervised objective family?

Design: hold EVERYTHING fixed from the frozen 2:1 recipe — same TaskModel
architecture, same train data, same cyclic streams, same 280/140 update
quotas, same 80 epochs, same batch size, same seeds (11/23/47) — and swap
only the intrinsic loss: triplet margin (contrastive family) -> Barlow
Twins cross-correlation (redundancy-reduction family) on the same
(query, positive) views. Negatives are unused by construction.

Protocol status: this is a CONTROL experiment, not candidate selection.
Train on train split only; evaluate on train/valid only; no reserved or
external reads; no lock is modified. The selected model remains 2:1.

Outputs: artifacts/phase3/ssl_family_control_v1/ with checkpoints,
task metrics (valid), intervention metrics (frozen operator, eps=0.25),
summary.json, manifest.json, SHA256SUMS.
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
OUTPUT = ROOT / "artifacts" / "phase3" / "ssl_family_control_v1"
SEEDS = (11, 23, 47)
EPOCHS = 80
CONTEXT_QUOTA = 280
INTRINSIC_QUOTA = 140
BT_LAMBDA = 5e-3
N_INTERVENTION_SNAPSHOTS = 40
TORCH_THREADS = 8


def parse_args() -> Any:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="run a single seed (worker mode); omit with --merge")
    parser.add_argument("--merge", action="store_true", help="merge per-seed records into summary")
    parser.add_argument("--threads", type=int, default=TORCH_THREADS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


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


def barlow_twins_loss(model: Any, z: torch.Tensor, query_count: int) -> torch.Tensor:
    """Barlow Twins cross-correlation loss on mode embeddings of (query, positive)."""
    query_z, positive_z = z.split(query_count, dim=0)
    za = model.mode_embedding(query_z)
    zb = model.mode_embedding(positive_z)
    za = (za - za.mean(dim=0)) / (za.std(dim=0) + 1e-9)
    zb = (zb - zb.mean(dim=0)) / (zb.std(dim=0) + 1e-9)
    n = za.shape[0]
    c = (za.T @ zb) / n
    on_diag = (torch.diagonal(c) - 1.0).pow(2).sum()
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
    return on_diag + BT_LAMBDA * off_diag


def train_control(model: Any, T5R3: Any, R2: Any, train: Any, adjacency: np.ndarray, seed: int) -> dict[str, Any]:
    T5R3.set_seed(seed)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    batch = T5R3.tensors(train, adjacency)
    context_stream = R2.build_cyclic_stream(len(train.raw), seed * 100_000, CONTEXT_QUOTA, EPOCHS)
    pair_stream = R2.build_cyclic_stream(len(train.pair_query), seed * 200_000, INTRINSIC_QUOTA, EPOCHS)
    started = time.perf_counter()
    last = {"context": float("nan"), "intrinsic": float("nan")}
    for epoch in range(EPOCHS):
        model.train()
        ctx_total, ctx_seen = 0.0, 0
        for slot in range(CONTEXT_QUOTA):
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
        bt_total, bt_seen = 0.0, 0
        for slot in range(INTRINSIC_QUOTA):
            index = torch.from_numpy(pair_stream[epoch, slot])
            qi = batch["pair_query"][index]
            pi = batch["pair_positive"][index]
            both = torch.cat([qi, pi], dim=0)
            z = model.encode(batch["centered"][both], batch["team"][both], batch["adjacency"][both], "team_mean")
            loss = barlow_twins_loss(model, z, len(index))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bt_total += float(loss.detach()) * len(index)
            bt_seen += len(index)
        last = {"context": ctx_total / max(ctx_seen, 1), "intrinsic": bt_total / max(bt_seen, 1)}
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == EPOCHS:
            print(f"[barlow_twins_2to1] seed={seed} epoch={epoch + 1}/{EPOCHS} "
                  f"context={last['context']:.5f} bt={last['intrinsic']:.5f} "
                  f"elapsed={time.perf_counter() - started:.0f}s", flush=True)
    return {"epochs": EPOCHS, "context_updates_per_epoch": CONTEXT_QUOTA,
            "intrinsic_updates_per_epoch": INTRINSIC_QUOTA, "objective": "barlow_twins",
            "bt_lambda": BT_LAMBDA, "last_loss": last,
            "elapsed_seconds": round(time.perf_counter() - started, 1)}

def run_one_seed(seed: int, threads: int, output: Path) -> None:
    """Worker mode: train + evaluate a single seed, write record_seed{seed}.json."""
    started = time.perf_counter()
    torch.set_num_threads(threads)
    T5R3 = load_module(ROOT / "scripts" / "run_t5r3_sanity.py", "t5r3_for_ssl_control")
    R2 = load_module(ROOT / "scripts" / "run_t5r4_round2.py", "r2_for_ssl_control")
    sweep = load_module(ROOT / "scripts" / "run_p3_epsilon_sweep.py", "sweep_for_ssl_control")
    sweep.EPSILONS = (0.25,)
    sweep.N_SNAPSHOTS = N_INTERVENTION_SNAPSHOTS
    T5R5 = load_module(ROOT / "scripts" / "run_t5r5_hidden_confirmation.py", "t5r5_for_ssl_control")
    FC = load_module(ROOT / "scripts" / "p2_fracture_controls.py", "fc_for_ssl_control")
    split_lock = json.loads((T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    train_adj = T5R3.knn_adjacency_batch(train.raw)
    valid_adj = T5R3.knn_adjacency_batch(valid.raw)

    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== barlow_twins_2to1 seed={seed} threads={threads} ===", flush=True)
    T5R3.set_seed(seed)
    model = T5R3.TaskModel()
    params = int(sum(p.numel() for p in model.parameters()))
    training = train_control(model, T5R3, R2, train, train_adj, seed)
    ckpt = models_dir / f"barlow_twins_2to1_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt)
    spec = {"name": "barlow_twins_2to1", "context_view": "raw", "intrinsic_view": "centered", "pooling": "team_mean"}
    with torch.inference_mode():
        train_ctx = T5R3.encode_split(model, train, train_adj, "raw", "team_mean")
        train_int = T5R3.encode_split(model, train, train_adj, "centered", "team_mean")
        valid_ctx = T5R3.encode_split(model, valid, valid_adj, "raw", "team_mean")
        valid_int = T5R3.encode_split(model, valid, valid_adj, "centered", "team_mean")
        record = {"candidate_id": "barlow_twins_2to1", "seed": seed, "parameter_count": params,
                  "train": {"context": T5R3.prediction_metrics(train, model, train_ctx),
                            "intrinsic": T5R3.pair_metrics(train, model, train_int),
                            "translation": T5R3.translation_metrics(model, train, train_adj, spec, train_ctx, train_int)},
                  "valid": {"context": T5R3.prediction_metrics(valid, model, valid_ctx),
                            "intrinsic": T5R3.pair_metrics(valid, model, valid_int),
                            "translation": T5R3.translation_metrics(model, valid, valid_adj, spec, valid_ctx, valid_int),
                            "cross_readout": T5R3.cross_readout_metrics(valid, model, valid_ctx, valid_int)},
                  "training": training,
                  "checkpoint": {"path": str(ckpt.relative_to(ROOT)), "sha256": sha256_file(ckpt)}}
    print(f"seed={seed} valid zone={record['valid']['context']['field_zone_match_grouped_macro_f1']:.4f} "
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
    n_sets = records[0]["n_intervention_sets"]
    summary = {"control_id": "barlow_twins_2to1", "role": "ssl_objective_family_control_NOT_selection",
               "protocol": "train split only; train/valid eval only; no reserved/external reads",
               "n_intervention_sets": n_sets, "records": records}
    write_json(output / "summary.json", summary)
    manifest = {"status": "P3_SSL_FAMILY_CONTROL_COMPLETE", "objective": "barlow_twins",
                "frozen_recipe": "2:1 quotas/streams/epochs/seeds identical to t5r4_round2",
                "seeds": list(SEEDS), "models": 3,
                "parallelism": {"seed_workers": 3,
                                 "torch_threads_train_per_worker": records[0]["torch_threads_train"]},
                "reproducibility_note": "thread count fixed and recorded; CPU reduction order may vary with thread count",
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
    if args.seed is None:
        print("REFUSED: pass --seed N (worker) or --merge", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"record_seed{args.seed}.json").exists():
        print(f"REFUSED: record exists for seed {args.seed}", file=sys.stderr)
        return 2
    run_one_seed(args.seed, args.threads, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
