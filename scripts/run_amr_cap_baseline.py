#!/usr/bin/env python3
"""P4-N4 control: CAP/M0 baseline (docs/02_METHOD_SPEC_AMR.md section 8).

Matched-capacity control for AMR M1: SAME AMRModel architecture, SAME
update budget (210 ctx / 140 pair / 70 cap = 420/epoch), SAME seeds and
evaluation as run_amr_m1.py v2. Only the mechanism objective differs:
instead of spectral band routing, the model learns to predict the
intervention from the representation difference,

    L_CAP = || B (z_mode(x+delta) - z_mode(x)) - delta ||^2 / eps^2,

with B a restricted linear head (spec M0: verifies the minimal principle
"remember the difference; common modes fall into the null space").

Same whitened support-conditioned intervention sampling as M1 (identical
intervention distribution, isolating the objective as the only difference).
Train on train split only; dev evaluation only. No reserved/external reads.

Outputs: artifacts/phase4_amr/cap_m0/{record_seed*.json, models/,
summary.json, manifest.json, SHA256SUMS}.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "artifacts" / "phase4_amr"
M1 = None  # loaded in main

ROUTE_QUOTA_CAP = 70


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CAPModel(nn.Module):
    """AMRModel backbone + restricted linear CAP readout head."""

    def __init__(self, amr_module: Any) -> None:
        super().__init__()
        self.base = amr_module.AMRModel()
        self.cap_head = nn.Linear(amr_module.MODE_DIM, 40)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--output", type=Path, default=P4 / "cap_m0")
    args = parser.parse_args()
    output = args.output.resolve()
    m1 = load_module(ROOT / "scripts" / "run_amr_m1.py", "m1_for_cap")
    if args.merge:
        return m1.merge(output)
    if args.seed is None:
        print("REFUSED: pass --seed N (worker) or --merge", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"record_seed{args.seed}.json").exists():
        print("REFUSED: record exists", file=sys.stderr)
        return 2

    # Patch the M1 training loop: replace route_batch with the CAP objective.
    amr = load_module(ROOT / "scripts" / "amr_model.py", f"amr_for_cap_{args.seed}")
    T5R3 = load_module(ROOT / "scripts" / "run_t5r3_sanity.py", f"t5r3_for_cap_{args.seed}")
    R2 = load_module(ROOT / "scripts" / "run_t5r4_round2.py", f"r2_for_cap_{args.seed}")

    def cap_batch(model: CAPModel, batch: dict[str, torch.Tensor], snap_idx: np.ndarray,
                  band: int, support_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
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
        coeff = model.base.band_coeffs[band]
        filtered = amr.apply_chebyshev(lap, xi, coeff)
        norm = filtered.reshape(len(snap_idx), -1).norm(dim=1).clamp_min(1e-12)
        delta = m1.EPSILON * filtered / norm[:, None, None]
        z0 = model.base.mode_embedding(model.base.encode(centered, team, adj, "team_mean"))
        z1 = model.base.mode_embedding(model.base.encode(centered + delta, team, adj, "team_mean"))
        pred = model.cap_head(z1 - z0)
        target = delta.reshape(len(snap_idx), -1)
        l_cap = ((pred - target) / m1.EPSILON).pow(2).sum(dim=-1).mean()
        return {"loss": l_cap, "l_route": l_cap.detach(), "l_inv": torch.zeros(()),
                "l_eqv": l_cap.detach(), "l_var": torch.zeros(()), "l_orth": torch.zeros(()),
                "band": band, "support_size": support_size}

    # Reuse M1's run_one by monkey-patching model construction and route step.
    original_amr_load = m1.load_module

    def patched_train(model: CAPModel, T5R3_: Any, R2_: Any, amr_: Any, train: Any,
                      adjacency: np.ndarray, seed: int) -> dict[str, Any]:
        T5R3_.set_seed(seed)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
        batch = T5R3_.tensors(train, adjacency)
        ctx_stream = R2_.build_cyclic_stream(len(train.raw), seed * 100_000, m1.CTX_QUOTA, m1.EPOCHS)
        pair_stream = R2_.build_cyclic_stream(len(train.pair_query), seed * 200_000, m1.PAIR_QUOTA, m1.EPOCHS)
        rng = np.random.default_rng(seed * 300_000)
        import time as _time
        started = _time.perf_counter()
        for epoch in range(m1.EPOCHS):
            model.train()
            for slot in range(m1.CTX_QUOTA):
                index = torch.from_numpy(ctx_stream[epoch, slot])
                z = model.base.context_channel(batch["raw"][index], batch["team"][index], batch["adjacency"][index])
                loss = T5R3_.context_loss(model.base, z, {"phase": batch["phase"][index],
                                                          "zone": batch["zone"][index],
                                                          "centroids": batch["centroids"][index]})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for slot in range(m1.PAIR_QUOTA):
                index = torch.from_numpy(pair_stream[epoch, slot])
                qi, pi, ni = batch["pair_query"][index], batch["pair_positive"][index], batch["pair_negative"][index]
                all_idx = torch.cat([qi, pi, ni], dim=0)
                z = model.base.encode(batch["centered"][all_idx], batch["team"][all_idx], batch["adjacency"][all_idx], "team_mean")
                loss = T5R3_.pair_loss(model.base, z, len(index))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for _ in range(ROUTE_QUOTA_CAP):
                snap_idx = rng.integers(0, len(train.raw), size=m1.ROUTE_BATCH)
                band = int(rng.integers(0, 6))
                support_size = int(m1.SUPPORT_SIZES[rng.integers(0, len(m1.SUPPORT_SIZES))])
                out = cap_batch(model, batch, snap_idx, band, support_size, rng)
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                optimizer.step()
            if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == m1.EPOCHS:
                print(f"CAP seed={seed} epoch={epoch + 1}/{m1.EPOCHS} cap={float(out['l_eqv']):.4f} "
                      f"elapsed={_time.perf_counter() - started:.0f}s", flush=True)
        return {"epochs": m1.EPOCHS, "objective": "cap_m0_transformation_prediction",
                "quotas": {"context": m1.CTX_QUOTA, "pair": m1.PAIR_QUOTA, "cap": ROUTE_QUOTA_CAP},
                "last_loss": {"cap": float(out["l_eqv"])},
                "route_loss_by_band_support_train_mean": {},
                "elapsed_seconds": round(_time.perf_counter() - started, 1)}

    # Delegate to M1.run_one with patched pieces via a shim: easiest is to
    # replicate run_one here with model_cls and train_fn substituted.
    import hashlib
    import json
    import time
    torch.set_num_threads(args.threads)
    m1.verify_lock()
    m1.T5R3_REF = T5R3
    sweep = load_module(ROOT / "scripts" / "run_p3_epsilon_sweep.py", f"sweep_for_cap_{args.seed}")
    sweep.EPSILONS = (0.25,)
    sweep.N_SNAPSHOTS = m1.N_INTERVENTION_SNAPSHOTS
    T5R5 = load_module(ROOT / "scripts" / "run_t5r5_hidden_confirmation.py", f"t5r5_for_cap_{args.seed}")
    FC = load_module(ROOT / "scripts" / "p2_fracture_controls.py", f"fc_for_cap_{args.seed}")
    split_lock = json.loads((m1.T5R_ROOT / "split_lock.json").read_text(encoding="utf-8"))
    train = T5R3.load_split("train", split_lock)
    valid = T5R3.load_split("valid", split_lock)
    train_adj = T5R3.knn_adjacency_batch(train.raw)
    valid_adj = T5R3.knn_adjacency_batch(valid.raw)
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"=== CAP/M0 seed={args.seed} threads={args.threads} ===", flush=True)
    T5R3.set_seed(args.seed)
    model = CAPModel(amr)
    params = int(sum(p.numel() for p in model.parameters()))
    training = patched_train(model, T5R3, R2, amr, train, train_adj, args.seed)
    ckpt = models_dir / f"cap_m0_seed{args.seed}.pt"
    torch.save(model.state_dict(), ckpt)
    with torch.inference_mode():
        train_ctx = m1.encode_ctx(model.base, train, train_adj)
        train_bands = m1.encode_bands(model.base, train, train_adj)
        valid_ctx = m1.encode_ctx(model.base, valid, valid_adj)
        valid_bands = m1.encode_bands(model.base, valid, valid_adj)
        record = {"candidate_id": "cap_m0_transformation_prediction", "seed": args.seed,
                  "parameter_count": params,
                  "train": {"context": T5R3.prediction_metrics(train, model.base, train_ctx),
                            "intrinsic": T5R3.pair_metrics(train, model.base, train_bands),
                            "translation": m1.translation_response(model.base, train, train_adj, train_ctx, train_bands)},
                  "valid": {"context": T5R3.prediction_metrics(valid, model.base, valid_ctx),
                            "intrinsic": T5R3.pair_metrics(valid, model.base, valid_bands),
                            "translation": m1.translation_response(model.base, valid, valid_adj, valid_ctx, valid_bands)},
                  "cross_readout": "not_applicable_channel_separated_by_construction",
                  "training": training,
                  "checkpoint": {"path": str(ckpt.relative_to(ROOT)), "sha256": m1.sha256_file(ckpt)}}
    v = record["valid"]
    print(f"CAP seed={args.seed} valid zone={v['context']['field_zone_match_grouped_macro_f1']:.4f} "
          f"pair={v['intrinsic']['match_grouped_ranking_accuracy']:.4f}", flush=True)
    raw_all = np.concatenate([train.raw, valid.raw]).astype(np.float64)
    team_all = np.concatenate([train.team_slots, valid.team_slots])
    snap_ids = train.snapshot_ids.tolist() + valid.snapshot_ids.tolist()
    torch.set_num_threads(1)
    sets = sweep.build_sets(T5R3, T5R5, FC, raw_all, team_all, snap_ids)
    model2 = amr.AMRModel()
    model2.load_state_dict({k.replace("base.", "", 1): v for k, v in torch.load(ROOT / record["checkpoint"]["path"], map_location="cpu", weights_only=True).items() if k.startswith("base.")})
    model2.eval()
    record["intervention_dev"] = sweep.evaluate_model(model2, sets, FC, T5R3)
    record["n_intervention_sets"] = len(sets)
    record["lock_sha256_at_train"] = hashlib.sha256((P4 / "m1_config_lock.json").read_bytes()).hexdigest()
    record["torch_threads_train"] = args.threads
    rho = record["intervention_dev"]["0.25"].get("full_spearman")
    print(f"CAP seed={args.seed} intervention full rho={rho if rho is None else round(rho, 3)} "
          f"total={time.perf_counter() - started:.0f}s", flush=True)
    m1.write_json(output / f"record_seed{args.seed}.json", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
