#!/usr/bin/env python3
"""Run the complete, bounded Phase 0 pipeline on one SkillCorner match.

The pipeline is intentionally a smoke-scale research artifact, not a claim
about the scientific phenomenon.  It creates canonical samples, matched
interventions, two small embedding views, and auditable Phase 0 diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.linalg import eigh


SEED = 20260827
DEFAULT_MATCH = "1886347"
N_SAMPLES = 1000
N_BANDS = 6
INTERVENTION_ENERGY = 1.0


@dataclass(frozen=True)
class Player:
    player_id: int
    team_id: int
    role: str


def finite_xy(player: dict[str, Any]) -> bool:
    try:
        return math.isfinite(float(player["x"])) and math.isfinite(float(player["y"]))
    except (KeyError, TypeError, ValueError):
        return False


def parse_players(match_json: Path) -> tuple[dict[int, Player], float, float]:
    payload = json.loads(match_json.read_text(encoding="utf-8"))
    players: dict[int, Player] = {}
    for raw in payload["players"]:
        role = str(raw.get("player_role", {}).get("position_group", "unknown"))
        players[int(raw["id"])] = Player(int(raw["id"]), int(raw["team_id"]), role)
    length = float(payload.get("pitch_length") or 105.0)
    width = float(payload.get("pitch_width") or 68.0)
    return players, length, width


def iter_tracking(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def count_player_coverage(path: Path, players: dict[int, Player]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in iter_tracking(path):
        for raw in row.get("player_data", []):
            pid = int(raw.get("player_id", -1))
            if pid in players and "goalkeeper" not in players[pid].role.lower() and finite_xy(raw):
                counts[pid] += 1
    return counts


def choose_nodes(players: dict[int, Player], coverage: Counter[int], n_per_team: int = 10) -> list[Player]:
    team_ids = sorted({p.team_id for p in players.values()})
    chosen: list[Player] = []
    for team_id in team_ids:
        candidates = [p for p in players.values() if p.team_id == team_id and "goalkeeper" not in p.role.lower()]
        candidates.sort(key=lambda p: (-coverage[p.player_id], p.player_id))
        chosen.extend(candidates[:n_per_team])
    if len(chosen) < n_per_team * 2:
        candidates = [p for p in players.values() if p not in chosen and "goalkeeper" not in p.role.lower()]
        candidates.sort(key=lambda p: (-coverage[p.player_id], p.player_id))
        chosen.extend(candidates[: n_per_team * 2 - len(chosen)])
    if len(chosen) < n_per_team * 2:
        raise RuntimeError(f"Could not find 20 outfield players; found {len(chosen)}")
    return sorted(chosen[: n_per_team * 2], key=lambda p: (p.team_id, p.player_id))


def collect_samples(
    tracking: Path,
    nodes: list[Player],
    pitch_length: float,
    pitch_width: float,
    limit: int = N_SAMPLES,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    node_ids = [p.player_id for p in nodes]
    rows: list[tuple[int, str | None, np.ndarray]] = []
    for row in iter_tracking(tracking):
        by_id = {int(p.get("player_id", -1)): p for p in row.get("player_data", [])}
        if not all(pid in by_id and finite_xy(by_id[pid]) for pid in node_ids):
            continue
        positions = np.array([[float(by_id[pid]["x"]), float(by_id[pid]["y"])] for pid in node_ids], dtype=np.float64)
        positions[:, 0] /= pitch_length / 2.0
        positions[:, 1] /= pitch_width / 2.0
        rows.append((int(row.get("frame", len(rows))), row.get("timestamp"), positions))
    if len(rows) < limit:
        raise RuntimeError(f"Only {len(rows)} valid frames available; need {limit}")
    indices = np.linspace(0, len(rows) - 1, limit, dtype=int)
    selected = [rows[int(i)] for i in indices]
    positions = np.stack([x[2] for x in selected])
    metadata = [{"frame": x[0], "timestamp": x[1]} for x in selected]
    return positions, metadata


def knn_adjacency(position: np.ndarray, k: int = 4) -> np.ndarray:
    diff = position[:, None, :] - position[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    nonzero = dist[dist > 0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    adjacency = np.zeros((len(position), len(position)), dtype=np.float64)
    for i in range(len(position)):
        neighbors = np.argsort(dist[i])[1 : k + 1]
        for j in neighbors:
            weight = math.exp(-float(dist[i, j] ** 2) / max(scale**2, 1e-12))
            adjacency[i, j] = max(adjacency[i, j], weight)
            adjacency[j, i] = max(adjacency[j, i], weight)
    return adjacency


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    return np.eye(len(adjacency)) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def equal_energy(delta: np.ndarray, energy: float = INTERVENTION_ENERGY) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize a zero or non-finite intervention")
    return delta * (energy / norm)


def band_ranges(n_modes: int, n_bands: int = N_BANDS) -> list[np.ndarray]:
    edges = np.linspace(0, n_modes, n_bands + 1, dtype=int)
    result: list[np.ndarray] = []
    for i in range(n_bands):
        start, end = int(edges[i]), int(edges[i + 1])
        if end <= start:
            end = min(n_modes, start + 1)
        result.append(np.arange(start, end, dtype=int))
    return result


def spectral_intervention(eigenvectors: np.ndarray, modes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    coeff = rng.normal(size=len(modes))
    scalar = eigenvectors[:, modes] @ coeff
    orientation = rng.normal(size=2)
    orientation /= np.linalg.norm(orientation)
    return equal_energy(scalar[:, None] * orientation[None, :])


def support_intervention(n_nodes: int, support: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    delta = np.zeros((n_nodes, 2), dtype=np.float64)
    delta[support] = rng.normal(size=(len(support), 2))
    return equal_energy(delta)


def intervention_row(
    sample_id: str,
    frame: int,
    kind: str,
    band_id: int | None,
    delta: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    target_modes: np.ndarray | None,
) -> dict[str, Any]:
    coeff = eigenvectors.T @ delta[:, 0]
    support = np.linalg.norm(delta, axis=1) > 1e-12
    rayleigh = float(coeff @ np.diag(eigenvalues) @ coeff / max(coeff @ coeff, 1e-12))
    purity = None
    if target_modes is not None:
        denom = float(np.sum(coeff**2))
        purity = float(np.sum(coeff[target_modes] ** 2) / max(denom, 1e-12))
    return {
        "sample_id": sample_id,
        "frame": frame,
        "intervention_id": f"{sample_id}:{kind}",
        "kind": kind,
        "band_id": band_id,
        "energy": float(np.linalg.norm(delta)),
        "energy_error": float(abs(np.linalg.norm(delta) - INTERVENTION_ENERGY)),
        "support_size": int(support.sum()),
        "support_mask": json.dumps(support.astype(int).tolist()),
        "spectral_coefficients": json.dumps(np.round(coeff, 10).tolist()),
        "rayleigh_quotient": rayleigh,
        "target_mode_purity": purity,
    }


def make_image(positions: np.ndarray, team_ids: np.ndarray, width: int = 256, height: int = 160) -> Image.Image:
    image = Image.new("RGB", (width, height), (42, 122, 58))
    draw = ImageDraw.Draw(image)
    margin = 8
    draw.rectangle((margin, margin, width - margin, height - margin), outline="white", width=2)
    draw.line((width // 2, margin, width // 2, height - margin), fill="white", width=1)
    draw.ellipse((width // 2 - 18, height // 2 - 18, width // 2 + 18, height // 2 + 18), outline="white", width=1)
    colors: dict[int, tuple[int, int, int]] = {}
    palette = [(220, 55, 55), (55, 80, 220), (245, 190, 55), (220, 80, 190)]
    for team in sorted(set(int(x) for x in team_ids)):
        colors[team] = palette[len(colors) % len(palette)]
    for (x, y), team in zip(positions, team_ids):
        px = margin + (float(x) + 1.0) * 0.5 * (width - 2 * margin)
        py = height - (margin + (float(y) + 1.0) * 0.5 * (height - 2 * margin))
        r = 4
        draw.ellipse((px - r, py - r, px + r, py + r), fill=colors[int(team)], outline="white")
    return image


def write_sample_grid(positions: np.ndarray, team_ids: np.ndarray, path: Path) -> None:
    selected = np.linspace(0, len(positions) - 1, 6, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, idx in zip(axes.flat, selected):
        ax.set_facecolor("#2a7a3a")
        ax.scatter(positions[idx, :, 0], positions[idx, :, 1], c=["#dc3737" if t == team_ids[0] else "#374fdc" for t in team_ids], s=35, edgecolors="white")
        ax.axvline(0, color="white", linewidth=0.7)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"sample {idx}, frame {idx}")
        ax.set_xlabel("normalized x")
        ax.set_ylabel("normalized y")
    fig.suptitle("Phase 0 canonical football samples")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train_coordinate_autoencoder(features: np.ndarray, out_dir: Path) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    x = torch.from_numpy(features.astype(np.float32))
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, x.shape[1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = torch.mean((model(x) - x) ** 2)
        loss.backward()
        optimizer.step()
    encoder = nn.Sequential(*list(model.children())[:4])
    encoder.eval()
    with torch.no_grad():
        embedding = encoder(x).numpy()
        reconstruction_mse = float(torch.mean((model(x) - x) ** 2).item())
    np.save(out_dir / "coordinate_embedding.npy", embedding)
    torch.save(model.state_dict(), out_dir / "coordinate_autoencoder.pt")
    return {"model": "tiny_coordinate_autoencoder", "embedding_dim": int(embedding.shape[1]), "reconstruction_mse": reconstruction_mse, "epochs": 40}


def frozen_visual_embedding(images: list[Image.Image], out_dir: Path, root: Path) -> dict[str, Any]:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    torch.hub.set_dir(str(root / ".cache" / "torch"))
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = weights.transforms()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            batch = torch.stack([transform(img) for img in images[start : start + 64]])
            batches.append(model(batch).numpy())
    embedding = np.concatenate(batches, axis=0)
    np.save(out_dir / "frozen_resnet18_embedding.npy", embedding)
    return {"model": "torchvision_resnet18", "weights": str(weights), "frozen": True, "embedding_dim": int(embedding.shape[1]), "n_images": len(images)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--match-id", default=DEFAULT_MATCH)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    root = args.root.resolve()
    match_dir = root / "data" / "raw" / "sports" / "skillcorner" / "data" / "matches" / args.match_id
    match_json = match_dir / f"{args.match_id}_match.json"
    tracking = match_dir / f"{args.match_id}_tracking_extrapolated.jsonl"
    if not match_json.exists() or not tracking.exists():
        raise FileNotFoundError(f"SkillCorner match files missing under {match_dir}")
    out_dir = root / "artifacts" / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)
    players, pitch_length, pitch_width = parse_players(match_json)
    coverage = count_player_coverage(tracking, players)
    nodes = choose_nodes(players, coverage)
    positions, frame_meta = collect_samples(tracking, nodes, pitch_length, pitch_width, args.n_samples)
    team_ids = np.array([p.team_id for p in nodes], dtype=np.int64)
    roles = np.array([p.role for p in nodes], dtype=object)
    node_ids = np.array([p.player_id for p in nodes], dtype=np.int64)

    adjacency = np.stack([knn_adjacency(p) for p in positions])
    eigenvalues = np.empty((len(positions), len(nodes)), dtype=np.float64)
    eigenvectors = np.empty((len(positions), len(nodes), len(nodes)), dtype=np.float64)
    intervention_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    ranges = band_ranges(len(nodes))
    semantic_support = np.array([i for i, p in enumerate(nodes) if p.team_id == nodes[0].team_id][:5], dtype=int)
    if len(semantic_support) < 5:
        semantic_support = np.arange(5, dtype=int)
    for i, (position, meta) in enumerate(zip(positions, frame_meta)):
        vals, vecs = eigh(normalized_laplacian(adjacency[i]))
        eigenvalues[i], eigenvectors[i] = vals, vecs
        sample_id = f"sc_{args.match_id}_{i:04d}"
        common = equal_energy(np.ones((len(nodes), 1)) @ np.array([[1.0, 0.0]]))
        single = np.zeros((len(nodes), 2), dtype=np.float64)
        single[i % len(nodes), 0] = INTERVENTION_ENERGY
        semantic = support_intervention(len(nodes), semantic_support, rng)
        random_support = rng.choice(len(nodes), size=len(semantic_support), replace=False)
        random_delta = support_intervention(len(nodes), random_support, rng)
        for kind, delta in [("common", common), ("single", single), ("semantic_coalition", semantic), ("random_coalition", random_delta)]:
            intervention_rows.append(intervention_row(sample_id, int(meta["frame"]), kind, None, delta, vecs, vals, None))
        for band_id, modes in enumerate(ranges):
            delta = spectral_intervention(vecs, modes, rng)
            intervention_rows.append(intervention_row(sample_id, int(meta["frame"]), f"band_{band_id}", band_id, delta, vecs, vals, modes))
            coeff = vecs.T @ delta[:, 0]
            sanity_rows.append({"sample_id": sample_id, "frame": int(meta["frame"]), "band_id": band_id, "eigenvalue_min": float(vals[modes].min()), "eigenvalue_max": float(vals[modes].max()), "energy": float(np.linalg.norm(delta)), "energy_error": float(abs(np.linalg.norm(delta) - 1.0)), "mode_purity": float(np.sum(coeff[modes] ** 2) / max(np.sum(coeff**2), 1e-12)), "graph_degree_mean": float(adjacency[i].sum(axis=1).mean())})

    np.savez_compressed(out_dir / "canonical_samples.npz", positions=positions, adjacency=adjacency, eigenvalues=eigenvalues, frame=np.array([m["frame"] for m in frame_meta]), node_ids=node_ids, team_ids=team_ids, node_mask=np.ones((len(positions), len(nodes)), dtype=np.uint8))
    with (out_dir / "canonical_samples.jsonl").open("w", encoding="utf-8") as handle:
        for i, meta in enumerate(frame_meta):
            handle.write(json.dumps({"sample_id": f"sc_{args.match_id}_{i:04d}", "domain": "football_tracking", "match_id": args.match_id, "frame": meta["frame"], "timestamp": meta["timestamp"], "node_ids": node_ids.tolist(), "team_ids": team_ids.tolist(), "roles": roles.tolist(), "pitch": {"length": pitch_length, "width": pitch_width}}, ensure_ascii=False) + "\n")
    pd.DataFrame(intervention_rows).to_parquet(out_dir / "intervention_manifest.parquet", index=False)
    pd.DataFrame(sanity_rows).to_csv(out_dir / "spectrum_sanity.csv", index=False)
    write_sample_grid(positions, team_ids, out_dir / "sample_grid.png")
    images = [make_image(positions[i], team_ids) for i in range(len(positions))]
    images[0].save(out_dir / "phase0_first_minimap.png")
    coord_meta = train_coordinate_autoencoder(positions.reshape(len(positions), -1), out_dir)
    visual_meta = frozen_visual_embedding(images, out_dir, root)
    report = {"status": "completed", "match_id": args.match_id, "n_samples": len(positions), "n_nodes": len(nodes), "n_interventions": len(intervention_rows), "n_bands": len(ranges), "energy_max_error": float(max(x["energy_error"] for x in intervention_rows)), "band_purity_min": float(min(x["mode_purity"] for x in sanity_rows)), "coordinate_embedding": coord_meta, "visual_embedding": visual_meta, "node_ids": node_ids.tolist(), "team_ids": team_ids.tolist(), "pitch_length": pitch_length, "pitch_width": pitch_width, "seed": args.seed}
    (out_dir / "pipeline_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
