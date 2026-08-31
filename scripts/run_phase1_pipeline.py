#!/usr/bin/env python3
"""Run the Phase 1 phenomenon probe on the local SkillCorner matches.

Phase 0 created interventions but did not measure representation responses.  This
script adds that missing link: match-level data splits, matched interventions,
coordinate/phase models, an optional frozen-vision robustness branch, paired
response metrics, and a C1 exploration report.  It is intentionally
exploratory: a successful run means that the measurements are available, not
that H1/H2 hold.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.linalg import eigh


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw" / "sports" / "skillcorner" / "data" / "matches"
OUT = ROOT / "artifacts" / "phase1"
REPORT = ROOT / "reports" / "EXPLORATION_CHECKPOINT_1.md"
MATCH_IDS = [
    "1886347",
    "1899585",
    "1925299",
    "1953632",
    "1996435",
    "2006229",
    "2011166",
    "2013725",
    "2015213",
    "2017461",
]
MATCH_SPLIT = {
    "1886347": "train",
    "1899585": "train",
    "1925299": "train",
    "1953632": "train",
    "1996435": "train",
    "2006229": "train",
    "2011166": "dev",
    "2013725": "dev",
    "2015213": "heldout",
    "2017461": "heldout",
}
SEEDS = [11, 23, 47]
N_BANDS = 6
PRIMARY_EPSILON = 1.0
ENERGIES = [0.25, 0.5, 1.0, 2.0]
SUPPORT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
POINT_MODEL_IDS = {f"{family}_seed{seed}" for family in ("deepsets_ae", "gat_ae", "phase_gat") for seed in SEEDS}
VISION_MODEL_IDS = {"dinov2_vits14", "openclip_vit_b32"}


def build_execution_contract(
    n_samples_per_match: int,
    visual_stride: int,
    skip_vision: bool,
    point_only: bool,
) -> dict[str, Any]:
    """Describe the actual run scope so modality changes cannot be implicit."""
    return {
        "phase": "P1_PHENOMENON",
        "checkpoint": "C1",
        "study_mode": "exploratory_discovery",
        "data_domain": "football_tracking_only",
        "match_ids": MATCH_IDS,
        "match_split": MATCH_SPLIT,
        "n_samples_per_match": n_samples_per_match,
        "seeds": SEEDS,
        "energies": ENERGIES,
        "primary_energy": PRIMARY_EPSILON,
        "n_bands": N_BANDS,
        "exact_mode_policy": "seed-11-primary-energy-only-for-accessibility",
        "mainline_modality": "coordinate_point_set" if point_only else "coordinate_point_set_with_optional_minimap_branch",
        "vision_policy": "deferred_auxiliary" if point_only else "optional_deterministic_minimap_robustness",
        "visual_stride": visual_stride,
        "skip_vision": bool(skip_vision or point_only),
        "point_only": point_only,
        "graph_policy": "baseline_fixed_weighted_knn4",
        "preprocessing_fit_split": "train_matches_only",
    }


def load_phase0_module():
    path = ROOT / "scripts" / "run_phase0_pipeline.py"
    spec = importlib.util.spec_from_file_location("phase0_for_phase1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P0 = load_phase0_module()


@dataclass
class Sample:
    sample_id: str
    match_id: str
    split: str
    frame: int
    timestamp: str | None
    positions: np.ndarray
    raw_positions: np.ndarray
    centered_positions: np.ndarray
    adjacency: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    team_slots: np.ndarray
    roles: list[str]
    role_groups: list[str]
    phase_label: str | None
    phase_status: str
    pitch_length: float
    pitch_width: float


def stable_int(*parts: object) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(model: Any) -> str:
    """Hash a loaded frozen model without writing a second checkpoint file."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def finite_xy(player: dict[str, Any]) -> bool:
    try:
        return math.isfinite(float(player["x"])) and math.isfinite(float(player["y"]))
    except (KeyError, TypeError, ValueError):
        return False


def iter_tracking(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_role(role: str) -> str:
    value = role.lower().replace("_", " ").replace("-", " ")
    if "goal" in value:
        return "goalkeeper"
    if any(token in value for token in ["defend", "back", "centre back", "center back"]):
        return "defender"
    if any(token in value for token in ["midfield", "wing back"]):
        return "midfielder"
    if any(token in value for token in ["forward", "striker", "attack", "winger"]):
        return "forward"
    return "unknown"


def read_match_players(match_id: str) -> tuple[dict[int, P0.Player], float, float]:
    match_dir = DATA_ROOT / match_id
    payload = json.loads((match_dir / f"{match_id}_match.json").read_text(encoding="utf-8"))
    players: dict[int, P0.Player] = {}
    for raw in payload["players"]:
        role = str(raw.get("player_role", {}).get("position_group", "unknown"))
        players[int(raw["id"])] = P0.Player(int(raw["id"]), int(raw["team_id"]), role)
    return players, float(payload.get("pitch_length") or 105.0), float(payload.get("pitch_width") or 68.0)


def choose_nodes_and_collect(
    match_id: str,
    n_samples: int,
) -> tuple[list[P0.Player], float, float, list[dict[str, Any]], pd.DataFrame]:
    match_dir = DATA_ROOT / match_id
    players, pitch_length, pitch_width = read_match_players(match_id)
    tracking = match_dir / f"{match_id}_tracking_extrapolated.jsonl"
    candidate_ids = [pid for pid, player in sorted(players.items()) if "goalkeeper" not in player.role.lower()]
    candidate_index = {pid: index for index, pid in enumerate(candidate_ids)}
    coverage: Counter[int] = Counter()
    compact_rows: list[tuple[int, str | None, np.ndarray]] = []
    for row in iter_tracking(tracking):
        coordinates = np.full((len(candidate_ids), 2), np.nan, dtype=np.float32)
        for raw in row.get("player_data", []):
            pid = int(raw.get("player_id", -1))
            if pid in candidate_index and finite_xy(raw):
                coverage[pid] += 1
                coordinates[candidate_index[pid]] = (float(raw["x"]), float(raw["y"]))
        compact_rows.append((int(row.get("frame", len(compact_rows))), row.get("timestamp"), coordinates))
    nodes = P0.choose_nodes(players, coverage)
    node_ids = [p.player_id for p in nodes]
    node_columns = [candidate_index[pid] for pid in node_ids]
    rows: list[dict[str, Any]] = []
    for frame, timestamp, coordinates in compact_rows:
        selected_coordinates = coordinates[node_columns]
        if not np.all(np.isfinite(selected_coordinates)):
            continue
        raw_positions = selected_coordinates.astype(np.float64)
        positions = raw_positions.copy()
        positions[:, 0] /= pitch_length / 2.0
        positions[:, 1] /= pitch_width / 2.0
        # Keep the canonical renderer's fixed pitch view explicit. Tracking
        # extrapolation occasionally places a player a few centimetres beyond
        # the nominal pitch; such a frame is not a valid baseline observation.
        if not np.all(np.abs(positions) <= 1.0 + 1e-9):
            continue
        rows.append({"frame": frame, "timestamp": timestamp, "raw_positions": raw_positions, "positions": positions})
    if len(rows) < n_samples:
        raise RuntimeError(f"{match_id}: only {len(rows)} complete frames; need {n_samples}")
    indices = np.linspace(0, len(rows) - 1, n_samples, dtype=int)
    phase_path = match_dir / f"{match_id}_phases_of_play.csv"
    phases = pd.read_csv(phase_path, usecols=["frame_start", "frame_end", "team_out_of_possession_phase_type"])
    selected = [rows[int(i)] for i in indices]
    selected_meta: list[dict[str, Any]] = []
    for item in selected:
        frame = int(item["frame"])
        hits = phases[(phases["frame_start"] <= frame) & (phases["frame_end"] >= frame)]
        labels = [str(x) for x in hits["team_out_of_possession_phase_type"].dropna().tolist()]
        if len(labels) == 1:
            phase_label, phase_status = labels[0], "unique"
        elif len(labels) == 0:
            phase_label, phase_status = None, "missing"
        else:
            phase_label, phase_status = None, "ambiguous"
        selected_meta.append({**item, "phase_label": phase_label, "phase_status": phase_status})
    return nodes, pitch_length, pitch_width, selected_meta, phases


def collect_match_for_pool(args: tuple[str, int]) -> tuple[list[P0.Player], float, float, list[dict[str, Any]], pd.DataFrame]:
    """Process-pool entry point; each match remains an independent split unit."""
    return choose_nodes_and_collect(*args)


def split_rows() -> list[dict[str, str]]:
    return [{"match_id": m, "split": MATCH_SPLIT[m]} for m in MATCH_IDS]


def build_samples(n_samples_per_match: int) -> list[Sample]:
    samples: list[Sample] = []
    jobs = [(match_id, n_samples_per_match) for match_id in MATCH_IDS]
    max_workers = min(4, len(jobs))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        collected = executor.map(collect_match_for_pool, jobs)
        collected_by_match = dict(zip(MATCH_IDS, collected))
    for match_id in MATCH_IDS:
        nodes, pitch_length, pitch_width, selected, _ = collected_by_match[match_id]
        team_ids = np.array([p.team_id for p in nodes], dtype=np.int64)
        team_order = {team: i for i, team in enumerate(sorted(set(team_ids.tolist())))}
        team_slots = np.array([team_order[int(team)] for team in team_ids], dtype=np.int64)
        roles = [p.role for p in nodes]
        role_groups = [normalize_role(p.role) for p in nodes]
        for index, meta in enumerate(selected):
            positions = np.asarray(meta["positions"], dtype=np.float64)
            centered = positions - positions.mean(axis=0, keepdims=True)
            adjacency = P0.knn_adjacency(positions, k=4)
            laplacian = P0.normalized_laplacian(adjacency)
            eigenvalues, eigenvectors = eigh(laplacian)
            samples.append(
                Sample(
                    sample_id=f"sc_{match_id}_{index:04d}",
                    match_id=match_id,
                    split=MATCH_SPLIT[match_id],
                    frame=int(meta["frame"]),
                    timestamp=None if meta["timestamp"] is None else str(meta["timestamp"]),
                    positions=positions,
                    raw_positions=np.asarray(meta["raw_positions"], dtype=np.float64),
                    centered_positions=centered,
                    adjacency=adjacency,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    team_slots=team_slots.copy(),
                    roles=roles.copy(),
                    role_groups=role_groups.copy(),
                    phase_label=meta["phase_label"],
                    phase_status=meta["phase_status"],
                    pitch_length=pitch_length,
                    pitch_width=pitch_width,
                )
            )
    return samples


def equal_energy(delta: np.ndarray, energy: float) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("zero or non-finite intervention")
    return delta * (energy / norm)


def random_orientation(rng: np.random.Generator) -> np.ndarray:
    orientation = rng.normal(size=2)
    norm = float(np.linalg.norm(orientation))
    if norm <= 1e-12:
        return np.array([1.0, 0.0])
    return orientation / norm


def role_support(sample: Sample, team_slot: int, role_group: str) -> np.ndarray:
    return np.array([i for i, (team, role) in enumerate(zip(sample.team_slots, sample.role_groups)) if team == team_slot and role == role_group], dtype=int)


def choose_semantic_support(sample: Sample, team_slot: int) -> tuple[str, np.ndarray]:
    for role_group in ["defender", "midfielder", "forward"]:
        support = role_support(sample, team_slot, role_group)
        if len(support) >= 2:
            return role_group, support
    candidates = [(role, role_support(sample, team_slot, role)) for role in sorted(set(sample.role_groups))]
    candidates = [(role, support) for role, support in candidates if len(support) >= 2]
    if not candidates:
        return "unknown", np.array([], dtype=int)
    role, support = max(candidates, key=lambda pair: len(pair[1]))
    return role, support


def role_order(sample: Sample) -> list[int]:
    priority = {"defender": 0, "midfielder": 1, "forward": 2, "unknown": 3, "goalkeeper": 4}
    return sorted(range(len(sample.positions)), key=lambda i: (priority.get(sample.role_groups[i], 3), int(sample.team_slots[i]), i))


def intervention_record(
    sample: Sample,
    seed: int,
    epsilon: float,
    kind: str,
    delta: np.ndarray | None,
    target_modes: np.ndarray | None,
    support_type: str,
    support_indices: np.ndarray | None,
    coalition_id: str | None,
    attempts: int,
    visual_eval: bool,
) -> dict[str, Any]:
    valid = delta is not None
    if valid:
        delta = np.asarray(delta, dtype=np.float64)
        coeff = sample.eigenvectors.T @ delta[:, 0]
        energy = float(np.linalg.norm(delta))
        energy_error = abs(energy - epsilon)
        support = np.linalg.norm(delta, axis=1) > 1e-12
        rayleigh = float(coeff @ np.diag(sample.eigenvalues) @ coeff / max(coeff @ coeff, 1e-12))
        purity = None if target_modes is None else float(np.sum(coeff[target_modes] ** 2) / max(np.sum(coeff**2), 1e-12))
        delta_json = json.dumps(np.round(delta, 10).tolist(), separators=(",", ":"))
        coeff_json = json.dumps(np.round(coeff, 10).tolist(), separators=(",", ":"))
        support_json = json.dumps(np.where(support)[0].astype(int).tolist(), separators=(",", ":"))
    else:
        energy = energy_error = rayleigh = purity = None
        delta_json = coeff_json = support_json = None
    mode_rank = None if target_modes is None or len(target_modes) != 1 else int(target_modes[0])
    band_id = None
    if kind.startswith("band_"):
        band_id = int(kind.split("_")[-1])
    id_kind = kind if coalition_id is None else f"{kind}:{coalition_id}"
    intervention_id = f"{sample.sample_id}:{id_kind}:eps{epsilon:g}:seed{seed}"
    return {
        "intervention_id": intervention_id,
        "observation_id": sample.sample_id,
        "observation_key": f"{sample.match_id}:{sample.sample_id}",
        "pair_id": intervention_id,
        "pair_role": "perturbed",
        "source_sample_id": sample.sample_id,
        "sample_index": None,
        "match_id": sample.match_id,
        "split": sample.split,
        "frame": sample.frame,
        "seed": int(seed),
        "kind": kind,
        "mode_rank": mode_rank,
        "band_id": band_id,
        "target_modes": None if target_modes is None else json.dumps(np.asarray(target_modes, dtype=int).tolist()),
        "epsilon": float(epsilon),
        "energy": energy,
        "energy_error": energy_error,
        "support_type": support_type,
        "support_indices": None if support_indices is None else json.dumps(np.asarray(support_indices, dtype=int).tolist()),
        "support_size": None if support_indices is None else int(len(support_indices)),
        "coalition_id": coalition_id,
        "delta": delta_json,
        "spectral_coefficients": coeff_json,
        "rayleigh_quotient": rayleigh,
        "target_mode_purity": purity,
        "boundary_status": "valid" if valid else "invalid_after_retries",
        "valid": bool(valid),
        "rejection_attempts": int(attempts),
        "visual_eval": bool(visual_eval),
    }


def generate_interventions(samples: list[Sample], visual_stride: int = 1) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        visual_eval = sample_index % max(1, visual_stride) == 0
        bands = P0.band_ranges(len(sample.eigenvalues), N_BANDS)
        semantic_specs: list[tuple[str, np.ndarray]] = []
        for team_slot in [0, 1]:
            role_group, support = choose_semantic_support(sample, team_slot)
            if len(support) >= 2:
                semantic_specs.append((f"team{team_slot}_{role_group}", support))
        sweep_order = role_order(sample)
        identity_id = f"{sample.sample_id}:identity:eps0:seed0"
        identity = intervention_record(
            sample,
            seed=0,
            epsilon=0.0,
            kind="identity",
            delta=np.zeros_like(sample.positions),
            target_modes=None,
            support_type="none",
            support_indices=np.array([], dtype=int),
            coalition_id=None,
            attempts=1,
            visual_eval=visual_eval,
        )
        identity["intervention_id"] = identity_id
        identity["pair_id"] = identity_id
        identity["sample_index"] = sample_index
        rows.append(identity)
        for seed in SEEDS:
            for epsilon in ENERGIES:
                kinds: list[tuple[str, Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray | None, str, np.ndarray | None, str | None]]]] = []

                def common_fn(rng: np.random.Generator):
                    return np.ones((len(sample.positions), 1)) @ random_orientation(rng)[None, :], None, "global", np.arange(len(sample.positions)), None

                def single_fn(rng: np.random.Generator):
                    support = np.array([int(rng.integers(0, len(sample.positions)))], dtype=int)
                    delta = np.zeros_like(sample.positions)
                    delta[support] = random_orientation(rng)
                    return delta, None, "single", support, None

                kinds.extend([("common", common_fn), ("single", single_fn)])
                # Exact modes are the accessibility probe. One fixed seed and
                # the primary energy are sufficient for this diagnostic; all
                # seeds/energies remain represented by the band transfer grid.
                if seed == SEEDS[0] and epsilon == PRIMARY_EPSILON:
                    for mode_rank in range(len(sample.eigenvalues)):
                        def exact_mode_fn(rng: np.random.Generator, mode_rank=mode_rank):
                            modes = np.array([mode_rank], dtype=int)
                            return P0.spectral_intervention(sample.eigenvectors, modes, rng), modes, "spectral_exact", np.arange(len(sample.positions)), None

                        kinds.append((f"exact_mode_{mode_rank}", exact_mode_fn))
                for band_id, modes in enumerate(bands):
                    def band_fn(rng: np.random.Generator, modes=modes):
                        return P0.spectral_intervention(sample.eigenvectors, modes, rng), modes, "spectral", np.arange(len(sample.positions)), None

                    kinds.append((f"band_{band_id}", band_fn))
                for coalition_name, support in semantic_specs:
                    def semantic_fn(rng: np.random.Generator, support=support, coalition_name=coalition_name):
                        return P0.support_intervention(len(sample.positions), support, rng), None, "semantic", support, coalition_name

                    def random_fn(rng: np.random.Generator, support=support, coalition_name=coalition_name):
                        random_support = rng.choice(len(sample.positions), size=len(support), replace=False)
                        if np.array_equal(np.sort(random_support), np.sort(support)) and len(random_support) < len(sample.positions):
                            random_support = np.roll(random_support, 1)
                        return P0.support_intervention(len(sample.positions), random_support, rng), None, "random_matched_size", random_support, coalition_name

                    kinds.extend([("semantic_coalition", semantic_fn), ("random_coalition", random_fn)])
                for kind, builder in kinds:
                    rng = np.random.default_rng(stable_int(sample.sample_id, kind, seed, epsilon))
                    chosen: tuple[np.ndarray, np.ndarray | None, str, np.ndarray | None, str | None] | None = None
                    attempts = 0
                    for attempts in range(1, 33):
                        candidate = builder(rng)
                        delta = equal_energy(candidate[0], epsilon)
                        if np.all(np.abs(sample.positions + delta) <= 1.0 + 1e-9):
                            chosen = (delta, candidate[1], candidate[2], candidate[3], candidate[4])
                            break
                    if chosen is None:
                        candidate = builder(rng)
                        chosen = (None, candidate[1], candidate[2], candidate[3], candidate[4])
                    row_visual_eval = visual_eval and not kind.startswith("exact_mode_")
                    row = intervention_record(sample, seed, epsilon, kind, chosen[0], chosen[1], chosen[2], chosen[3], chosen[4], attempts, row_visual_eval)
                    row["sample_index"] = sample_index
                    rows.append(row)
            if seed == SEEDS[0]:
                for fraction in SUPPORT_FRACTIONS:
                    k = max(1, int(round(fraction * len(sample.positions))))
                    semantic_support = np.array(sweep_order[:k], dtype=int)
                    for support_kind, support in [("role_ordered", semantic_support)]:
                        kind = f"support_sweep_{fraction:g}"
                        rng = np.random.default_rng(stable_int(sample.sample_id, kind, seed))
                        chosen_delta = None
                        attempts = 0
                        for attempts in range(1, 33):
                            candidate = P0.support_intervention(len(sample.positions), support, rng)
                            if np.all(np.abs(sample.positions + candidate) <= 1.0 + 1e-9):
                                chosen_delta = candidate
                                break
                        row = intervention_record(sample, seed, PRIMARY_EPSILON, kind, chosen_delta, None, support_kind, support, None, attempts, visual_eval)
                        row["sample_index"] = sample_index
                        rows.append(row)
    return pd.DataFrame(rows)


def validate_interventions(samples: list[Sample], interventions: pd.DataFrame) -> dict[str, Any]:
    sample_map = {sample.sample_id: sample for sample in samples}
    errors: list[str] = []
    if interventions["intervention_id"].duplicated().any():
        errors.append("duplicate intervention_id")
    if not set(interventions["source_sample_id"]).issubset(sample_map):
        errors.append("intervention references unknown sample")
    pair_roles = set(interventions["pair_role"].dropna().astype(str))
    if pair_roles != {"perturbed"}:
        errors.append(f"unexpected pair roles: {sorted(pair_roles)}")
    boundary_failures = 0
    boundary_ids: list[str] = []
    energy_failures = 0
    for row in interventions.itertuples(index=False):
        if not bool(row.valid):
            continue
        sample = sample_map[str(row.source_sample_id)]
        delta = np.asarray(json.loads(row.delta), dtype=float)
        if not np.all(np.isfinite(delta)) or not np.all(np.abs(sample.positions + delta) <= 1.0 + 1e-9):
            boundary_failures += 1
            if len(boundary_ids) < 10:
                boundary_ids.append(str(row.intervention_id))
        if float(row.epsilon) > 0 and float(row.energy_error) > 1e-7:
            energy_failures += 1
    if boundary_failures:
        errors.append(f"{boundary_failures} valid rows exceed the pitch boundary: {boundary_ids}")
    if energy_failures:
        errors.append(f"{energy_failures} valid rows exceed the energy tolerance")
    valid = interventions["valid"].astype(bool)
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "n_rows": int(len(interventions)),
        "n_valid": int(valid.sum()),
        "n_invalid": int((~valid).sum()),
        "n_identity": int((interventions["kind"] == "identity").sum()),
        "max_energy_error": float(interventions.loc[valid, "energy_error"].max()) if valid.any() else None,
    }


def save_samples(samples: list[Sample], interventions: pd.DataFrame) -> None:
    positions = np.stack([s.positions for s in samples])
    raw_positions = np.stack([s.raw_positions for s in samples])
    centered = np.stack([s.centered_positions for s in samples])
    adjacency = np.stack([s.adjacency for s in samples])
    eigenvalues = np.stack([s.eigenvalues for s in samples])
    eigenvectors = np.stack([s.eigenvectors for s in samples])
    team_slots = np.stack([s.team_slots for s in samples])
    np.savez_compressed(
        OUT / "canonical_samples.npz",
        positions=positions,
        raw_positions=raw_positions,
        centered_positions=centered,
        adjacency=adjacency,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        team_slots=team_slots,
        node_mask=np.ones((len(samples), positions.shape[1]), dtype=np.uint8),
    )
    with (OUT / "canonical_samples.jsonl").open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "sample_index": index,
                        "domain": "football_tracking",
                        "match_id": sample.match_id,
                        "split": sample.split,
                        "frame": sample.frame,
                        "timestamp": sample.timestamp,
                        "roles": sample.roles,
                        "role_groups": sample.role_groups,
                        "team_slots": sample.team_slots.tolist(),
                        "phase_label": sample.phase_label,
                        "phase_status": sample.phase_status,
                        "pitch": {"length": sample.pitch_length, "width": sample.pitch_width},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    split_df = pd.DataFrame(split_rows())
    split_df.to_csv(OUT / "split_manifest.csv", index=False)
    interventions.to_parquet(OUT / "intervention_manifest.parquet", index=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def team_pool(h, team_slots):
    import torch

    pools = []
    for team_slot in [0, 1]:
        mask = (team_slots == team_slot).float().unsqueeze(-1)
        pools.append((h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))
    return torch.cat(pools, dim=-1)


def build_torch_models():
    import torch
    from torch import nn

    class DeepSetsAE(nn.Module):
        def __init__(self, n_nodes: int, latent_dim: int = 128):
            super().__init__()
            self.node = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.project = nn.Sequential(nn.Linear(256, latent_dim), nn.ReLU())
            self.slot = nn.Embedding(n_nodes, 16)
            self.decoder = nn.Sequential(nn.Linear(latent_dim + 16 + 2, 128), nn.ReLU(), nn.Linear(128, 2))

        def encode(self, positions, team_slots):
            features = torch.cat([positions, torch.nn.functional.one_hot(team_slots.long(), 2).float()], dim=-1)
            h = self.node(features)
            return self.project(team_pool(h, team_slots))

        def forward(self, positions, team_slots):
            z = self.encode(positions, team_slots)
            slots = self.slot.weight.unsqueeze(0).expand(len(positions), -1, -1)
            team_onehot = torch.nn.functional.one_hot(team_slots.long(), 2).float()
            recon = self.decoder(torch.cat([z.unsqueeze(1).expand(-1, positions.shape[1], -1), slots, team_onehot], dim=-1))
            return z, recon

    class GraphAttentionLayer(nn.Module):
        def __init__(self, hidden_dim: int = 128, head_dim: int = 32):
            super().__init__()
            self.query = nn.Linear(hidden_dim, head_dim, bias=False)
            self.key = nn.Linear(hidden_dim, head_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.self_proj = nn.Linear(hidden_dim, hidden_dim)
            self.neigh_proj = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, h, adjacency):
            query = self.query(h)
            key = self.key(h)
            scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
            mask = adjacency > 0
            # The graph is fixed from the baseline observation. Edge weights
            # affect attention as a deterministic prior, not as a learned graph.
            scores = scores.masked_fill(~mask, -1e4)
            scores = scores + torch.log(adjacency.clamp_min(1e-8))
            attention = torch.softmax(scores, dim=-1)
            neighbors = torch.bmm(attention, self.value(h))
            return torch.relu(self.self_proj(h) + self.neigh_proj(neighbors))

    class GraphEncoder(nn.Module):
        def __init__(self, latent_dim: int = 128):
            super().__init__()
            self.in_proj = nn.Linear(4, 128)
            self.layers = nn.ModuleList([GraphAttentionLayer(), GraphAttentionLayer()])
            self.project = nn.Sequential(nn.Linear(256, latent_dim), nn.ReLU())

        def forward_nodes(self, positions, team_slots, adjacency):
            features = torch.cat([positions, torch.nn.functional.one_hot(team_slots.long(), 2).float()], dim=-1)
            h = torch.relu(self.in_proj(features))
            for layer in self.layers:
                h = layer(h, adjacency)
            return h

        def encode(self, positions, team_slots, adjacency):
            return self.project(team_pool(self.forward_nodes(positions, team_slots, adjacency), team_slots))

    class GATAE(nn.Module):
        def __init__(self, n_nodes: int, latent_dim: int = 128):
            super().__init__()
            self.encoder = GraphEncoder(latent_dim)
            self.slot = nn.Embedding(n_nodes, 16)
            self.decoder = nn.Sequential(nn.Linear(latent_dim + 16 + 2, 128), nn.ReLU(), nn.Linear(128, 2))

        def encode(self, positions, team_slots, adjacency):
            return self.encoder.encode(positions, team_slots, adjacency)

        def forward(self, positions, team_slots, adjacency):
            z = self.encode(positions, team_slots, adjacency)
            slots = self.slot.weight.unsqueeze(0).expand(len(positions), -1, -1)
            team_onehot = torch.nn.functional.one_hot(team_slots.long(), 2).float()
            recon = self.decoder(torch.cat([z.unsqueeze(1).expand(-1, positions.shape[1], -1), slots, team_onehot], dim=-1))
            return z, recon

    class PhaseGAT(nn.Module):
        def __init__(self, n_classes: int, latent_dim: int = 128):
            super().__init__()
            self.encoder = GraphEncoder(latent_dim)
            self.head = nn.Linear(latent_dim, n_classes)

        def encode(self, positions, team_slots, adjacency):
            return self.encoder.encode(positions, team_slots, adjacency)

        def forward(self, positions, team_slots, adjacency):
            z = self.encode(positions, team_slots, adjacency)
            return z, self.head(z)

    return DeepSetsAE, GATAE, PhaseGAT


def batch_indices(indices: np.ndarray, batch_size: int = 64):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def train_coordinate_models(samples: list[Sample], phase_classes: list[str], epochs: int) -> list[dict[str, Any]]:
    import torch

    DeepSetsAE, GATAE, PhaseGAT = build_torch_models()
    positions = np.stack([s.positions for s in samples]).astype(np.float32)
    team_slots = np.stack([s.team_slots for s in samples]).astype(np.int64)
    adjacency = np.stack([s.adjacency for s in samples]).astype(np.float32)
    train_idx = np.array([i for i, s in enumerate(samples) if s.split == "train"], dtype=int)
    dev_idx = np.array([i for i, s in enumerate(samples) if s.split == "dev"], dtype=int)
    label_map = {label: i for i, label in enumerate(phase_classes)}
    phase_idx = np.array([label_map.get(s.phase_label, -1) for s in samples], dtype=int)
    outputs: list[dict[str, Any]] = []
    model_dir = OUT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    for seed in SEEDS:
        set_seed(seed)
        model = DeepSetsAE(positions.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(epochs):
            order = np.random.default_rng(seed + _).permutation(train_idx)
            for idx in batch_indices(order):
                x = torch.from_numpy(positions[idx])
                t = torch.from_numpy(team_slots[idx])
                optimizer.zero_grad()
                _, recon = model(x, t)
                loss = torch.mean((recon - x) ** 2)
                loss.backward()
                optimizer.step()
        path = model_dir / f"deepsets_ae_seed{seed}.pt"
        torch.save(model.state_dict(), path)
        outputs.append({"model_id": f"deepsets_ae_seed{seed}", "family": "coordinate", "architecture": "DeepSets-AE", "seed": seed, "path": str(path.relative_to(ROOT)), "checkpoint_sha256": file_sha256(path), "latent_dim": 128, "pooling": "team_mean"})

        set_seed(seed)
        model = GATAE(positions.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(epochs):
            order = np.random.default_rng(seed + 1000 + _).permutation(train_idx)
            for idx in batch_indices(order):
                x = torch.from_numpy(positions[idx])
                t = torch.from_numpy(team_slots[idx])
                a = torch.from_numpy(adjacency[idx])
                optimizer.zero_grad()
                _, recon = model(x, t, a)
                loss = torch.mean((recon - x) ** 2)
                loss.backward()
                optimizer.step()
        path = model_dir / f"gat_ae_seed{seed}.pt"
        torch.save(model.state_dict(), path)
        outputs.append({"model_id": f"gat_ae_seed{seed}", "family": "coordinate", "architecture": "GAT-small-AE", "seed": seed, "path": str(path.relative_to(ROOT)), "checkpoint_sha256": file_sha256(path), "latent_dim": 128, "pooling": "team_mean", "graph": "baseline_fixed_weighted_knn4"})

        set_seed(seed)
        phase_model = PhaseGAT(max(1, len(phase_classes)))
        optimizer = torch.optim.Adam(phase_model.parameters(), lr=1e-3)
        valid_train = train_idx[phase_idx[train_idx] >= 0]
        phase_model.train()
        for _ in range(epochs):
            order = np.random.default_rng(seed + 2000 + _).permutation(valid_train)
            for idx in batch_indices(order):
                x = torch.from_numpy(positions[idx])
                t = torch.from_numpy(team_slots[idx])
                a = torch.from_numpy(adjacency[idx])
                y = torch.from_numpy(phase_idx[idx]).long()
                optimizer.zero_grad()
                _, logits = phase_model(x, t, a)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
        path = model_dir / f"phase_gat_seed{seed}.pt"
        torch.save(phase_model.state_dict(), path)
        with torch.no_grad():
            phase_model.eval()
            dev_valid = dev_idx[phase_idx[dev_idx] >= 0]
            if len(dev_valid):
                logits = []
                for idx in batch_indices(dev_valid):
                    z, out = phase_model(torch.from_numpy(positions[idx]), torch.from_numpy(team_slots[idx]), torch.from_numpy(adjacency[idx]))
                    logits.append(out.numpy())
                pred = np.argmax(np.concatenate(logits), axis=1)
                from sklearn.metrics import f1_score

                dev_f1 = float(f1_score(phase_idx[dev_valid], pred, average="macro"))
            else:
                dev_f1 = None
        outputs.append({"model_id": f"phase_gat_seed{seed}", "family": "task_trained", "architecture": "Phase-GAT", "seed": seed, "path": str(path.relative_to(ROOT)), "checkpoint_sha256": file_sha256(path), "latent_dim": 128, "pooling": "team_mean", "graph": "baseline_fixed_weighted_knn4", "phase_classes": phase_classes, "training_status": "completed" if len(valid_train) else "blocked_no_train_labels", "dev_macro_f1": dev_f1})
    return outputs


def render_minimap(positions: np.ndarray, team_slots: np.ndarray) -> Image.Image:
    colors = np.array([1805 if int(x) == 0 else 4177 for x in team_slots], dtype=np.int64)
    image = P0.make_image(positions, colors, width=256, height=160)
    canvas = Image.new("RGB", (224, 224), (245, 245, 245))
    canvas.paste(image.resize((224, 140), Image.Resampling.BILINEAR), (0, 42))
    return canvas


def load_vision_models(skip_vision: bool = False) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    if skip_vision:
        return [
            {"model_id": "dinov2_vits14", "family": "frozen_image", "architecture": "DINOv2 ViT-S/14", "error": "disabled_by_cli"},
            {"model_id": "openclip_vit_b32", "family": "frozen_image", "architecture": "OpenCLIP ViT-B/32", "error": "disabled_by_cli"},
        ]
    try:
        import timm
        from timm.data import create_transform, resolve_data_config

        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        model.eval()
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        loaded.append({"model_id": "dinov2_vits14", "family": "frozen_image", "architecture": "DINOv2 ViT-S/14", "model": model, "transform": transform, "dimension": int(model.num_features), "checkpoint_sha256": state_dict_sha256(model), "preprocess": "timm_resolve_data_config_v1", "pooling": "model_default"})
    except Exception as exc:
        loaded.append({"model_id": "dinov2_vits14", "family": "frozen_image", "architecture": "DINOv2 ViT-S/14", "error": repr(exc)})
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model.eval()
        loaded.append({"model_id": "openclip_vit_b32", "family": "frozen_image", "architecture": "OpenCLIP ViT-B/32", "model": model, "transform": preprocess, "dimension": int(model.visual.output_dim), "checkpoint_sha256": state_dict_sha256(model), "preprocess": "open_clip_preprocess_openai_v1", "pooling": "visual_projection"})
    except Exception as exc:
        loaded.append({"model_id": "openclip_vit_b32", "family": "frozen_image", "architecture": "OpenCLIP ViT-B/32", "error": repr(exc)})
    return loaded


def embed_torch_batches(model, tensors, encode: Callable, batch_size: int = 64) -> np.ndarray:
    import torch

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size])
            outputs.append(encode(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


def run_embeddings(
    samples: list[Sample],
    interventions: pd.DataFrame,
    model_specs: list[dict[str, Any]],
    visual_stride: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    import torch

    DeepSetsAE, GATAE, PhaseGAT = build_torch_models()
    positions = np.stack([s.positions for s in samples]).astype(np.float32)
    team_slots = np.stack([s.team_slots for s in samples]).astype(np.int64)
    adjacency = np.stack([s.adjacency for s in samples]).astype(np.float32)
    embedding_dir = OUT / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    baseline_cache: dict[str, np.ndarray] = {}
    intervention_cache: dict[str, dict[str, np.ndarray]] = {}
    coordinate_specs = [x for x in model_specs if x.get("family") in {"coordinate", "task_trained"}]
    for spec in coordinate_specs:
        model_id = spec["model_id"]
        if spec.get("training_status") not in {None, "completed"}:
            statuses.append({"model_id": model_id, "status": "blocked", "family": spec.get("family"), "error": spec["training_status"]})
            continue
        if "path" not in spec:
            statuses.append({"model_id": model_id, "status": "blocked", "family": spec.get("family"), "error": "missing checkpoint path"})
            continue
        if spec["family"] == "coordinate" and spec["architecture"] == "DeepSets-AE":
            model = DeepSetsAE(positions.shape[1])
            encode = lambda x, t, a=None: model.encode(x, t)
        elif spec["family"] == "coordinate":
            model = GATAE(positions.shape[1])
            encode = lambda x, t, a: model.encode(x, t, a)
        else:
            model = PhaseGAT(max(1, len(spec.get("phase_classes", []))))
            encode = lambda x, t, a: model.encode(x, t, a)
        model.load_state_dict(torch.load(ROOT / spec["path"], map_location="cpu"))
        model.eval()
        base_parts: list[np.ndarray] = []
        with torch.no_grad():
            for idx in batch_indices(np.arange(len(samples))):
                x = torch.from_numpy(positions[idx])
                t = torch.from_numpy(team_slots[idx])
                a = torch.from_numpy(adjacency[idx])
                base_parts.append(encode(x, t, a).numpy())
        baseline = np.concatenate(base_parts, axis=0)
        baseline_cache[model_id] = baseline
        base_path = embedding_dir / f"{model_id}_baseline.npy"
        np.save(base_path, baseline)
        for row_index, sample in enumerate(samples):
            records.append({"model_id": model_id, "embedding_kind": "baseline", "sample_id": sample.sample_id, "sample_index": row_index, "observation_key": f"{sample.match_id}:{sample.sample_id}", "intervention_id": None, "pair_id": None, "pair_role": "baseline", "path": str(base_path.relative_to(ROOT)), "row_index": row_index, "embedding_dim": int(baseline.shape[1]), "dtype": str(baseline.dtype), "normalization": "raw_saved_l2_at_metric", "checkpoint_sha256": spec.get("checkpoint_sha256"), "preprocess": spec.get("preprocess", "coordinate_normalized_xy_v1"), "pooling": spec.get("pooling", "team_mean")})
        eligible = interventions[(interventions["valid"])].copy()
        int_rows: list[dict[str, Any]] = []
        int_positions: list[np.ndarray] = []
        for _, row in eligible.iterrows():
            sample_index = int(row["sample_index"])
            delta = np.asarray(json.loads(row["delta"]), dtype=np.float32)
            int_positions.append(positions[sample_index] + delta)
            int_rows.append(row.to_dict())
        int_embeddings: list[np.ndarray] = []
        for start in range(0, len(int_rows), 64):
            part_rows = int_rows[start : start + 64]
            x = torch.from_numpy(np.stack(int_positions[start : start + 64]).astype(np.float32))
            t = torch.from_numpy(np.stack([team_slots[int(r["sample_index"])] for r in part_rows]).astype(np.int64))
            a = torch.from_numpy(np.stack([adjacency[int(r["sample_index"])] for r in part_rows]).astype(np.float32))
            with torch.no_grad():
                int_embeddings.append(encode(x, t, a).numpy())
        intervention_embedding = np.concatenate(int_embeddings, axis=0) if int_embeddings else np.empty((0, baseline.shape[1]), dtype=np.float32)
        int_path = embedding_dir / f"{model_id}_intervention.npy"
        np.save(int_path, intervention_embedding)
        intervention_cache[model_id] = {}
        for row_index, row in enumerate(int_rows):
            intervention_cache[model_id][str(row["intervention_id"])] = intervention_embedding[row_index]
            records.append({"model_id": model_id, "embedding_kind": "intervention", "sample_id": row["source_sample_id"], "sample_index": int(row["sample_index"]), "observation_key": row["observation_key"], "intervention_id": row["intervention_id"], "pair_id": row["pair_id"], "pair_role": "perturbed", "path": str(int_path.relative_to(ROOT)), "row_index": row_index, "embedding_dim": int(intervention_embedding.shape[1]), "dtype": str(intervention_embedding.dtype), "normalization": "raw_saved_l2_at_metric", "checkpoint_sha256": spec.get("checkpoint_sha256"), "preprocess": spec.get("preprocess", "coordinate_normalized_xy_v1"), "pooling": spec.get("pooling", "team_mean")})
            before = baseline[int(row["sample_index"])]
            after = intervention_embedding[row_index]
            before_norm = before / max(float(np.linalg.norm(before)), 1e-12)
            after_norm = after / max(float(np.linalg.norm(after)), 1e-12)
            response = float(1.0 - np.dot(before_norm, after_norm))
            response_row = row.copy()
            response_row.update({"model_id": model_id, "response_distance": response, "normalized_response": response if float(row["epsilon"]) == 0 else response / float(row["epsilon"])})
            responses.append(response_row)
        statuses.append({"model_id": model_id, "status": "completed", "family": spec.get("family"), "architecture": spec.get("architecture"), "seed": spec.get("seed"), "checkpoint_sha256": spec.get("checkpoint_sha256"), "n_baseline": len(baseline), "n_interventions": len(int_rows)})
    vision_specs = [x for x in model_specs if x.get("family") == "frozen_image" and "model" in x]
    for spec in vision_specs:
        model_id = spec["model_id"]
        model = spec["model"]
        transform = spec["transform"]
        model.eval()
        eligible = interventions[(interventions["valid"]) & (interventions["visual_eval"])].copy()
        visual_sample_indices = sorted(eligible["sample_index"].astype(int).unique().tolist())
        base_tensors = [transform(render_minimap(samples[index].positions, samples[index].team_slots)) for index in visual_sample_indices]
        if model_id.startswith("openclip"):
            encode = model.encode_image
        else:
            encode = model
        baseline_subset = embed_torch_batches(model, base_tensors, encode)
        # Keep a full-index cache with explicit NaNs for non-visual-stride
        # observations. Only rows marked visual_eval may read a finite entry.
        baseline = np.full((len(samples), baseline_subset.shape[1]), np.nan, dtype=baseline_subset.dtype)
        baseline[visual_sample_indices] = baseline_subset
        baseline_cache[model_id] = baseline
        base_path = embedding_dir / f"{model_id}_baseline.npy"
        np.save(base_path, baseline)
        for row_index in visual_sample_indices:
            sample = samples[row_index]
            records.append({"model_id": model_id, "embedding_kind": "baseline", "sample_id": sample.sample_id, "sample_index": row_index, "observation_key": f"{sample.match_id}:{sample.sample_id}", "intervention_id": None, "pair_id": None, "pair_role": "baseline", "path": str(base_path.relative_to(ROOT)), "row_index": row_index, "embedding_dim": int(baseline.shape[1]), "dtype": str(baseline.dtype), "normalization": "raw_saved_l2_at_metric", "checkpoint_sha256": spec.get("checkpoint_sha256"), "preprocess": spec.get("preprocess", "render_minimap_rgb_v1"), "pooling": spec.get("pooling", "visual_projection")})
        int_rows: list[dict[str, Any]] = []
        int_tensors = []
        for _, row in eligible.iterrows():
            sample_index = int(row["sample_index"])
            delta = np.asarray(json.loads(row["delta"]), dtype=np.float32)
            int_tensors.append(transform(render_minimap(samples[sample_index].positions + delta, samples[sample_index].team_slots)))
            int_rows.append(row.to_dict())
        intervention_embedding = embed_torch_batches(model, int_tensors, encode)
        int_path = embedding_dir / f"{model_id}_intervention.npy"
        np.save(int_path, intervention_embedding)
        intervention_cache[model_id] = {}
        for row_index, row in enumerate(int_rows):
            intervention_cache[model_id][str(row["intervention_id"])] = intervention_embedding[row_index]
            records.append({"model_id": model_id, "embedding_kind": "intervention", "sample_id": row["source_sample_id"], "sample_index": int(row["sample_index"]), "observation_key": row["observation_key"], "intervention_id": row["intervention_id"], "pair_id": row["pair_id"], "pair_role": "perturbed", "path": str(int_path.relative_to(ROOT)), "row_index": row_index, "embedding_dim": int(intervention_embedding.shape[1]), "dtype": str(intervention_embedding.dtype), "normalization": "raw_saved_l2_at_metric", "checkpoint_sha256": spec.get("checkpoint_sha256"), "preprocess": spec.get("preprocess", "render_minimap_rgb_v1"), "pooling": spec.get("pooling", "visual_projection")})
            before = baseline[int(row["sample_index"])]
            after = intervention_embedding[row_index]
            before_norm = before / max(float(np.linalg.norm(before)), 1e-12)
            after_norm = after / max(float(np.linalg.norm(after)), 1e-12)
            response = float(1.0 - np.dot(before_norm, after_norm))
            response_row = row.copy()
            response_row.update({"model_id": model_id, "response_distance": response, "normalized_response": response if float(row["epsilon"]) == 0 else response / float(row["epsilon"])})
            responses.append(response_row)
        statuses.append({"model_id": model_id, "status": "completed", "family": spec.get("family"), "architecture": spec.get("architecture"), "checkpoint_sha256": spec.get("checkpoint_sha256"), "n_baseline": len(visual_sample_indices), "n_observations": len(samples), "n_interventions": len(int_rows), "visual_stride": visual_stride})
    for spec in [x for x in model_specs if x.get("family") == "frozen_image" and "model" not in x]:
        statuses.append({"model_id": spec["model_id"], "status": "blocked", "family": spec.get("family"), "architecture": spec.get("architecture"), "error": spec.get("error", "model not loaded")})
    response_df = pd.DataFrame(responses)
    response_df.to_parquet(OUT / "spectrum_response.parquet", index=False)
    pd.DataFrame(records).to_parquet(OUT / "embedding_manifest.parquet", index=False)
    return response_df, statuses, baseline_cache, intervention_cache


def grouped_bootstrap(df: pd.DataFrame, value: str, group_cols: list[str], n_bootstrap: int = 500, seed: int = 20260827) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [x for x in group_cols if x in df.columns]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        per_match = group.groupby("match_id")[value].mean()
        match_values = per_match.to_numpy(dtype=float)
        draws = rng.choice(match_values, size=(n_bootstrap, len(match_values)), replace=True).mean(axis=1) if len(match_values) else np.array([])
        row = dict(zip(group_cols, keys))
        row.update({"mean_response": float(match_values.mean()) if len(match_values) else None, "ci_low": float(np.quantile(draws, 0.025)) if len(draws) else None, "ci_high": float(np.quantile(draws, 0.975)) if len(draws) else None, "n_matches": int(len(match_values)), "n_records": int(len(group))})
        rows.append(row)
    return pd.DataFrame(rows)


def compute_accessibility(
    response_df: pd.DataFrame,
    baseline_cache: dict[str, np.ndarray],
    intervention_cache: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import accuracy_score, r2_score

    out: list[dict[str, Any]] = []
    for model_id, group in response_df.groupby("model_id"):
        exact = group[group["kind"].astype(str).str.startswith("exact_mode_") & group["epsilon"].eq(PRIMARY_EPSILON)].copy()
        if exact.empty:
            out.append({"model_id": model_id, "status": "not_evaluated_exact_modes"})
            continue
        X: list[np.ndarray] = []
        y: list[int] = []
        splits: list[str] = []
        baseline = baseline_cache[model_id]
        embedding_by_id = intervention_cache.get(model_id, {})
        for _, row in exact.iterrows():
            int_embedding = embedding_by_id.get(str(row["intervention_id"]))
            if int_embedding is None:
                continue
            base = baseline[int(row["sample_index"])]
            X.append(int_embedding - base)
            y.append(int(row["mode_rank"]))
            splits.append(str(row["split"]))
        if len(X) < 12 or len(set(y)) < 2:
            out.append({"model_id": model_id, "status": "not_enough_rows", "n": len(X)})
            continue
        X_np = np.stack(X)
        y_np = np.asarray(y)
        train = np.array([s == "train" for s in splits])
        test = np.array([s == "heldout" for s in splits])
        if test.sum() < 2 or train.sum() < 2:
            out.append({"model_id": model_id, "status": "no_heldout"})
            continue
        classes = sorted(set(y_np.tolist()))
        onehot = np.zeros((len(y_np), len(classes)), dtype=float)
        for i, value in enumerate(y_np):
            onehot[i, classes.index(int(value))] = 1.0
        reg = Ridge(alpha=1.0).fit(X_np[train], onehot[train])
        pred = reg.predict(X_np[test])
        out.append({"model_id": model_id, "status": "completed", "target": "exact_mode_rank", "n_train": int(train.sum()), "n_heldout": int(test.sum()), "r2": float(r2_score(onehot[test], pred, multioutput="variance_weighted")), "mode_accuracy": float(accuracy_score(y_np[test], np.array(classes)[np.argmax(pred, axis=1)]))})
    return out


def make_figures(response_df: pd.DataFrame, summary_df: pd.DataFrame, errr_df: pd.DataFrame) -> list[str]:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model_id, group in summary_df[summary_df["epsilon"].eq(PRIMARY_EPSILON)].groupby("model_id"):
            curve = group[group["band_id"].notna()].sort_values("band_id")
            if curve.empty:
                continue
            ax.plot(curve["band_id"], curve["mean_response"], marker="o", label=model_id)
            ax.fill_between(curve["band_id"], curve["ci_low"], curve["ci_high"], alpha=0.12)
        ax.set_xlabel("band id (equal-count Laplacian modes)")
        ax.set_ylabel("mean cosine response / epsilon")
        ax.set_title("P1 structural transfer function")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        path = figure_dir / "transfer_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))
    if not errr_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        valid = errr_df[errr_df["errr"].notna()]
        if not valid.empty:
            ax.bar(valid["model_id"], valid["errr"], color="#3b6ea8")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("preliminary ERRR")
        ax.set_title("Common vs semantic coalition: exploratory only")
        fig.tight_layout()
        path = figure_dir / "preliminary_reversal.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))
    if not response_df.empty:
        group = response_df[response_df["kind"].astype(str).str.startswith("support_sweep_")]
        if not group.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for model_id, sub in group.groupby("model_id"):
                means = sub.groupby("support_size")["normalized_response"].mean().sort_index()
                ax.plot(means.index, means.values, marker="o", label=model_id)
            ax.set_xlabel("support size")
            ax.set_ylabel("mean cosine response / epsilon")
            ax.set_title("Preliminary support sweep")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            path = figure_dir / "support_sweep.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(str(path.relative_to(ROOT)))
    return paths


def write_report(
    samples: list[Sample],
    interventions: pd.DataFrame,
    statuses: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    errr_df: pd.DataFrame,
    notch_rows: list[dict[str, Any]],
    accessibility: list[dict[str, Any]],
    intervention_validation: dict[str, Any],
    figure_paths: list[str],
    elapsed: float,
    point_only: bool = False,
) -> dict[str, Any]:
    required_models = set(POINT_MODEL_IDS)
    if not point_only:
        required_models.update(VISION_MODEL_IDS)
    status_map = {x["model_id"]: x for x in statuses}
    missing = sorted(model_id for model_id in required_models if status_map.get(model_id, {}).get("status") != "completed")
    overall = "completed" if not missing and intervention_validation["status"] == "pass" else "partial"
    valid = interventions[interventions["valid"]]
    max_energy_error = float(valid["energy_error"].max()) if not valid.empty else None
    min_purity = float(valid["target_mode_purity"].dropna().min()) if not valid["target_mode_purity"].dropna().empty else None
    summary = {
        "phase": "P1_PHENOMENON",
        "checkpoint": "C1",
        "status": overall,
        "missing_or_blocked_models": missing,
        "n_matches": len(set(s.match_id for s in samples)),
        "n_samples": len(samples),
        "n_interventions": int(len(interventions)),
        "n_valid_interventions": int(len(valid)),
        "n_invalid_interventions": int((~interventions["valid"]).sum()),
        "intervention_validation": intervention_validation,
        "max_energy_error": max_energy_error,
        "min_target_mode_purity": min_purity,
        "mainline_modality": "coordinate_point_set" if point_only else "coordinate_point_set_with_optional_minimap_branch",
        "vision_policy": "deferred_auxiliary" if point_only else "optional_deterministic_minimap_robustness",
        "models": statuses,
        "errr": errr_df.to_dict("records"),
        "notch": notch_rows,
        "mode_accessibility": accessibility,
        "figures": figure_paths,
        "elapsed_seconds": elapsed,
        "scientific_boundary": "C1 measures candidate response patterns; it does not establish H1/H2 or the P2 same-frequency organization claim.",
    }
    (OUT / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    vision_line = (
        "- Mainline input is the observed player point set; the minimap/vision branch is deferred and excluded from this checkpoint."
        if point_only
        else "- Vision inputs are deterministic minimap rasters generated from the same tracking state; they are modality-specific sensitivity measurements, not video-based architecture comparisons."
    )
    lines = [
        "# Exploration Checkpoint 1 — Phenomenon Screening",
        "",
        f"- **Status**: `{overall}`",
        "- **Study mode**: `exploratory_discovery`",
        f"- **Samples**: {len(samples)} across {summary['n_matches']} SkillCorner matches",
        f"- **Interventions**: {len(interventions)} total; {len(valid)} valid; {summary['n_invalid_interventions']} boundary-invalid",
        f"- **Maximum energy error**: `{max_energy_error}`",
        f"- **Minimum target-mode purity**: `{min_purity}`",
        "",
        "## Models",
        "",
    ]
    for item in statuses:
        lines.append(f"- `{item['model_id']}`: **{item['status']}**" + (f" — {item.get('error')}" if item.get("error") else ""))
    lines.extend([
        "",
        "## Measurements",
        "",
        "- Primary response: normalized cosine distance between baseline and intervened embeddings.",
        "- Transfer curves: six equal-count Laplacian bands across all seeds and energies; exact modes are a primary-energy/seed-11 accessibility diagnostic.",
        "- Preliminary ERRR: common versus role-ordered semantic coalition; not a P2 strict same-frequency result.",
        "- Confidence intervals: match-level bootstrap, not frame-level independent bootstrap.",
        f"- Pairing/validity gate: `{intervention_validation['status']}`; observation key is `match_id:sample_id`, and each intervention has one baseline counterpart.",
        vision_line,
        "",
        "## Exploratory findings",
        "",
        "The numerical tables below are generated from the response artifact. A positive or null result is retained as an exploration outcome; no mechanical positive threshold is applied.",
        "",
        "### Preliminary ERRR",
        "",
        "```json",
        json.dumps(errr_df.to_dict("records"), indent=2, ensure_ascii=False),
        "```",
        "",
        "### Notch candidates",
        "",
        "```json",
        json.dumps(notch_rows, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Artifacts",
        "",
    ])
    lines.extend(f"- `{path}`" for path in figure_paths)
    lines.extend([
        "- `artifacts/phase1/spectrum_response.parquet`",
        "- `artifacts/phase1/embedding_manifest.parquet`",
        "- `artifacts/phase1/pipeline_summary.json`",
        "- `artifacts/phase1/model_manifest.json`",
        "- `artifacts/phase1/execution_contract.json`",
        "- `artifacts/phase1/intervention_validation.json`",
        "",
        "## Boundary and next action",
        "",
        "P1 is a response-measurement checkpoint. It does not claim that common motion is more salient, that a mid-frequency notch exists, or that organization exceeds frequency. Those interpretations require the matched controls and alternative graph constructions planned for P2.",
        "",
        f"Runtime: {elapsed:.1f} seconds.",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def public_model_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if key not in {"model", "transform"}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples-per-match", type=int, default=500)
    parser.add_argument("--visual-stride", type=int, default=1, help="Use every Nth sample for frozen vision interventions; default 1 means all.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--skip-vision", action="store_true", help="Do not load frozen vision checkpoints; for interface smoke only.")
    parser.add_argument("--point-only", action="store_true", help="Run the primary point-set branch and defer the optional minimap/vision branch.")
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    global ROOT, DATA_ROOT, OUT, REPORT
    args = parse_args()
    ROOT = args.root.resolve()
    DATA_ROOT = ROOT / "data" / "raw" / "sports" / "skillcorner" / "data" / "matches"
    OUT = ROOT / "artifacts" / "phase1"
    REPORT = ROOT / "reports" / "EXPLORATION_CHECKPOINT_1.md"
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    (OUT / "execution_contract.json").write_text(
        json.dumps(
            build_execution_contract(
                n_samples_per_match=args.n_samples_per_match,
                visual_stride=args.visual_stride,
                skip_vision=args.skip_vision,
                point_only=args.point_only,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    samples = build_samples(args.n_samples_per_match)
    print(f"[P1] samples ready: {len(samples)}", flush=True)
    # The supervised phase vocabulary is learned from train only. Held-out
    # labels remain for post-hoc descriptive stratification, never for model
    # dimension selection or graph construction.
    phase_classes = sorted({s.phase_label for s in samples if s.split == "train" and s.phase_label is not None})
    interventions = generate_interventions(samples, visual_stride=args.visual_stride)
    print(f"[P1] interventions ready: {len(interventions)}", flush=True)
    intervention_validation = validate_interventions(samples, interventions)
    (OUT / "intervention_validation.json").write_text(json.dumps(intervention_validation, indent=2) + "\n", encoding="utf-8")
    if intervention_validation["status"] != "pass":
        raise RuntimeError(f"P1 intervention validity gate failed: {intervention_validation['errors']}")
    save_samples(samples, interventions)
    print("[P1] canonical and intervention manifests saved", flush=True)
    model_specs = train_coordinate_models(samples, phase_classes, epochs=args.epochs)
    print(f"[P1] coordinate models trained: {len(model_specs)}", flush=True)
    vision_specs = [] if args.point_only else load_vision_models(skip_vision=args.skip_vision)
    print(f"[P1] vision model load statuses: {len(vision_specs)}", flush=True)
    (OUT / "model_manifest.json").write_text(
        json.dumps([public_model_spec(spec) for spec in model_specs + vision_specs], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    response_df, statuses, baseline_cache, intervention_cache = run_embeddings(samples, interventions, model_specs + vision_specs, visual_stride=args.visual_stride)
    print(f"[P1] responses ready: {len(response_df)}", flush=True)
    summary_df = grouped_bootstrap(response_df, "normalized_response", ["model_id", "kind", "epsilon", "band_id", "mode_rank", "support_size"])
    summary_df.to_parquet(OUT / "spectrum_summary.parquet", index=False)
    errr_rows: list[dict[str, Any]] = []
    if not response_df.empty:
        for model_id, group in response_df.groupby("model_id"):
            common = group[group["kind"] == "common"].groupby(["source_sample_id", "seed", "epsilon"])["response_distance"].mean().rename("common")
            semantic = group[group["kind"] == "semantic_coalition"].groupby(["source_sample_id", "seed", "epsilon"])["response_distance"].mean().rename("semantic")
            paired = pd.concat([common, semantic], axis=1).dropna()
            errr_rows.append({"model_id": model_id, "n_pairs": int(len(paired)), "errr": float((paired["common"] > paired["semantic"]).mean()) if len(paired) else None, "mean_common_minus_semantic": float((paired["common"] - paired["semantic"]).mean()) if len(paired) else None})
    errr_df = pd.DataFrame(errr_rows)
    notch_rows: list[dict[str, Any]] = []
    if not summary_df.empty:
        for model_id, group in summary_df[summary_df["epsilon"].eq(PRIMARY_EPSILON)].groupby("model_id"):
            curve = group[group["band_id"].notna()].groupby("band_id")["mean_response"].mean()
            if {0, 2, 3, 5}.issubset(set(curve.index)):
                low_high = 0.5 * (curve.loc[0] + curve.loc[5])
                mid = 0.5 * (curve.loc[2] + curve.loc[3])
                notch_rows.append({"model_id": model_id, "notch_depth_candidate": float(1.0 - mid / max(low_high, 1e-12)), "low_high_reference": float(low_high), "mid_response": float(mid), "interpretation": "candidate only; inspect monotonicity and P2 controls"})
    accessibility = compute_accessibility(response_df, baseline_cache, intervention_cache)
    figure_paths = make_figures(response_df, summary_df, errr_df)
    write_report(
        samples,
        interventions,
        statuses,
        summary_df,
        errr_df,
        notch_rows,
        accessibility,
        intervention_validation,
        figure_paths,
        time.time() - started,
        point_only=args.point_only,
    )
    print(json.dumps(json.loads((OUT / "pipeline_summary.json").read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
