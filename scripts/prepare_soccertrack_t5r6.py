#!/usr/bin/env python3
"""Convert SoccerTrack-v2 GSR halves to T5R6 canonical snapshots + task views (N2/N3).

Reads data/raw/sports/soccertrack_v2/gsr/<match>/<match>_<half>.json
(COCO-style, streamed with ijson). For each half, takes every 25th frame
(1Hz @25fps), keeps frames with exactly 20 non-goalkeeper players split
10/10 by team side, aligns physical teams across halves via player_id
majority vote (deterministic, response-blind), and writes IDSSE-compatible
task views plus per-match conversion receipts.

Outputs under artifacts/phase3/task_semantic_repair_v1/data_views_soccertrack/
and artifacts/data_v2/soccertrack/canonical_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "sports" / "soccertrack_v2"
ST_ROOT = ROOT / "artifacts" / "data_v2" / "soccertrack"
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"
VIEW_ROOT = T5R_ROOT / "data_views_soccertrack"

MATCHES = ["118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]
HALVES = ["1st", "2nd"]
PITCH_X, PITCH_Y = 105.0, 68.0
FPS = 25
SAMPLE_STRIDE = 25
N_PLAYER_NODES = 20
HALF_TO_SECTION = {"1st": "firstHalf", "2nd": "secondHalf"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def zone_label(value: float) -> str:
    if value < -1.0 / 3.0:
        return "defensive_third"
    if value < 1.0 / 3.0:
        return "middle_third"
    return "attacking_third"


def person_key(attributes: dict) -> str:
    pid = attributes.get("player_id")
    if pid is not None:
        return f"id:{pid}"
    return f"side:{attributes.get('team')}:jersey:{attributes.get('jersey')}"


def stream_half(path: Path):
    """Yield (frame_order, image_id, list of object annotations) in file order."""
    import ijson
    with path.open("rb") as f:
        images = [(img["image_id"], n) for n, img in enumerate(ijson.items(f, "images.item"))]
    order_of = {img_id: n for img_id, n in images}
    frames: dict[int, list] = defaultdict(list)
    with path.open("rb") as f:
        for ann in ijson.items(f, "annotations.item"):
            if ann.get("supercategory") != "object":
                continue
            order = order_of.get(ann.get("image_id"))
            if order is None:
                continue
            frames[order].append(ann)
    for order in sorted(frames):
        yield order, frames[order]


def extract_entities(annotations: list) -> list[dict]:
    out = []
    for ann in annotations:
        attrs = ann.get("attributes", {}) or {}
        role = attrs.get("role")
        if role not in ("player", "goalkeeper"):
            continue
        try:
            box = ann["bbox_pitch"]
            x = float(box["x_bottom_middle"])
            y = float(box["y_bottom_middle"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        out.append({"role": role, "team_side": attrs.get("team"),
                    "key": person_key(attrs), "x": x, "y": y})
    return out


def align_teams(halves_entities: dict[str, dict[int, list]]) -> dict[str, dict[str, int]]:
    """Map per-half team_side -> stable slot (0/1) via player_id majority vote.

    Slot 0 = the physical team that is 'left' in the 1st half. Frozen,
    deterministic, response-blind (uses only identity fields).
    """
    first = halves_entities["1st"]
    left_ids, right_ids = set(), set()
    for ents in first.values():
        for e in ents:
            if e["key"].startswith("id:"):
                (left_ids if e["team_side"] == "left" else right_ids).add(e["key"])
    mapping = {"1st": {"left": 0, "right": 1}}
    second = halves_entities["2nd"]
    votes = Counter()
    for ents in second.values():
        for e in ents:
            if e["key"] in left_ids:
                votes[(e["team_side"], 0)] += 1
            elif e["key"] in right_ids:
                votes[(e["team_side"], 1)] += 1
    # If 2nd-half 'left' mostly holds 1st-half-left players, keep mapping;
    # otherwise flip. Ties (or no votes) keep identity mapping.
    left_is_0 = votes[("left", 0)] + votes[("right", 1)]
    left_is_1 = votes[("left", 1)] + votes[("right", 0)]
    if left_is_1 > left_is_0:
        mapping["2nd"] = {"left": 1, "right": 0}
    else:
        mapping["2nd"] = {"left": 0, "right": 1}
    return mapping


def convert_match(match: str) -> dict:
    halves_entities: dict[str, dict[int, list]] = {}
    frame_counts: dict[str, int] = {}
    for half in HALVES:
        path = RAW_ROOT / "gsr" / match / f"{match}_{half}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        sampled: dict[int, list] = {}
        n_frames = 0
        for order, anns in stream_half(path):
            n_frames = order + 1
            if order % SAMPLE_STRIDE == 0:
                sampled[order] = extract_entities(anns)
        halves_entities[half] = sampled
        frame_counts[half] = n_frames
    slot_of = align_teams(halves_entities)
    records, raw_list, team_list = [], [], []
    qc = {"frames_per_half": frame_counts, "sampled_per_half": {},
          "kept": 0, "dropped_not20": 0, "dropped_split": 0, "dropped_nonfinite": 0,
          "slot_mapping": slot_of}
    for half in HALVES:
        section = HALF_TO_SECTION[half]
        kept_half = 0
        for order in sorted(halves_entities[half]):
            ents = halves_entities[half][order]
            qc["sampled_per_half"].setdefault(half, 0)
            qc["sampled_per_half"][half] += 1
            outfield = [e for e in ents if e["role"] == "player"]
            if len(outfield) != N_PLAYER_NODES:
                qc["dropped_not20"] += 1
                continue
            slots = [slot_of[half].get(e["team_side"], -1) for e in outfield]
            if any(s < 0 for s in slots) or sorted(slots) != [0] * 10 + [1] * 10:
                qc["dropped_split"] += 1
                continue
            table = sorted(zip(slots, [e["key"] for e in outfield],
                               [e["x"] for e in outfield], [e["y"] for e in outfield]))
            xs = np.array([t[2] for t in table], dtype=np.float64) / (PITCH_X / 2.0)
            ys = np.array([t[3] for t in table], dtype=np.float64) / (PITCH_Y / 2.0)
            raw = np.stack([xs, ys], axis=1).astype(np.float32)
            if not np.isfinite(raw).all():
                qc["dropped_nonfinite"] += 1
                continue
            team_slots = np.array([t[0] for t in table], dtype=np.int64)
            home_c = raw[team_slots == 0].mean(axis=0)
            away_c = raw[team_slots == 1].mean(axis=0)
            records.append({
                "snapshot_id": f"st_{match}_{section}_{order}",
                "source_match_id": match, "split": "external",
                "game_section": section, "frame_n": int(order),
                "timestamp_ms": int(order * (1000 // FPS)),
                "player_count": N_PLAYER_NODES, "ball_present": False,
                "home_team_id": f"ST{match}_A", "away_team_id": f"ST{match}_B",
                "player_ids": json.dumps([t[1] for t in table], separators=(",", ":")),
                "phase_label": section,
                "home_field_zone": zone_label(float(home_c[0])),
                "away_field_zone": zone_label(float(away_c[0])),
                "home_centroid_x": float(home_c[0]), "home_centroid_y": float(home_c[1]),
                "away_centroid_x": float(away_c[0]), "away_centroid_y": float(away_c[1]),
            })
            raw_list.append(raw)
            team_list.append(team_slots)
            kept_half += 1
        qc["kept_per_half"] = qc.get("kept_per_half", {})
        qc["kept_per_half"][half] = kept_half
    qc["kept"] = len(records)
    index = pd.DataFrame.from_records(records)
    if index.empty:
        raise ValueError(f"no valid snapshots for {match}")
    raw_arr = np.stack(raw_list).astype(np.float32)
    centered = (raw_arr - raw_arr.mean(axis=1, keepdims=True)).astype(np.float32)
    team_arr = np.stack(team_list).astype(np.int64)
    VIEW_ROOT.mkdir(parents=True, exist_ok=True)
    index.to_parquet(VIEW_ROOT / f"snapshot_index_{match}.parquet", index=False, compression="zstd")
    index.to_parquet(VIEW_ROOT / f"context_labels_{match}.parquet", index=False, compression="zstd")
    np.save(VIEW_ROOT / f"positions_raw_{match}.npy", raw_arr)
    np.save(VIEW_ROOT / f"positions_centered_{match}.npy", centered)
    np.save(VIEW_ROOT / f"team_slots_{match}.npy", team_arr)
    files = {}
    for name in (f"snapshot_index_{match}.parquet", f"context_labels_{match}.parquet",
                 f"positions_raw_{match}.npy", f"positions_centered_{match}.npy",
                 f"team_slots_{match}.npy"):
        files[name] = sha256_file(VIEW_ROOT / name)
    # natural pairs with frozen T5R2 thresholds
    sys.path.insert(0, str(ROOT / "scripts"))
    import prepare_idsse_t5r2 as prep
    pairs, diagnostics = prep.build_natural_pairs(index, raw_arr)
    pairs.to_parquet(VIEW_ROOT / f"natural_pair_ranking_{match}.parquet", index=False, compression="zstd")
    files[f"natural_pair_ranking_{match}.parquet"] = sha256_file(VIEW_ROOT / f"natural_pair_ranking_{match}.parquet")
    receipt = {
        "status": "SOCCERTRACK_CANONICAL_CONVERSION_COMPLETE", "source_match_id": match,
        "source_files": {f"{match}_{h}.json": sha256_file(RAW_ROOT / "gsr" / match / f"{match}_{h}.json") for h in HALVES},
        "pitch_assumed": [PITCH_X, PITCH_Y], "sample_hz": FPS / SAMPLE_STRIDE,
        "team_slot_rule": "slot0 = physical team that is left in 1st half (player_id majority vote)",
        "ball_present": False, "ball_reason": "ball tracks not downloaded by design (positions only)",
        "qc": qc,
        "snapshot_count": int(len(index)),
        "pair_count": int(len(pairs)),
        "pair_diagnostics": {str(k): (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in dict(diagnostics).items()},
        "output_files": files,
    }
    write_json(ST_ROOT / "canonical" / "conversion_receipts" / f"{match}.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", nargs="*", default=MATCHES)
    args = parser.parse_args()
    (ST_ROOT / "canonical" / "conversion_receipts").mkdir(parents=True, exist_ok=True)
    started = time.time()
    receipts = [convert_match(m) for m in args.matches]
    manifest = {
        "status": "SOCCERTRACK_T5R6_CANONICAL_CONVERSION_COMPLETE",
        "matches": [r["source_match_id"] for r in receipts],
        "total_snapshots": sum(r["snapshot_count"] for r in receipts),
        "total_pairs": sum(r["pair_count"] for r in receipts),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    write_json(ST_ROOT / "canonical_manifest.json", manifest)
    print(json.dumps(manifest, indent=1), flush=True)
    for r in receipts:
        print(f"{r['source_match_id']}: snaps={r['snapshot_count']} pairs={r['pair_count']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
