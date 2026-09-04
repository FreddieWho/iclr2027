#!/usr/bin/env python3
"""Prepare the local IDSSE mirror for the P3-T5R2 task and baseline lock.

The script has two deliberately separate stages:

* ``convert`` streams the 21 XML files into versioned canonical parquet files
  and writes source/QC receipts for all seven matches.
* ``tasks`` reads only the pre-declared development and validation matches,
  builds context labels and natural ranking triplets, and writes the T5R2
  split/task/baseline locks.  The reserved match is never loaded in this
  stage.

No model response, old heldout result, or external-confirmation result is read.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "sports" / "idsse-data"
IDSSE_ARTIFACT_ROOT = ROOT / "artifacts" / "data_v2" / "idsse"
T5R_ROOT = ROOT / "artifacts" / "phase3" / "task_semantic_repair_v1"

OFFICIAL_DATASET_URL = "https://huggingface.co/datasets/pysport/idsse-data"
OFFICIAL_TREE_URL = "https://huggingface.co/datasets/pysport/idsse-data/tree/main"
OFFICIAL_REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
FIGSHARE_URL = (
    "https://springernature.figshare.com/articles/dataset/"
    "An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177"
)

# This map is an ID-only decision made before task construction.  It is not
# inferred from scores, event frequency, or any model output.
MATCH_SPLITS: dict[str, str] = {
    "J03WMX": "train",
    "J03WOH": "train",
    "J03WPY": "train",
    "J03WR9": "train",
    "J03WN1": "valid",
    "J03WOY": "valid",
    "J03WQQ": "reserved_holdout",
}
SEEDS = [11, 23, 47]
SAMPLE_HZ = 1.0
SOURCE_FPS = 25.0
SAMPLE_STRIDE = int(round(SOURCE_FPS / SAMPLE_HZ))
N_PLAYER_NODES = 20
MIN_PAIR_TIME_SEPARATION_S = 4.0
PAIR_ANCHOR_STRIDE = 5
PAIR_NEIGHBOR_K = 512

# Natural-pair thresholds are fixed protocol values in normalized pitch
# coordinates.  They are not selected from model results.
POSITIVE_INTERNAL_MAX = 0.28
POSITIVE_CENTROID_MIN = 0.35
HARD_NEGATIVE_CENTROID_MAX = 0.12
HARD_NEGATIVE_INTERNAL_MIN = 0.35

MATCH_RE = re.compile(r"DFL-MAT-(J[A-Z0-9]+)\.xml$")

FRAME_SCHEMA = pa.schema(
    [
        ("source_match_id", pa.string()),
        ("game_section", pa.string()),
        ("team_id", pa.string()),
        ("person_id", pa.string()),
        ("entity_type", pa.string()),
        ("frame_n", pa.int64()),
        ("timestamp_ms", pa.int64()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("z", pa.float32()),
        ("direction", pa.float32()),
        ("speed", pa.float32()),
        ("acceleration", pa.float32()),
        ("measurement_status", pa.int32()),
        ("ball_possession", pa.int32()),
        ("ball_status", pa.int32()),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        ("source_match_id", pa.string()),
        ("event_index", pa.int64()),
        ("event_id", pa.string()),
        ("timestamp_ms", pa.int64()),
        ("event_type", pa.string()),
        ("event_root_type", pa.string()),
        ("game_section", pa.string()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("source_x", pa.float32()),
        ("source_y", pa.float32()),
        ("player_id", pa.string()),
        ("team_id", pa.string()),
        ("recipient_id", pa.string()),
        ("outcome", pa.string()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def strip_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_goalkeeper(player: dict[str, Any] | None) -> bool:
    role = str((player or {}).get("position_role") or "").upper()
    return role in {"TW", "GK", "GOALKEEPER"} or "GOALKEEP" in role


def iso_to_epoch_ms(value: str) -> int:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(round(parsed.timestamp() * 1000.0))


def short_match_id(raw_match_id: str) -> str:
    return raw_match_id.removeprefix("DFL-MAT-")


def discover_files(raw_root: Path) -> dict[str, dict[str, Path]]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"IDSSE raw directory missing: {raw_root}")
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(raw_root.glob("*.xml")):
        match = MATCH_RE.search(path.name)
        if match is None:
            raise ValueError(f"cannot identify match ID from XML filename: {path.name}")
        match_id = match.group(1)
        if "02_01_matchinformation" in path.name:
            kind = "metadata"
        elif "03_02_events_raw" in path.name:
            kind = "events"
        elif "04_03_positions_raw_observed" in path.name:
            kind = "positions"
        else:
            raise ValueError(f"unknown IDSSE XML file kind: {path.name}")
        if kind in grouped[match_id]:
            raise ValueError(f"duplicate {kind} file for {match_id}")
        grouped[match_id][kind] = path
    expected = {"metadata", "events", "positions"}
    if set(grouped) != set(MATCH_SPLITS):
        raise ValueError(f"raw match IDs {sorted(grouped)} do not equal locked IDs {sorted(MATCH_SPLITS)}")
    for match_id, files in grouped.items():
        if set(files) != expected:
            raise ValueError(f"{match_id} has file kinds {sorted(files)}, expected {sorted(expected)}")
    return dict(grouped)


def parse_metadata(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    general = next((e for e in root.iter() if strip_tag(e.tag) == "General"), None)
    environment = next((e for e in root.iter() if strip_tag(e.tag) == "Environment"), None)
    if general is None or environment is None:
        raise ValueError(f"metadata missing General/Environment: {path.name}")
    raw_match_id = str(general.attrib.get("MatchId", ""))
    match_id = short_match_id(raw_match_id)
    if not match_id:
        raise ValueError(f"metadata missing MatchId: {path.name}")

    teams: dict[str, dict[str, Any]] = {}
    players: dict[str, dict[str, Any]] = {}
    for team in (e for e in root.iter() if strip_tag(e.tag) == "Team"):
        team_id = team.attrib.get("TeamId")
        if not team_id:
            continue
        role = str(team.attrib.get("Role", "")).lower()
        team_record = {
            "team_id": team_id,
            "team_name": team.attrib.get("TeamName"),
            "role": role,
        }
        teams[team_id] = team_record
        for player in (e for e in team.iter() if strip_tag(e.tag) == "Player"):
            person_id = player.attrib.get("PersonId")
            if not person_id:
                continue
            players[person_id] = {
                "person_id": person_id,
                "team_id": team_id,
                "position_role": player.attrib.get("PlayingPosition"),
                "starting": player.attrib.get("Starting"),
            }

    home = next((item for item in teams.values() if item["role"] == "home"), None)
    away = next((item for item in teams.values() if item["role"] == "guest"), None)
    if home is None or away is None:
        ordered = sorted(teams.values(), key=lambda item: str(item["team_id"]))
        if len(ordered) != 2:
            raise ValueError(f"{match_id}: expected two teams, got {sorted(teams)}")
        home, away = ordered
    pitch_x = safe_float(environment.attrib.get("PitchX"))
    pitch_y = safe_float(environment.attrib.get("PitchY"))
    if pitch_x is None or pitch_y is None or pitch_x <= 0 or pitch_y <= 0:
        raise ValueError(f"{match_id}: invalid pitch dimensions")
    return {
        "source_match_id": match_id,
        "raw_match_id": raw_match_id,
        "competition": general.attrib.get("CompetitionName"),
        "competition_id": general.attrib.get("CompetitionId"),
        "season": general.attrib.get("Season"),
        "match_day": safe_int(general.attrib.get("MatchDay")),
        "match_title": general.attrib.get("MatchTitle"),
        "result": general.attrib.get("Result"),
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "teams": teams,
        "home_team_id": home["team_id"],
        "away_team_id": away["team_id"],
        "players": players,
    }


def first_leaf(element: ET.Element) -> str:
    children = list(element)
    if not children:
        return strip_tag(element.tag)
    return first_leaf(children[0])


def first_attr(element: ET.Element, names: Iterable[str]) -> str | None:
    wanted = set(names)
    for node in element.iter():
        for name in wanted:
            value = node.attrib.get(name)
            if value:
                return value
    return None


def parse_events(path: Path, match_id: str) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    timestamps: list[int] = []
    for _, event in ET.iterparse(path, events=("end",)):
        if strip_tag(event.tag) != "Event":
            continue
        event_time = event.attrib.get("EventTime")
        if not event_time:
            event.clear()
            continue
        timestamp_ms = iso_to_epoch_ms(event_time)
        children = list(event)
        root_type = strip_tag(children[0].tag) if children else "Unknown"
        leaf_type = first_leaf(children[0]) if children else root_type
        event_type = root_type if root_type == leaf_type else f"{root_type}:{leaf_type}"
        row = {
            "source_match_id": match_id,
            "event_index": len(rows),
            "event_id": event.attrib.get("EventId"),
            "timestamp_ms": timestamp_ms,
            "event_type": event_type,
            "event_root_type": root_type,
            "game_section": first_attr(event, ["GameSection"]),
            "x": safe_float(event.attrib.get("X-Position")),
            "y": safe_float(event.attrib.get("Y-Position")),
            "source_x": safe_float(event.attrib.get("X-Source-Position")),
            "source_y": safe_float(event.attrib.get("Y-Source-Position")),
            "player_id": first_attr(event, ["Player", "Winner", "Loser"]),
            "team_id": first_attr(event, ["Team", "WinnerTeam", "LoserTeam"]),
            "recipient_id": first_attr(event, ["Recipient"]),
            "outcome": first_attr(event, ["Evaluation", "Outcome", "Result"]),
        }
        rows.append(row)
        timestamps.append(timestamp_ms)
        event.clear()
    return rows, timestamps


def row_from_frame(
    frame: ET.Element,
    current: dict[str, str],
    match_id: str,
) -> dict[str, Any]:
    attrs = frame.attrib
    frame_n = safe_int(attrs.get("N"))
    timestamp = attrs.get("T")
    if frame_n is None or timestamp is None:
        raise ValueError(f"{match_id}: Frame missing N or T")
    team_id = current.get("TeamId", "")
    if team_id == "BALL":
        entity_type = "ball"
    elif team_id == "referee":
        entity_type = "referee"
    else:
        entity_type = "player"
    return {
        "source_match_id": match_id,
        "game_section": current.get("GameSection"),
        "team_id": team_id,
        "person_id": current.get("PersonId"),
        "entity_type": entity_type,
        "frame_n": frame_n,
        "timestamp_ms": iso_to_epoch_ms(timestamp),
        "x": safe_float(attrs.get("X")),
        "y": safe_float(attrs.get("Y")),
        "z": safe_float(attrs.get("Z")),
        "direction": safe_float(attrs.get("D")),
        "speed": safe_float(attrs.get("S")),
        "acceleration": safe_float(attrs.get("A")),
        "measurement_status": safe_int(attrs.get("M")),
        "ball_possession": safe_int(attrs.get("BallPossession")),
        "ball_status": safe_int(attrs.get("BallStatus")),
    }


def flush_writer(writer: pq.ParquetWriter, records: list[dict[str, Any]], schema: pa.Schema) -> None:
    if records:
        writer.write_table(pa.Table.from_pylist(records, schema=schema))
        records.clear()


def write_events(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=EVENT_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def convert_positions(
    path: Path,
    match_id: str,
    metadata: dict[str, Any],
    event_timestamps: list[int],
    frames_path: Path,
    sampled_path: Path,
) -> dict[str, Any]:
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frame_writer = pq.ParquetWriter(frames_path, FRAME_SCHEMA, compression="zstd")
    sample_writer = pq.ParquetWriter(sampled_path, FRAME_SCHEMA, compression="zstd")
    frame_buffer: list[dict[str, Any]] = []
    sample_buffer: list[dict[str, Any]] = []
    current: dict[str, str] | None = None
    section_first_n: dict[str, int] = {}
    last_n: int | None = None
    last_t: int | None = None
    frame_rows = 0
    sampled_rows = 0
    frame_set_count = 0
    frame_number_gap_count = 0
    time_gap_count_gt_80ms = 0
    max_time_gap_ms = 0
    min_frame_n: int | None = None
    max_frame_n: int | None = None
    min_timestamp_ms: int | None = None
    max_timestamp_ms: int | None = None
    missing_xy = 0
    out_of_pitch = 0
    ball_rows = 0
    player_rows = 0
    referee_rows = 0
    entity_ids_by_section: dict[str, set[str]] = defaultdict(set)
    sample_keys: set[tuple[str, int]] = set()
    sample_player_counts: Counter[tuple[str, int]] = Counter()
    sample_outfield_counts: Counter[tuple[str, int]] = Counter()
    sample_ball_keys: set[tuple[str, int]] = set()
    sampled_timestamps: set[int] = set()

    try:
        for event, element in ET.iterparse(path, events=("start", "end")):
            tag = strip_tag(element.tag)
            if event == "start" and tag == "FrameSet":
                current = {str(k): str(v) for k, v in element.attrib.items()}
                frame_set_count += 1
                last_n = None
                last_t = None
                continue
            if event != "end" or tag != "Frame":
                if event == "end" and tag == "FrameSet":
                    element.clear()
                continue
            if current is None:
                raise ValueError(f"{match_id}: Frame found outside FrameSet")
            row = row_from_frame(element, current, match_id)
            frame_n = int(row["frame_n"])
            timestamp_ms = int(row["timestamp_ms"])
            section = str(row["game_section"])
            if section not in section_first_n:
                section_first_n[section] = frame_n
            if last_n is not None:
                frame_gap = frame_n - last_n - 1
                if frame_gap > 0:
                    frame_number_gap_count += 1
                if last_t is not None:
                    delta_ms = timestamp_ms - last_t
                    if delta_ms > max_time_gap_ms:
                        max_time_gap_ms = delta_ms
                    if delta_ms > 80:
                        time_gap_count_gt_80ms += 1
            last_n = frame_n
            last_t = timestamp_ms
            frame_rows += 1
            min_frame_n = frame_n if min_frame_n is None else min(min_frame_n, frame_n)
            max_frame_n = frame_n if max_frame_n is None else max(max_frame_n, frame_n)
            min_timestamp_ms = timestamp_ms if min_timestamp_ms is None else min(min_timestamp_ms, timestamp_ms)
            max_timestamp_ms = timestamp_ms if max_timestamp_ms is None else max(max_timestamp_ms, timestamp_ms)
            entity_type = str(row["entity_type"])
            if entity_type == "ball":
                ball_rows += 1
            elif entity_type == "player":
                player_rows += 1
            else:
                referee_rows += 1
            person_id = str(row["person_id"])
            entity_ids_by_section[section].add(person_id)
            x, y = row["x"], row["y"]
            if x is None or y is None:
                missing_xy += 1
            elif not (-float(metadata["pitch_x"]) / 2.0 <= float(x) <= float(metadata["pitch_x"]) / 2.0 and -float(metadata["pitch_y"]) / 2.0 <= float(y) <= float(metadata["pitch_y"]) / 2.0):
                out_of_pitch += 1
            frame_buffer.append(row)
            if len(frame_buffer) >= 65536:
                flush_writer(frame_writer, frame_buffer, FRAME_SCHEMA)

            if (frame_n - section_first_n[section]) % SAMPLE_STRIDE == 0:
                sampled_rows += 1
                sample_buffer.append(row)
                sample_key = (section, frame_n)
                sample_keys.add(sample_key)
                sampled_timestamps.add(timestamp_ms)
                if entity_type == "player":
                    sample_player_counts[sample_key] += 1
                    if not is_goalkeeper(metadata.get("players", {}).get(str(row["person_id"]))):
                        sample_outfield_counts[sample_key] += 1
                elif entity_type == "ball":
                    sample_ball_keys.add(sample_key)
                if len(sample_buffer) >= 65536:
                    flush_writer(sample_writer, sample_buffer, FRAME_SCHEMA)
            element.clear()
    finally:
        flush_writer(frame_writer, frame_buffer, FRAME_SCHEMA)
        flush_writer(sample_writer, sample_buffer, FRAME_SCHEMA)
        frame_writer.close()
        sample_writer.close()

    sample_counts = list(sample_player_counts.values())
    outfield_sample_counts = list(sample_outfield_counts.values())
    sample_keys_with_players = len(sample_player_counts)
    in_range_events = [
        timestamp
        for timestamp in event_timestamps
        if min_timestamp_ms is not None
        and max_timestamp_ms is not None
        and min_timestamp_ms <= timestamp <= max_timestamp_ms
    ]
    sorted_sample_times = sorted(sampled_timestamps)
    nearest_sample_deltas = []
    for timestamp in event_timestamps:
        if not sorted_sample_times:
            continue
        index = bisect.bisect_left(sorted_sample_times, timestamp)
        candidates = []
        if index < len(sorted_sample_times):
            candidates.append(abs(sorted_sample_times[index] - timestamp))
        if index > 0:
            candidates.append(abs(sorted_sample_times[index - 1] - timestamp))
        if candidates:
            nearest_sample_deltas.append(min(candidates))
    return {
        "source_match_id": match_id,
        "frame_rows": frame_rows,
        "sampled_rows": sampled_rows,
        "frame_set_count": frame_set_count,
        "section_entity_counts": {key: len(value) for key, value in sorted(entity_ids_by_section.items())},
        "player_rows": player_rows,
        "ball_rows": ball_rows,
        "referee_rows": referee_rows,
        "frame_n_range": [min_frame_n, max_frame_n],
        "timestamp_ms_range": [min_timestamp_ms, max_timestamp_ms],
        "missing_xy_rows": missing_xy,
        "out_of_pitch_rows": out_of_pitch,
        "frame_number_gap_count": frame_number_gap_count,
        "time_gap_count_gt_80ms": time_gap_count_gt_80ms,
        "max_within_frameset_time_gap_ms": max_time_gap_ms,
        "sampled_snapshot_count": len(sample_keys),
        "sampled_snapshot_with_player_rows": sample_keys_with_players,
        "sampled_player_count_min": min(sample_counts) if sample_counts else None,
        "sampled_player_count_max": max(sample_counts) if sample_counts else None,
        "sampled_source_player_count_expected": 22,
        "sampled_exact_22_source_player_fraction": (
            float(sum(count == 22 for count in sample_counts) / len(sample_counts))
            if sample_counts
            else None
        ),
        "sampled_outfield_count_min": min(outfield_sample_counts) if outfield_sample_counts else None,
        "sampled_outfield_count_max": max(outfield_sample_counts) if outfield_sample_counts else None,
        "sampled_exact_20_outfield_fraction": (
            float(sum(count == N_PLAYER_NODES for count in outfield_sample_counts) / len(outfield_sample_counts))
            if outfield_sample_counts
            else None
        ),
        "sampled_ball_snapshot_fraction": (
            float(len(sample_ball_keys) / len(sample_keys)) if sample_keys else None
        ),
        "event_count": len(event_timestamps),
        "event_time_in_full_position_range_fraction": (
            float(len(in_range_events) / len(event_timestamps)) if event_timestamps else None
        ),
        "event_nearest_sample_median_ms": (
            float(np.median(nearest_sample_deltas)) if nearest_sample_deltas else None
        ),
        "event_nearest_sample_max_ms": (
            int(max(nearest_sample_deltas)) if nearest_sample_deltas else None
        ),
        "event_alignment_reference": "1Hz sampled source timestamps; full-range check uses all canonical frames",
        "canonical_coordinate_units": "meters in source pitch coordinates centered at pitch origin; task views divide x by pitch_x/2 and y by pitch_y/2",
    }


def inventory_raw(raw_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = []
    hashes: dict[str, str] = {}
    for path in sorted(raw_root.iterdir()):
        if not path.is_file():
            continue
        digest = sha256_file(path)
        rel = path.relative_to(raw_root).as_posix()
        hashes[rel] = digest
        rows.append({"relative_path": rel, "bytes": path.stat().st_size, "sha256": digest})
    return rows, hashes


def convert_all(raw_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.time()
    grouped = discover_files(raw_root)
    inventory, raw_hashes = inventory_raw(raw_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "canonical" / "frames").mkdir(parents=True, exist_ok=True)
    (output_root / "canonical" / "sampled_frames").mkdir(parents=True, exist_ok=True)
    (output_root / "canonical" / "events").mkdir(parents=True, exist_ok=True)
    (output_root / "canonical" / "conversion_receipts").mkdir(parents=True, exist_ok=True)

    metadata_records = []
    match_receipts = []
    for match_id in sorted(grouped):
        files = grouped[match_id]
        metadata = parse_metadata(files["metadata"])
        if metadata["source_match_id"] != match_id:
            raise ValueError(f"metadata/file ID mismatch for {match_id}")
        events, event_timestamps = parse_events(files["events"], match_id)
        events_path = output_root / "canonical" / "events" / f"{match_id}.parquet"
        write_events(events, events_path)
        qc = convert_positions(
            files["positions"],
            match_id,
            metadata,
            event_timestamps,
            output_root / "canonical" / "frames" / f"{match_id}.parquet",
            output_root / "canonical" / "sampled_frames" / f"{match_id}.parquet",
        )
        metadata_public = {key: value for key, value in metadata.items() if key not in {"players"}}
        metadata_public["player_count_in_match_info"] = len(metadata["players"])
        metadata_public["players"] = metadata["players"]
        metadata_records.append(metadata_public)
        receipt = {
            "status": "CANONICAL_CONVERSION_COMPLETE",
            "source_match_id": match_id,
            "split": MATCH_SPLITS[match_id],
            "source_files": {
                kind: {
                    "path": relative_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": raw_hashes[path.relative_to(raw_root).as_posix()],
                }
                for kind, path in sorted(files.items())
            },
            "outputs": {
                "frames": relative_path(output_root / "canonical" / "frames" / f"{match_id}.parquet"),
                "sampled_frames": relative_path(output_root / "canonical" / "sampled_frames" / f"{match_id}.parquet"),
                "events": relative_path(events_path),
            },
            "output_sha256": {
                "frames": sha256_file(output_root / "canonical" / "frames" / f"{match_id}.parquet"),
                "sampled_frames": sha256_file(output_root / "canonical" / "sampled_frames" / f"{match_id}.parquet"),
                "events": sha256_file(events_path),
            },
            "metadata": {
                "pitch_x": metadata["pitch_x"],
                "pitch_y": metadata["pitch_y"],
                "home_team_id": metadata["home_team_id"],
                "away_team_id": metadata["away_team_id"],
            },
            "qc": qc,
        }
        write_json(output_root / "canonical" / "conversion_receipts" / f"{match_id}.json", receipt)
        match_receipts.append(receipt)

    write_json(output_root / "match_metadata.json", metadata_records)
    write_json(output_root / "source_file_manifest.json", {
        "status": "RAW_SOURCE_MANIFEST_COMPLETE",
        "raw_root": relative_path(raw_root),
        "file_count": len(inventory),
        "total_bytes": sum(int(row["bytes"]) for row in inventory),
        "files": inventory,
    })
    (output_root / "RAW_SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['relative_path']}\n" for row in inventory),
        encoding="utf-8",
    )
    canonical_manifest = {
        "status": "IDSSE_T5R1_CANONICAL_CONVERSION_COMPLETE",
        "source_dataset": "IDSSE",
        "official_dataset_url": OFFICIAL_DATASET_URL,
        "official_file_tree_url": OFFICIAL_TREE_URL,
        "official_revision": OFFICIAL_REVISION,
        "figshare_url": FIGSHARE_URL,
        "raw_root": relative_path(raw_root),
        "sample_hz": SAMPLE_HZ,
        "source_fps": SOURCE_FPS,
        "sample_stride": SAMPLE_STRIDE,
        "coordinate_policy": "canonical frames retain source meter coordinates; task arrays normalize x about pitch center and y about field center",
        "match_count": len(match_receipts),
        "match_ids": sorted(MATCH_SPLITS),
        "split_lock": MATCH_SPLITS,
        "frame_schema": [field.name for field in FRAME_SCHEMA],
        "event_schema": [field.name for field in EVENT_SCHEMA],
        "matches": [
            {
                "source_match_id": receipt["source_match_id"],
                "split": receipt["split"],
                "frame_rows": receipt["qc"]["frame_rows"],
                "sampled_rows": receipt["qc"]["sampled_rows"],
                "event_count": receipt["qc"]["event_count"],
                "receipt": relative_path(output_root / "canonical" / "conversion_receipts" / f"{receipt['source_match_id']}.json"),
            }
            for receipt in match_receipts
        ],
        "raw_manifest_sha256": sha256_file(output_root / "source_file_manifest.json"),
        "raw_sha256sums_sha256": sha256_file(output_root / "RAW_SHA256SUMS"),
        "generated_by": relative_path(Path(__file__).resolve()),
    }
    write_json(output_root / "canonical_manifest.json", canonical_manifest)
    write_json(output_root / "source_provenance.json", {
        "status": "OFFICIAL_PAGE_VERIFIED_RAW_MANIFEST_AND_CANONICAL_QC_COMPLETE",
        "dataset": "IDSSE",
        "official_dataset_url": OFFICIAL_DATASET_URL,
        "official_file_tree_url": OFFICIAL_TREE_URL,
        "official_revision": OFFICIAL_REVISION,
        "original_source": FIGSHARE_URL,
        "license": "CC BY 4.0",
        "official_file_tree_match_ids": sorted(MATCH_SPLITS),
        "official_card_text_id_note": "official card text also lists J03WPF and J03WQF; current main file tree lists J03WQQ and J03WR9; raw identity follows the file tree",
        "viewer_note": "Dataset Viewer default/train was unavailable with FeaturesError/FileNotFoundError; schema evidence comes from local XML conversion and receipts",
        "raw_root": relative_path(raw_root),
        "raw_file_manifest": relative_path(output_root / "source_file_manifest.json"),
        "raw_sha256sums": relative_path(output_root / "RAW_SHA256SUMS"),
        "canonical_manifest": relative_path(output_root / "canonical_manifest.json"),
        "canonical_qc_receipts": len(match_receipts),
        "elapsed_seconds": round(time.time() - started, 3),
    })
    return canonical_manifest


def zone_label(value: float) -> str:
    if value < -1.0 / 3.0:
        return "defensive_third"
    if value < 1.0 / 3.0:
        return "middle_third"
    return "attacking_third"


def pairwise_signature(positions: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(positions.shape[1], k=1)
    differences = positions[:, :, None, :] - positions[:, None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=-1, dtype=np.float32))
    return distances[:, upper[0], upper[1]]


def shape_descriptor(signatures: np.ndarray, n_nodes: int = N_PLAYER_NODES) -> np.ndarray:
    """Return a compact, response-blind descriptor used only for candidate lookup.

    Final pair acceptance still uses the full pairwise-distance signature.  The
    descriptor avoids an O(n^2) full distance matrix while keeping the lookup
    deterministic in environments where scipy/sklearn cannot be imported.
    """
    upper = np.triu_indices(n_nodes, k=1)
    home = (upper[0] < n_nodes // 2) & (upper[1] < n_nodes // 2)
    away = (upper[0] >= n_nodes // 2) & (upper[1] >= n_nodes // 2)
    cross = ~(home | away)
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    parts = [
        np.quantile(signatures[:, mask], quantiles, axis=1).T
        for mask in (home, away, cross)
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def nearest_indices(query: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Deterministic top-k Euclidean lookup using NumPy only."""
    distances = np.sum((values - query[None, :]) ** 2, axis=1, dtype=np.float32)
    k = min(int(k), len(values))
    candidates = np.argpartition(distances, k - 1)[:k]
    return candidates[np.argsort(distances[candidates], kind="mergesort")]


def normalize_positions(group: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    pitch_x = float(metadata["pitch_x"])
    pitch_y = float(metadata["pitch_y"])
    x = group["x"].to_numpy(dtype=np.float32) / (pitch_x / 2.0)
    y = group["y"].to_numpy(dtype=np.float32) / (pitch_y / 2.0)
    return np.column_stack([x, y]).astype(np.float32)


def load_metadata_map(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["source_match_id"]): record for record in records}


def build_snapshots(
    canonical_root: Path,
    task_root: Path,
    metadata_map: dict[str, dict[str, Any]],
    split: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    match_ids = [match_id for match_id, value in MATCH_SPLITS.items() if value == split]
    records: list[dict[str, Any]] = []
    raw_arrays: list[np.ndarray] = []
    centered_arrays: list[np.ndarray] = []
    team_arrays: list[np.ndarray] = []
    for match_id in sorted(match_ids):
        metadata = metadata_map[match_id]
        home = str(metadata["home_team_id"])
        away = str(metadata["away_team_id"])
        sample_path = canonical_root / "canonical" / "sampled_frames" / f"{match_id}.parquet"
        sampled = pd.read_parquet(sample_path)
        player_rows = sampled[sampled["entity_type"] == "player"].copy()
        ball_keys = set(
            zip(
                sampled.loc[sampled["entity_type"] == "ball", "game_section"].astype(str),
                sampled.loc[sampled["entity_type"] == "ball", "frame_n"].astype(int),
            )
        )
        for (section, frame_n), group in player_rows.groupby(["game_section", "frame_n"], sort=True):
            group = group.copy()
            group["team_slot"] = group["team_id"].map({home: 0, away: 1})
            group = group[group["team_slot"].notna()]
            group = group[
                ~group["person_id"].astype(str).map(
                    lambda person_id: is_goalkeeper(metadata.get("players", {}).get(person_id))
                )
            ]
            if len(group) != N_PLAYER_NODES or group["person_id"].nunique() != N_PLAYER_NODES:
                continue
            if group["team_slot"].value_counts().to_dict() != {0: 10, 1: 10}:
                continue
            group = group.sort_values(["team_slot", "person_id"], kind="mergesort")
            raw = normalize_positions(group, metadata)
            if not np.isfinite(raw).all():
                continue
            centered = raw - raw.mean(axis=0, keepdims=True)
            team_slots = group["team_slot"].to_numpy(dtype=np.int64)
            timestamp_values = group["timestamp_ms"].to_numpy(dtype=np.int64)
            timestamp_ms = int(np.median(timestamp_values))
            if int(timestamp_values.max() - timestamp_values.min()) > 1:
                continue
            home_centroid = raw[team_slots == 0].mean(axis=0)
            away_centroid = raw[team_slots == 1].mean(axis=0)
            snapshot_id = f"idsse_{match_id}_{section}_{int(frame_n)}"
            raw_arrays.append(raw)
            centered_arrays.append(centered)
            team_arrays.append(team_slots)
            records.append(
                {
                    "snapshot_id": snapshot_id,
                    "source_match_id": match_id,
                    "split": split,
                    "game_section": str(section),
                    "frame_n": int(frame_n),
                    "timestamp_ms": timestamp_ms,
                    "player_count": int(len(group)),
                    "ball_present": (str(section), int(frame_n)) in ball_keys,
                    "home_team_id": home,
                    "away_team_id": away,
                    "player_ids": json.dumps(group["person_id"].astype(str).tolist(), separators=(",", ":")),
                    "phase_label": str(section),
                    "home_field_zone": zone_label(float(home_centroid[0])),
                    "away_field_zone": zone_label(float(away_centroid[0])),
                    "home_centroid_x": float(home_centroid[0]),
                    "home_centroid_y": float(home_centroid[1]),
                    "away_centroid_x": float(away_centroid[0]),
                    "away_centroid_y": float(away_centroid[1]),
                }
            )
    index = pd.DataFrame.from_records(records)
    if index.empty:
        raise ValueError(f"no complete 20-player snapshots for split {split}")
    arrays = (
        np.stack(raw_arrays).astype(np.float32),
        np.stack(centered_arrays).astype(np.float32),
        np.stack(team_arrays).astype(np.int64),
    )
    task_root.mkdir(parents=True, exist_ok=True)
    index.to_parquet(task_root / f"snapshot_index_{split}.parquet", index=False, compression="zstd")
    np.save(task_root / f"positions_raw_{split}.npy", arrays[0])
    np.save(task_root / f"positions_centered_{split}.npy", arrays[1])
    np.save(task_root / f"team_slots_{split}.npy", arrays[2])
    return index, arrays[0], arrays[1], arrays[2]


def build_natural_pairs(index: pd.DataFrame, positions: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    positions_by_id = {str(row.snapshot_id): positions[i] for i, row in index.reset_index(drop=True).iterrows()}
    records: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    for (match_id, section), group in index.groupby(["source_match_id", "game_section"], sort=True):
        group = group.sort_values(["timestamp_ms", "frame_n"], kind="mergesort").reset_index(drop=True)
        ids = group["snapshot_id"].astype(str).tolist()
        matrix = np.stack([positions_by_id[snapshot_id] for snapshot_id in ids]).astype(np.float32)
        signatures = pairwise_signature(matrix)
        centroids = matrix.mean(axis=1)
        times = group["timestamp_ms"].to_numpy(dtype=np.int64)
        if len(group) < 3:
            continue
        neighbor_k = min(PAIR_NEIGHBOR_K, len(group))
        descriptors = shape_descriptor(signatures)
        anchor_indices = range(0, len(group), PAIR_ANCHOR_STRIDE)
        for row_offset, anchor_local in enumerate(anchor_indices):
            internal_neighbors = nearest_indices(descriptors[anchor_local], descriptors, neighbor_k)
            centroid_neighbors = nearest_indices(centroids[anchor_local], centroids, neighbor_k)
            positive: tuple[int, float, float, int] | None = None
            for rank, candidate_local in enumerate(internal_neighbors):
                candidate_local = int(candidate_local)
                if candidate_local == anchor_local:
                    continue
                time_gap_s = abs(int(times[candidate_local]) - int(times[anchor_local])) / 1000.0
                if time_gap_s < MIN_PAIR_TIME_SEPARATION_S:
                    continue
                centroid_distance = float(np.linalg.norm(centroids[candidate_local] - centroids[anchor_local]))
                internal_distance = float(
                    np.sqrt(np.mean((signatures[candidate_local] - signatures[anchor_local]) ** 2))
                )
                if centroid_distance >= POSITIVE_CENTROID_MIN and internal_distance <= POSITIVE_INTERNAL_MAX:
                    positive = (candidate_local, centroid_distance, internal_distance, rank)
                    break
            if positive is None:
                diagnostics["missing_positive"] += 1
                continue
            negative: tuple[int, float, float, int] | None = None
            for rank, candidate_local in enumerate(centroid_neighbors):
                candidate_local = int(candidate_local)
                if candidate_local == anchor_local or candidate_local == positive[0]:
                    continue
                time_gap_s = abs(int(times[candidate_local]) - int(times[anchor_local])) / 1000.0
                if time_gap_s < MIN_PAIR_TIME_SEPARATION_S:
                    continue
                centroid_distance = float(np.linalg.norm(centroids[candidate_local] - centroids[anchor_local]))
                internal_distance = float(
                    np.sqrt(np.mean((signatures[candidate_local] - signatures[anchor_local]) ** 2))
                )
                if centroid_distance <= HARD_NEGATIVE_CENTROID_MAX and internal_distance >= HARD_NEGATIVE_INTERNAL_MIN:
                    negative = (candidate_local, centroid_distance, internal_distance, rank)
                    break
            if negative is None:
                diagnostics["missing_hard_negative"] += 1
                continue
            query_id = ids[anchor_local]
            positive_id = ids[positive[0]]
            negative_id = ids[negative[0]]
            records.append(
                {
                    "pair_id": f"{query_id}__pos__{positive_id}__neg__{negative_id}",
                    "query_snapshot_id": query_id,
                    "positive_snapshot_id": positive_id,
                    "negative_snapshot_id": negative_id,
                    "source_match_id": str(match_id),
                    "game_section": str(section),
                    "split": str(group.loc[0, "split"]),
                    "query_timestamp_ms": int(times[anchor_local]),
                    "positive_timestamp_ms": int(times[positive[0]]),
                    "negative_timestamp_ms": int(times[negative[0]]),
                    "positive_time_gap_s": abs(int(times[positive[0]]) - int(times[anchor_local])) / 1000.0,
                    "negative_time_gap_s": abs(int(times[negative[0]]) - int(times[anchor_local])) / 1000.0,
                    "positive_centroid_distance": positive[1],
                    "positive_internal_geometry_distance": positive[2],
                    "negative_centroid_distance": negative[1],
                    "negative_internal_geometry_distance": negative[2],
                    "positive_internal_neighbor_rank": positive[3],
                    "negative_centroid_neighbor_rank": negative[3],
                    "construction_rule": "natural_temporal_snapshots_centered_pairwise_geometry_close_abs_centroid_far_vs_abs_centroid_close_geometry_far",
                }
            )
    pairs = pd.DataFrame.from_records(records)
    if pairs.empty:
        raise ValueError(
            "natural pair construction produced no triplets; inspect fixed thresholds before changing the protocol"
        )
    diagnostics.update({"pairs": int(len(pairs)), "anchor_stride": PAIR_ANCHOR_STRIDE})
    return pairs, dict(diagnostics)


def make_split_lock(output_path: Path) -> dict[str, Any]:
    value = {
        "status": "MATCH_SPLIT_FROZEN_BEFORE_T5R2_TASK_CONSTRUCTION",
        "primary_unit": "source_match_id",
        "split_map": MATCH_SPLITS,
        "train_match_count": sum(value == "train" for value in MATCH_SPLITS.values()),
        "valid_match_count": sum(value == "valid" for value in MATCH_SPLITS.values()),
        "reserved_holdout_match_count": sum(value == "reserved_holdout" for value in MATCH_SPLITS.values()),
        "selection_rule": "fixed ID-only assignment; no event count, score, model result, or holdout result used",
        "frame_split_forbidden": True,
        "event_split_forbidden": True,
        "reserved_holdout_policy": "do not load into T5R2 task views; candidate lock required before one read",
        "same_source_not_independent_provider": True,
        "official_revision": OFFICIAL_REVISION,
    }
    write_json(output_path, value)
    return value


def baseline_lock(
    task_root: Path,
    idsse_root: Path,
    split_lock: dict[str, Any],
    snapshot_indices: dict[str, pd.DataFrame],
    pair_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    lock = {
        "status": "T5R2_CLOSED_BASELINE_AND_METRIC_LOCK",
        "task": "P3-T5R2",
        "phase": "P3_CAUSAL_MECHANISM",
        "dataset": {
            "id": "IDSSE",
            "role": "temporary_substitute_for_sngar_development",
            "official_dataset_url": OFFICIAL_DATASET_URL,
            "official_revision": OFFICIAL_REVISION,
            "canonical_manifest": relative_path(idsse_root / "canonical_manifest.json"),
            "canonical_manifest_sha256": sha256_file(idsse_root / "canonical_manifest.json"),
            "source_manifest": relative_path(idsse_root / "source_file_manifest.json"),
            "source_manifest_sha256": sha256_file(idsse_root / "source_file_manifest.json"),
        },
        "split_lock": split_lock,
        "task_views": {
            split: {
                "snapshot_index": relative_path(task_root / f"snapshot_index_{split}.parquet"),
                "snapshot_index_sha256": sha256_file(task_root / f"snapshot_index_{split}.parquet"),
                "positions_raw": relative_path(task_root / f"positions_raw_{split}.npy"),
                "positions_raw_sha256": sha256_file(task_root / f"positions_raw_{split}.npy"),
                "positions_centered": relative_path(task_root / f"positions_centered_{split}.npy"),
                "positions_centered_sha256": sha256_file(task_root / f"positions_centered_{split}.npy"),
                "team_slots": relative_path(task_root / f"team_slots_{split}.npy"),
                "team_slots_sha256": sha256_file(task_root / f"team_slots_{split}.npy"),
                "snapshot_count": int(len(snapshot_indices[split])),
                "pair_table": relative_path(task_root / f"natural_pair_ranking_{split}.parquet"),
                "pair_table_sha256": sha256_file(task_root / f"natural_pair_ranking_{split}.parquet"),
                "pair_count": int(len(pair_tables[split])),
            }
            for split in ("train", "valid")
        },
        "context_tasks": {
            "phase_deployment_classification": {
                "label": "phase_label",
                "classes": ["firstHalf", "secondHalf"],
                "primary_metric": "match_grouped_macro_f1",
                "secondary_metric": "match_grouped_accuracy",
                "representation": "z_ctx",
                "input_view": "raw_positions",
            },
            "absolute_team_centroid_regression": {
                "targets": ["home_centroid_x", "home_centroid_y", "away_centroid_x", "away_centroid_y"],
                "primary_metric": "match_grouped_mae",
                "secondary_metric": "match_grouped_r2",
                "representation": "z_ctx",
                "input_view": "raw_positions",
            },
            "absolute_field_zone_classification": {
                "label": "home_field_zone",
                "classes": ["defensive_third", "middle_third", "attacking_third"],
                "primary_metric": "match_grouped_macro_f1",
                "representation": "z_ctx",
                "input_view": "raw_positions",
            },
            "translation_reporting": "report z_ctx accessibility and sensitivity; do not optimize for lower response",
        },
        "intrinsic_task": {
            "name": "natural_pair_ranking",
            "positive_rule": f"same_match_same_half natural snapshot with centered pairwise geometry distance <= {POSITIVE_INTERNAL_MAX} and absolute centroid distance >= {POSITIVE_CENTROID_MIN}",
            "hard_negative_rule": f"same_match_same_half natural snapshot with absolute centroid distance <= {HARD_NEGATIVE_CENTROID_MAX} and centered pairwise geometry distance >= {HARD_NEGATIVE_INTERNAL_MIN}",
            "minimum_time_separation_seconds": MIN_PAIR_TIME_SEPARATION_S,
            "pair_anchor_stride_in_one_hz_snapshots": PAIR_ANCHOR_STRIDE,
            "representation": "z_mode",
            "input_view": "globally_centered_positions",
            "primary_metrics": ["match_grouped_ranking_accuracy", "match_grouped_mrr_at_2"],
            "secondary_metrics": ["hard_negative_ranking_accuracy", "raw_coordinate_shortcut_audit"],
            "intervention_labels_as_task": False,
            "cross_split_pairs": False,
        },
        "representation_contract": {
            "z_ctx": "shared encoder pooled from pitch-normalized raw positions; absolute deployment retained",
            "z_mode": "shared encoder pooled from positions after subtracting the 20-player global centroid",
            "team_features": "two-hot team slot is allowed; player role is not a required runtime input",
            "coordinate_normalization": "x=X/(pitch_x/2), y=Y/(pitch_y/2); source X/Y are centered at the pitch origin",
            "graph": "fixed weighted kNN-4 from each snapshot normalized coordinates; no learned graph",
        },
        "baseline_models": [
            {"id": "raw_single_channel_phase_gat_team_mean", "input": "raw_positions", "pooling": "team_mean", "channel": "z_ctx"},
            {"id": "centered_single_channel_phase_gat_team_mean", "input": "centered_positions", "pooling": "team_mean", "channel": "z_mode"},
            {"id": "raw_relational_pooling_phase_gat", "input": "raw_positions", "pooling": "relational_pairwise", "channel": "z_ctx"},
            {"id": "fixed_dual_channel_shared_phase_gat", "input": "raw_and_centered_positions", "pooling": "team_mean", "channel": "z_ctx_and_z_mode", "status": "T5R3_minimal_sanity_only"},
            {"id": "raw_coordinate_procrustes", "input": "raw_coordinates", "pooling": "geometry_free_control", "channel": "intrinsic_control"},
        ],
        "training_lock": {
            "seeds": SEEDS,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 80,
            "batch_size": 64,
            "latent_dim": 128,
            "hidden_dim": 128,
            "message_passing_layers": 2,
            "parameter_matching": "same encoder width/depth; dual channel shares encoder and uses fixed task heads",
            "early_stopping": False,
            "class_weighting": False,
            "augmentation": "none",
        },
        "decision_rules": {
            "context_noninferiority_margin": {"macro_f1_absolute_drop_max": 0.05, "centroid_mae_increase_max": 0.05},
            "intrinsic_improvement": "positive direction on primary ranking metric across majority of matches and at least 2 of 3 seeds",
            "translation_robustness": "z_mode global translation response reported separately; lower is better for intrinsic channel only",
            "geometry_floor": "geometry-response Spearman must not fall below locked baseline tolerance in T5R3",
            "no_post_hoc_reweighting": True,
        },
        "statistics": {
            "primary_unit": "source_match_id",
            "seed_is_not_independent_match": True,
            "bootstrap": "match-level percentile bootstrap",
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 20260904,
            "leave_one_match_out": True,
        },
        "firewall": {
            "train_match_ids": [key for key, value in MATCH_SPLITS.items() if value == "train"],
            "valid_match_ids": [key for key, value in MATCH_SPLITS.items() if value == "valid"],
            "reserved_holdout_match_ids": [key for key, value in MATCH_SPLITS.items() if value == "reserved_holdout"],
            "reserved_holdout_task_arrays_written": False,
            "old_heldout_used": False,
            "model_results_used": False,
            "external_results_used": False,
            "candidate_lock_required_before_reserved_read": True,
        },
        "candidate_budget": {"max_rounds": 2, "max_new_candidates_per_round": 6, "candidate_lock_status": "NOT_CREATED"},
        "generated_by": relative_path(Path(__file__).resolve()),
    }
    write_json(T5R_ROOT / "baseline_and_metric_lock.json", lock)
    return lock


def build_tasks(idsse_root: Path, t5r_root: Path) -> dict[str, Any]:
    metadata_map = load_metadata_map(idsse_root / "match_metadata.json")
    split_lock = make_split_lock(t5r_root / "split_lock.json")
    task_root = t5r_root / "data_views"
    snapshot_indices: dict[str, pd.DataFrame] = {}
    positions_raw: dict[str, np.ndarray] = {}
    pair_tables: dict[str, pd.DataFrame] = {}
    pair_diagnostics: dict[str, Any] = {}
    label_summary: dict[str, Any] = {}
    for split in ("train", "valid"):
        index, raw, _, _ = build_snapshots(idsse_root, task_root, metadata_map, split)
        pairs, diagnostics = build_natural_pairs(index, raw)
        pairs.to_parquet(task_root / f"natural_pair_ranking_{split}.parquet", index=False, compression="zstd")
        index.to_parquet(task_root / f"context_labels_{split}.parquet", index=False, compression="zstd")
        snapshot_indices[split] = index
        positions_raw[split] = raw
        pair_tables[split] = pairs
        pair_diagnostics[split] = diagnostics
        label_summary[split] = {
            "match_counts": index.groupby("source_match_id").size().to_dict(),
            "phase_counts": index["phase_label"].value_counts().sort_index().to_dict(),
            "home_field_zone_counts": index["home_field_zone"].value_counts().sort_index().to_dict(),
            "ball_present_fraction": float(index["ball_present"].mean()),
        }
    lock = baseline_lock(task_root, idsse_root, split_lock, snapshot_indices, pair_tables)
    task_manifest = {
        "status": "T5R2_TASK_CONSTRUCTION_COMPLETE",
        "task": "P3-T5R2",
        "split_lock": relative_path(t5r_root / "split_lock.json"),
        "split_lock_sha256": sha256_file(t5r_root / "split_lock.json"),
        "baseline_and_metric_lock": relative_path(T5R_ROOT / "baseline_and_metric_lock.json"),
        "baseline_and_metric_lock_sha256": sha256_file(T5R_ROOT / "baseline_and_metric_lock.json"),
        "source_canonical_manifest": relative_path(idsse_root / "canonical_manifest.json"),
        "source_canonical_manifest_sha256": sha256_file(idsse_root / "canonical_manifest.json"),
        "task_views": {
            split: {
                "snapshot_count": int(len(snapshot_indices[split])),
                "pair_count": int(len(pair_tables[split])),
                "snapshot_index": relative_path(task_root / f"snapshot_index_{split}.parquet"),
                "context_labels": relative_path(task_root / f"context_labels_{split}.parquet"),
                "natural_pair_ranking": relative_path(task_root / f"natural_pair_ranking_{split}.parquet"),
                "positions_raw": relative_path(task_root / f"positions_raw_{split}.npy"),
                "positions_raw_sha256": sha256_file(task_root / f"positions_raw_{split}.npy"),
                "positions_centered": relative_path(task_root / f"positions_centered_{split}.npy"),
                "positions_centered_sha256": sha256_file(task_root / f"positions_centered_{split}.npy"),
                "team_slots": relative_path(task_root / f"team_slots_{split}.npy"),
                "team_slots_sha256": sha256_file(task_root / f"team_slots_{split}.npy"),
                "reserved_holdout_data_written": False,
            }
            for split in ("train", "valid")
        },
        "label_summary": label_summary,
        "pair_diagnostics": pair_diagnostics,
        "natural_pair_protocol": {
            "positive_internal_max": POSITIVE_INTERNAL_MAX,
            "positive_centroid_min": POSITIVE_CENTROID_MIN,
            "hard_negative_centroid_max": HARD_NEGATIVE_CENTROID_MAX,
            "hard_negative_internal_min": HARD_NEGATIVE_INTERNAL_MIN,
            "minimum_time_separation_s": MIN_PAIR_TIME_SEPARATION_S,
            "same_match_same_half": True,
            "no_cross_split": True,
        },
        "holdout_firewall": {
            "reserved_match_ids": [key for key, value in MATCH_SPLITS.items() if value == "reserved_holdout"],
            "reserved_match_loaded": False,
            "reserved_match_labels_used": False,
            "candidate_lock_required_before_read": True,
        },
        "status_boundary": "T5R2 closed; T5R3 fixed dual-channel sanity not run",
        "generated_by": relative_path(Path(__file__).resolve()),
    }
    write_json(t5r_root / "task_manifest.json", task_manifest)
    return {"split_lock": split_lock, "baseline_lock": lock, "task_manifest": task_manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("convert", "tasks", "all"), default="all")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--idsse-output", type=Path, default=IDSSE_ARTIFACT_ROOT)
    parser.add_argument("--t5r-output", type=Path, default=T5R_ROOT)
    args = parser.parse_args()
    result: dict[str, Any] = {}
    if args.stage in ("convert", "all"):
        result["conversion"] = convert_all(args.raw_root.resolve(), args.idsse_output.resolve())
    if args.stage in ("tasks", "all"):
        result["tasks"] = build_tasks(args.idsse_output.resolve(), args.t5r_output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
