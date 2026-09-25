"""Build the writing index from the independently reviewed claim records.

This command reads reports and explicitly referenced evidence only. It does not
train models, run inference, open sealed numerical pools, or regenerate papers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/submission_audit_20260925"
ARTIFACT = ROOT / "artifacts/submission_audit_20260925"
LANES = ("core", "mechanism", "routes", "vision", "controls")
FIELDS = (
    "id", "lane", "claim", "verdict", "allowed_wording", "forbidden_wording",
    "denominator", "code_paths", "evidence_paths", "recomputation",
    "limitations", "fixes",
)
SEALED = re.compile(r"holdout_?909|confirm_?1007|holdout_?895|J03WQQ|SoccerTrack-v2", re.I)
NUMERIC = {".npz", ".npy", ".pt", ".pth", ".parquet", ".h5", ".hdf5"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def path_value(value) -> str:
    if isinstance(value, dict):
        value = value.get("path", value.get("file", ""))
    value = str(value).strip().strip("`")
    value = re.sub(r":\d+(?:-\d+)?$", "", value)
    value = re.sub(r"#L\d+(?:-L?\d+)?$", "", value)
    return value


def evidence_record(raw_path, tracked: set[str]):
    raw_path = path_value(raw_path)
    candidate = Path(raw_path)
    path = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return {"path": raw_path, "availability": "OUTSIDE_REPOSITORY", "sha256": None}
    record = {"path": relative, "tracked_at_start": relative in tracked, "sha256": None}
    # Metadata only for sealed numerical evidence. No hashing or model loading.
    if SEALED.search(relative) and path.suffix.lower() in NUMERIC:
        record["availability"] = "SEALED_NOT_REOPENED"
    elif not path.exists():
        record["availability"] = "MISSING"
    elif path.is_dir():
        record["availability"] = "DIRECTORY_REFERENCE_ONLY"
    elif path.is_file():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256(path)
        record["availability"] = "TRACKED" if relative in tracked else "LOCAL_ONLY"
    else:
        record["availability"] = "UNSUPPORTED_FILE_TYPE"
    return record


def records_from(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("claims", data.get("records", data))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of claims")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true",
                        help="development only; final receipt must include every lane")
    args = parser.parse_args()
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    claims, errors, seen, sources = [], [], set(), []
    references = {}
    for lane in LANES:
        path = REPORT / lane / "claims.json"
        if not path.exists():
            errors.append(f"Missing lane: {lane}")
            continue
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
        for entry in records_from(path):
            row = {k: entry.get(k, "") for k in FIELDS}
            row["lane"] = lane
            if not row["id"]:
                errors.append(f"{lane}: claim without id")
                continue
            row["id"] = f"{lane}:{row['id']}"
            if row["id"] in seen:
                errors.append(f"Duplicate id: {row['id']}")
            seen.add(row["id"])
            for key in ("claim", "verdict", "allowed_wording", "denominator"):
                if not row[key]:
                    errors.append(f"{row['id']}: missing {key}")
            for key in ("code_paths", "evidence_paths"):
                row[key] = as_list(row[key])
                if not row[key]:
                    errors.append(f"{row['id']}: no {key}")
                for reference in row[key]:
                    p = path_value(reference)
                    if p not in references:
                        references[p] = evidence_record(reference, tracked)
                        references[p]["claim_ids"] = []
                    if row["id"] not in references[p]["claim_ids"]:
                        references[p]["claim_ids"].append(row["id"])
                    if references[p]["availability"] in {"MISSING", "OUTSIDE_REPOSITORY"}:
                        errors.append(f"{row['id']}: unresolved {key}: {p}")
            claims.append(row)
    manifest = sorted(references.values(), key=lambda x: x["path"])
    dump = lambda p, obj: p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    dump(REPORT / "CLAIMS.json", {"baseline_head": head, "claims": claims})
    with (REPORT / "CLAIMS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: text(v) for k, v in row.items()} for row in claims)
    dump(ARTIFACT / "EVIDENCE_MANIFEST.json", {"baseline_head": head, "files": manifest})
    with (REPORT / "ASSET_AVAILABILITY.csv").open("w", newline="") as handle:
        fields = ["path", "availability", "tracked_at_start", "bytes", "sha256", "claim_ids"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: text(row.get(k, "")) for k in fields} for row in manifest)
    versions = {}
    for package in ("numpy", "scipy", "pandas", "torch", "torchvision", "pytest", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    counts = {}
    for row in manifest:
        status = row["availability"]
        counts[status] = counts.get(status, 0) + 1
    receipt = {
        "status": "PASS" if not errors else "INCOMPLETE",
        "baseline_head": head, "python": sys.version, "package_versions": versions,
        "lane_sources": sources, "claims": len(claims), "evidence_entries": len(manifest),
        "availability_counts": counts, "errors": errors,
        "limitations": [
            "File existence and current SHA256 are provenance checks, not scientific validation.",
            "LOCAL_ONLY entries are not publicly downloadable merely because a hash exists.",
            "DIRECTORY_REFERENCE_ONLY entries do not certify the files inside the directory.",
            "Frozen numerical pools are not reopened; archived summaries are distinguished by each lane.",
            "Current source hashes describe the repaired checkout, not historical training bytes.",
        ],
    }
    dump(ARTIFACT / "INDEX_CHECK.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "claims", "evidence_entries", "availability_counts", "errors")}, ensure_ascii=False))
    if errors and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
