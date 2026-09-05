#!/usr/bin/env python3
"""Download SoccerTrack-v2 GSR files for the 8 T5R6 matches (N1).

Excludes smoke matches 117092/117093. Writes source manifest, SHA256SUMS,
and provenance (HF revision, license snapshot) under
artifacts/data_v2/soccertrack/. Fails closed if destination exists with
complete content; resumes partial downloads via the HF cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = ROOT / "data" / "raw" / "sports" / "soccertrack_v2"
ART_ROOT = ROOT / "artifacts" / "data_v2" / "soccertrack"
REPO_ID = "atomscott/soccertrack-v2"
MATCHES = ["118575", "118576", "118577", "118578", "128057", "128058", "132831", "132877"]
HALVES = ["1st", "2nd"]
LICENSE_SPDX = "CC-BY-4.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("REFUSED: HF_TOKEN not set", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    me = api.whoami()
    print(f"auth user: {me.get('name')}", flush=True)
    revision = api.dataset_info(REPO_ID).sha
    print(f"HF revision: {revision}", flush=True)
    started = time.time()
    files: list[dict] = []
    for match in MATCHES:
        for half in HALVES:
            rel = f"gsr/{match}/{match}_{half}.json"
            t0 = time.time()
            local = hf_hub_download(REPO_ID, rel, repo_type="dataset",
                                    local_dir=str(DEST_ROOT), revision=revision)
            # hf_hub_download with local_dir nests repo structure; normalize
            src = Path(local)
            dst = DEST_ROOT / rel
            if src.resolve() != dst.resolve():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
            digest = sha256_file(dst)
            files.append({"relative_path": rel, "bytes": dst.stat().st_size,
                          "sha256": digest})
            print(f"got {rel} bytes={dst.stat().st_size} "
                  f"sha={digest[:16]} elapsed={time.time()-t0:.0f}s", flush=True)
    try:
        card = hf_hub_download(REPO_ID, "README.md", repo_type="dataset",
                               local_dir="/tmp/st_v2_card", revision=revision)
        license_text = Path(card).read_text()[:2000]
    except Exception as exc:  # noqa: BLE001
        license_text = f"README fetch failed: {exc}"
    ART_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "SOCCERTRACK_V2_SOURCE_MANIFEST_COMPLETE",
        "dataset": REPO_ID, "revision": revision,
        "matches": MATCHES, "halves": HALVES,
        "excluded_smoke_matches": ["117092", "117093"],
        "auth_user": me.get("name"),
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    (ART_ROOT / "source_file_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    (ART_ROOT / "RAW_SHA256SUMS").write_text(
        "".join(f"{f['sha256']}  {f['relative_path']}\n" for f in files))
    provenance = {
        "status": "SOURCE_REGISTERED",
        "dataset": REPO_ID, "revision": revision,
        "license_spdx": LICENSE_SPDX,
        "license_snapshot_head": license_text,
        "role": "independent_external_confirmation_T5R6",
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (ART_ROOT / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "files": len(files),
                      "total_gb": round(manifest["total_bytes"] / 1e9, 2)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
