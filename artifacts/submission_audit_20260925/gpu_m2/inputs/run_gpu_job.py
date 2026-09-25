#!/usr/bin/env python3
"""Execute an immutable, explicitly enumerated repair job on an authorized GPU.

Connection details belong to private SSH configuration, not this runner or its
receipts. Each step has its own output directory and an explicit time limit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    receipt = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
               "job_sha256": digest(args.job), "python": sys.version, "steps": [],
               "inputs": [], "outputs": []}
    receipt_path = out / "RUN_RECEIPT.json"
    save(receipt_path, receipt)
    start = time.monotonic()
    try:
        import torch
        import torchvision
        receipt["environment"] = {
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; this job must not fall back to CPU")
        for entry in job["input_files"]:
            actual = digest(entry["path"])
            receipt["inputs"].append({**entry, "actual_sha256": actual})
            if actual != entry["sha256"]:
                raise RuntimeError(f"input hash mismatch: {entry['path']}")
        env = os.environ.copy()
        env.update(job.get("environment", {}))
        for step in job["steps"]:
            step_id = step["id"]
            if not step_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in step_id):
                raise ValueError(f"invalid step id: {step_id!r}")
            command = [token.replace("{run_root}", str(out)) for token in step["argv"]]
            row = {"id": step_id, "argv": command, "status": "RUNNING"}
            receipt["steps"].append(row)
            save(receipt_path, receipt)
            print(json.dumps({"event": "step_start", "id": step_id}), flush=True)
            step_start = time.monotonic()
            with (out / "logs" / f"{step_id}.log").open("w") as log:
                completed = subprocess.run(command, cwd=job["cwd"], env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           timeout=step.get("timeout_seconds", 1800), check=False)
            row.update(returncode=completed.returncode, seconds=time.monotonic() - step_start,
                       status="SUCCEEDED" if completed.returncode == 0 else "FAILED")
            save(receipt_path, receipt)
            print(json.dumps({"event": "step_end", **row}), flush=True)
            if completed.returncode:
                raise RuntimeError(f"step {step_id} exited {completed.returncode}")
            for declared in step["outputs"]:
                path = out / declared
                if not path.is_file():
                    raise RuntimeError(f"missing declared output: {path}")
        for path in sorted(out.rglob("*")):
            if path.is_file() and path != receipt_path:
                receipt["outputs"].append({"path": path.relative_to(out).as_posix(),
                                           "bytes": path.stat().st_size, "sha256": digest(path)})
        receipt["status"] = "SUCCEEDED"
    except BaseException as exc:
        receipt["status"] = "FAILED"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["seconds"] = time.monotonic() - start
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(receipt_path, receipt)
    # Separate small scientific evidence from checkpoint transport. Both use
    # the same manifest, so omission of weights cannot masquerade as collection.
    for kind, include_weights in (("evidence", False), ("checkpoints", True)):
        archive = out.parent / f"{out.name}_{kind}.tar.gz"
        with tarfile.open(archive, "x:gz") as tar:
            for path in sorted(out.rglob("*")):
                if not path.is_file():
                    continue
                is_weight = path.suffix in {".pt", ".pth"}
                if is_weight == include_weights:
                    tar.add(path, arcname=path.relative_to(out).as_posix())
        print(json.dumps({"event": "archive", "path": str(archive),
                          "bytes": archive.stat().st_size, "sha256": digest(archive)}), flush=True)
    print(json.dumps({"event": "job_complete", "status": receipt["status"]}), flush=True)


if __name__ == "__main__":
    main()
