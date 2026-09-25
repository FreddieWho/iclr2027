#!/usr/bin/env python3
"""Verify and unpack previously downloaded GPU repair archives, without SSH."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unpack(archive, expected, out):
    actual = digest(archive)
    if actual != expected:
        raise ValueError(f"archive hash mismatch: {archive}")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (out / member.name).resolve()
            if not target.is_relative_to(out.resolve()) or not member.isfile():
                raise ValueError(f"unsafe archive member: {member.name}")
            data = tar.extractfile(member)
            if target.exists():
                # Idempotent verification; never replace a pre-existing file.
                h = hashlib.sha256()
                for block in iter(lambda: data.read(1024 * 1024), b""):
                    h.update(block)
                if digest(target) != h.hexdigest():
                    raise FileExistsError(f"refusing to replace {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as f:
                    for block in iter(lambda: data.read(1024 * 1024), b""):
                        f.write(block)
    return {"path": archive.name, "sha256": actual, "bytes": archive.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--checkpoints-sha256")
    args = parser.parse_args()
    out = args.directory / "out"
    archives = [unpack(args.directory / f"{args.stem}_evidence.tar.gz", args.evidence_sha256, out)]
    if args.checkpoints_sha256:
        archives.append(unpack(args.directory / f"{args.stem}_checkpoints.tar.gz", args.checkpoints_sha256, out))
    run = json.loads((out / "RUN_RECEIPT.json").read_text())
    if run["status"] != "SUCCEEDED":
        raise ValueError("remote computation did not succeed")
    verified, pending = [], []
    for entry in run["outputs"]:
        p = (out / entry["path"]).resolve()
        if not p.is_relative_to(out.resolve()):
            raise ValueError("unsafe output path in run receipt")
        if p.suffix in {".pt", ".pth"} and not args.checkpoints_sha256:
            pending.append(entry["path"])
            continue
        if not p.is_file() or p.stat().st_size != entry["bytes"] or digest(p) != entry["sha256"]:
            raise ValueError(f"collected output mismatch: {p}")
        verified.append(entry["path"])
    receipt = {"status": "COMPLETE" if not pending else "EVIDENCE_VERIFIED_CHECKPOINTS_PENDING",
               "archives": archives, "verified_count": len(verified), "verified": verified,
               "pending_checkpoints": pending}
    (args.directory / "COLLECTION_CHECK.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "verified_count": len(verified),
                      "pending_checkpoints": len(pending)}))


if __name__ == "__main__":
    main()
