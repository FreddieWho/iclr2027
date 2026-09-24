"""Rebuild the paper and write PAPER_BUILD_RECEIPT.json.

The receipt is regenerated from a real build rather than hand-edited, so the recorded
page counts, overfull count and source hashes always correspond to a build that happened.

Usage:
    python experiments/e1a933_review/paper_build_receipt.py [--out-dir DIR] [--publish]

--publish additionally copies the built PDF to reports/e1a933_review/paper_revised.pdf
(the current-round deliverable; paper/main.pdf is never replaced). The superseded hash is
recorded in the receipt so the previous build stays traceable.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
RECEIPT = ROOT / "reports/e1a933_review/PAPER_BUILD_RECEIPT.json"
PDFLATEX = Path("/home/huyudi/.TinyTeX/bin/x86_64-linux/pdflatex")

SOURCE_FILES = sorted(
    [p for p in (PAPER / "sections").glob("*.tex")] + [PAPER / "main.tex"]
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PDFLATEX),
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={out_dir}",
        "main.tex",
    ]
    logs = []
    for _ in range(2):  # twice so cross-references settle
        proc = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
        logs.append(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            tail = "\n".join(logs[-1].splitlines()[-25:])
            raise SystemExit(f"pdflatex failed (exit {proc.returncode}):\n{tail}")
    # First pass always warns on new labels. Count the settled pass only.
    return out_dir / "main.pdf", logs[-1]


def pdf_page_of(pdf, needle):
    """Physical page containing `needle`, via pdftotext (authoritative over aux labels)."""
    for page in range(1, 40):
        txt = subprocess.run(
            ["pdftotext", "-q", "-f", str(page), "-l", str(page), str(pdf), "-"],
            capture_output=True, text=True,
        ).stdout
        if needle in re.sub(r"\s+", " ", txt):
            return page
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/e1a933-paper-build"))
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()

    pdf, log = build(a.out_dir)
    total_pages = int(
        re.search(r"Output written on .*?\((\d+) pages", log).group(1)
    )
    # 'Underfull'/'Overfull' boxes reported by LaTeX
    overfull = len(re.findall(r"Overfull \\[hv]box", log))
    undefined = len(re.findall(r"LaTeX Warning: (?:Reference|Citation) .* undefined", log))
    conclusion_page = pdf_page_of(pdf, "oracle-known changes")

    receipt = {
        "compiler": str(PDFLATEX),
        "working_directory": "paper",
        "command": " ".join(
            [
                str(PDFLATEX),
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={a.out_dir}",
                "main.tex",
            ]
        ),
        "total_pages": total_pages,
        "conclusion_page": conclusion_page,
        "conclusion_page_method": "pdftotext physical page containing the conclusion text",
        "undefined_references": undefined,
        "overfull_boxes": overfull,
        "pdf_sha256": sha256(pdf),
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in SOURCE_FILES},
        "note": (
            "PDF hashes are not byte-reproducible across runs (pdflatex embeds a creation "
            "timestamp and document id); reproduce by comparing extracted text or page counts."
        ),
    }

    if RECEIPT.exists():
        try:
            prev = json.loads(RECEIPT.read_text())
            if prev.get("pdf_sha256") and prev["pdf_sha256"] != receipt["pdf_sha256"]:
                receipt["supersedes"] = {
                    "pdf_sha256": prev["pdf_sha256"],
                    "total_pages": prev.get("total_pages"),
                    "conclusion_page": prev.get("conclusion_page"),
                }
        except Exception:
            pass

    if a.publish:
        dest = ROOT / "reports/e1a933_review/paper_revised.pdf"
        shutil.copy2(pdf, dest)
        receipt["published_to"] = str(dest.relative_to(ROOT))
        receipt["published_pdf_sha256"] = sha256(dest)

    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "source_hashes"}, indent=2))
    print(f"source files hashed: {len(receipt['source_hashes'])}")
    assert overfull == 0, f"{overfull} overfull boxes"
    assert undefined == 0, f"{undefined} undefined references"


if __name__ == "__main__":
    main()
