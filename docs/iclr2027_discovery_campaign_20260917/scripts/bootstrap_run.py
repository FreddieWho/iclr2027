#!/usr/bin/env python3
"""Read-only, bounded asset inventory; creates only a fresh output directory.

Does not clone, download, load checkpoints, validate historical experiments,
train models, call providers, or execute old approval gates.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

KNOWN = [
    'scripts/run_t5r3_sanity.py', 'scripts/p3_support_geometry.py',
    'artifacts/phase1/models', 'artifacts/phase3/task_semantic_repair_v1',
    'artifacts/data_v2/idsse', 'artifacts/data_v2/soccertrack',
    'artifacts/phase4_amr', 'paper',
]
BASELINE = 'f86dbdf6107bad3d156b250a3c23bd5dcd8bb9d2'


def git_output(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(['git', '-C', str(repo), *args], check=False,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def inventory(repo: Path) -> dict:
    entries = []
    for relative in KNOWN:
        p = repo / relative
        entries.append({'path': relative, 'exists': p.exists(),
                        'kind': 'directory' if p.is_dir() else 'file' if p.is_file() else 'missing'})
    found = []
    roots = [repo/'artifacts/phase1/models', repo/'artifacts/phase3/task_semantic_repair_v1', repo/'artifacts/phase4_amr']
    inspected = 0
    truncated = False
    for root in roots:
        if not root.is_dir():
            continue
        for here, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {'.git', '__pycache__', 'raw', 'data_views'})
            inspected += 1
            if inspected > 1000:
                truncated = True
                break
            for filename in sorted(files):
                p = Path(here)/filename
                if p.suffix.lower() in {'.pt','.pth','.ckpt'}:
                    found.append(str(p.relative_to(repo)))
            if len(found) >= 500:
                truncated = True
                break
        if truncated:
            break
    current = git_output(repo, 'rev-parse', 'HEAD')
    return {
        'kind': 'asset_inventory_not_scientific_audit',
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'repo': str(repo), 'pack_baseline_commit': BASELINE, 'current_commit': current,
        'commit_differs_from_pack': current is not None and current != BASELINE,
        'logical_cpus': os.cpu_count(), 'python': sys.version.split()[0],
        'known_entries': entries, 'checkpoint_candidate_paths': found[:500],
        'checkpoint_scan_truncated': truncated,
        'notes': [
            'File existence is not a checksum, compatibility, or authorization check.',
            'Checkpoint state keys and model adapters must be inspected by the agent.',
            'Missing historical assets do not block procedural-scene or simulator branches.',
            'No old audit, training, provider, remote write, or download was executed.',
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        parser.error('--repo must be an existing directory')
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('--output must be a new directory; choose a new run path')
    result = inventory(repo)
    output.mkdir(parents=True, exist_ok=False)
    (output/'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    text = '# Campaign state — initial inventory\n\n'
    text += 'This file is an engineering starting point, not a research verdict.\n\n'
    text += f"Current commit: `{result['current_commit']}`\n\n"
    text += '\n'.join(f"- {'FOUND' if row['exists'] else 'MISSING'} `{row['path']}`" for row in result['known_entries'])
    text += '\n\nNext: minimal adapters + shared oracle; run R01/R02/R05. No old-audit prerequisite.\n'
    (output/'CAMPAIGN_STATE.md').write_text(text, encoding='utf-8')
    print(f'Inventory written to {output}. No project experiment executed.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
