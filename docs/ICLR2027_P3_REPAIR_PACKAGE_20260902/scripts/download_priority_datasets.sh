#!/usr/bin/env bash
# Download helper for the ICLR2027 P3 task-semantic repair package.
# This script intentionally separates development from hidden-test downloads.
set -euo pipefail

ROOT="${DATA_ROOT:-data/raw}"
MODE="${1:-help}"

log() { printf '[data] %s\n' "$*"; }
die() { printf '[data][ERROR] %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    printf 'hf'
  elif command -v huggingface-cli >/dev/null 2>&1; then
    printf 'huggingface-cli'
  else
    die "Hugging Face CLI not found. Install: python -m pip install -U huggingface_hub"
  fi
}

require_candidate_lock() {
  local lock_path="${CANDIDATE_LOCK_PATH:-artifacts/phase3/task_semantic_repair_v1/candidate_lock.json}"
  [[ -f "$lock_path" ]] || die "hidden test blocked: candidate lock missing at $lock_path"
  python - "$lock_path" <<'PY'
import json, sys
p = sys.argv[1]
x = json.load(open(p, encoding='utf-8'))
if x.get('status') != 'CANDIDATE_LOCKED_BEFORE_TEST':
    raise SystemExit(f'invalid candidate lock status in {p}')
fw = x.get('test_firewall_declaration', {})
if fw.get('test_files_present_during_search') is not False or fw.get('test_labels_read') is not False:
    raise SystemExit(f'test firewall declaration failed in {p}')
print(f'[data] candidate lock accepted: {p}')
PY
}

sngar_dev() {
  local hf
  hf="$(hf_bin)"
  mkdir -p "$ROOT/sports/sngar_tracking"
  log "downloading SNGAR train+valid only; test remains inaccessible"
  "$hf" download OpenSportsLab/SNGAR-Action-Spotting-Tracking \
    --repo-type dataset \
    --local-dir "$ROOT/sports/sngar_tracking" \
    --include README.md \
    --include MANIFEST.sha256 \
    --include annotations_train.json \
    --include annotations_valid.json \
    --include 'train/videos/*' \
    --include 'valid/videos/*'
  log "verify with: (cd '$ROOT/sports/sngar_tracking' && sha256sum -c MANIFEST.sha256)"
}

sngar_dry_run() {
  local hf
  hf="$(hf_bin)"
  "$hf" download OpenSportsLab/SNGAR-Action-Spotting-Tracking \
    --repo-type dataset \
    --local-dir "$ROOT/sports/sngar_tracking" \
    --include README.md \
    --include MANIFEST.sha256 \
    --include annotations_train.json \
    --include annotations_valid.json \
    --include 'train/videos/*' \
    --include 'valid/videos/*' \
    --dry-run
}

sngar_test() {
  require_candidate_lock
  local hf
  hf="$(hf_bin)"
  mkdir -p "$ROOT/sports/sngar_tracking"
  log "candidate lock verified; downloading SNGAR hidden test once"
  "$hf" download OpenSportsLab/SNGAR-Action-Spotting-Tracking \
    --repo-type dataset \
    --local-dir "$ROOT/sports/sngar_tracking" \
    --include annotations_test.json \
    --include 'test/videos/*'
}

skillcorner() {
  need_cmd git
  local dest="$ROOT/sports/skillcorner_open"
  if [[ -d "$dest/.git" ]]; then
    log "SkillCorner already present; refusing implicit pull. Current commit:"
    git -C "$dest" rev-parse HEAD
  elif [[ -e "$dest" ]]; then
    die "$dest exists but is not a git repository"
  else
    git clone https://github.com/SkillCorner/opendata.git "$dest"
    git -C "$dest" rev-parse HEAD
  fi
}

idsse() {
  need_cmd python
  local dest="$ROOT/sports/idsse"
  mkdir -p "$dest"
  IDSSE_DEST="$dest" python <<'PY'
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import requests

article_id = 28196177
out = Path(os.environ['IDSSE_DEST'])
out.mkdir(parents=True, exist_ok=True)
url = f'https://api.figshare.com/v2/articles/{article_id}'
r = requests.get(url, timeout=60)
r.raise_for_status()
metadata = r.json()
(out / 'figshare_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
records = []
for item in metadata.get('files', []):
    target = out / item['name']
    expected_size = int(item.get('size') or 0)
    if target.exists() and (not expected_size or target.stat().st_size == expected_size):
        print(f'[data] keep existing {target.name}')
    else:
        print(f'[data] download {target.name} ({expected_size} bytes)')
        with requests.get(item['download_url'], stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with target.open('wb') as handle:
                for chunk in resp.iter_content(8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    digest = hashlib.sha256()
    with target.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    records.append({'name': target.name, 'bytes': target.stat().st_size, 'sha256': digest.hexdigest()})
(out / 'download_receipt.json').write_text(json.dumps({'article_id': article_id, 'files': records}, indent=2), encoding='utf-8')
print(f'[data] IDSSE receipt: {out / "download_receipt.json"}')
PY
}

soccertrack_smoke() {
  need_cmd git
  local repo="external/SoccerTrack-v2"
  if [[ ! -d "$repo/.git" ]]; then
    git clone https://github.com/AtomScott/SoccerTrack-v2.git "$repo"
  fi
  log "downloading parser-smoke matches 117092 and 117093"
  "$repo/scripts/download.sh" \
    --dest "$ROOT/sports/soccertrack_v2_smoke" \
    --match 117092 --match 117093
}

soccertrack_full_after_lock() {
  require_candidate_lock
  need_cmd git
  local repo="external/SoccerTrack-v2"
  if [[ ! -d "$repo/.git" ]]; then
    git clone https://github.com/AtomScott/SoccerTrack-v2.git "$repo"
  fi
  log "candidate lock verified; downloading full SoccerTrack v2"
  "$repo/scripts/download.sh" --dest "$ROOT/sports/soccertrack_v2"
}

hcsu() {
  local hf
  hf="$(hf_bin)"
  mkdir -p "$ROOT/calligraphy/hcsu"
  log "HCSU must not be used to select sports mechanisms"
  "$hf" download Tongji209/HCSU \
    --repo-type dataset \
    --local-dir "$ROOT/calligraphy/hcsu"
}

mccd_repo() {
  need_cmd git
  local dest="external/MCCD"
  if [[ ! -d "$dest/.git" ]]; then
    git clone https://github.com/SCUT-DLVCLab/MCCD.git "$dest"
  fi
  log "MCCD data require an application and decompression password; repository cloned only"
}

usage() {
  cat <<'TXT'
Usage: scripts/download_priority_datasets.sh MODE

Modes:
  sngar-dry-run          inspect SNGAR train+valid download
  sngar-dev              download SNGAR train+valid only
  sngar-test             download SNGAR test; requires valid CANDIDATE_LOCK_PATH
  skillcorner            clone/pin SkillCorner Open Data
  idsse                   download public Figshare IDSSE files and write SHA receipt
  soccertrack-smoke      download two parser-smoke matches
  soccertrack-full       download full SoccerTrack; requires candidate lock
  hcsu                    download HCSU calligraphy data (post sports lock only)
  mccd-repo               clone MCCD application/loader repository only
  help                    show this message

Environment:
  DATA_ROOT               default data/raw
  CANDIDATE_LOCK_PATH     default artifacts/phase3/task_semantic_repair_v1/candidate_lock.json
  HF_TOKEN                optional Hugging Face token for gated datasets
TXT
}

case "$MODE" in
  sngar-dry-run) sngar_dry_run ;;
  sngar-dev) sngar_dev ;;
  sngar-test) sngar_test ;;
  skillcorner) skillcorner ;;
  idsse) idsse ;;
  soccertrack-smoke) soccertrack_smoke ;;
  soccertrack-full) soccertrack_full_after_lock ;;
  hcsu) hcsu ;;
  mccd-repo) mccd_repo ;;
  help|-h|--help) usage ;;
  *) usage; die "unknown mode: $MODE" ;;
esac
