#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

python "$ROOT/scripts/spectral_smoke.py"
python "$ROOT/scripts/verify_data.py" --manifest "$ROOT/configs/data_manifest.yaml" || {
  echo "Required data are missing. Run download_public_data.sh --core." >&2
  exit 1
}

python - <<'PY'
from pathlib import Path
root = Path.cwd()
print("phase0 package smoke: PASS")
PY
