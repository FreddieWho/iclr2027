#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV_PATH:-$ROOT/.venv}"
MODELS=1

for arg in "$@"; do
  case "$arg" in
    --minimal) MODELS=0 ;;
    --with-models) MODELS=1 ;;
    -h|--help)
      echo "Usage: $0 [--minimal|--with-models]"
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null || { echo "python3 not found" >&2; exit 1; }

if [[ ! -d "$VENV" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$ROOT/requirements-core.txt"

if [[ "$MODELS" -eq 1 ]]; then
  python -m pip install -r "$ROOT/requirements-models.txt"
fi

python - <<'PY'
import platform
import numpy, scipy, pandas, networkx, yaml
print("Environment ready")
print("Python:", platform.python_version())
print("NumPy:", numpy.__version__)
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available())
except Exception as exc:
    print("Torch not installed or failed to import:", exc)
PY
