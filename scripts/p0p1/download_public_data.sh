#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/raw}"
mkdir -p "$DATA_ROOT/sports" "$DATA_ROOT/calligraphy" "$DATA_ROOT/mathwriting" "$ROOT/data/manual"

DO_CORE=0
DO_SPORTS=0
DO_CALLIGRAPHY_OPEN=0
DO_BASKETBALL=0
DO_CCSE=0
DO_MATH_FULL=0
DO_ALL=0

usage() {
  cat <<USAGE
Usage: $0 [options]
  --core               SkillCorner, Metrica, MakeMeAHanzi, repo metadata,
                       kirosc and MathWriting excerpt
  --sports             Football repos only
  --calligraphy-open   Download zhuojg 747MB open calligraphy archive
  --basketball         TrackID3x3 repo and Google Drive folder
  --ccse               CCSE repo and two dataset archives
  --math-full          MathWriting full 2.9GB archive
  --all                All automated downloads except MCCD restricted data
  -h, --help           Show help
USAGE
}

if [[ $# -eq 0 ]]; then usage; exit 2; fi
for arg in "$@"; do
  case "$arg" in
    --core) DO_CORE=1 ;;
    --sports) DO_SPORTS=1 ;;
    --calligraphy-open) DO_CALLIGRAPHY_OPEN=1 ;;
    --basketball) DO_BASKETBALL=1 ;;
    --ccse) DO_CCSE=1 ;;
    --math-full) DO_MATH_FULL=1 ;;
    --all) DO_ALL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done

if [[ "$DO_ALL" -eq 1 ]]; then
  DO_CORE=1; DO_CALLIGRAPHY_OPEN=1; DO_BASKETBALL=1; DO_CCSE=1; DO_MATH_FULL=1
fi
if [[ "$DO_CORE" -eq 1 ]]; then DO_SPORTS=1; fi

clone_repo() {
  local url="$1" dest="$2"
  if [[ -d "$dest/.git" ]]; then
    echo "[skip] repo exists: $dest"
  elif [[ -e "$dest" ]]; then
    echo "[warn] path exists but is not a git repo: $dest" >&2
  else
    echo "[clone] $url"
    git clone --depth 1 "$url" "$dest"
  fi
}

download_url() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "[skip] file exists: $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[download] $url"
  if command -v curl >/dev/null; then
    curl -fL --retry 3 --retry-delay 2 "$url" -o "$dest"
  else
    wget -O "$dest" "$url"
  fi
}

gdown_file() {
  local id="$1" dest="$2"
  command -v gdown >/dev/null || { echo "gdown not found; run bootstrap_env.sh" >&2; return 1; }
  if [[ -s "$dest" ]]; then echo "[skip] $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  gdown "$id" -O "$dest"
}

if [[ "$DO_SPORTS" -eq 1 ]]; then
  clone_repo https://github.com/SkillCorner/opendata.git "$DATA_ROOT/sports/skillcorner"
  clone_repo https://github.com/metrica-sports/sample-data.git "$DATA_ROOT/sports/metrica"
fi

if [[ "$DO_CORE" -eq 1 ]]; then
  clone_repo https://github.com/skishore/makemeahanzi.git "$DATA_ROOT/calligraphy/makemeahanzi"
  clone_repo https://github.com/SCUT-DLVCLab/MCCD.git "$DATA_ROOT/calligraphy/MCCD_repo"
  clone_repo https://github.com/zhuojg/chinese-calligraphy-dataset.git "$DATA_ROOT/calligraphy/zhuojg_repo"
  clone_repo https://github.com/kirosc/chinese-calligraphy-dataset.git "$DATA_ROOT/calligraphy/kirosc"
  clone_repo https://github.com/lizhaoliu-Lec/CCSE.git "$DATA_ROOT/calligraphy/CCSE_repo"
  download_url \
    https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz \
    "$DATA_ROOT/mathwriting/mathwriting-2024-excerpt.tgz"
  cat > "$ROOT/data/manual/MCCD_APPLICATION.md" <<'NOTE'
# MCCD manual action

1. Open the official repository: https://github.com/SCUT-DLVCLab/MCCD
2. Complete the application linked in its README.
3. Download through the official Baidu/OneDrive link.
4. Store the archive under data/raw/calligraphy/MCCD_download/.
5. Never commit or redistribute the images; preserve the CC BY-NC-ND 4.0 terms.
NOTE
fi

if [[ "$DO_CALLIGRAPHY_OPEN" -eq 1 ]]; then
  gdown_file 1k849yUZhkUfbupZT0kRR2ZzZj5g89yLw \
    "$DATA_ROOT/calligraphy/zhuojg_characters.zip"
fi

if [[ "$DO_BASKETBALL" -eq 1 ]]; then
  clone_repo https://github.com/open-starlab/TrackID3x3.git "$DATA_ROOT/sports/trackid3x3_repo"
  command -v gdown >/dev/null || { echo "gdown not found; run bootstrap_env.sh" >&2; exit 1; }
  if [[ ! -d "$DATA_ROOT/sports/trackid3x3_drive" ]]; then
    gdown --folder \
      https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX \
      -O "$DATA_ROOT/sports/trackid3x3_drive" || {
        echo "TrackID3x3 folder download failed. Download manually from the URL in the data guide." >&2
      }
  fi
fi

if [[ "$DO_CCSE" -eq 1 ]]; then
  clone_repo https://github.com/lizhaoliu-Lec/CCSE.git "$DATA_ROOT/calligraphy/CCSE_repo"
  gdown_file 1U8mLLb_qWSqC4yRnJlzoVaI2ELF2lGAH "$DATA_ROOT/calligraphy/ccse_hw.zip"
  gdown_file 1-2VFuiWHSd3fzl9qYMoEi0mlO_BoSgCd "$DATA_ROOT/calligraphy/ccse_kai.zip"
fi

if [[ "$DO_MATH_FULL" -eq 1 ]]; then
  download_url \
    https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz \
    "$DATA_ROOT/mathwriting/mathwriting-2024.tgz"
fi

echo "Download step complete. Run verify_data.py next."
