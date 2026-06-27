#!/usr/bin/env bash
# Generate denoised test results and package as result.zip.
#
# Usage:
#   bash scripts/make_submission.sh <checkpoint.pkl> [config.yaml] [name]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PKL="${1:?Usage: bash scripts/make_submission.sh <checkpoint.pkl> [config.yaml] [name]}"
CONFIG="${2:-configs/default.yaml}"
NAME="${3:-$(basename "$PKL" .pkl)}"
OUT_DIR="results_${NAME}"
ZIP_FILE="result_${NAME}.zip"

echo "Checkpoint : $PKL"
echo "Config     : $CONFIG"
echo "Output dir : $OUT_DIR"
echo "Zip file   : $ZIP_FILE"
echo ""

if [[ -e "$OUT_DIR" ]]; then
  echo "Output dir already exists: $OUT_DIR" >&2
  exit 1
fi
if [[ -e "$ZIP_FILE" ]]; then
  echo "Zip file already exists: $ZIP_FILE" >&2
  exit 1
fi

CMD=(python predict.py --config "$CONFIG" --model "$PKL" --out "$OUT_DIR")
"${CMD[@]}"

cd "$OUT_DIR"
zip -r "../$ZIP_FILE" shapenet/
cd ..

echo ""
echo "Done -> $ZIP_FILE"
