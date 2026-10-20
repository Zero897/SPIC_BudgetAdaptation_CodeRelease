#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"
[[ "$MODE" == preflight || "$MODE" == smoke || "$MODE" == full ]] || { echo "Usage: $0 {preflight|smoke|full}" >&2; exit 2; }
: "${A2I2_DATASET_ROOT:?Set A2I2_DATASET_ROOT to the provider-obtained data directory}"
: "${A2I2_MANIFEST:?Set A2I2_MANIFEST to the reviewed evaluation manifest}"
: "${A2I2_LAUNCH_MANIFEST:?Set A2I2_LAUNCH_MANIFEST with detector/checkpoint paths and hashes}"
: "${A2I2_OUTPUT_DIR:?Set A2I2_OUTPUT_DIR to a new output directory}"
"$PYTHON_BIN" "$HERE/eval_a2i2_downstream_v1.py" --mode "$MODE" --dataset_root "$A2I2_DATASET_ROOT" --manifest "$A2I2_MANIFEST" --launch_manifest "$A2I2_LAUNCH_MANIFEST" --output_dir "$A2I2_OUTPUT_DIR"

