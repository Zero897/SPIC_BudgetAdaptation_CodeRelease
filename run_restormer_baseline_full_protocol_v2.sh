#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
if [[ -n "${ARCHITECTURE:-}" ]]; then
  ARCHITECTURE=$ARCHITECTURE
elif [[ -f "$PROJECT_ROOT/restormer_arch.py" ]]; then
  ARCHITECTURE="$PROJECT_ROOT/restormer_arch.py"
elif [[ -f "$PROJECT_ROOT/comparative/restormer/restormer_arch.py" ]]; then
  ARCHITECTURE="$PROJECT_ROOT/comparative/restormer/restormer_arch.py"
else
  ARCHITECTURE="$PROJECT_ROOT/restormer_arch.py"
fi
DATASET_SOURCE=${DATASET_SOURCE:-$PROJECT_ROOT/data/dataset_crossmodal.py}
TRAIN_ROOT=${TRAIN_ROOT:-$PROJECT_ROOT/data/dataset/uav/train}
VAL_ROOT=${VAL_ROOT:-$PROJECT_ROOT/data/dataset/uav/synthetic_smoke_val_v2}

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$PROJECT_ROOT/runs_restormer_baseline_v2/full_seed3407_${STAMP}"
LOG_DIR="$PROJECT_ROOT/runs_restormer_baseline_v2/launch_logs"
LOG="$LOG_DIR/full_seed3407_${STAMP}.log"
SMOKE_LOG=${SMOKE_LOG:-}
mkdir -p "$LOG_DIR"

verify_hash() {
  local path="$1"
  local expected="$2"
  local label="$3"
  [[ -f "$path" ]] || { echo "MISSING REQUIRED FILE: $path" >&2; return 2; }
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "HASH MISMATCH: $label expected=$expected actual=$actual" >&2
    return 3
  fi
  echo "HASH PASS: $label sha256=$actual"
}

main() {
  cd "$PROJECT_ROOT" || return 1
  [[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; return 2; }
  [[ -n "$SMOKE_LOG" && -f "$SMOKE_LOG" ]] || { echo "Set SMOKE_LOG to the completed Restormer Smoke log." >&2; return 2; }
  grep -q '^RESTORMER_BASELINE_SMOKE_PROTOCOL_V2_BATCH4X4_PASS$' "$SMOKE_LOG" || { echo "Restormer batch-4/accumulation-4 Smoke pass marker not found in $SMOKE_LOG" >&2; return 2; }
  [[ -d "$TRAIN_ROOT" ]] || { echo "Missing training root: $TRAIN_ROOT" >&2; return 2; }
  [[ ! -e "$RUN_ROOT" ]] || { echo "Refusing existing run root: $RUN_ROOT" >&2; return 2; }
  for split in lq gt; do
    [[ -d "$VAL_ROOT/$split" ]] || { echo "Missing validation directory: $VAL_ROOT/$split" >&2; return 2; }
  done

  verify_hash "$PROJECT_ROOT/train_restormer_baseline_protocol_v2.py" 453055c6132adcde5b49ba6af7d443e90427dad860ce211e67dbfd86a9ad98c3 train_restormer_baseline_protocol_v2.py || return $?
  verify_hash "$ARCHITECTURE" 3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a restormer_arch.py || return $?
  verify_hash "$DATASET_SOURCE" 5e8a7c975a3f90c54c393c53ab7528eacc70e5b95aff4b8d9009107456daa34d data/dataset_crossmodal.py || return $?
  verify_hash "$VAL_ROOT/generation_manifest.jsonl" 943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128 validation_manifest || return $?

  echo "RESTORMER FULL PROTOCOL"
  echo "initialization=random_from_epoch_1"
  echo "epochs=100 eval_every=2 batch=4 grad_accum_steps=4 effective_batch=16"
  echo "early_stop_min_epoch=50 patience_validations=4 min_delta_db=0.01"
  echo "run_root=$RUN_ROOT"
  echo "log=$LOG"

  export OMP_NUM_THREADS=1
  "$PYTHON_BIN" -c 'import torch; print("torch={} cuda={} gpu={} bf16={}".format(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported()))' || return $?

  "$PYTHON_BIN" -u "$PROJECT_ROOT/train_restormer_baseline_protocol_v2.py" \
    --architecture "$ARCHITECTURE" \
    --dataset_source "$DATASET_SOURCE" \
    --train_root "$TRAIN_ROOT" \
    --val_lq "$VAL_ROOT/lq" \
    --val_gt "$VAL_ROOT/gt" \
    --val_manifest "$VAL_ROOT/generation_manifest.jsonl" \
    --save_dir "$RUN_ROOT" \
    --size 256 \
    --epochs 100 \
    --batch 4 \
    --grad_accum_steps 4 \
    --workers 8 \
    --lr 2e-4 \
    --eval_every 2 \
    --max_train_batches -1 \
    --eval_max_batches -1 \
    --seed 3407 \
    --deterministic || return $?

  local artifact
  for artifact in run_config.json history.jsonl best_psnr.pth last.pth sample_selection.json; do
    [[ -f "$RUN_ROOT/$artifact" ]] || { echo "MISSING OUTPUT: $RUN_ROOT/$artifact" >&2; return 12; }
  done
  sha256sum "$RUN_ROOT/run_config.json" "$RUN_ROOT/history.jsonl" "$RUN_ROOT/best_psnr.pth" "$RUN_ROOT/last.pth"
  echo "run_root=$RUN_ROOT"
  echo "RESTORMER_BASELINE_FULL_PROTOCOL_V2_PASS"
}

main 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "launch_status=$status" | tee -a "$LOG"
echo "log=$LOG" | tee -a "$LOG"
exit "$status"


