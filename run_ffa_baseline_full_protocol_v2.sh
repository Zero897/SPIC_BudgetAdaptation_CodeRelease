#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
ARCHITECTURE=${ARCHITECTURE:-$PROJECT_ROOT/FFA-Net-master/net/models/FFA.py}
PRETRAINED=${PRETRAINED:-$PROJECT_ROOT/weights/ffa-net/its_train_ffa_3_19.pk}
DATASET_SOURCE=${DATASET_SOURCE:-$PROJECT_ROOT/data/dataset_crossmodal.py}
TRAIN_ROOT=${TRAIN_ROOT:-$PROJECT_ROOT/data/dataset/uav/train}
VAL_ROOT=${VAL_ROOT:-$PROJECT_ROOT/data/dataset/uav/synthetic_smoke_val_v2}

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$PROJECT_ROOT/runs_ffa_baseline_v2/full_${STAMP}/seed3407"
LOG_DIR="$PROJECT_ROOT/runs_ffa_baseline_v2/launch_logs"
LOG="$LOG_DIR/ffa_baseline_full_protocol_v2_${STAMP}.log"
mkdir -p "$LOG_DIR"

verify_hash() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ ! -f "$path" ]]; then
    echo "MISSING REQUIRED FILE: $path" >&2
    return 2
  fi
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "HASH MISMATCH: $label" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    return 3
  fi
  echo "HASH PASS: $label sha256=$actual"
}

main() {
  cd "$PROJECT_ROOT" || return 1
  [[ -x "$PYTHON_BIN" ]] || {
    echo "Python is not executable: $PYTHON_BIN" >&2
    return 2
  }
  [[ -d "$TRAIN_ROOT" ]] || {
    echo "Missing training root: $TRAIN_ROOT" >&2
    return 2
  }
  [[ ! -e "$RUN_ROOT" ]] || {
    echo "Refusing existing run root: $RUN_ROOT" >&2
    return 2
  }
  for split in lq gt; do
    [[ -d "$VAL_ROOT/$split" ]] || {
      echo "Missing validation directory: $VAL_ROOT/$split" >&2
      return 2
    }
  done

  verify_hash \
    "$PROJECT_ROOT/train_ffa_baseline_protocol_v2.py" \
    2387f61969cf347104d3306a4412d9a10fc1ecdd6c2fa86d918e2c5d005bfddd \
    train_ffa_baseline_protocol_v2.py || return $?
  verify_hash \
    "$ARCHITECTURE" \
    8e2e7122cb510f157ccf7e1ae09b8885d63634f5e572d12c43ef06e95e3bc199 \
    FFA.py || return $?
  verify_hash \
    "$DATASET_SOURCE" \
    5e8a7c975a3f90c54c393c53ab7528eacc70e5b95aff4b8d9009107456daa34d \
    data/dataset_crossmodal.py || return $?
  verify_hash \
    "$VAL_ROOT/generation_manifest.jsonl" \
    943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128 \
    validation_manifest || return $?
  verify_hash \
    "$PRETRAINED" \
    f53e056d3ada3c00487fc22baab2b911e698d2a608472e56a92889487078b199 \
    official_its_train_ffa_3_19.pk || return $?

  echo "FFA FULL PROTOCOL"
  echo "max_successful_steps=200000"
  echo "validation_every_steps=5000"
  echo "early_stop_min_steps=50000"
  echo "early_stop_patience_validations=6"
  echo "early_stop_min_delta_db=0.01"
  echo "run_root=$RUN_ROOT"
  echo "log=$LOG"

  export OMP_NUM_THREADS=1
  "$PYTHON_BIN" -c 'import torch; print("torch={} cuda={} gpu={}".format(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)))' || return $?

  "$PYTHON_BIN" -u "$PROJECT_ROOT/train_ffa_baseline_protocol_v2.py" \
    --architecture "$ARCHITECTURE" \
    --pretrained "$PRETRAINED" \
    --dataset_source "$DATASET_SOURCE" \
    --train_root "$TRAIN_ROOT" \
    --val_lq "$VAL_ROOT/lq" \
    --val_gt "$VAL_ROOT/gt" \
    --val_manifest "$VAL_ROOT/generation_manifest.jsonl" \
    --save_dir "$RUN_ROOT" \
    --size 256 \
    --batch 8 \
    --grad_accum_steps 2 \
    --workers 8 \
    --max_steps 200000 \
    --eval_every_steps 5000 \
    --checkpoint_every_steps 1000 \
    --log_every_steps 100 \
    --lr 1e-4 \
    --seed 3407 \
    --eval_max_batches -1 \
    --deterministic || return $?

  local artifact
  for artifact in run_config.json history.jsonl best_psnr.pth last.pth; do
    [[ -f "$RUN_ROOT/$artifact" ]] || {
      echo "MISSING OUTPUT: $RUN_ROOT/$artifact" >&2
      return 12
    }
  done
  sha256sum \
    "$RUN_ROOT/run_config.json" \
    "$RUN_ROOT/history.jsonl" \
    "$RUN_ROOT/best_psnr.pth" \
    "$RUN_ROOT/last.pth"
  echo "run_root=$RUN_ROOT"
  echo "FFA_BASELINE_FULL_PROTOCOL_V2_PASS"
}

main 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "launch_status=$status" | tee -a "$LOG"
echo "log=$LOG" | tee -a "$LOG"
exit "$status"


