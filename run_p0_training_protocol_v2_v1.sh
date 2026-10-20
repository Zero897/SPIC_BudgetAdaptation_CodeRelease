#!/usr/bin/env bash

# Protocol V2 P0 launcher. This script never starts a run without an explicit mode.

set -u
set -o pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAIN_ROOT=${P0_TRAIN_ROOT:-$PROJECT_ROOT/data/dataset/uav/train}
VAL_ROOT=${P0_VAL_ROOT:-$PROJECT_ROOT/data/dataset/uav/synthetic_smoke_val_v2}
PRETRAINED=${P0_PRETRAINED:-$PROJECT_ROOT/weights/dehazeformer/latest.pth}
RUN_ROOT=${P0_RUN_ROOT:-$PROJECT_ROOT/runs_p0_protocol_v2}
MODE=${1:-}

verify_hash() {
  local relative_path="$1"
  local expected="$2"
  local absolute_path="$PROJECT_ROOT/$relative_path"

  if [[ ! -f "$absolute_path" ]]; then
    echo "MISSING REQUIRED FILE: $absolute_path"
    return 2
  fi

  local actual
  actual=$(sha256sum "$absolute_path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "HASH MISMATCH: $relative_path"
    echo "expected=$expected"
    echo "actual=$actual"
    return 3
  fi
  echo "HASH PASS: $relative_path"
}

preflight() {
  cd "$PROJECT_ROOT" || return 1
  export PYTHONPATH="$PROJECT_ROOT/DehazeFormer-main:${PYTHONPATH:-}"
  export OMP_NUM_THREADS=1

  verify_hash train_cm_dehazeformer_v2_residual.py cd9dfb97ce55e035e3669071e5ae9ccd38e8ab6394534ea31ee30ba640f404d6 || return $?
  verify_hash crossmodal_dehazeformer_residual_v2.py c52a7051483e7a8706eb883d73af511ebf8ffd3b15b672736796a96f7a2a207b || return $?
  verify_hash cm_dehazeformer_v2_loss.py 2531f3704fab0b72dd5d6466674c2a7454c6f8a06521e71c111067fc71b7ba64 || return $?
  verify_hash data/dataset_crossmodal.py 5e8a7c975a3f90c54c393c53ab7528eacc70e5b95aff4b8d9009107456daa34d || return $?
  verify_hash DehazeFormer-main/models/dehazeformer.py 1363d18fb2beb9ecc0649c41da4526323c8d1823b9bc8195586582b3bdc0b149 || return $?
  verify_hash DehazeFormer-main/models/__init__.py efac465c348279d2e8b8866c3d7e8055efcca02c5852a8b8eab339f8be1fe3c1 || return $?
  verify_hash verify_p0_training_assets_v1.py 71250a55ee1eb03b332bf5f44eab52091a69619f4fdf165eef23cdcb8ab69afe || return $?

  if [[ ! -f "$PRETRAINED" ]]; then
    echo "MISSING PRETRAINED WEIGHT: $PRETRAINED"
    return 2
  fi
  local weight_hash
  weight_hash=$(sha256sum "$PRETRAINED" | awk '{print $1}')
  if [[ "$weight_hash" != "e5af35e6254916c3bfbcdcd25bdbfd856e65942e0a682a12d986ca951183ef88" ]]; then
    echo "HASH MISMATCH: pretrained weight"
    echo "expected=e5af35e6254916c3bfbcdcd25bdbfd856e65942e0a682a12d986ca951183ef88"
    echo "actual=$weight_hash"
    return 3
  fi
  echo "HASH PASS: pretrained weight"

  if [[ ! -d "$TRAIN_ROOT" ]]; then
    echo "MISSING TRAIN ROOT: $TRAIN_ROOT"
    return 2
  fi
  local split
  for split in lq ir gt; do
    if [[ ! -d "$VAL_ROOT/$split" ]]; then
      echo "MISSING VALIDATION SPLIT: $VAL_ROOT/$split"
      return 2
    fi
  done
  if [[ ! -f "$VAL_ROOT/generation_manifest.jsonl" ]]; then
    echo "MISSING VALIDATION MANIFEST: $VAL_ROOT/generation_manifest.jsonl"
    return 2
  fi
  local manifest_hash
  manifest_hash=$(sha256sum "$VAL_ROOT/generation_manifest.jsonl" | awk '{print $1}')
  if [[ "$manifest_hash" != "943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128" ]]; then
    echo "HASH MISMATCH: validation manifest"
    echo "expected=943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128"
    echo "actual=$manifest_hash"
    return 3
  fi
  echo "HASH PASS: validation manifest"

  python -m py_compile \
    train_cm_dehazeformer_v2_residual.py \
    crossmodal_dehazeformer_residual_v2.py \
    cm_dehazeformer_v2_loss.py \
    data/dataset_crossmodal.py \
    DehazeFormer-main/models/dehazeformer.py \
    verify_p0_training_assets_v1.py || return $?
  python verify_p0_training_assets_v1.py --pretrained "$PRETRAINED" || return $?
}

run_one() {
  local group_dir="$1"
  local seed="$2"
  local smoke="$3"
  local save_dir
  local epochs
  local eval_every
  local train_max_batches
  local eval_max_batches
  local smoke_flag=()

  if [[ "$smoke" == "true" ]]; then
    save_dir="$group_dir/P0_smoke_seed${seed}"
    epochs=1
    eval_every=1
    train_max_batches=2
    eval_max_batches=2
    smoke_flag=(--smoke_test)
  else
    save_dir="$group_dir/P0_seed${seed}"
    epochs=10
    eval_every=2
    train_max_batches=-1
    eval_max_batches=-1
  fi

  local log_file="$group_dir/P0_seed${seed}.log"
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "REFUSING TO OVERWRITE: $save_dir or $log_file"
    return 10
  fi

  echo "START P0 seed=$seed smoke=$smoke"
  echo "save_dir=$save_dir"
  python train_cm_dehazeformer_v2_residual.py \
    --experiment_id P0 \
    --train_mode adapter \
    --train_root "$TRAIN_ROOT" \
    --val_lq "$VAL_ROOT/lq" \
    --val_ir "$VAL_ROOT/ir" \
    --val_gt "$VAL_ROOT/gt" \
    --save_dir "$save_dir" \
    --backbone b \
    --pretrained "$PRETRAINED" \
    --size 256 \
    --epochs "$epochs" \
    --batch 16 \
    --grad_accum_steps 1 \
    --workers 8 \
    --lr_base 0 \
    --lr_new 1e-4 \
    --freeze_base_epochs "$epochs" \
    --eval_every "$eval_every" \
    --train_max_batches "$train_max_batches" \
    --eval_max_batches "$eval_max_batches" \
    --init_gamma 0.01 \
    --ablation wo_ir_physics \
    --wo_gate \
    --adapter_variant plain \
    --refiner_width 25 \
    --refiner_blocks 4 \
    --loss_profile common \
    --seed "$seed" \
    --deterministic \
    "${smoke_flag[@]}" \
    2>&1 | tee "$log_file"
  local train_status=${PIPESTATUS[0]}
  if [[ "$train_status" -ne 0 ]]; then
    echo "FAILED P0 seed=$seed exit_code=$train_status"
    return "$train_status"
  fi

  local required
  for required in run_config.json history.jsonl best_psnr.pth last.pth; do
    if [[ ! -f "$save_dir/$required" ]]; then
      echo "MISSING OUTPUT AFTER SUCCESS: $save_dir/$required"
      return 12
    fi
  done
  cp "$log_file" "$save_dir/console.log"
  echo "PASS P0 seed=$seed"
}

bundle_outputs() {
  local group_dir="$1"
  local archive="${group_dir}.tar.gz"
  tar -C "$(dirname "$group_dir")" -czf "$archive" "$(basename "$group_dir")" || return $?
  sha256sum "$archive"
  echo "bundle=$archive"
}

main() {
  case "$MODE" in
    smoke|seed3407|multiseed) ;;
    *)
      echo "Usage: bash ./run_p0_training_protocol_v2_v1.sh {smoke|seed3407|multiseed}"
      return 64
      ;;
  esac

  preflight || {
    local status=$?
    echo "P0 PREFLIGHT FAILED: exit_code=$status"
    return "$status"
  }

  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local group_dir="$RUN_ROOT/p0_${MODE}_${stamp}"
  if [[ -e "$group_dir" ]]; then
    echo "REFUSING TO OVERWRITE RUN GROUP: $group_dir"
    return 10
  fi
  mkdir -p "$group_dir"

  case "$MODE" in
    smoke)
      run_one "$group_dir" 3407 true || return $?
      ;;
    seed3407)
      run_one "$group_dir" 3407 false || return $?
      ;;
    multiseed)
      local seed
      for seed in 3407 2026 8888; do
        run_one "$group_dir" "$seed" false || return $?
      done
      ;;
  esac

  bundle_outputs "$group_dir" || return $?
  echo "P0_TRAINING_PROTOCOL_V2_V1_PASS mode=$MODE"
}

main
exit $?

