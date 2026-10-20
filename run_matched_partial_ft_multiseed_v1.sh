#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "preflight" && "$MODE" != "seed2026" && "$MODE" != "seed8888" && "$MODE" != "queue" ]]; then
  echo "Usage: bash run_matched_partial_ft_multiseed_v1.sh {preflight|seed2026|seed8888|queue}" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_ROOT="$PROJECT_ROOT/data/dataset/uav/train"
VAL_ROOT="$PROJECT_ROOT/data/dataset/uav/synthetic_smoke_val_v2"
PRETRAINED="$PROJECT_ROOT/weights/dehazeformer/latest.pth"
RUN_ROOT="$PROJECT_ROOT/runs_protocol_v2/matched_partial_ft_v1"
LOG_ROOT="$PROJECT_ROOT/runs_protocol_v2/launch_logs"
PILOT_ROOT="$RUN_ROOT/MFT_seed3407_20261010"

TRAIN_SCRIPT="$PROJECT_ROOT/train_matched_partial_ft_v1.py"
EVAL_SCRIPT="$PROJECT_ROOT/eval_matched_partial_ft_v1.py"

check_sha256() {
  local path="$1"
  local expected="$2"
  if [[ ! -f "$path" ]]; then
    echo "MISSING_FILE $path" >&2
    return 1
  fi
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256_MISMATCH $path actual=$actual expected=$expected" >&2
    return 1
  fi
  echo "SHA256_OK $actual  $path"
}

static_checks() {
  cd "$PROJECT_ROOT"
  check_sha256 "$TRAIN_SCRIPT" "b3de46bcf54346127f881277f02baca762bfe800e9641ca8e0749cbd02a33fa6"
  check_sha256 "$EVAL_SCRIPT" "feeee7bab99d74b3a8c75c506a075a0d2ecacea0bf85af9163bde2539bf9d646"
  check_sha256 "$PROJECT_ROOT/crossmodal_dehazeformer_residual_v2.py" "c52a7051483e7a8706eb883d73af511ebf8ffd3b15b672736796a96f7a2a207b"
  check_sha256 "$PROJECT_ROOT/cm_dehazeformer_v2_loss.py" "2531f3704fab0b72dd5d6466674c2a7454c6f8a06521e71c111067fc71b7ba64"
  check_sha256 "$PROJECT_ROOT/data/dataset_crossmodal.py" "5e8a7c975a3f90c54c393c53ab7528eacc70e5b95aff4b8d9009107456daa34d"
  check_sha256 "$PROJECT_ROOT/DehazeFormer-main/models/dehazeformer.py" "1363d18fb2beb9ecc0649c41da4526323c8d1823b9bc8195586582b3bdc0b149"
  check_sha256 "$PROJECT_ROOT/DehazeFormer-main/models/__init__.py" "efac465c348279d2e8b8866c3d7e8055efcca02c5852a8b8eab339f8be1fe3c1"
  check_sha256 "$PRETRAINED" "e5af35e6254916c3bfbcdcd25bdbfd856e65942e0a682a12d986ca951183ef88"
  check_sha256 "$VAL_ROOT/generation_manifest.jsonl" "943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128"
  check_sha256 "$PILOT_ROOT/pilot_analysis.json" "44e7060fe319fb5397689af204326f4df5b97f113abc4fc85b85ef8b9085a602"
}

data_and_pilot_checks() {
  "$PYTHON_BIN" - "$TRAIN_ROOT" "$VAL_ROOT" "$PILOT_ROOT/pilot_analysis.json" <<'PY'
import json
import sys
from pathlib import Path

train_root = Path(sys.argv[1])
val_root = Path(sys.argv[2])
analysis_path = Path(sys.argv[3])
if not train_root.is_dir():
    raise FileNotFoundError(train_root)
sets = []
for part in ("lq", "ir", "gt"):
    folder = val_root / part
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    stems = {path.stem for path in folder.iterdir() if path.is_file()}
    if len(stems) != 1469:
        raise ValueError(f"{folder}: expected 1469 files, found {len(stems)}")
    sets.append(stems)
if not (sets[0] == sets[1] == sets[2]):
    raise ValueError("Validation lq/ir/gt stems do not match")
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
if analysis.get("valid") is not True:
    raise ValueError("Seed-3407 Pilot is not valid")
if analysis.get("decision") != "CONTINUE_MFT_MULTI_SEED":
    raise ValueError("Seed-3407 Pilot did not release multi-seed continuation")
if not all(analysis.get("quality_gate", {}).values()):
    raise ValueError("Seed-3407 Pilot quality gate is incomplete or failed")
print("MFT_MULTI_SEED_DATA_AND_PILOT_GATE_PASS validation_pairs=1469")
PY
}

run_preflight() {
  static_checks
  data_and_pilot_checks
  echo "MFT_MULTI_SEED_PREFLIGHT_PASS"
}

assert_idle_gpu() {
  "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA_OK", torch.cuda.get_device_name(0))'
  local compute_pids
  compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "$compute_pids" ]]; then
    echo "GPU_NOT_IDLE compute_pids=$compute_pids" >&2
    return 1
  fi
  echo "GPU_IDLE_CHECK_PASS"
}

run_seed() {
  local seed="$1"
  local save_dir="$RUN_ROOT/MFT_seed${seed}_20261010"
  local log_file="$LOG_ROOT/MFT_seed${seed}_20261010.log"
  if [[ -e "$save_dir" || -e "$log_file" ]]; then
    echo "REFUSING_TO_OVERWRITE save_dir=$save_dir log_file=$log_file" >&2
    return 1
  fi
  mkdir -p "$LOG_ROOT"
  assert_idle_gpu

  set +e
  "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --experiment_id MFT \
    --train_mode matched_partial \
    --train_root "$TRAIN_ROOT" \
    --val_lq "$VAL_ROOT/lq" \
    --val_ir "$VAL_ROOT/ir" \
    --val_gt "$VAL_ROOT/gt" \
    --save_dir "$save_dir" \
    --backbone b \
    --pretrained "$PRETRAINED" \
    --size 256 \
    --batch 16 \
    --grad_accum_steps 1 \
    --workers 8 \
    --lr_base 1e-6 \
    --lr_new 0 \
    --freeze_base_epochs 0 \
    --init_gamma 0.01 \
    --ablation wo_ir_physics \
    --wo_gate \
    --adapter_variant fscatterres \
    --refiner_width 24 \
    --refiner_blocks 4 \
    --loss_profile common \
    --seed "$seed" \
    --deterministic \
    --epochs 10 \
    --eval_every 2 \
    --train_max_batches -1 \
    --eval_max_batches -1 \
    2>&1 | tee "$log_file"
  local train_status=${PIPESTATUS[0]}
  set -e
  if [[ -d "$save_dir" ]]; then
    cp "$log_file" "$save_dir/console.log"
  fi
  if [[ "$train_status" -ne 0 ]]; then
    return "$train_status"
  fi

  local required
  for required in run_config.json history.jsonl best_psnr.pth last.pth console.log; do
    if [[ ! -f "$save_dir/$required" ]]; then
      echo "MISSING_OUTPUT $save_dir/$required" >&2
      return 1
    fi
  done

  local eval_dir="$save_dir/stratified_validation"
  if [[ -e "$eval_dir" ]]; then
    echo "REFUSING_TO_OVERWRITE $eval_dir" >&2
    return 1
  fi
  set +e
  "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --experiment_id MFT \
    --checkpoint "$save_dir/best_psnr.pth" \
    --val_lq "$VAL_ROOT/lq" \
    --val_ir "$VAL_ROOT/ir" \
    --val_gt "$VAL_ROOT/gt" \
    --manifest "$VAL_ROOT/generation_manifest.jsonl" \
    --output_dir "$eval_dir" \
    --size 256 --batch 8 --workers 8 --seed "$seed" \
    2>&1 | tee "$save_dir/stratified_validation.log"
  local eval_status=${PIPESTATUS[0]}
  set -e
  if [[ "$eval_status" -ne 0 ]]; then
    return "$eval_status"
  fi

  "$PYTHON_BIN" - "$save_dir" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
args = config["args"]
expected = {
    "experiment_id": "MFT", "train_mode": "matched_partial", "seed": seed,
    "epochs": 10, "batch": 16, "grad_accum_steps": 1, "lr_base": 1e-6,
    "lr_new": 0.0, "loss_profile": "common", "smoke_test": False,
    "train_max_batches": -1, "eval_max_batches": -1, "expected_trainable": 45412,
}
for key, value in expected.items():
    if args.get(key) != value:
        raise ValueError(f"Config mismatch {key}: expected={value!r} actual={args.get(key)!r}")
counts = config["parameter_counts"]
if (counts.get("trainable_total"), counts.get("base_trainable"), counts.get("adapter_trainable")) != (45412, 45412, 0):
    raise ValueError(f"Parameter-count mismatch: {counts}")
history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
validations = [row for row in history if row.get("event") == "validation"]
if [row.get("epoch") for row in validations] != [2, 4, 6, 8, 10]:
    raise ValueError("Validation epochs are not exactly 2,4,6,8,10")
metrics = json.loads((run_dir / "stratified_validation/MFT_stratified_metrics.json").read_text(encoding="utf-8"))
if metrics.get("sample_count") != 1469 or metrics.get("seed") != seed:
    raise ValueError("Stratified evaluator sample count or seed mismatch")
if not all(math.isfinite(float(metrics["aggregates"]["all"][key])) for key in ("psnr", "ssim", "lpips")):
    raise ValueError("Non-finite aggregate metric")
print(f"MFT_SEED_ARTIFACT_AUDIT_PASS seed={seed}")
PY

  find "$save_dir" -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$save_dir/SHA256SUMS"
  echo "MFT_V1_SEED_RUNTIME_PASS seed=$seed output=$save_dir"
}

export OMP_NUM_THREADS=1
case "$MODE" in
  preflight) run_preflight ;;
  seed2026) run_preflight; run_seed 2026 ;;
  seed8888) run_preflight; run_seed 8888 ;;
  queue) run_preflight; run_seed 2026; run_seed 8888; echo "MFT_V1_MULTI_SEED_QUEUE_PASS" ;;
esac


