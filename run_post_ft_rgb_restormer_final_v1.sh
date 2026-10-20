#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
ARCH="${RESTORMER_ARCH:?Set to an external Restormer architecture file; source is not included}"
BASE_CKPT="$PROJECT_ROOT/weights/restormer/best.pth"
SEED3407_CKPT="$PROJECT_ROOT/runs_post_ft_rgb_residual_v1/pilot_both_seed3407_20261014_152331/restormer_post_ft_rgb_residual_seed3407/best_psnr.pth"
SEED2026_CKPT="$PROJECT_ROOT/runs_post_ft_rgb_residual_v1/confirm_restormer_20261015_000021/restormer_post_ft_rgb_residual_seed2026/best_psnr.pth"
SEED8888_CKPT="$PROJECT_ROOT/runs_post_ft_rgb_residual_v1/confirm_restormer_20261015_000021/restormer_post_ft_rgb_residual_seed8888/best_psnr.pth"
TEST_ROOT="$PROJECT_ROOT/data/dataset/uav/synthetic_smoke_test_v2"
OUTPUT_PARENT="$PROJECT_ROOT/results_post_ft_rgb_restormer_final_v1"
OUTPUT_ROOT="$OUTPUT_PARENT/locked_v1"
BUNDLE="$OUTPUT_PARENT/post_ft_rgb_restormer_final_locked_v1.tar.gz"
LOG_DIR="$PROJECT_ROOT/runs_post_ft_rgb_residual_v1/launch_logs"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/post_ft_rgb_restormer_final_v1_${STAMP}.log"

verify_file() {
  local path="$1"
  local expected="$2"
  local label="$3"
  [[ -f "$path" ]] || { echo "MISSING: $label path=$path" >&2; return 2; }
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "HASH MISMATCH: $label expected=$expected actual=$actual" >&2
    return 3
  }
  echo "HASH PASS: $label sha256=$actual"
}

preflight() {
  cd "$PROJECT_ROOT" || return 1
  [[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; return 2; }

  verify_file "$PROJECT_ROOT/eval_post_ft_rgb_restormer_final_v1.py" 5c62edb2435123af715a05ac89f72ee759084af3cb128e9538a272eb52c16490 evaluator || return $?
  verify_file "$PROJECT_ROOT/analyze_post_ft_rgb_restormer_final_v1.py" 1b7677bb51d132147e7221472be740e357d56178f159f1e9e34b1772ee36cccd analyzer || return $?
  verify_file "$PROJECT_ROOT/check_post_ft_rgb_restormer_final_v1.py" 3bf0692bba43edac64a7a3f5f48806509ff481243a6c27562fbde9f7fd8660a1 self_check || return $?
  verify_file "$PROJECT_ROOT/POST_FT_RGB_RESTORMER_FINAL_PROTOCOL_V1.md" 41ef5a177574aba4c4eca113e7cc99ee1dffa932ce8028d1dd4f1c50680f310e protocol || return $?
  verify_file "$PROJECT_ROOT/post_ft_rgb_restormer_final_launch_manifest_v1.json" 149dfa78435ada5f55068eb7871385e70def07b5c3d53da982f293359150c726 launch_manifest || return $?
  verify_file "$PROJECT_ROOT/post_ft_rgb_residual_v1.py" 7ce05a16b62b462ee5b8cd129c45cdab4bc3d891abc16fa39a583dd3f773f952 model || return $?
  verify_file "$PROJECT_ROOT/native_partial_ft_v1.py" 0fcbfaa74890f727b4b30ce7c0aaee0451bac3f8c6448bdd98117d9bf30de6e8 native_loader || return $?
  verify_file "$ARCH" 3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a architecture || return $?
  verify_file "$BASE_CKPT" a272ba1c49254e9b59ad7eb3299a16e8c169ab3f939e31171716064ddd1a5e37 corrected_ft_base || return $?
  verify_file "$SEED3407_CKPT" 54c03b54309a7d7760c78fe6b3bd457afb6998bf720ab27f917e4d84555acadb residual_seed3407 || return $?
  verify_file "$SEED2026_CKPT" b436eb209015042fb338225465a20dda32f03322348009471e2c4a1317ec5e4a residual_seed2026 || return $?
  verify_file "$SEED8888_CKPT" d419c2212834f46c4cf03a41ac2b034eb4ad7630be2a8a627f9dddb6c864d944 residual_seed8888 || return $?

  [[ ! -e "$OUTPUT_ROOT" ]] || { echo "FINAL OUTPUT ALREADY EXISTS: $OUTPUT_ROOT" >&2; return 10; }
  [[ ! -e "$BUNDLE" && ! -e "$BUNDLE.sha256" ]] || { echo "FINAL BUNDLE ALREADY EXISTS: $BUNDLE" >&2; return 10; }
  [[ -d "$TEST_ROOT/lq" && -d "$TEST_ROOT/gt" ]] || { echo "Missing locked test lq/gt: $TEST_ROOT" >&2; return 2; }
  verify_file "$TEST_ROOT/generation_manifest.jsonl" 4f1782e5d4c6e946d7819c5bb647d7f1a5f49355414b54a37cd935d5e4a3d0e2 locked_test_manifest || return $?

  "$PYTHON_BIN" -m py_compile \
    eval_post_ft_rgb_restormer_final_v1.py \
    analyze_post_ft_rgb_restormer_final_v1.py \
    check_post_ft_rgb_restormer_final_v1.py || return $?
  "$PYTHON_BIN" check_post_ft_rgb_restormer_final_v1.py || return $?
  "$PYTHON_BIN" -m json.tool post_ft_rgb_restormer_final_launch_manifest_v1.json >/dev/null || return $?
  export OMP_NUM_THREADS=1
}

run_eval() {
  local mode="$1"
  local seed="$2"
  local output_name="$3"
  local checkpoint="${4:-}"
  local command=(
    "$PYTHON_BIN" -u "$PROJECT_ROOT/eval_post_ft_rgb_restormer_final_v1.py"
    --mode "$mode"
    --seed "$seed"
    --architecture "$ARCH"
    --base_checkpoint "$BASE_CKPT"
    --test_lq "$TEST_ROOT/lq"
    --test_gt "$TEST_ROOT/gt"
    --manifest "$TEST_ROOT/generation_manifest.jsonl"
    --output_dir "$OUTPUT_ROOT/$output_name"
    --workers 4
  )
  if [[ "$mode" == "residual" ]]; then
    command+=(--checkpoint "$checkpoint")
  fi
  "${command[@]}"
}

main() {
  preflight || return $?
  mkdir -p "$OUTPUT_ROOT" || return $?
  cp "$PROJECT_ROOT/post_ft_rgb_restormer_final_launch_manifest_v1.json" "$OUTPUT_ROOT/" || return $?
  cp "$PROJECT_ROOT/POST_FT_RGB_RESTORMER_FINAL_PROTOCOL_V1.md" "$OUTPUT_ROOT/" || return $?

  run_eval base 3407 base || return $?
  run_eval residual 3407 residual_seed3407 "$SEED3407_CKPT" || return $?
  run_eval residual 2026 residual_seed2026 "$SEED2026_CKPT" || return $?
  run_eval residual 8888 residual_seed8888 "$SEED8888_CKPT" || return $?
  "$PYTHON_BIN" -u "$PROJECT_ROOT/analyze_post_ft_rgb_restormer_final_v1.py" \
    --output_root "$OUTPUT_ROOT" || return $?

  sha256sum \
    "$PROJECT_ROOT/eval_post_ft_rgb_restormer_final_v1.py" \
    "$PROJECT_ROOT/analyze_post_ft_rgb_restormer_final_v1.py" \
    "$PROJECT_ROOT/check_post_ft_rgb_restormer_final_v1.py" \
    "$PROJECT_ROOT/POST_FT_RGB_RESTORMER_FINAL_PROTOCOL_V1.md" \
    "$PROJECT_ROOT/post_ft_rgb_restormer_final_launch_manifest_v1.json" \
    "$PROJECT_ROOT/post_ft_rgb_residual_v1.py" \
    "$PROJECT_ROOT/native_partial_ft_v1.py" \
    "$ARCH" \
    "$BASE_CKPT" \
    "$SEED3407_CKPT" \
    "$SEED2026_CKPT" \
    "$SEED8888_CKPT" \
    "$TEST_ROOT/generation_manifest.jsonl" > "$OUTPUT_ROOT/source_hashes.sha256" || return $?
  touch "$OUTPUT_ROOT/POST_FT_RGB_RESTORMER_FINAL_V1_COMPLETE"
  tar -czf "$BUNDLE" -C "$OUTPUT_PARENT" "$(basename "$OUTPUT_ROOT")" || return $?
  sha256sum "$BUNDLE" | tee "$BUNDLE.sha256" || return $?
  echo "result=$OUTPUT_ROOT/final_summary.json"
  echo "bundle=$BUNDLE"
  echo "POST_FT_RGB_RESTORMER_FINAL_V1_COMPLETE"
}

mkdir -p "$LOG_DIR"
main 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "launch_status=$status"
echo "log=$LOG"
exit "$status"



