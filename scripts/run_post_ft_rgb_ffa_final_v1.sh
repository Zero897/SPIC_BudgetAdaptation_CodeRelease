#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
: "${FFA_ARCHITECTURE:?Set to an external FFA-Net architecture file; source is not included}"
: "${FFA_BASE_CHECKPOINT:?Set to the fine-tuned FFA-Net base checkpoint}"
: "${TEST_LQ:?Set test LQ directory}"
: "${TEST_GT:?Set test GT directory}"
: "${TEST_MANIFEST:?Set the frozen manifest after release review}"
: "${OUTPUT_DIR:?Set a new output directory}"
: "${FFA_RESIDUAL_CHECKPOINT_3407:?Set seed 3407 adapter checkpoint}"
: "${FFA_RESIDUAL_CHECKPOINT_2026:?Set seed 2026 adapter checkpoint}"
: "${FFA_RESIDUAL_CHECKPOINT_8888:?Set seed 8888 adapter checkpoint}"
"$PYTHON_BIN" "$ROOT/eval_post_ft_rgb_ffa_final_v1.py" --mode base --seed 3407 --architecture "$FFA_ARCHITECTURE" --base_checkpoint "$FFA_BASE_CHECKPOINT" --test_lq "$TEST_LQ" --test_gt "$TEST_GT" --manifest "$TEST_MANIFEST" --output_dir "$OUTPUT_DIR/base"
for seed in 3407 2026 8888; do
  var="FFA_RESIDUAL_CHECKPOINT_${seed}"
  checkpoint="${!var}"
  "$PYTHON_BIN" "$ROOT/eval_post_ft_rgb_ffa_final_v1.py" --mode residual --seed "$seed" --architecture "$FFA_ARCHITECTURE" --base_checkpoint "$FFA_BASE_CHECKPOINT" --checkpoint "$checkpoint" --test_lq "$TEST_LQ" --test_gt "$TEST_GT" --manifest "$TEST_MANIFEST" --output_dir "$OUTPUT_DIR/seed${seed}"
done
