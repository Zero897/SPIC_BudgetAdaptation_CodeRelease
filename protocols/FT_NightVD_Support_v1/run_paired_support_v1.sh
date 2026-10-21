#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-preflight}
ARM=${2:-}
SEED=${3:-}
ROOT=/root/autodl-tmp/CrossFlowIE
PKG=$ROOT/FT_NightVD_Support_v1
SRC=$PKG/source
VAL=$ROOT/data/dataset/uav/synthetic_smoke_val_v2
TRAIN=$ROOT/data/dataset/uav/train
WEIGHT=$ROOT/weights/dehazeformer/latest.pth
ORIGINAL=$ROOT/data/dataset_crossmodal.py
OUT=$ROOT/analysis-output/ft_nightvd_support_v1
export PYTHONPATH="$ROOT/DehazeFormer-main:$SRC${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

check_identity() {
  (cd "$PKG" && sha256sum -c SHA256SUMS.txt)
  test -f "$SRC/train_cm_dehazeformer_v2_residual.py"
  test -f "$ORIGINAL"
  test -f "$WEIGHT"
  test -f "$VAL/generation_manifest.jsonl"
  test -d "$TRAIN/gt"
  test -d "$TRAIN/ir"
  printf '%s  %s\n' \
    'cd9dfb97ce55e035e3669071e5ae9ccd38e8ab6394534ea31ee30ba640f404d6' "$SRC/train_cm_dehazeformer_v2_residual.py" \
    'c52a7051483e7a8706eb883d73af511ebf8ffd3b15b672736796a96f7a2a207b' "$SRC/crossmodal_dehazeformer_residual_v2.py" \
    '2531f3704fab0b72dd5d6466674c2a7454c6f8a06521e71c111067fc71b7ba64' "$SRC/cm_dehazeformer_v2_loss.py" \
    '5e8a7c975a3f90c54c393c53ab7528eacc70e5b95aff4b8d9009107456daa34d' "$ORIGINAL" \
    'e5af35e6254916c3bfbcdcd25bdbfd856e65942e0a682a12d986ca951183ef88' "$WEIGHT" \
    '943252c7e269d277a0bfa36d397c11022e040113e94c43d32b392cbb76031128' "$VAL/generation_manifest.jsonl" \
    | sha256sum -c -
  python "$PKG/verify_generator.py" --original-dataset "$ORIGINAL"
}

check_identity
if [[ "$MODE" == preflight ]]; then
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print("CUDA_PREFLIGHT_PASS")'
  python "$PKG/audit_selection.py" --train-root "$TRAIN" --output "$OUT/preflight_selection.json"
  exit 0
fi
if [[ "$MODE" != train || ! "$ARM" =~ ^(default|expanded)$ || ! "$SEED" =~ ^(3407|2026|8888)$ ]]; then
  echo 'usage: bash run_paired_support_v1.sh preflight | train default|expanded 3407|2026|8888' >&2
  exit 2
fi
DEST=$OUT/FT_${ARM}_seed${SEED}
test ! -e "$DEST" || { echo "Refusing existing output: $DEST" >&2; exit 1; }
mkdir -p "$OUT"
export FT_NIGHT_VD_ARM=$ARM FT_NIGHT_VD_SEED=$SEED
echo "FT_NIGHT_VD_ARM=$ARM FT_NIGHT_VD_SEED=$SEED"
exec python -u "$SRC/train_cm_dehazeformer_v2_residual.py" \
  --experiment_id FT --train_mode base_only \
  --train_root "$TRAIN" --val_lq "$VAL/lq" --val_ir "$VAL/ir" --val_gt "$VAL/gt" \
  --save_dir "$DEST" --backbone b --pretrained "$WEIGHT" \
  --size 256 --epochs 10 --batch 8 --grad_accum_steps 2 --workers 8 \
  --lr_base 1e-6 --lr_new 0 --freeze_base_epochs 0 \
  --eval_every 2 --eval_max_batches -1 \
  --init_gamma 0.01 --adapter_variant fscatterres \
  --ablation wo_ir_physics --wo_gate \
  --refiner_width 24 --refiner_blocks 4 \
  --seed "$SEED" --deterministic
