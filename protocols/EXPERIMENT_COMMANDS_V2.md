# Protocol-v2 experiment commands

These commands use the fixed 1,469-image validation set for checkpoint
selection. They never read the locked final test set.

Run from `$PROJECT_ROOT` after copying the two modified Python
files into that directory. Replace `SEED` with `3407`, `2027`, or `8888`.

```bash
cd $PROJECT_ROOT || exit 1
export OMP_NUM_THREADS=1

SEED=3407
TRAIN=$PROJECT_ROOT/data/dataset/uav/train
VAL=$PROJECT_ROOT/data/dataset/uav/synthetic_smoke_val_v2
PRETRAINED=$PROJECT_ROOT/weights/dehazeformer/latest.pth
RUNS=$PROJECT_ROOT/runs_protocol_v2
```

## B0: frozen DehazeFormer-B

```bash
python train_cm_dehazeformer_v2_residual.py \
  --experiment_id B0 --train_mode frozen_baseline \
  --train_root "$TRAIN" \
  --val_lq "$VAL/lq" --val_ir "$VAL/ir" --val_gt "$VAL/gt" \
  --save_dir "$RUNS/B0_frozen/seed${SEED}" \
  --backbone b --pretrained "$PRETRAINED" \
  --size 256 --epochs 40 --batch 16 --workers 8 \
  --lr_base 0 --lr_new 0 --freeze_base_epochs 40 \
  --eval_every 5 --eval_max_batches -1 \
  --init_gamma 0.01 --adapter_variant fscatterres \
  --ablation wo_ir_physics --wo_gate \
  --refiner_width 24 --refiner_blocks 4 \
  --seed "$SEED" --deterministic
```

## A0: ordinary RGB residual correction

```bash
python train_cm_dehazeformer_v2_residual.py \
  --experiment_id A0 --train_mode adapter \
  --train_root "$TRAIN" \
  --val_lq "$VAL/lq" --val_ir "$VAL/ir" --val_gt "$VAL/gt" \
  --save_dir "$RUNS/A0_core_residual/seed${SEED}" \
  --backbone b --pretrained "$PRETRAINED" \
  --size 256 --epochs 40 --batch 16 --workers 8 \
  --lr_base 0 --lr_new 1e-4 --freeze_base_epochs 40 \
  --eval_every 5 --eval_max_batches -1 \
  --init_gamma 0.01 --adapter_variant fscatterres \
  --ablation wo_ir_physics --wo_gate \
  --refiner_width 24 --refiner_blocks 4 \
  --seed "$SEED" --deterministic
```

## P0: parameter-matched plain RGB residual adapter

P0 uses only `[base prediction, smoke RGB]`, four ordinary residual blocks,
and width 25. It has approximately the same active parameter budget as A2 but
does not use scattering guidance, infrared, AttnRes, or an output gate.

```bash
python train_cm_dehazeformer_v2_residual.py \
  --experiment_id P0 --train_mode adapter \
  --train_root "$TRAIN" \
  --val_lq "$VAL/lq" --val_ir "$VAL/ir" --val_gt "$VAL/gt" \
  --save_dir "$RUNS/P0_plain_matched/seed${SEED}" \
  --backbone b --pretrained "$PRETRAINED" \
  --size 256 --epochs 40 --batch 16 --workers 8 \
  --lr_base 0 --lr_new 1e-4 --freeze_base_epochs 40 \
  --eval_every 5 --eval_max_batches -1 \
  --init_gamma 0.01 --adapter_variant plain \
  --ablation wo_ir_physics --wo_gate \
  --refiner_width 25 --refiner_blocks 4 \
  --seed "$SEED" --deterministic
```

## A2: complete RGB adapter

```bash
python train_cm_dehazeformer_v2_residual.py \
  --experiment_id A2 --train_mode adapter \
  --train_root "$TRAIN" \
  --val_lq "$VAL/lq" --val_ir "$VAL/ir" --val_gt "$VAL/gt" \
  --save_dir "$RUNS/A2_full_rgb/seed${SEED}" \
  --backbone b --pretrained "$PRETRAINED" \
  --size 256 --epochs 40 --batch 16 --workers 8 \
  --lr_base 0 --lr_new 1e-4 --freeze_base_epochs 40 \
  --eval_every 5 --eval_max_batches -1 \
  --init_gamma 0.01 --adapter_variant fscatterres \
  --ablation wo_ir --wo_gate --use_attnres \
  --refiner_width 24 --refiner_blocks 4 \
  --seed "$SEED" --deterministic
```

## A3: complete RGB adapter plus infrared

```bash
python train_cm_dehazeformer_v2_residual.py \
  --experiment_id A3 --train_mode adapter \
  --train_root "$TRAIN" \
  --val_lq "$VAL/lq" --val_ir "$VAL/ir" --val_gt "$VAL/gt" \
  --save_dir "$RUNS/A3_rgb_ir/seed${SEED}" \
  --backbone b --pretrained "$PRETRAINED" \
  --size 256 --epochs 40 --batch 16 --workers 8 \
  --lr_base 0 --lr_new 1e-4 --freeze_base_epochs 40 \
  --eval_every 5 --eval_max_batches -1 \
  --init_gamma 0.01 --adapter_variant fscatterres \
  --ablation full --wo_gate --use_attnres \
  --refiner_width 24 --refiner_blocks 4 \
  --seed "$SEED" --deterministic
```

## FT: DehazeFormer-B-only fine-tuning

The adapter is bypassed and frozen. The initial pilot uses a predeclared
backbone learning rate of `1e-5`; change it only on validation evidence and
record the selection.

```bash
python train_cm_dehazeformer_v2_residual.py \
  --experiment_id FT --train_mode base_only \
  --train_root "$TRAIN" \
  --val_lq "$VAL/lq" --val_ir "$VAL/ir" --val_gt "$VAL/gt" \
  --save_dir "$RUNS/FT_base_only/seed${SEED}" \
  --backbone b --pretrained "$PRETRAINED" \
  --size 256 --epochs 40 --batch 16 --workers 8 \
  --lr_base 1e-5 --lr_new 0 --freeze_base_epochs 0 \
  --eval_every 5 --eval_max_batches -1 \
  --init_gamma 0.01 --adapter_variant fscatterres \
  --ablation wo_ir_physics --wo_gate \
  --refiner_width 24 --refiner_blocks 4 \
  --seed "$SEED" --deterministic
```

For a short implementation smoke test, add `--smoke_test --epochs 1
--eval_every 1 --train_max_batches 2 --eval_max_batches 2` and use a separate
save directory containing the experiment ID, seed, and the word `smoke`.

## First runtime gate: A0 two-batch smoke test

Run this before any 40-epoch experiment:

```bash
python train_cm_dehazeformer_v2_residual.py \
  --experiment_id A0 --train_mode adapter --smoke_test \
  --train_root "$TRAIN" \
  --val_lq "$VAL/lq" --val_ir "$VAL/ir" --val_gt "$VAL/gt" \
  --save_dir "$RUNS/smoke/A0_core_residual/seed${SEED}" \
  --backbone b --pretrained "$PRETRAINED" \
  --size 256 --epochs 1 --batch 2 --workers 0 \
  --lr_base 0 --lr_new 1e-4 --freeze_base_epochs 1 \
  --eval_every 1 --train_max_batches 2 --eval_max_batches 2 \
  --init_gamma 0.01 --adapter_variant fscatterres \
  --ablation wo_ir_physics --wo_gate \
  --refiner_width 24 --refiner_blocks 4 \
  --seed "$SEED" --deterministic
```

The gate passes only if the run has finite loss, completes forward/backward,
writes `epoch_1.pth`, `last.pth`, `best_psnr.pth`, `run_config.json`, and
`history.jsonl`, and reports the expected validation manifest hash.

