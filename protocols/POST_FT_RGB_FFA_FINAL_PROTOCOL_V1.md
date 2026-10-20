# Post-FT RGB-only FFA-Net locked final-test protocol V1

Frozen 2026-10-15 before the first FFA evaluation in this extension. This is a
fixed FFA evaluation on the existing project test set, which was previously
evaluated for other model families. Do not describe the whole project test set
as never accessed. FFA gamma/checkpoint selection used development validation.

## Fixed question and model

Does the gated RGB residual adapter retain a positive restoration gain over
the identical frozen already fine-tuned FFA base on the locked synthetic-smoke
test? This evaluation does not independently establish full-FT convergence.

- Base: FFA-Net-master/net/trained_models/its_train_ffa_3_19_epoch_50.pk.
  SHA256: 113587bb1c656fd77b1c0999f4f0d8e60e55b5f351fa2bd53f39444d8bd1d206.
- Architecture: ffa_arch_protocol_v2.py, SHA256
  8e2e7122cb510f157ccf7e1ae09b8885d63634f5e572d12c43ef06e95e3bc199.
- Seed 3407: validation-selected epoch 12, SHA256
  a852a3b3bfb1b5eb65ba1d36cad2de50cd5bee863611a406775d6015b607a725.
- Seed 2026: validation-selected epoch 15, SHA256
  b303fb9b69f6b8264135a5b6b803ca9e3d9c0bd5851080047ab4776585cdedc0.
- Seed 8888: validation-selected epoch 9, SHA256
  ef92d0621f3f6b04c52944b44064e11983136bb8e10d13d327d32056429268b6.
- Checkpoint paths point to the audited confirm_20261015_132440 bundle directory,
  including its identity-checked copy of seed 3407. No training is performed.
- Complete checkpoints load strictly; embedded base tensors must equal the
  strictly loaded shared FT baseline. Base: 4,455,913 frozen parameters. RGB
  adapter including active spatial gate: 46,254 checkpoint parameters.
- Trainable gamma was initialized at 0.01, clipped to [0,0.12]; retain the
  saved learned gamma, gate, residual scale and adapter weights. Do not reset
  gamma to its initialization or substitute gamma-0.08 checkpoints.
- Degraded RGB only. No IR, transmission, coarse output or AttnRes inputs.
  FFA input normalization stays inside the frozen model wrapper.

## Locked data and inference

- Root: $PROJECT_ROOT/data/dataset/uav/synthetic_smoke_test_v2.
- generation_manifest.jsonl SHA256:
  4f1782e5d4c6e946d7819c5bb647d7f1a5f49355414b54a37cd935d5e4a3d0e2.
- Exactly 768 unique output_stem and 768 unique source_stem identities.
- OpenCV RGB INTER_AREA resize to 256x256; batch 2, workers 4; no augmentation,
  test-time averaging, image output or subset selection. FP32 inference.
- Per-image PSNR, SSIM and AlexNet LPIPS use the same implementations as the
  Restormer final evaluator. Fixed strata: all/light/dense/very_dense/day/night.
- Shared FT base once, then each of the three fixed residual checkpoints once.
- Output: results_post_ft_rgb_ffa_final_v1/locked_v1. Existing output/bundle is
  refused. Keep failed artifacts for audit instead of deleting them to rerun.

## Analysis and predeclared support rule

For each source compute residual minus the shared baseline for each seed, then
average the three seed deltas at that source. Use 50,000 percentile bootstrap
resamples over source_stem, RNG seed 20261015, with the same fixed group RNG
offsets as Restormer. Report every seed, overall metrics and all fixed strata.
Intervals describe source-resampling variability for these fixed checkpoints;
they do not capture training-seed uncertainty or prove scene/video independence.

All five conditions must hold for FINAL_TEST_SUPPORTS_RGB_RESIDUAL:
1. Every seed has positive overall mean PSNR delta.
2. The three-seed overall mean PSNR delta is positive.
3. Its paired 95% source-resampling interval has a positive lower bound.
4. The three-seed overall mean SSIM delta is nonnegative.
5. The three-seed overall mean LPIPS delta is nonpositive.

Otherwise report FINAL_TEST_DOES_NOT_SUPPORT_RGB_RESIDUAL. The validation
continuation threshold of +0.10 dB is not a new final-test threshold. Do not
select seeds, epochs, gamma, caps, metrics, strata or revised methods using
test results. Do not tune or add seeds after an unfavorable result. Gate
causality, IR/physics benefit and external generalization remain unestablished.

## Execution boundary

Local preparation/self-checks use synthetic tensors and already returned
checkpoints only. They do not evaluate locked test images. The experiment owner
manually runs run_post_ft_rgb_ffa_final_v1.sh on the server once. Return its log,
post_ft_rgb_ffa_final_locked_v1.tar.gz and .sha256. Pipeline COMPLETE is distinct
from the scientific support decision in final_summary.json.

