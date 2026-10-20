# Post-FT RGB-Only Restormer Locked Final-Test Protocol V1

Status: frozen before final-test access on 2026-10-15.

## Question

After official-weight full fine-tuning no longer improves Restormer, does the
frozen-backbone RGB-only residual adapter retain a positive restoration gain on
the previously locked 768-image synthetic-smoke test set?

## Fixed comparison

- Shared baseline: corrected full-FT Restormer checkpoint
  `$PROJECT_ROOT/weights/restormer/best.pth`, SHA256
  `a272ba1c49254e9b59ad7eb3299a16e8c169ab3f939e31171716064ddd1a5e37`.
- Residual seed 3407: validation-selected epoch 6 checkpoint under
  `runs_post_ft_rgb_residual_v1/pilot_both_seed3407_20261014_152331/`;
  SHA256 `54c03b54309a7d7760c78fe6b3bd457afb6998bf720ab27f917e4d84555acadb`.
- Residual seed 2026: validation-selected epoch 6, SHA256
  `b436eb209015042fb338225465a20dda32f03322348009471e2c4a1317ec5e4a`.
- Residual seed 8888: validation-selected epoch 12, SHA256
  `d419c2212834f46c4cf03a41ac2b034eb4ad7630be2a8a627f9dddb6c864d944`.
- Each residual checkpoint must strictly load the complete model and embed
  tensor-identical frozen base weights matching the shared baseline.
- Architecture SHA256:
  `3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a`.
- Model SHA256:
  `7ce05a16b62b462ee5b8cd129c45cdab4bc3d891abc16fa39a583dd3f773f952`.

## Locked data and inference

- Manifest:
  `$PROJECT_ROOT/data/dataset/uav/synthetic_smoke_test_v2/generation_manifest.jsonl`.
- Manifest SHA256:
  `4f1782e5d4c6e946d7819c5bb647d7f1a5f49355414b54a37cd935d5e4a3d0e2`.
- Exactly 768 unique `output_stem` and 768 unique `source_stem` records.
- Input is degraded RGB only. IR, transmission, coarse output, and spatial gate
  are absent.
- Restormer base has 13,926,281 frozen parameters. The ungated RGB adapter has
  43,637 trainable checkpoint parameters.
- Resize to 256 by 256 with OpenCV `INTER_AREA`; batch size 2; no image saving.
- Metrics are per-image PSNR, SSIM, and AlexNet LPIPS, plus fixed severity and
  day/night strata.
- The shared FT baseline is evaluated once. Each of the three already selected
  residual checkpoints is evaluated once against the same baseline.

## Fixed analysis and decision

For each source, compute residual minus baseline for every seed, then average
the three seed deltas for that source. Use 50,000 percentile bootstrap resamples
over the 768 unique source identities with RNG seed 20261015. Report the overall
result and the pre-existing light, dense, very-dense, day, and night strata.

The result supports the RGB residual adapter only if all conditions hold:

1. Every seed has positive mean PSNR delta.
2. The three-seed mean PSNR delta is positive.
3. Its paired 95% bootstrap confidence interval has a positive lower bound.
4. The three-seed mean SSIM delta is nonnegative.
5. The three-seed mean LPIPS delta is nonpositive.

Failure of any condition yields `FINAL_TEST_DOES_NOT_SUPPORT_RGB_RESIDUAL`.
Final-test results cannot select a seed, checkpoint, threshold, subgroup, or
revised method. A failed condition closes this branch without tuning or rerun.

