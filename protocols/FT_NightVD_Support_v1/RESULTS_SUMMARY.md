# Paired evaluation summary

The table reports expanded minus default FT. Checkpoint selection used the aggregate-PSNR rule as primary; six-stratum-balanced selection is secondary. Metrics are paired on the same validation examples. No checkpoints, per-image outputs, or validation images are included.

| Selection rule | Seed | Selected epochs (default / expanded) | All PSNR (dB) | Night very-dense PSNR (dB) | Day very-dense PSNR (dB) | Night improved / regressed |
|---|---:|---:|---:|---:|---:|---:|
| Aggregate (primary) | 3407 | 4 / 4 | +0.013 | +0.201 | -0.094 | 44 / 20 |
| Aggregate (primary) | 2026 | 4 / 4 | +0.024 | +0.281 | -0.060 | 46 / 18 |
| Aggregate (primary) | 8888 | 6 / 6 | +0.017 | +0.367 | -0.074 | 50 / 14 |
| **Aggregate three-seed mean** | — | — | **+0.018** | **+0.283** | **-0.076** | — |
| Six-stratum balanced (secondary) | 3407 | 4 / 4 | +0.013 | +0.201 | -0.094 | 44 / 20 |
| Six-stratum balanced (secondary) | 2026 | 2 / 2 | +0.009 | +0.169 | -0.082 | 45 / 19 |
| Six-stratum balanced (secondary) | 8888 | 2 / 2 | +0.012 | +0.230 | -0.083 | 47 / 17 |
| **Balanced three-seed mean** | — | — | **+0.012** | **+0.200** | **-0.086** | — |

The expanded range produced higher selected-checkpoint PSNR for the night very-dense stratum in all three seeds, alongside a smaller decrease for day very-dense samples. Because the intervention changes `constant_bias` and `transmission_power` jointly, the experiment does not identify either parameter as the cause of the observed differences. It also does not establish scene/video independence for the validation cohort.

Source: `paired_comparison_after_eval_20261021.json`, transferred evaluation summary. The complete metrics and per-image artifacts are not included in this code repository.
