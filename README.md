# SPIC Budget Adaptation Code Release (draft)

Curated source snapshot for the SPIC manuscript on budgeted RGB restoration adaptation. This repository contains project-owned model wrappers, training and evaluation entry points, smoke-generation code, and experiment protocols. It does not contain DehazeFormer, Restormer, or FFA-Net upstream source, datasets, checkpoints, or experiment outputs.

## Experiments covered

- Main DehazeFormer-B comparison: frozen baseline (B0), RGB output adapter (P0), and full fine-tuning (FT).
- Matched native partial fine-tuning control (M-FT).
- Post-FT RGB adapter extensions on Restormer and FFA-Net.
- Stratified and locked synthetic-smoke evaluation, RRSHID zero-shot evaluation, A2I2 vehicle detection, and efficiency measurement.

## Main entry points

- `train_cm_dehazeformer_v2_residual.py`: B0/P0/FT training.
- `train_matched_partial_ft_v1.py`: M-FT training.
- `eval_protocol_v2_stratified.py`, `eval_protocol_v2_final_test.py`: primary evaluators.
- `train_post_ft_rgb_residual_v1.py`, `post_ft_rgb_residual_v1.py`: post-FT RGB adapter.
- `eval_post_ft_rgb_restormer_final_v1.py`, `eval_post_ft_rgb_ffa_final_v1.py`: cross-backbone evaluation.
- `eval_rrshid_zero_shot_protocol_v1.py`, `a2i2/eval_a2i2_downstream_v1.py`: external endpoints.
- `benchmark_protocol_v2_efficiency_v1.py`: efficiency measurements.
- `generate_fixed_testset.py`, `generate_fixed_valset.py`: smoke data generation and fixed validation set.

## Source and checkpoint provenance

See `UPSTREAM_VERSIONS.md` for file-verified upstream references and license findings. See `CHECKPOINT_PROVENANCE.md` for the checkpoint hashes and origins that are supported by the archived records. The project `LICENSE` applies only to original project-authored files.

## External assets and data

Obtain model source and checkpoints separately from the upstream providers listed in `THIRD_PARTY_NOTICES.md`; this project intentionally does not vendor them. The FFA-Net source identity currently has no explicit upstream license grant, and Restormer's experiment-matched source commit is non-commercial; review `UPSTREAM_VERSIONS.md` before using those branches. Obtain DroneVehicle, RRSHID, and A2I2-Haze from the original providers under their terms. No images, labels, generated samples, or frozen evaluation manifests are included. See `DATASET_NOTICE.md`.

The principal trainer accepts explicit paths for the training data, validation folders, pretrained checkpoint, and output directory. Historical protocol documents are under `protocols/`; some retain AutoDL paths and require adaptation before direct use.

## Environment

The available run records identify Python 3.9, PyTorch 2.1.2 with CUDA 12.1, and OpenCV 4.12.0. `requirements.txt` is a preliminary package list; versions for the remaining packages have not been recovered, so it is not a complete environment lock. See `ENVIRONMENT.md`.

## Release status

This is a local preparation repository, not yet a public-ready release. See `RELEASE_STATUS.md`. No project license has been chosen; keep the repository private until the authors decide reuse terms.

