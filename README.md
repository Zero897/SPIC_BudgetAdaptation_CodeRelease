# SPIC Budget Adaptation Code Release

Curated source snapshot for the SPIC manuscript on budgeted RGB restoration adaptation. This repository contains project-owned model wrappers, training and evaluation entry points, smoke-generation code, and experiment protocols. It does not contain DehazeFormer, Restormer, or FFA-Net upstream source, datasets, checkpoints, or experiment outputs.

## Experiments covered

- Main DehazeFormer-B comparison: frozen baseline (B0), RGB output adapter (P0), and full fine-tuning (FT).
- Matched native partial fine-tuning control (M-FT).
- Post-FT RGB adapter extensions on Restormer and FFA-Net.
- Stratified and locked synthetic-smoke evaluation, RRSHID zero-shot evaluation, A2I2 vehicle detection, and efficiency measurement.
- Follow-up FT smoke-range experiment, with its source snapshot, protocol, and summarized paired results in [`protocols/FT_NightVD_Support_v1/`](protocols/FT_NightVD_Support_v1/).

## Reproduction scope

This repository releases the project-owned code paths and protocols; inclusion of an experiment's code does not mean that every manuscript result has been rerun from this repository. During release verification, one single-seed A0 adapter training run completed for 40 epochs on AutoDL. Its saved configuration and validation history were checked, and the recorded hashes for the training source files match this release. This verifies that specific training path only. It is not a claim that all experiments or reported results in the manuscript have been reproduced.

Datasets, pretrained weights, upstream model source, and experiment outputs are not bundled. Reproducing other experiment arms requires those external assets, compatible runtime setup, and potentially substantial compute; see the data, provenance, and environment notes below.

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

The principal trainer accepts explicit paths for the training data, validation folders, pretrained checkpoint, and output directory. Historical protocol documents are under `protocols/`; some retain AutoDL paths and require adaptation before direct use. The FT night-range package is an archived follow-up experiment, not a standalone reproduction environment; its bundled run scripts expect the original CrossFlowIE layout and external data, checkpoint, and upstream source.

## Environment

The AutoDL runtime probe identifies Python 3.9.25, PyTorch 2.1.2+cu121, and CUDA 12.1. Exact package inventories are archived under `environment/`; they contain conflicting Torch and overlapping OpenCV metadata, so they are not a clean portable lock. The clean target pins OpenCV 4.11.0.86 to remain compatible with NumPy 1.26.4 and PyTorch 2.1.2. The successful A0 run does not validate a fresh installation from the dependency files; a full transitive lock is not included. See `ENVIRONMENT.md`.

```bash
conda env create -f environment.yml
conda activate spic-budget-adapt
python -m pip install -r requirements.txt
```

## Release status

Original project-authored files are licensed under MIT. Third-party source, weights, and data remain under their own terms; the experiment-matched Restormer source commit is non-commercial and FFA-Net has no explicit license grant at the identified commit. These third-party sources and weights are not redistributed. Review `THIRD_PARTY_NOTICES.md` and `RELEASE_STATUS.md` before using those branches.

