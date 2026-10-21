# FT nighttime very-dense smoke range-support experiment

This package records the completed paired three-seed experiment that expanded the synthetic-smoke parameter ranges for a hash-selected subset of nighttime training sources. It changes the joint `constant_bias` and `transmission_power` ranges; it does not isolate either parameter or frequency alone. The default arm was checked for bit-exact generator behavior, and the paired arms used matched source images, initialization, optimizer settings, and update budgets. The selected source fraction was approximately one fifth.

`PROTOCOL.md` is the archived pre-run protocol. The training and paired evaluation are complete; the observed aggregate-rule results are summarized in `RESULTS_SUMMARY.md`. Checkpoints, datasets, validation outputs, and logs are not included.

## Execution context

The scripts were written for the original AutoDL CrossFlowIE layout and expect the project at `/root/autodl-tmp/CrossFlowIE`, including its `flowie` environment, training data, validation set, DehazeFormer source, and pretrained weights. Those assets are not included in this code release. The scripts are provided for protocol and source inspection and require path adaptation before use in another environment. This package does not establish that the experiment can be reproduced from this repository alone.

The `source/` subdirectory preserves the exact source snapshot used by the run. The training entry, model, and loss files match the original FT sources; `source/data/dataset_crossmodal.py` contains the range-support modification. `SHA256SUMS.txt` records package-file integrity. The shell scripts also verify external source, weight, and manifest hashes against the AutoDL run inputs.

## Archived entry points

- `run_paired_support_v1.sh`: identity checks, generator verification, preflight, and one training arm.
- `evaluate_paired_support_v1.sh`: paired checkpoint-step audit, stratified evaluations, and checkpoint selection.
- `selection_rule_v1.py`, `compare_arms.py`: selection and paired comparison.
- `audit_selection.py`, `verify_generator.py`: selection audit and generator/RNG checks.
