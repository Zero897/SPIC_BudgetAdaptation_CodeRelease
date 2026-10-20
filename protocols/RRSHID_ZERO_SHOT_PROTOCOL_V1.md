# RRSHID Protocol V1: strict zero-shot external evaluation

Frozen: 2026-10-09, before any Protocol V1 RRSHID model metric was computed.

## Question

Does the already selected Protocol V2 P0 residual adapter retain a practically
meaningful gain over the frozen B0 checkpoint on an independent paired real
remote-sensing haze dataset without target-domain adaptation?

This experiment supports, at most, zero-shot cross-domain remote-sensing haze
evidence. It is not paired real-smoke evidence.

## Fixed comparison

- B0: one deterministic evaluation with evaluator seed 3407.
- P0: existing validation-selected seeds 3407, 2026, and 8888.
- FT: existing validation-selected seeds 3407, 2026, and 8888 as the
  high-update-parameter control.
- No RRSHID training, validation, calibration, checkpoint selection, residual
  scaling, threshold tuning, or subset selection is allowed.
- Test data: all 304 paired RRSHID test images, stratified as 76 thin, 152
  moderate, and 76 thick.
- Metrics: PSNR, SSIM, and LPIPS-Alex, reported for all/thin/moderate/thick.
- Input: native 256 x 256 RGB uint8 converted deterministically to float [0,1].
  No resizing or batch-dependent range inference is used.

## Prospective decision rule

The external-evidence gate passes only if all conditions hold:

1. every P0 seed has positive all-sample PSNR change versus B0;
2. mean all-sample P0-minus-B0 PSNR is at least +0.10 dB;
3. mean all-sample SSIM change is at least -0.001;
4. mean all-sample LPIPS change is at most +0.001;
5. at least two of thin/moderate/thick have positive mean PSNR change.

On PASS, the result may support a bounded zero-shot cross-domain claim in the
CAS Zone 3 manuscript. On FAIL, do not train or tune on RRSHID to rescue that
claim; record the failure and close this external-evidence branch.

## Frozen identities

| Artifact | SHA256 |
|---|---|
| `eval_rrshid_zero_shot_protocol_v1.py` | `dd34ca74df2db6a20a9f0bb0531d4f1eade2079a300579f65f27d6b1d6940a55` |
| `analyze_rrshid_zero_shot_protocol_v1.py` | `d4091013a6abc8b297828c265c13f22dc070e076c284c836889e4fce554d4670` |
| `run_rrshid_zero_shot_v1.sh` | `d42c7a3e141007757db050132be1de811a8b2b810bea3e8b6af6747ed4dc4c74` |
| `rrshid_zero_shot_launch_manifest_v1.json` | `78bc63477fe33dd6dddd48ef8e97f99b7ac8fc56fed50e9aec6ee49f12866d0a` |
| `rrshid_zero_shot_test_manifest_v1.jsonl` | `3a0a349a3a92eff0e7467def332c2dda4e1ec38047855cb485762db894221053` |
| RRSHID `metadata.csv` | `68088a2365c2252a0e63a4ca546ff62ad8661af275c6a78815d8eee044846693` |
| `crossmodal_dehazeformer_residual_v2.py` | `c52a7051483e7a8706eb883d73af511ebf8ffd3b15b672736796a96f7a2a207b` |
| `dataset_crossmodal.py` | `5e8a7c975a3f90c54c393c53ab7528eacc70e5b95aff4b8d9009107456daa34d` |
| `train_cm_dehazeformer_v2_residual.py` | `cd9dfb97ce55e035e3669071e5ae9ccd38e8ab6394534ea31ee30ba640f404d6` |

Checkpoint paths, hashes, byte counts, selected epochs, and selected validation
scores are fixed in `rrshid_zero_shot_launch_manifest_v1.json` and match the
completed Protocol V2 final-test comparison set.

## Validation completed locally

- Python compilation: PASS.
- CLI import/help: PASS.
- locked 304-row manifest, file-size, per-file SHA256, pairing, image shape,
  and density counts: PASS.
- rejection of train, val, validation, and non-test roots: PASS.
- metric implementation bit-exact with the locked final-test evaluator on a
  deterministic tensor check: PASS.
- real FT seed-2026 checkpoint config validation and strict load: PASS;
  checkpoint SHA256 matched the frozen identity.
- synthetic seven-run analyzer reconstruction and PASS-gate test: PASS.
- Bash syntax: PASS.

No RRSHID model inference or metric calculation was performed locally.


