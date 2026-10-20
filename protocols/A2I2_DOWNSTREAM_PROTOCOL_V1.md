# A2I2 downstream vehicle-detection protocol V1

Status: `FULL_RELEASE_PREPARED_NOT_RUN`

## Objective

Evaluate whether restoration improves zero-shot vehicle detection on the 177
A2I2 hazy upper frames, without A2I2 fitting or result-dependent selection.

## Frozen arms

The comparison contains raw hazy input, B0, three P0 seeds, and three M-FT
seeds. Checkpoint paths and SHA256 values are fixed in the launch manifest.
P0 versus B0 is primary. M-FT versus paired-seed P0 is a matched-budget control.

## Geometry-preserving restoration

Each 1845x750 upper frame is restored using 256x256 tiles, 64-pixel overlap,
batch four, and a fixed clamped-Hann blending window. Every restored output is
written as an 8-bit 1845x750 PNG. All restoration arms use the same rule and an
all-zero unused IR map. No output scaling or target-domain calibration is used.

## Frozen detectors

- YOLOv8s: image size 640, AP prediction floor 0.001, NMS IoU 0.7, max 300.
- Faster R-CNN ResNet-50-FPN V2: two classes including background, transform
  min/max 512/640, AP prediction floor 0.001, max 300.
- Fixed IoU-0.50 precision/recall thresholds are 0.25 for YOLO and
  0.70195472240448 for Faster R-CNN. AP50:95 and AP50 are primary/secondary;
  smoke metrics cannot change these settings.
- AP uses the frozen single-class COCO-style 101-point interpolation at IoU
  0.50:0.95 with at most 300 predictions per image. A2I2 contains one vehicle
  class and no crowd annotations in this protocol.

## Stages

1. `prepare` extracts the frozen 177-image private data archive.
2. `preflight` verifies every source image/label and checkpoint hash and strict
   loads all seven restoration models plus both detector families. It performs
   no image inference.
3. `smoke` runs only manifest-first/middle/last IDs `000`, `092`, and `192`.
   Its metrics are technical diagnostics only.
4. The returned technical Smoke archive
   `smoke_20261013_170912.tar.gz` passed external/internal identity, geometry,
   eight-arm, fixed-ID, and two-detector completeness review. Its SHA256 is
   `1420d6519394afad576ce0a102d479a848e4b4de5cecbbd412b0f082ba86b8c3`.
5. `full` evaluates all 177 frozen manifest rows with the unchanged arms,
   checkpoints, tiling, detectors, thresholds, and metrics. It writes all exact
   detector input images, predictions, per-arm manifests, ground truth, a full
   summary, launch log, and internal hashes to a fresh timestamped output.
6. The launcher fixes `OMP_NUM_THREADS=1` because the returned Smoke inherited
   a malformed value. This is a runtime hygiene correction; it does not change
   inference outputs, model settings, thresholds, or metrics.

No A2I2 image, label, threshold, checkpoint, subset, or smoke result may be used
to fit or select a method. Image bootstrap cannot be called video- or
scene-independent because source video IDs are unavailable.

The full run remains user-started. A failed run is preserved in its unique
output directory and log and must not be renamed or interpreted as complete.

