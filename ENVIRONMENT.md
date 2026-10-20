# Environment evidence and version audit

## Observed in transferred AutoDL run logs

- Python: 3.9
- PyTorch: 2.1.2+cu121
- CUDA runtime: 12.1
- OpenCV: 4.12.0
- GPU used for the relevant runs: NVIDIA GeForce RTX 4090

These are the versions printed by AutoDL run records. The record does not include a full `pip freeze`, exact Python patch version, or exact versions/build strings for torchvision, NumPy, timm, LPIPS, TensorBoard, tqdm, einops, and transitive dependencies.

## Local environment cross-check

The Windows `FlowIE` environment currently reports Python 3.9.25, PyTorch 2.1.2+cu118, torchvision 0.16.2+cu118, timm 0.6.12, LPIPS 0.1.4, TensorBoard 2.11.2, tqdm 4.67.1, einops 0.8.1, NumPy 1.24.4, and scikit-image 0.24.0. It has OpenCV Python packages 4.11.0.86 and 4.12.0.88 installed together. This is a separate CUDA 11.8 Windows environment and is not a substitute for an exact lock of the CUDA 12.1 AutoDL runs.

`requirements.txt` and `environment.yml` are therefore a best-effort install recipe, not a complete lock file. The exact AutoDL dependency lock cannot be recovered from the files currently archived. To finalize it, export `python -m pip freeze` and `conda list --explicit` from the original AutoDL `flowie` environment, then preserve those exports with the run metadata.

