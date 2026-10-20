# Environment evidence and remaining version audit

## Observed in transferred AutoDL run logs

- Python: 3.9
- PyTorch: 2.1.2+cu121
- CUDA runtime: 12.1
- OpenCV: 4.12.0
- GPU used for the relevant runs: NVIDIA GeForce RTX 4090

## Not yet locked

The archived project does not contain a complete `pip freeze` or Conda lock for the exact manuscript runs. Versions of NumPy, torchvision, timm, LPIPS, TensorBoard, tqdm, einops, and other transitive packages still need verification. `requirements.txt` therefore lists needed packages and only pins versions supported by the available records; it must not be presented as a fully reproducible lock file yet.

For publication, recover the original environment export if available, or recreate the environment and run the release smoke tests before freezing exact package/build versions.

