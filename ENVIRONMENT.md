# Environment and dependency audit

## Runtime observed on AutoDL

The saved runtime probe from the AutoDL `flowie` environment reports Python 3.9.25, PyTorch 2.1.2+cu121, CUDA 12.1, torchvision 0.16.2+cu121, OpenCV 4.12.0, LPIPS 0.1.4, and NumPy 1.26.4. The broader package inventory also reports timm 1.0.22, TensorBoard 2.20.0, tqdm 4.67.1, einops 0.8.1, scikit-image 0.24.0, SciPy 1.13.1, Pillow 10.4.0, PyYAML 6.0.3, imageio 2.37.2, matplotlib 3.9.4, BasicSR 1.4.2, and yacs 0.1.8.

The captured files are retained under [`environment/`](environment/):

- `runtime-autodl-flowie-20261020.txt`: imported runtime versions.
- `conda-export-autodl-flowie-20261020.txt`: `conda list --export` snapshot for Linux-64.
- `pip-freeze-autodl-flowie-20261020.txt`: pip package inventory.

These are evidence snapshots, not a clean install lock. The pip inventory contains both `torch==1.13.1+cu116` and `torch==2.1.2`, plus three OpenCV distributions (`opencv-python`, `opencv-python-headless`, and `opencv-contrib-python`). The Conda export lists Torch 2.1.2, and the runtime probe imports 2.1.2+cu121. The original AutoDL environment is therefore not a reliable source for a byte-for-byte lock.

## Clean reproduction recipe

`environment.yml` and `requirements.txt` pin the Python, PyTorch/CUDA, and direct Python dependency versions for the clean Linux target. OpenCV is pinned to 4.11.0.86 because 4.12.0.88 requires NumPy 2.x on Python 3.9, while the verified training runtime uses NumPy 1.26.4 and PyTorch 2.1.2. Use one OpenCV distribution only. These pins are not a full transitive lock, and the clean target has not yet been installed and smoke-tested as a fresh environment.

The archived `conda list --export` is a useful Linux package/build inventory, but includes pip-origin entries marked `pypi_0`; it should not be treated as a portable standalone lock. A resolver dry-run and a fresh Linux install are still required before reporting the target as validated.

## Local Windows environment

The Windows `FlowIE` environment is separate from the AutoDL Linux run environment: Python 3.9.25, PyTorch 2.1.2+cu118, torchvision 0.16.2+cu118, timm 0.6.12, LPIPS 0.1.4, TensorBoard 2.11.2, tqdm 4.67.1, einops 0.8.1, NumPy 1.24.4, and scikit-image 0.24.0. It is not the environment used for the reported CUDA 12.1 runs.
