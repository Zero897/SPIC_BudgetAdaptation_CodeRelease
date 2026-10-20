# Third-party software references

This repository does not redistribute DehazeFormer, Restormer, FFA-Net, LPIPS, or Ultralytics source code or model weights. The wrappers require separately obtained compatible sources and checkpoints. See [UPSTREAM_VERSIONS.md](UPSTREAM_VERSIONS.md) for source identities and license findings, and [CHECKPOINT_PROVENANCE.md](CHECKPOINT_PROVENANCE.md) for checkpoint hashes and known origins.

- DehazeFormer: https://github.com/IDKiro/DehazeFormer — MIT at the pinned reference commit.
- Restormer: https://github.com/swz30/Restormer — the experiment-matched architecture commit carries the Academic Public License (non-commercial); the current upstream main has a later MIT license change.
- FFA-Net: https://github.com/zhilin007/FFA-Net — no explicit license was present at the experiment-matched architecture commit. No FFA source or weight is included here.
- LPIPS metric implementation: https://github.com/richzhang/PerceptualSimilarity
- Ultralytics detector: https://github.com/ultralytics/ultralytics

The project `LICENSE` covers only original project-authored files. It grants no rights to third-party code, checkpoints, datasets, or package dependencies.

