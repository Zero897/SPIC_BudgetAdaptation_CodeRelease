# Upstream implementation identities

This file records upstream code required by the release without vendoring it. The pinned commits below are reference checkouts for reproducing the model implementations. The original training logs did not record full upstream Git commits, so these file-level matches do not prove that every experiment used the same full checkout.

| Model | Reference commit | File-level evidence from the local source used by the project | License at reference commit |
|---|---|---|---|
| DehazeFormer | [`5af7c34d6f8e0784a88a2132a7add296bf99ca9c`](https://github.com/IDKiro/DehazeFormer/tree/5af7c34d6f8e0784a88a2132a7add296bf99ca9c) | `models/dehazeformer.py` SHA-256 `1363d18fb2beb9ecc0649c41da4526323c8d1823b9bc8195586582b3bdc0b149`; byte-identical at this reference commit. | MIT; see upstream [`LICENSE`](https://github.com/IDKiro/DehazeFormer/blob/5af7c34d6f8e0784a88a2132a7add296bf99ca9c/LICENSE). |
| Restormer | [`8b1d498826ea8eeafcef794a6c75f34e3a9de7df`](https://github.com/swz30/Restormer/tree/8b1d498826ea8eeafcef794a6c75f34e3a9de7df) | `basicsr/models/archs/restormer_arch.py` SHA-256 `3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a`; byte-identical at this commit and matches the experiment protocol hash. | Academic Public License at this commit; non-commercial research use only. See the pinned [`LICENSE.md`](https://github.com/swz30/Restormer/blob/8b1d498826ea8eeafcef794a6c75f34e3a9de7df/LICENSE.md). |
| FFA-Net | [`d5e0cdbc82c92bea2ba1931a7d9a80684977bc0b`](https://github.com/zhilin007/FFA-Net/tree/d5e0cdbc82c92bea2ba1931a7d9a80684977bc0b) | `net/models/FFA.py` SHA-256 `8e2e7122cb510f157ccf7e1ae09b8885d63634f5e572d12c43ef06e95e3bc199`; byte-identical at this commit and matches the experiment protocol hash. | No license file or explicit license grant was present at the pinned commit. Do not redistribute its source or weights; obtain permission or an upstream license before public reuse. |

The current Restormer `main` has since moved to commit `68dc6ac472db26f16361150cb7a96a1bc87da93f` and its current license file is MIT. The experiment's architecture hash instead identifies the older commit above, whose license is non-commercial. This release therefore records the terms of the version used, not the later license state.

The DehazeFormer working directory had no `.git` metadata. Its model file matches the reference commit above, but the source directory as a whole is not a versioned snapshot. The Restormer and FFA-Net identities above are likewise architecture-file matches; the original full checkouts were not preserved with Git metadata.

The project `LICENSE` applies only to original project-authored files in this repository. It does not relicense upstream implementations, pretrained weights, datasets, or third-party packages.
