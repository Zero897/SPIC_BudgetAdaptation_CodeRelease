# Checkpoint provenance

Checkpoint binaries are intentionally excluded. The hashes below are transcribed from the project's archived protocols and run records; they let a user verify a separately obtained artifact.

| Role | Checkpoint identity | Provenance supported by the archived record |
|---|---|---|
| Main DehazeFormer shared start (B0/P0/FT) | `weights/dehazeformer/latest.pth`; SHA-256 `e5af35e6254916c3bfbcdcd25bdbfd856e65942e0a682a12d986ca951183ef88` | Author-trained from scratch, not an official DehazeFormer pretrained checkpoint. The archived audit identifies it as epoch 127. |
| Restormer Protocol-V2 baseline | `best_psnr.pth`; SHA-256 `0b0657dd65a1476da9b791f0530f0bad7aedea02378adf690f7e7990076bebd5` | The baseline protocol says random initialization and training on the project's 12,000 training pairs; validation-selected at epoch 94. |
| FFA-Net OTS baseline initializer | `weights/ffa-net/ots_train_ffa_3_19.pk`; SHA-256 `1c80e4ad4f4e8ae82ca773b66cbe47fd85ea78375e0a7f503b2585ed54da7723` | Identified in the protocol as the official outdoor OTS initialization. Its original download URL and upstream checksum were not retained in the release evidence. |
| FFA-Net OTS trained baseline | `best_psnr.pth`; SHA-256 `5dcb1f843bfd467445a4ace1673fdb6a7e2d00c1de15b4c11e131f86d08f7f7e` | Author-trained from the OTS initializer; validation-selected at step 44,000. |
| FFA-Net post-FT extension base | `its_train_ffa_3_19_epoch_50.pk`; SHA-256 `113587bb1c656fd77b1c0999f4f0d8e60e55b5f351fa2bd53f39444d8bd1d206` | The final extension protocol names this as its base, but the archived evidence does not establish its original source or how it relates to the later OTS-trained baseline. Treat its provenance as unresolved. |
| Restormer post-FT extension base | SHA-256 `a272ba1c49254e9b59ad7eb3299a16e8c169ab3f939e31171716064ddd1a5e37` | Described as a corrected full-FT checkpoint. The available records do not establish its initialization checkpoint or training lineage. It is distinct from the Protocol-V2 scratch-trained checkpoint above. |

Adapter checkpoints are derived from their corresponding frozen or fine-tuned base checkpoints and are not included. For a public reproduction package, attach signed or archived run metadata with the checkpoint file name, byte count, SHA-256, initialization source, training-data identity, and selection rule. Do not describe an upstream checkpoint as “official” unless its source URL and hash have been verified.
