# Matched Partial Fine-Tuning V1（冻结于 2026-10-10）

## 研究问题

P0 的收益是否来自外置残差适配方式，而不是简单地增加约 4.7 万个可训练参数？

对照方法 MFT 不增加网络模块，只更新 DehazeFormer-B 输出侧的一组原生参数。P0 与 MFT 使用相同训练数据、10 epoch、有效 batch 16、common loss、固定验证集和三个随机种子。MFT 的主干学习率固定为完整 FT 已使用的 `1e-6`，不进行新的学习率搜索。

## 固定参数预算

MFT 只解冻：

- `base.layer4.blocks.7.attn.*`
- `base.patch_split2.*`
- `base.fusion2.*`
- `base.layer5.*`
- `base.patch_unembed.*`

MFT 可训练参数为 45,412，P0 为 47,255；MFT/P0 比例为 96.10%，差异为 −3.90%，满足预设 ±10% 同预算范围。所选参数全部位于解码器末端和输出重建路径。

## 固定训练设置

| 项目 | MFT |
|---|---:|
| epochs | 10 |
| micro-batch | 16 |
| gradient accumulation | 1 |
| effective batch | 16 |
| optimizer | AdamW |
| base learning rate | `1e-6` |
| new-parameter learning rate | 0 |
| loss | Protocol V2 common |
| validation cadence | 每 2 epoch |
| checkpoint selection | 固定验证集 PSNR |
| seeds | 3407、2026、8888 |

## 分阶段执行

1. `preflight`：核对代码、数据清单、预训练权重和参数名单，不训练。
2. `smoke`：seed 3407，1 epoch、2 个训练 batch、2 个验证 batch，只验证前向、反向、优化器、保存和加载链路。
3. `pilot`：seed 3407，完整 10 epoch 和 1,469 张固定验证集。
4. 只有 Pilot 有效且通过候选门槛，才准备 seeds 2026/8888。

Pilot 候选门槛沿用 Protocol V2：相对 B0，PSNR 至少 `+0.05 dB`、SSIM 变化至少 `−0.002`、LPIPS 变化不超过 `+0.005`；AMP 跳过窗口比例不超过 0.1%。

## 最终解释门槛

三种子完成后，满足以下任一条件，才支持“P0 的残差适配位置优于同预算部分微调”：

1. P0−MFT 平均总体 PSNR 至少 `+0.05 dB`，平均 SSIM 变化至少 `−0.0005`，平均 LPIPS 变化不超过 `+0.001`；
2. P0−MFT 平均总体 PSNR 至少 `−0.02 dB`，平均 very-dense PSNR 至少 `+0.05 dB`，并满足相同 SSIM/LPIPS边界。

若两项均不满足，不声称残差适配位置优于同预算部分微调；论文只保留应用场景、外部泛化和参数更新效率结论。

## 评测边界

当前阶段只允许使用固定验证集。三种子验证结论冻结前，不读取 768-source 锁定测试或重新评测 RRSHID。不得根据测试结果更改参数层、学习率、训练轮数或门槛。

## 用户启动命令

```bash
cd $PROJECT_ROOT
bash run_matched_partial_ft_v1.sh preflight
bash run_matched_partial_ft_v1.sh smoke
```

Smoke 通过并完成审计后，再运行：

```bash
bash run_matched_partial_ft_v1.sh pilot
```

