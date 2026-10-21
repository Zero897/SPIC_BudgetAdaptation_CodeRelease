import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.dehazeformer import dehazeformer_t, dehazeformer_s, dehazeformer_b
except Exception:
    from dehazeformer import dehazeformer_t, dehazeformer_s, dehazeformer_b


def to_01(x):
    return ((x + 1.0) / 2.0).clamp(0, 1)


def to_pm1(x):
    return (x * 2.0 - 1.0).clamp(-1, 1)


class RGBSmokeIRAssistedCoarseDehazer(nn.Module):
    """
    RGB-smoke-driven physics-inspired coarse prior with IR structural guidance.

    Compared with the previous IR-predicted T/A design, this version estimates
    the visible-spectrum auxiliary variables T and A from the smoke-degraded RGB
    image, because the atmospheric scattering model describes RGB degradation.
    The infrared image is used as complementary structural guidance to refine
    the coarse RGB prior.
    """
    def __init__(self):
        super().__init__()

        # Predict a transmission-like visibility map from smoke-degraded RGB.
        self.t_predictor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )

        # Predict an atmospheric-light-like auxiliary component from smoke RGB.
        self.A_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(3, 16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1), nn.Sigmoid()
        )

        # Use IR only as structural guidance for coarse-prior refinement.
        # Input: coarse RGB prior (3 ch) + IR (1 ch) = 4 ch.
        self.refine = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1)
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, lq_rgb_01, ir, use_ir_refinement=True):
        """
        Args:
            lq_rgb_01: smoke-degraded RGB image in [0, 1], shape [B, 3, H, W].
            ir: infrared image in [-1, 1] or [0, 1], shape [B, 1, H, W] or [B, C, H, W].

        Returns:
            coarse: RGB-smoke-driven coarse prior refined by IR guidance, [B, 3, H, W].
            t_map: transmission-like visibility map predicted from smoke RGB, [B, 1, H, W].
            A: atmospheric-light-like auxiliary component predicted from smoke RGB, [B, 3, 1, 1].
        """
        # Prepare IR guidance.
        if ir.shape[1] > 1:
            ir = ir.mean(dim=1, keepdim=True)
        ir_01 = to_01(ir) if ir.min() < 0 else ir.clamp(0, 1)

        if ir_01.shape[-2:] != lq_rgb_01.shape[-2:]:
            ir_01 = F.interpolate(
                ir_01,
                size=lq_rgb_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # ------------------------------------------------------------
        # 1) Estimate visible-spectrum auxiliary variables from smoke RGB.
        # ------------------------------------------------------------
        t_map = self.t_predictor(lq_rgb_01)
        t_map = F.interpolate(
            t_map,
            size=lq_rgb_01.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        lq_intensity = lq_rgb_01.mean(dim=1, keepdim=True)
        dark_mask_t = torch.clamp((0.2 - lq_intensity) / 0.1, 0.0, 1.0)
        t_min_dynamic = 0.1 + 0.2 * dark_mask_t
        t_map = torch.clamp(t_map, min=t_min_dynamic, max=torch.ones_like(t_map))

        A = self.A_predictor(lq_rgb_01).mean(dim=[-2, -1], keepdim=True)
        A_mean = A.mean(dim=1, keepdim=True)
        A = 0.8 * A + 0.2 * A_mean
        lq_max = lq_rgb_01.amax(dim=[2, 3], keepdim=True)
        A = torch.clamp(A, min=torch.full_like(A, 0.05), max=lq_max + 0.1)

        # ------------------------------------------------------------
        # 2) Scattering-inspired coarse inversion using smoke RGB.
        # ------------------------------------------------------------
        noise_mask = torch.clamp((0.15 - lq_intensity) / 0.1, 0.0, 1.0)
        lq_gray = lq_intensity.repeat(1, 3, 1, 1)
        lq_smooth = F.avg_pool2d(lq_gray, kernel_size=3, stride=1, padding=1)
        lq_clean = lq_rgb_01 * (1.0 - noise_mask) + lq_smooth * noise_mask

        coarse = (lq_clean - A * (1.0 - t_map)) / t_map
        coarse = coarse.clamp(0.0, 1.0)

        # ------------------------------------------------------------
        # 3) IR-guided structural refinement of the coarse prior.
        # ------------------------------------------------------------
        if use_ir_refinement:
            residual = self.refine(torch.cat([coarse, ir_01], dim=1))
            coarse = (coarse + residual).clamp(0.0, 1.0)
        return coarse, t_map, A


# Backward-compatible alias. New experiments should use the RGBSmokeIRAssistedCoarseDehazer name.
IRAssistedCoarseDehazer = RGBSmokeIRAssistedCoarseDehazer


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x) * 0.1

class BlockAttnResAggregator(nn.Module):
    """
    AttnRes-style block-level depth-wise residual aggregation.

    Given features sampled at the head and after every second residual block,
    it predicts
    input-adaptive softmax weights and aggregates them as:

        F_agg = sum_i alpha_i * F_i

    This is inspired by Attention Residuals, but adapted to
    2D image restoration residual features.
    """
    def __init__(self, channels, num_states=4, hidden=64):
        super().__init__()
        self.channels = int(channels)
        self.num_states = int(num_states)

        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )

    def forward(self, states):
        """
        states: list of feature tensors, each [B, C, H, W]
        """
        if len(states) != self.num_states:
            raise ValueError(
                f"Expected {self.num_states} depth states, got {len(states)}"
            )
        if len(states) == 1:
            return states[0]

        # [B, N, C]
        pooled = torch.stack(
            [F.adaptive_avg_pool2d(s, 1).flatten(1) for s in states],
            dim=1,
        )

        b, n, c = pooled.shape

        # [B, N, 1]
        scores = self.score(pooled.reshape(b * n, c)).reshape(b, n, 1)
        weights = torch.softmax(scores, dim=1)

        # [B, N, C, H, W]
        feat_stack = torch.stack(states, dim=1)

        # [B, N, 1, 1, 1]
        weights = weights.view(b, n, 1, 1, 1)

        out = (feat_stack * weights).sum(dim=1)
        out = self.out_proj(out)

        return out

class IRPhysicsResidualRefiner(nn.Module):
    def __init__(
        self,
        in_ch=11,
        width=48,
        num_blocks=6,
        min_delta_scale=0.05,
        max_delta_scale=0.30,
        init_delta_scale=0.10,
        use_attnres=False,
        enable_latent_pred=False,
        latent_ch=6,
        latent_size=16,
    ):
        super().__init__()

        self.use_attnres = bool(use_attnres)
        self.enable_latent_pred = bool(enable_latent_pred)
        self.latent_ch = int(latent_ch)
        self.latent_size = int(latent_size)

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GELU(),
        )

        # 改成 ModuleList，方便取中间层特征
        self.body = nn.ModuleList(
            [ResBlock(width) for _ in range(num_blocks)]
        )

        if self.use_attnres:
            # 默认聚合 x0, x2, x4, x6，共 4 个状态
            self.attnres = BlockAttnResAggregator(
                channels=width,
                num_states=1 + num_blocks // 2,
                hidden=max(16, width),
            )
        else:
            self.attnres = None

        self.gate = nn.Sequential(
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, 1),
            nn.Sigmoid(),
        )

        self.tail = nn.Conv2d(width, 3, 3, padding=1)

        # 初始化为 0，保证初始不会破坏 DehazeFormer base
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

        self.min_delta_scale = float(min_delta_scale)
        self.max_delta_scale = float(max_delta_scale)

        init_ratio = (init_delta_scale - self.min_delta_scale) / (
            self.max_delta_scale - self.min_delta_scale
        )
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)

        import math
        init_logit = math.log(init_ratio / (1.0 - init_ratio))
        self.delta_scale_logit = nn.Parameter(
            torch.tensor(float(init_logit))
        )

        if self.enable_latent_pred:
            self.latent_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((self.latent_size, self.latent_size)),
                nn.Conv2d(width, width // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width // 2, self.latent_ch, 1),
            )
            nn.init.zeros_(self.latent_head[-1].weight)
            nn.init.zeros_(self.latent_head[-1].bias)
        else:
            self.latent_head = None

    def get_delta_scale(self):
        scale = self.min_delta_scale + (
            self.max_delta_scale - self.min_delta_scale
        ) * torch.sigmoid(self.delta_scale_logit)
        return scale

    def forward(self, x, compute_gate=True):
        feat = self.head(x)

        if self.use_attnres:
            # x0
            states = [feat]

            for idx, block in enumerate(self.body):
                feat = block(feat)

                # 收集 x2, x4, x6
                if (idx + 1) % 2 == 0:
                    states.append(feat)

            feat = self.attnres(states)
        else:
            for block in self.body:
                feat = block(feat)

        gate = self.gate(feat) if compute_gate else None

        delta_scale = self.get_delta_scale()
        delta = torch.tanh(self.tail(feat)) * delta_scale

        if self.enable_latent_pred:
            latent_delta = self.latent_head(feat)
            return delta, gate, latent_delta

        return delta, gate


class CrossModalDehazeFormerResidualV2(nn.Module):
    def __init__(
        self,
        backbone="b",
        init_gamma=0.01,
        ablation="full",
        wo_gate=False,
        use_attnres=False,
        enable_latent_pred=False,
        latent_size=16,
        refiner_width=48,
        refiner_blocks=6,
        adapter_enabled=True,
        adapter_variant="fscatterres",
    ):
        super().__init__()

        # Backward compatibility:
        # Old command used "--ablation wo_gate".
        # New design separates component ablation from output gate.
        if ablation == "wo_gate":
            ablation = "full"
            wo_gate = True
        
        assert ablation in ["full", "wo_ir", "wo_physics", "wo_ir_physics"], \
            f"Unsupported ablation mode: {ablation}"
        
        self.ablation = ablation
        self.wo_gate = bool(wo_gate)
        self.adapter_enabled = bool(adapter_enabled)
        if adapter_variant not in ["fscatterres", "plain"]:
            raise ValueError(f"Unsupported adapter variant: {adapter_variant}")
        self.adapter_variant = adapter_variant

        if backbone == "t":
            self.base = dehazeformer_t()
        elif backbone == "s":
            self.base = dehazeformer_s()
        elif backbone == "b":
            self.base = dehazeformer_b()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.coarse_dehazer = RGBSmokeIRAssistedCoarseDehazer()
        
        self.refiner = IRPhysicsResidualRefiner(
            in_ch=6 if self.adapter_variant == "plain" else 11,
            width=refiner_width,
            num_blocks=refiner_blocks,
            use_attnres=use_attnres,
            enable_latent_pred=enable_latent_pred,
            latent_ch=6,
            latent_size=latent_size,
        )
        
        self.use_attnres = bool(use_attnres)
        self.enable_latent_pred = bool(enable_latent_pred)
        self.latent_size = int(latent_size)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

        self._configure_adapter_parameters()

    def _configure_adapter_parameters(self):
        """Freeze branches that are inactive for the selected protocol."""
        for name, parameter in self.named_parameters():
            if not name.startswith("base."):
                parameter.requires_grad = self.adapter_enabled

        if not self.adapter_enabled:
            return

        if self.adapter_variant == "plain" or self.ablation in ["wo_physics", "wo_ir_physics"]:
            for parameter in self.coarse_dehazer.parameters():
                parameter.requires_grad = False
        elif self.ablation == "wo_ir":
            for parameter in self.coarse_dehazer.refine.parameters():
                parameter.requires_grad = False

        if self.wo_gate:
            for parameter in self.refiner.gate.parameters():
                parameter.requires_grad = False

    def set_adapter_enabled(self, enabled):
        self.adapter_enabled = bool(enabled)
        self._configure_adapter_parameters()

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False
        print("[Freeze] DehazeFormer base frozen.")

    def unfreeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = True
        print("[Unfreeze] DehazeFormer base unfrozen.")

    def forward(self, lq, ir, return_aux=False, detach_base=False):
        """
        ablation:
            full          : pred_base + lq + ir + coarse + t
            wo_ir         : remove IR from refiner and physics branch
            wo_physics    : keep IR, remove coarse/t physics prior
            wo_ir_physics : remove both IR and physics prior
        
        wo_gate:
            remove final spatial output gate, i.e., use gate = 1 everywhere.
        """
        lq_01 = lq.clamp(0, 1)

        if detach_base:
            with torch.no_grad():
                pred_base = self.base(lq_01).clamp(0, 1)
        else:
            pred_base = self.base(lq_01).clamp(0, 1)

        if not self.adapter_enabled:
            if return_aux:
                return pred_base, {
                    "pred_base": pred_base,
                    "adapter_enabled": False,
                    "adapter_variant": self.adapter_variant,
                    "ablation": self.ablation,
                    "wo_gate": self.wo_gate,
                    "use_attnres": self.use_attnres,
                }
            return pred_base

        # ------------------------------
        # Prepare IR
        # ------------------------------
        if ir.shape[1] > 1:
            ir1 = ir.mean(dim=1, keepdim=True)
        else:
            ir1 = ir

        ir01 = (ir1 + 1.0) / 2.0 if ir1.min() < 0 else ir1.clamp(0, 1)

        if ir01.shape[-2:] != lq_01.shape[-2:]:
            ir01 = F.interpolate(
                ir01,
                size=lq_01.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # ------------------------------
        # Ablation 1: w/o IR
        # Remove IR from both explicit IR input and physics branch.
        # ------------------------------
        if self.ablation in ["wo_ir", "wo_ir_physics"]:
            ir_for_refiner = torch.zeros_like(ir01)
            ir_for_physics = torch.zeros_like(ir1)
        else:
            ir_for_refiner = ir01
            ir_for_physics = ir

        # ------------------------------
        # Physics branch
        # ------------------------------
        if self.ablation in ["wo_physics", "wo_ir_physics"]:
            coarse = torch.zeros_like(pred_base)
            t_map = torch.zeros_like(ir01)
            A = torch.zeros(
                (lq_01.size(0), 3, 1, 1),
                dtype=lq_01.dtype,
                device=lq_01.device,
            )
        else:
            coarse, t_map, A = self.coarse_dehazer(
                lq_01,
                ir_for_physics,
                use_ir_refinement=self.ablation != "wo_ir",
            )

            if coarse.shape[-2:] != lq_01.shape[-2:]:
                coarse = F.interpolate(
                    coarse,
                    size=lq_01.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if t_map.shape[-2:] != lq_01.shape[-2:]:
                t_map = F.interpolate(
                    t_map,
                    size=lq_01.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        # ------------------------------
        # Refiner input:
        # pred_base(3) + lq(3) + ir(1) + coarse(3) + t(1) = 11 channels
        # ------------------------------
        if self.adapter_variant == "plain":
            ref_in = torch.cat([pred_base, lq_01], dim=1)
        else:
            ref_in = torch.cat(
                [pred_base, lq_01, ir_for_refiner, coarse, t_map],
                dim=1,
            )

        refiner_out = self.refiner(ref_in, compute_gate=not self.wo_gate)

        if isinstance(refiner_out, tuple) and len(refiner_out) == 3:
            delta, gate, latent_delta = refiner_out
        else:
            delta, gate = refiner_out
            latent_delta = None

        # Output setting: remove spatial gate.
        # This is independent from component ablations such as wo_ir/wo_physics.
        if self.wo_gate:
            gate = torch.ones_like(delta[:, :1])
        
        # Optional real-domain inference trick:
        # force the residual gate to keep a small activation on real smoke images.
        # This is only enabled when self.real_gate_floor is set externally.
        elif hasattr(self, "real_gate_floor") and self.real_gate_floor is not None:
            gate = torch.clamp(gate, min=float(self.real_gate_floor), max=1.0)

        pred = (pred_base + self.gamma * gate * delta).clamp(0, 1)

        if return_aux:
            aux = {
                "pred_base": pred_base,
                "coarse": coarse,
                "t": t_map,
                "A": A,
                "delta": delta,
                "gate": gate,
                "gamma": self.gamma.detach(),
                "ablation": self.ablation,
                "wo_gate": self.wo_gate,
                "latent_delta": latent_delta,
                "use_attnres": self.use_attnres,
                "adapter_enabled": self.adapter_enabled,
                "adapter_variant": self.adapter_variant,
            }

            if hasattr(self.refiner, "get_delta_scale"):
                aux["delta_scale"] = self.refiner.get_delta_scale().detach()

            return pred, aux

        return pred


def load_dehazeformer_base_weights(model, ckpt_path, strict=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {}
    for k, v in raw.items():
        k = k.replace("module.", "")
        if k.startswith("base."):
            k = k[5:]
        if k.startswith("net."):
            k = k[4:]
        state[k] = v
    missing, unexpected = model.base.load_state_dict(state, strict=strict)
    print(f"[Load DehazeFormer base] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing examples:", missing[:10])
    if unexpected:
        print("  unexpected examples:", unexpected[:10])
    return missing, unexpected


def load_v2_checkpoint(model, ckpt_path, strict=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in raw.items()}

    # strict=False in PyTorch still raises errors for shape-mismatched tensors.
    # This filter allows loading older checkpoints after changing the coarse-prior
    # predictors from IR-input to RGB-input. Shape-mismatched layers are skipped.
    skipped = []
    if not strict:
        current = model.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in current and tuple(current[k].shape) != tuple(v.shape):
                skipped.append((k, tuple(v.shape), tuple(current[k].shape)))
                continue
            filtered[k] = v
        state = filtered

    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[Load V2] missing={len(missing)}, unexpected={len(unexpected)}, skipped_shape={len(skipped)}")
    if skipped:
        print("  skipped shape-mismatch examples:", skipped[:10])
    if missing:
        print("  missing examples:", missing[:10])
    if unexpected:
        print("  unexpected examples:", unexpected[:10])
    return missing, unexpected
