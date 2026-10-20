"""RGB-only residual adapters on frozen, already fine-tuned Restormer/FFA backbones."""

import math
from pathlib import Path

import torch
import torch.nn as nn

from native_partial_ft_v1 import FFA_MEAN, FFA_STD, build_base


BACKBONES = ("restormer", "ffa")


def _unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "state_dict", "params", "net", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _clean_base_state(state):
    cleaned = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[7:]
        if key.startswith("base."):
            key = key[5:]
        cleaned[key] = value
    return cleaned


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value):
        return value + 0.1 * self.body(value)


class RGBResidualRefiner(nn.Module):
    def __init__(self, use_gate, width=24, blocks=4):
        super().__init__()
        self.use_gate = bool(use_gate)
        self.head = nn.Sequential(nn.Conv2d(6, width, 3, padding=1), nn.GELU())
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, 3, 3, padding=1)
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(width, width // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width // 2, 1, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate = None
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
        ratio = (0.10 - 0.05) / (0.30 - 0.05)
        self.delta_scale_logit = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def get_delta_scale(self):
        return 0.05 + 0.25 * torch.sigmoid(self.delta_scale_logit)

    def forward(self, value):
        feature = self.body(self.head(value))
        delta = torch.tanh(self.tail(feature)) * self.get_delta_scale()
        gate = self.gate(feature) if self.gate is not None else torch.ones_like(delta[:, :1])
        return delta, gate


class PostFTRGBResidual(nn.Module):
    def __init__(self, backbone: str, architecture_path: Path, init_gamma: float):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError("Unsupported backbone: {}".format(backbone))
        self.backbone = backbone
        self.use_gate = backbone == "ffa"
        self.base = build_base(backbone, architecture_path)
        self.refiner = RGBResidualRefiner(use_gate=self.use_gate, width=24, blocks=4)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))
        self.freeze_base()

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def load_base_weights_strict(self, checkpoint_path: Path):
        state = _clean_base_state(
            _unwrap_checkpoint(torch.load(str(checkpoint_path), map_location="cpu"))
        )
        incompatible = self.base.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Strict base checkpoint load failed")
        self.freeze_base()
        return len(incompatible.missing_keys), len(incompatible.unexpected_keys)

    @staticmethod
    def _to_01(value):
        return ((value + 1.0) / 2.0).clamp(0, 1)

    def _base_input(self, lq_01):
        if self.backbone == "restormer":
            return lq_01
        mean = lq_01.new_tensor(FFA_MEAN).view(1, 3, 1, 1)
        std = lq_01.new_tensor(FFA_STD).view(1, 3, 1, 1)
        return (lq_01 - mean) / std

    def forward(self, lq, ir=None, return_aux=False, adapter_enabled=True):
        del ir
        lq_01 = self._to_01(lq)
        with torch.no_grad():
            pred_base = self.base(self._base_input(lq_01))
        if isinstance(pred_base, (tuple, list)):
            pred_base = pred_base[0]
        pred_base = pred_base.float().clamp(0, 1)

        if not adapter_enabled:
            if return_aux:
                return pred_base, {
                    "pred_base": pred_base,
                    "delta": None,
                    "gate": None,
                    "gamma": None,
                    "adapter_enabled": False,
                    "rgb_only": True,
                }
            return pred_base

        delta, gate = self.refiner(torch.cat([pred_base, lq_01], dim=1))
        gamma_eff = (
            torch.clamp(self.gamma, 0.0, 0.12)
            if self.backbone == "ffa"
            else self.gamma
        )
        prediction = (pred_base + gamma_eff * gate * delta).clamp(0, 1)
        if return_aux:
            return prediction, {
                "pred_base": pred_base,
                "delta": delta,
                "gate": gate,
                "gamma": gamma_eff,
                "gamma_raw": self.gamma,
                "delta_scale": self.refiner.get_delta_scale(),
                "adapter_enabled": True,
                "rgb_only": True,
                "wo_gate": not self.use_gate,
            }
        return prediction

    def parameter_counts(self):
        base_total = sum(parameter.numel() for parameter in self.base.parameters())
        adapter_total = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        )
        return {
            "total": base_total + adapter_total,
            "base_total": base_total,
            "base_trainable": sum(
                parameter.numel() for parameter in self.base.parameters() if parameter.requires_grad
            ),
            "adapter_total": adapter_total,
            "adapter_trainable": sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if not name.startswith("base.") and parameter.requires_grad
            ),
        }


def build_model(backbone: str, architecture_path: Path, init_gamma: float):
    return PostFTRGBResidual(backbone, architecture_path, init_gamma)
