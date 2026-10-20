"""Native partial fine-tuning controls for Restormer and FFA-Net."""

import hashlib
import importlib.util
import random
from pathlib import Path

import torch
import torch.nn as nn


FFA_MEAN = (0.64, 0.60, 0.58)
FFA_STD = (0.14, 0.15, 0.152)


LATE_PREFIXES = {
    "restormer": ("refinement.", "output."),
    "ffa": ("ca.", "palayer.", "post."),
}

FORWARD_PREFIX_ORDER = {
    "restormer": (
        "patch_embed.",
        "encoder_level1.",
        "down1_2.",
        "encoder_level2.",
        "down2_3.",
        "encoder_level3.",
        "down3_4.",
        "latent.",
        "up4_3.",
        "reduce_chan_level3.",
        "decoder_level3.",
        "up3_2.",
        "reduce_chan_level2.",
        "decoder_level2.",
        "up2_1.",
        "decoder_level1.",
        "refinement.",
        "output.",
    ),
    "ffa": ("pre.", "g1.", "g2.", "g3.", "ca.", "palayer.", "post."),
}


def _load_module(module_name: str, source_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(source_path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load module from {}".format(source_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_restormer(source_path: Path) -> nn.Module:
    module = _load_module("native_partial_ft_restormer", source_path)
    return module.Restormer(
        inp_channels=3,
        out_channels=3,
        dim=40,
        num_blocks=[3, 4, 4, 6],
        num_refinement_blocks=3,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )


def build_ffanet(source_path: Path) -> nn.Module:
    module = _load_module("native_partial_ft_ffa", source_path)
    return module.FFA(gps=3, blocks=19)


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


def build_base(backbone: str, architecture_path: Path) -> nn.Module:
    if backbone == "restormer":
        return build_restormer(architecture_path)
    if backbone == "ffa":
        return build_ffanet(architecture_path)
    raise ValueError("Unsupported backbone: {}".format(backbone))


def load_base_weights_strict(model, checkpoint_path: Path):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = _clean_base_state(_unwrap_checkpoint(checkpoint))
    incompatible = model.base.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict base checkpoint load failed")
    model.freeze_base()


def _forward_ordered_parameters(base: nn.Module, backbone: str):
    named = dict(base.named_parameters())
    ordered = []
    seen = set()
    for prefix in FORWARD_PREFIX_ORDER[backbone]:
        for name, parameter in named.items():
            if name.startswith(prefix):
                ordered.append((name, parameter))
                seen.add(name)
    if seen != set(named):
        missing = sorted(set(named) - seen)
        raise RuntimeError("Unmapped parameters: {}".format(missing))
    return ordered


def _nearest_contiguous_prefix(items, target):
    selected = []
    count = 0
    for name, parameter in items:
        before = count
        count += parameter.numel()
        selected.append(name)
        if count >= target:
            if abs(before - target) < abs(count - target):
                return selected[:-1]
            return selected
    return selected


def _random_tensor_subset(items, target, mask_seed):
    shuffled = list(items)
    random.Random(mask_seed).shuffle(shuffled)
    selected = []
    count = 0
    for name, parameter in shuffled:
        if count + parameter.numel() <= target:
            selected.append(name)
            count += parameter.numel()
    return selected


def configure_native_partial(model, policy: str, mask_seed: int = 1701):
    if policy not in ("late", "early", "random"):
        raise ValueError("Unsupported policy: {}".format(policy))
    model.freeze_base()
    ordered = _forward_ordered_parameters(model.base, model.backbone)
    late_prefixes = LATE_PREFIXES[model.backbone]
    late_names = [name for name, _ in ordered if name.startswith(late_prefixes)]
    target = sum(dict(ordered)[name].numel() for name in late_names)
    eligible = [(name, parameter) for name, parameter in ordered if name not in late_names]

    if policy == "late":
        selected = late_names
    elif policy == "early":
        selected = _nearest_contiguous_prefix(eligible, target)
    else:
        selected = _random_tensor_subset(eligible, target, mask_seed)

    selected_set = set(selected)
    for name, parameter in model.base.named_parameters():
        parameter.requires_grad = name in selected_set
    trainable = sum(parameter.numel() for parameter in model.base.parameters() if parameter.requires_grad)
    if not selected or trainable <= 0:
        raise RuntimeError("No parameters selected")
    if abs(trainable - target) / target > 0.10:
        raise RuntimeError(
            "Budget mismatch exceeds 10%: selected={}, target={}".format(trainable, target)
        )
    selection_hash = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
    return {
        "policy": policy,
        "mask_seed": mask_seed if policy == "random" else None,
        "target_late_parameters": target,
        "trainable_parameters": trainable,
        "base_parameters": sum(parameter.numel() for parameter in model.base.parameters()),
        "trainable_fraction": trainable / sum(parameter.numel() for parameter in model.base.parameters()),
        "tensor_count": len(selected),
        "parameter_names": selected,
        "parameter_names_sha256": selection_hash,
    }


class NativePartialFT(nn.Module):
    def __init__(self, base: nn.Module, backbone: str):
        super().__init__()
        self.base = base
        self.backbone = backbone
        self.freeze_base()

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, lq, ir=None, detach_base=False):
        del ir
        lq_01 = ((lq + 1.0) / 2.0).clamp(0, 1)
        if self.backbone == "ffa":
            mean = lq_01.new_tensor(FFA_MEAN).view(1, 3, 1, 1)
            std = lq_01.new_tensor(FFA_STD).view(1, 3, 1, 1)
            base_input = (lq_01 - mean) / std
        else:
            base_input = lq_01
        if detach_base:
            with torch.no_grad():
                prediction = self.base(base_input)
        else:
            prediction = self.base(base_input)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        return prediction.float().clamp(0, 1)


def build_model(backbone: str, architecture_path: Path) -> NativePartialFT:
    return NativePartialFT(build_base(backbone, architecture_path), backbone)

