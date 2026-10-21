
# cm_dehazeformer_v2_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def simple_ssim_loss(x, y, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, 1, window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, window_size // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, 1, window_size // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, window_size // 2) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + 1e-8
    )
    return 1.0 - ssim.mean()


def sobel_mag(x):
    if x.shape[1] == 3:
        x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def gradient_magnitude(x):
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    return torch.sqrt(dx.square() + dy.square() + 1e-6)


def box_blur(x, kernel_size=5):
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1)


@torch.no_grad()
def artifact_metrics(pred, gt):
    false_edge = F.relu(
        gradient_magnitude(pred) - gradient_magnitude(gt) - 0.02
    ).mean(dim=(1, 2, 3))
    high_error = (
        (pred - box_blur(pred, 5)) - (gt - box_blur(gt, 5))
    ).abs().mean(dim=(1, 2, 3))
    return false_edge, high_error


class CrossModalResidualLoss(nn.Module):
    def __init__(
        self,
        w_char=1.0,
        w_ssim=0.12,
        w_edge=0.03,
        w_color=0.01,
        w_delta=0.001,
        w_gate=0.0005,
        w_not_worse=0.2,
    ):
        super().__init__()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.w_color = w_color
        self.w_delta = w_delta
        self.w_gate = w_gate
        self.w_not_worse = w_not_worse

    def forward(self, pred, gt, aux=None):
        losses = {}
        losses["char"] = charbonnier_loss(pred, gt)
        losses["ssim"] = simple_ssim_loss(pred, gt)
        losses["edge"] = F.l1_loss(sobel_mag(pred), sobel_mag(gt))
        losses["color"] = F.l1_loss(pred.mean(dim=[2, 3]), gt.mean(dim=[2, 3]))

        losses["delta"] = aux["delta"].abs().mean() if aux is not None and "delta" in aux else pred.new_tensor(0.0)
        if aux is not None and "gate" in aux and not aux.get("wo_gate", False):
            losses["gate"] = aux["gate"].mean()
        else:
            losses["gate"] = pred.new_tensor(0.0)

        if aux is not None and "pred_base" in aux:
            final_l1 = (pred - gt).abs().mean(dim=[1,2,3])
            base_l1 = (aux["pred_base"].detach() - gt).abs().mean(dim=[1,2,3])
            losses["not_worse"] = F.relu(final_l1 - base_l1).mean()
        else:
            losses["not_worse"] = pred.new_tensor(0.0)

        total = (
            self.w_char * losses["char"]
            + self.w_ssim * losses["ssim"]
            + self.w_edge * losses["edge"]
            + self.w_color * losses["color"]
            + self.w_delta * losses["delta"]
            + self.w_gate * losses["gate"]
            + self.w_not_worse * losses["not_worse"]
        )
        losses["total"] = total
        return total, losses
