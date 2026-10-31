"""
FlowHead module for v4 geometric generator.

A lightweight convolutional head that takes G's intermediate features
(256x256 feature map after the resnet blocks / upsampling) and predicts
a 2-channel flow field (delta_u, delta_v) in normalized grid coordinates.

Design notes:
  - Last layer is zero-initialized so initial flow is identically zero,
    making v4's output at epoch 0 mathematically equivalent to v3's output.
  - Uses 3 conv layers + tanh to bound |flow| <= 1 in normalized coords.
  - Predicts a flow field whose unit is one pixel (see warp_with_flow).
  - The axial monotonicity of the vertical component is encouraged by a
    downstream loss, not by construction.
"""
import torch
import torch.nn as nn


class FlowHead(nn.Module):
    """Predicts a 2-channel flow field from G's bottleneck features.

    Input:  (B, C, H, W)  feature map (typically C = ngf = 64)
    Output: (B, 2, H, W)  flow field, bounded in [-1, 1] via tanh.
    """

    def __init__(self, in_channels=64, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden, 2, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        # zero-init last layer: flow starts at exactly zero, so v4 == v3 at epoch 0.
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        flow = torch.tanh(self.conv3(h))  # bounded in [-1, 1]
        return flow


def warp_with_flow(x, flow):
    """Warp tensor x with flow field via grid_sample.

    x:    (B, C, H, W) source image
    flow: (B, 2, H, W) displacement field in normalized grid coords.
          flow[..., 0, :] is dx (along width), flow[..., 1, :] is dy (along height).
    Returns warped image of same shape.

    Units: the flow is scaled by 2/W and 2/H respectively, so a flow component of
    value v displaces the image content by exactly v pixels. The tanh bound of
    [...][-1, 1] is therefore a one-pixel bound, not a whole-image shift.
    """
    B, C, H, W = x.shape
    # base identity grid in normalized coords [-1, 1]
    device = x.device
    dtype = x.dtype
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    # IMPORTANT: grid_sample expects (x, y) ordering, where y is vertical axis.
    # Scaling by 2/W and 2/H makes the flow unit pixels (see the docstring above).
    flow_x = flow[:, 0:1, :, :] * (2.0 / W)
    flow_y = flow[:, 1:2, :, :] * (2.0 / H)
    flow_grid = torch.stack([flow_x.squeeze(1), flow_y.squeeze(1)], dim=-1)
    sampling_grid = base_grid + flow_grid
    return torch.nn.functional.grid_sample(
        x, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True
    )