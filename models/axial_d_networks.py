"""
Axial-Directional Multi-scale Discriminator.

A two-branch discriminator designed for CycleGAN axial-stretch tasks:

  Branch 1 (spatial):  standard 70x70 PatchGAN that captures local 2D texture.
  Branch 2 (axial):    long-strip 1xK conv stack that captures morphology
                       along the axial (vertical) direction specifically.

Outputs of the two branches are concatenated along the channel axis and fused
by a 1x1 conv into a single 1-channel prediction map with the same spatial
size as the original PatchGAN output.

All conv layers are wrapped with Spectral Normalization to stabilize training
and prevent D_B collapse (Innovation 3's primary KPI improvement is D stability).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import functools


def _get_norm_layer(norm_type="instance"):
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "none":
        return lambda x: nn.Identity()
    else:
        raise ValueError(f"norm {norm_type} not supported")


class _SpatialPatchBranch(nn.Module):
    """Standard 3-layer 70x70 PatchGAN branch (with spectral norm)."""

    def __init__(self, input_nc, ndf=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw, padw = 4, 1
        seq = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seq += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** 3, 8)
        seq += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        self.body = nn.Sequential(*seq)

    def forward(self, x):
        return self.body(x)


class _AxialStripBranch(nn.Module):
    """1xK long-strip conv stack capturing axial-direction morphology.

    With kernel=1x8 and stride=1x4, padding=0, after 3 layers:
    input 256 -> 249 -> 62 -> 15 along width.  Off-the-grid for 256x256.
    We use kw=1, kw_h=8 with stride (1,4) and SAME padding via F.pad.
    """

    def __init__(self, input_nc, ndf=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 1xK kernel, stride 1x2 each layer.
        # Heights preserved with kernel_size=1; along width we downsample.
        # But the strip still captures axial (vertical) co-occurrence columns.
        kw_w = 8
        seq = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=(3, kw_w), stride=(1, 2), padding=(1, 3))),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seq += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(3, kw_w), stride=(1, 2), padding=(1, 3), bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** 3, 8)
        seq += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(3, kw_w), stride=(1, 1), padding=(1, 3), bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        self.body = nn.Sequential(*seq)

    def forward(self, x):
        return self.body(x)


class AxialMultiscaleDiscriminator(nn.Module):
    """Two-branch discriminator: spatial PatchGAN + axial-strip conv stack.

    Both branches are wrapped with spectral norm.

    Outputs are concatenated along the channel axis and fused by a 1x1 conv
    + final 1x1 conv -> 1 channel prediction map.
    """

    def __init__(self, input_nc, ndf=64, norm="instance"):
        super().__init__()
        norm_layer = _get_norm_layer(norm)
        self.spatial = _SpatialPatchBranch(input_nc, ndf, norm_layer)
        self.axial = _AxialStripBranch(input_nc, ndf, norm_layer)
        # fuse: spatial gives ndf*8 channels, axial gives ndf*8 channels
        fused_channels = ndf * 8 * 2
        self.fuse = nn.Sequential(
            spectral_norm(nn.Conv2d(fused_channels, ndf * 4, kernel_size=1, stride=1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * 4, 1, kernel_size=1, stride=1)),
        )

    def forward(self, x):
        s = self.spatial(x)
        a = self.axial(x)
        # Match spatial size (axial output width may differ slightly from spatial's).
        if s.shape[-1] != a.shape[-1] or s.shape[-2] != a.shape[-2]:
            a = F.interpolate(a, size=s.shape[-2:], mode="bilinear", align_corners=False)
        cat = torch.cat([s, a], dim=1)
        return self.fuse(cat)
