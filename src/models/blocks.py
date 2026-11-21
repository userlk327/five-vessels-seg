"""Building blocks for the 2x2 ablation (models 2-4) and the U-Net they share.

``UNetVariant(attention, multiscale)`` has exactly the baseline U-Net backbone of
``src/models/unet.py`` (same depth, channels, pooling, ConvTranspose upsampling, 1x1 output,
same module names). With attention=False and multiscale=False its state dict is identical in
keys and shapes to the notebook ``UNET`` (tests/test_models.py). Only the added modules differ:

AttentionGate (Oktay et al., 2018; additive "grid" attention), one on each of the 4 skips
    x = skip features (C_x channels, H x W) from the encoder
    g = gating signal: the decoder features one level coarser (C_g channels, H/2 x W/2),
        i.e. the bottleneck output for the deepest skip, then d4, d3, d2
    q = psi( ReLU( theta_x(x) + phi_g(g) ) )      theta_x: 2x2 conv, stride 2, no bias (x -> g's grid)
                                                  phi_g:   1x1 conv with bias
                                                  psi:     1x1 conv with bias -> 1 channel
    alpha = sigmoid(q), upsampled bilinearly to H x W;  output = x * alpha
    The intermediate width is C_x / 2. The gated skip replaces the plain skip in the
    concatenation; the decoder blocks are unchanged.

MultiScaleConv: "parallel convolutions at different scales"
    three parallel 3x3 convolutions with dilation 1, 2 and 4 (padding = dilation), whose
    output channels are SPLIT across the branches (e.g. 64 -> 22 + 21 + 21, not 3 x 64),
    concatenated -> BN -> ReLU -> 1x1 conv -> BN -> ReLU (fusion).
    It replaces the SECOND 3x3 conv of a block (out -> out); the first conv (in -> out,
    BN, ReLU) is kept. Used in all 9 blocks: encoder 1-4, bottleneck, decoder 4-1.
    The split keeps the branch convolutions at the cost of one ordinary 3x3 conv; the only
    added parameters are the 1x1 fusion and one extra BN per block.

Attention gates are created after every backbone module, so with the same seed the
Attention U-Net starts from the same backbone weights as the U-Net (tests/test_models.py).

The x2 bilinear upsampling of alpha uses ``upsample2x_bilinear`` (slicing and
concatenation only) instead of F.interpolate, whose CUDA backward uses atomic adds and is
not deterministic; the two are numerically identical (tests/test_models.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import convolution_block

CHANNELS = (64, 128, 256, 512, 1024)


def upsample_bilinear_int(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Bilinear upsampling by an integer factor with half-pixel centres, identical to
    F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False), built from
    slicing, concatenation and stacking only, so its backward is deterministic on CUDA
    (interpolate's backward uses atomic adds). Output f*i + k blends x[i] with its
    neighbour at offset d = (k + 0.5) / f - 0.5 (x[i-1] if d < 0, x[i+1] if d > 0; edges clamped)."""
    if factor == 1:
        return x

    def along(t: torch.Tensor, dim: int) -> torch.Tensor:
        n = t.shape[dim]
        prev = torch.cat([t.narrow(dim, 0, 1), t.narrow(dim, 0, n - 1)], dim)
        nxt = torch.cat([t.narrow(dim, 1, n - 1), t.narrow(dim, n - 1, 1)], dim)
        outs = []
        for k in range(factor):
            d = (k + 0.5) / factor - 0.5
            outs.append((1 + d) * t - d * prev if d < 0 else (1 - d) * t + d * nxt)
        return torch.stack(outs, dim + 1).flatten(dim, dim + 1)

    return along(along(x, 2), 3)


def upsample2x_bilinear(x: torch.Tensor) -> torch.Tensor:
    return upsample_bilinear_int(x, 2)


class AttentionGate(nn.Module):
    def __init__(self, x_channels: int, g_channels: int, inter_channels: int | None = None):
        super().__init__()
        inter = inter_channels or x_channels // 2
        self.theta_x = nn.Conv2d(x_channels, inter, kernel_size=2, stride=2, bias=False)
        self.phi_g = nn.Conv2d(g_channels, inter, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter, 1, kernel_size=1, bias=True)

    def coefficients(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Attention coefficients alpha in (0, 1) at x's resolution, shape (N, 1, H, W)."""
        a = torch.sigmoid(self.psi(F.relu(self.theta_x(x) + self.phi_g(g))))
        if tuple(x.shape[2:]) == (2 * a.shape[2], 2 * a.shape[3]):
            return upsample2x_bilinear(a)
        return F.interpolate(a, size=x.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return x * self.coefficients(x, g)


def split_channels(total: int, parts: int) -> list[int]:
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations=(1, 2, 4)):
        super().__init__()
        self.dilations = tuple(dilations)
        self.branches = nn.ModuleList(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=d, dilation=d, bias=False)
            for c, d in zip(split_channels(out_channels, len(self.dilations)), self.dilations))
        self.bn = nn.BatchNorm2d(out_channels)
        self.fuse = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(F.relu(self.bn(torch.cat([b(x) for b in self.branches], dim=1)), inplace=True))


def multiscale_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """The baseline block with its second 3x3 conv replaced by MultiScaleConv."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        MultiScaleConv(out_channels, out_channels))


class UNetVariant(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, *, attention: bool = False,
                 multiscale: bool = False):
        super().__init__()
        self.attention, self.multiscale = attention, multiscale
        block = multiscale_block if multiscale else convolution_block
        c1, c2, c3, c4, c5 = CHANNELS
        # backbone: same modules, names and order as src/models/unet.py
        self.pool = nn.MaxPool2d(2)
        self.enc1 = block(in_channels, c1)
        self.enc2 = block(c1, c2)
        self.enc3 = block(c2, c3)
        self.enc4 = block(c3, c4)
        self.bottleneck = block(c4, c5)
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = block(2 * c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = block(2 * c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = block(2 * c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = block(2 * c1, c1)
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)
        if attention:                                   # created last: backbone init unchanged
            self.att4 = AttentionGate(c4, c5)           # skip e4, gated by the bottleneck output
            self.att3 = AttentionGate(c3, c4)           # skip e3, gated by d4
            self.att2 = AttentionGate(c2, c3)           # skip e2, gated by d3
            self.att1 = AttentionGate(c1, c2)           # skip e1, gated by d2

    def _skip(self, level: int, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return getattr(self, f"att{level}")(x, g) if self.attention else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), self._skip(4, e4, b)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), self._skip(3, e3, d4)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self._skip(2, e2, d3)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self._skip(1, e1, d2)], dim=1))
        return self.out(d1)
