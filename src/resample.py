"""Resizing between the native 2048 grid and the model grids (512 / 1024 / 256).

Downsampling training masks (Phase 2 comparison, then the mask cache):
  nearest     nearest neighbour at pixel centres (cv2.INTER_NEAREST_EXACT, identical to
              PIL NEAREST, i.e. the notebook's torchvision resize)
  area_t50    block mean, then >= 0.5
  area_t25    block mean, then >= 0.25
  maxpool     block contains any vessel pixel (block mean > 0)
  soft        block mean, no threshold (soft target in [0, 1])
The block mean is exact because 2048 is an integer multiple of every target size
(identical to cv2.INTER_AREA for integer factors).

Upsampling predictions (and reconstructed masks) back to 2048: bilinear with half-pixel
centres (cv2.INTER_LINEAR on CPU; torch interpolate(mode="bilinear",
align_corners=False) on GPU; the two agree to 1e-7, see tests), then probability >= t.
"""
from __future__ import annotations

import cv2
import numpy as np

MASK_METHODS = ("nearest", "area_t50", "area_t25", "maxpool", "soft")


def block_mean(mask: np.ndarray, size: int) -> np.ndarray:
    f = mask.shape[0] // size
    if mask.shape != (size * f, size * f):
        raise ValueError(f"{mask.shape} is not an integer multiple of {size}")
    return mask.astype(np.float32).reshape(size, f, size, f).mean(axis=(1, 3))


def downsample_mask(mask: np.ndarray, size: int, method: str) -> np.ndarray:
    """Binary 2048 mask -> float32 target at ``size`` (0/1, or soft in [0, 1])."""
    m = mask.astype(bool)
    if method == "nearest":
        return cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST_EXACT).astype(np.float32)
    mean = block_mean(m, size)
    if method == "area_t50":
        return (mean >= 0.5).astype(np.float32)
    if method == "area_t25":
        return (mean >= 0.25).astype(np.float32)
    if method == "maxpool":
        return (mean > 0).astype(np.float32)
    if method == "soft":
        return mean
    raise ValueError(f"unknown method {method!r}; choose from {MASK_METHODS}")


def upsample_prob(prob: np.ndarray, size: int = 2048) -> np.ndarray:
    """Bilinear upsampling (half-pixel centres) of a probability / target map."""
    return cv2.resize(prob.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


def upsample_prob_torch(prob, size: int = 2048):
    """GPU twin of ``upsample_prob`` for (N, 1, h, w) tensors."""
    import torch.nn.functional as F
    return F.interpolate(prob, size=(size, size), mode="bilinear", align_corners=False)


def binarize(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return prob >= threshold
