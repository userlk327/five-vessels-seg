"""Resizing helpers: every mask method, and CPU/GPU upsampling agreement."""
import cv2
import numpy as np
import pytest
import torch

from src.resample import MASK_METHODS, binarize, block_mean, downsample_mask, upsample_prob, upsample_prob_torch


def test_block_mean_equals_inter_area_for_integer_factors():
    m = (np.random.RandomState(0).rand(2048, 2048) > 0.8)
    a = cv2.resize(m.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
    assert np.allclose(block_mean(m, 512), a, atol=1e-6)


def test_one_px_line_under_each_method_at_4x():
    """A 1-px line covers 25% of each 4x4 block: kept by area_t25 / maxpool / soft (0.25),
    deleted by area_t50, dotted or lost by nearest (depends on the sampling phase)."""
    m = np.zeros((2048, 2048), bool); m[:, 1001] = True
    out = {k: downsample_mask(m, 512, k) for k in MASK_METHODS}
    assert out["area_t50"].sum() == 0
    assert out["area_t25"][:, 250].all() and out["maxpool"][:, 250].all()
    assert np.allclose(out["soft"][:, 250], 0.25)
    assert out["nearest"].sum() in (0, 512)                                  # pixel-centre sampling: all or nothing here
    for k, v in out.items():
        assert v.shape == (512, 512) and v.dtype == np.float32, k


def test_cpu_and_gpu_upsampling_agree():
    x = np.random.RandomState(1).rand(256, 256).astype(np.float32)
    a = upsample_prob(x, 1024)
    b = upsample_prob_torch(torch.from_numpy(x)[None, None], 1024)[0, 0].numpy()
    assert np.abs(a - b).max() < 1e-6


def test_binarize_is_inclusive():
    assert binarize(np.array([0.49, 0.5, 0.51]), 0.5).tolist() == [False, True, True]
