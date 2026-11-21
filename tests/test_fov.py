"""Circle-fit FOV on a synthetic eye: dim side, notch, burned-in text, eye beyond the frame."""
import cv2
import numpy as np
import pytest

from src.fov import circle_mask, fit_fov_circle, fov_mask


def synthetic_eye(cx=1030.0, cy=1010.0, r=1030.0, dim_side=True):
    rng = np.random.RandomState(0)
    img = np.full((2048, 2048, 3), 3, np.uint8)                              # background ~3
    img = np.clip(img.astype(int) + rng.randint(0, 2, img.shape), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[:2048, :2048]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    bright = np.where(xx < cx, 120, 8) if dim_side else 120                  # right half very dim (8)
    for ch, f in enumerate((1.0, 0.5, 0.25)):
        img[..., ch] = np.where(disk, np.maximum((bright * f).astype(int), 7 if ch == 0 else 3), img[..., ch])
    notch = (xx > cx + 0.62 * r) & (xx < cx + 0.62 * r + 40) & (yy > cy - 0.80 * r - 20) & (yy < cy - 0.80 * r + 20)
    img[notch & ~disk] = (110, 55, 28)                                       # bright tab outside the circle
    cv2.putText(img, "XX", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)   # burned-in text
    return img, disk, notch & ~disk


def test_circle_recovered_with_dim_side_notch_text_and_frame():
    img, disk, notch = synthetic_eye()
    c, inl = fit_fov_circle(img)
    # residual bias toward the bright side is ~1 px with sigma 1 (see src/fov.py)
    assert c.cx == pytest.approx(1030, abs=2) and c.cy == pytest.approx(1010, abs=2)
    assert c.r == pytest.approx(1030, abs=2)
    assert c.touches_frame                                                    # r = 1030 > distance to the frame
    fov, _ = fov_mask(img)
    assert fov[disk].mean() > 0.999                                          # dim half included
    assert fov[notch].mean() > 0.9                                           # notch included
    assert not fov[:70, :160].any()                                          # text excluded


def test_circle_mask_scaling_matches_downsampling():
    c = {"cx": 1023.5, "cy": 1023.5, "r": 1000.0}
    full = circle_mask(c, (2048, 2048)).astype(np.float32)
    small = circle_mask(c, (512, 512), scale=0.25)
    down = cv2.resize(full, (512, 512), interpolation=cv2.INTER_AREA) > 0.5
    assert (small != down).mean() < 0.002
