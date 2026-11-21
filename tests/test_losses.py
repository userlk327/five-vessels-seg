"""Shared loss: soft targets enter both terms (BCE and soft Dice) unchanged, in float32,
and nothing between the mask cache and the loss binarizes them."""
import cv2
import numpy as np
import pytest
import torch

from src.data import FIVESDataset, load_mask_target, write_npz_archive
from src.losses import BCEDiceLoss


def manual_terms(z, t, eps=1.0):
    """BCE (mean over pixels) and mean over images of (1 - soft Dice), float64, written out."""
    p, t = torch.sigmoid(z.double()), t.double()
    bce = -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()
    dims = (1, 2, 3)
    dice = (2 * (p * t).sum(dims) + eps) / (p.sum(dims) + t.sum(dims) + eps)
    return bce.item(), (1 - dice).mean().item()


def manual_loss(z, t, eps=1.0):
    return sum(manual_terms(z, t, eps))


def soft_batch(seed=0):
    rng = np.random.RandomState(seed)
    t = torch.from_numpy((rng.randint(0, 17, (2, 1, 32, 32)) / 16).astype(np.float32))   # k/16, like 4x4 block means
    z = torch.from_numpy(rng.randn(2, 1, 32, 32).astype(np.float32))
    return z, t


def test_soft_targets_enter_both_terms_unchanged():
    z, t = soft_batch()
    loss, parts = BCEDiceLoss()(z, t)
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(manual_loss(z, t), abs=1e-5)
    assert parts["bce"] + parts["dice_loss"] == pytest.approx(loss.item(), abs=1e-6)
    # each term is the formula with the soft target t, not with a binarized one
    bce_soft, dice_soft = manual_terms(z, t)
    bce_bin, dice_bin = manual_terms(z, (t >= 0.5).float())
    assert parts["bce"] == pytest.approx(bce_soft, abs=1e-6) and parts["dice_loss"] == pytest.approx(dice_soft, abs=1e-6)
    assert abs(parts["bce"] - bce_bin) > 1e-3 and abs(parts["dice_loss"] - dice_bin) > 1e-3


def test_bce_term_is_minimized_at_the_soft_target():
    t = torch.tensor([[[[1 / 16, 0.25, 0.5, 0.75]]]])
    z = torch.logit(t).requires_grad_()
    BCEDiceLoss(1.0, 0.0)(z, t)[0].backward()
    assert z.grad.abs().max() < 1e-7


def test_loss_is_computed_in_float32():
    z, t = soft_batch(1)
    ref = BCEDiceLoss()(z.half().float(), t)[0]
    assert BCEDiceLoss()(z.half(), t)[0].item() == ref.item()          # fp16 logits are upcast first
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = BCEDiceLoss()(z, t)[0]
    assert out.dtype == torch.float32 and out.item() == pytest.approx(BCEDiceLoss()(z, t)[0].item(), abs=1e-6)


def test_integer_targets_are_refused():
    z, t = soft_batch()
    with pytest.raises(TypeError):
        BCEDiceLoss()(z, (t > 0.5).long())


def test_training_path_keeps_soft_values(tmp_path):
    """Mask archive (float16) -> FIVESDataset (train, augmentation) -> DataLoader -> loss:
    every soft value arrives unchanged (augmentation only permutes pixels)."""
    rng = np.random.RandomState(0)
    ids = ["train_1_A", "train_2_D"]
    soft = {i: (rng.randint(0, 17, (16, 16)) / 16).astype(np.float16) for i in ids}
    write_npz_archive(tmp_path / "masks_soft_16.npz", soft)
    for i in ids:
        cv2.imwrite(str(tmp_path / f"{i}.png"), rng.randint(0, 255, (16, 16, 3), np.uint8))
    ds = FIVESDataset(ids, image_dir=tmp_path, mask_source=tmp_path / "masks_soft_16.npz", normalization="cnn",
                      norm_stats={"mean": [0.5] * 3, "std": [0.2] * 3}, train=True,
                      augmentation={"hflip_p": 0.5, "vflip_p": 0.5, "rot90": True})
    np.random.seed(3)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)))
    m = batch["mask"]
    assert m.dtype == torch.float32 and m.shape == (2, 1, 16, 16)
    for k, i in enumerate(batch["image_id"]):
        assert np.array_equal(np.sort(m[k, 0].numpy().ravel()), np.sort(soft[i].astype(np.float32).ravel()))
    assert ((m > 0) & (m < 1)).any()                                        # soft values present, not binarized
    z = torch.randn(2, 1, 16, 16)
    assert BCEDiceLoss()(z, m)[0].item() == pytest.approx(manual_loss(z, m), abs=1e-5)


def test_mask_archive_members_and_range_check(tmp_path):
    a = {"train_1_A": np.array([[0, 0.0625], [0.25, 1]], np.float16)}
    write_npz_archive(tmp_path / "m.npz", a)
    assert np.array_equal(load_mask_target(tmp_path / "m.npz", key="train_1_A"),
                          np.array([[0, 0.0625], [0.25, 1]], np.float32))
    write_npz_archive(tmp_path / "bad.npz", {"x": np.array([[0, 1.5]], np.float16)})
    with pytest.raises(ValueError):
        load_mask_target(tmp_path / "bad.npz", key="x")
