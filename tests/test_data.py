"""Tests for src/data.py on small synthetic inputs (safe on the login node)."""
import collections

import cv2
import numpy as np
import pytest

from src.data import (_sums_one, augment, clahe_rgb, load_mask_target, make_splits, normalize_cnn,
                      parse_image_id, read_rgb, reduce_channel_sums, to_sam_input, write_rgb)


def test_norm_stats_match_direct_computation(tmp_path):
    """cmd_stats' per-image sums + reduction == mean/std over all pixels of all images."""
    rng = np.random.RandomState(5)
    paths, imgs = [], []
    for i in range(4):   # different brightness per image and per channel
        img = np.clip(rng.normal([60 + 30 * i, 90, 140 - 20 * i], 25, (64, 64, 3)), 0, 255).astype(np.uint8)
        write_rgb(tmp_path / f"{i}.png", img)
        paths.append(tmp_path / f"{i}.png"); imgs.append(img)
    mean, std, n = reduce_channel_sums([_sums_one((str(p),)) for p in paths])
    allpx = np.concatenate([read_rgb(p).reshape(-1, 3) / 255.0 for p in paths])
    assert n == 4 * 64 * 64
    assert np.allclose(mean, allpx.mean(0), atol=1e-12) and np.allclose(std, allpx.std(0), atol=1e-9)
    assert all(np.array_equal(read_rgb(p), im) for p, im in zip(paths, imgs))   # RGB PNG round trip is lossless


def _pair(seed=0, size=32):
    """A random square image (our images are square) and a mask that is a pixelwise
    function of it, so any image/mask mismatch after augmentation is detected."""
    img = np.random.RandomState(seed).randint(0, 256, (size, size, 3), dtype=np.uint8)
    return img, (img[..., 0] > 127).astype(np.float32)


def test_augment_moves_image_and_mask_together():
    img, mask = _pair()
    rng = np.random.RandomState(1)
    seen = set()
    for _ in range(400):
        a, m = augment(img, mask, rng)
        assert a.shape == img.shape and m.shape == mask.shape
        assert np.array_equal(m, (a[..., 0] > 127).astype(np.float32))   # same transform on both
        assert a.flags["C_CONTIGUOUS"] and m.flags["C_CONTIGUOUS"]
        seen.add(a.tobytes())
    assert len(seen) == 8                                                  # all 8 square symmetries


def test_augment_symmetries_are_uniform():
    img, mask = _pair()
    rng = np.random.RandomState(2)
    counts = collections.Counter(augment(img, mask, rng)[0].tobytes() for _ in range(8000))
    assert len(counts) == 8
    assert all(abs(c - 1000) < 150 for c in counts.values()), sorted(counts.values())   # ~4.7 sd


def test_augment_off_is_identity():
    img, mask = _pair()
    a, m = augment(img, mask, np.random.RandomState(3), hflip_p=0, vflip_p=0, rot90=False)
    assert np.array_equal(a, img) and np.array_equal(m, mask)


def test_clahe_changes_only_lightness():
    rng = np.random.RandomState(4)
    base = cv2.GaussianBlur(rng.randint(0, 256, (256, 256, 3), dtype=np.uint8), (0, 0), 8)
    out = clahe_rgb(base, 2.0, (8, 8))
    assert out.shape == base.shape and out.dtype == np.uint8
    lab_in, lab_out = cv2.cvtColor(base, cv2.COLOR_RGB2LAB).astype(int), cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(int)
    assert np.abs(lab_out[..., 0] - lab_in[..., 0]).mean() > 1          # L was enhanced
    assert np.median(np.abs(lab_out[..., 1:] - lab_in[..., 1:])) <= 1    # a, b kept (up to RGB rounding)


def test_normalizations_are_separate():
    img = np.full((4, 4, 3), 128, np.uint8)
    cnn = normalize_cnn(img, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    sam = to_sam_input(img)
    assert cnn.shape == sam.shape == (3, 4, 4) and cnn.dtype == sam.dtype == np.float32
    assert np.allclose(cnn, (128 / 255 - 0.5) / 0.25, rtol=0, atol=1e-6)   # float32 rounding
    assert np.allclose(sam, 128.0)                                         # raw pixels; SAM normalizes itself


def test_soft_mask_targets_are_not_binarized(tmp_path):
    soft = np.array([[0.0, 0.25], [0.5, 1.0]], np.float32)
    np.savez_compressed(tmp_path / "x.npz", mask=soft)
    assert np.array_equal(load_mask_target(tmp_path / "x.npz"), soft)
    cv2.imwrite(str(tmp_path / "y.png"), np.array([[0, 255]], np.uint8))
    assert np.array_equal(load_mask_target(tmp_path / "y.png"), np.array([[0.0, 1.0]], np.float32))


def _official_ids():
    train = [f"train_{n}_{c}" for c, start in zip("ADGN", (1, 151, 301, 451)) for n in range(start, start + 150)]
    test = [f"test_{n}_{c}" for c, start in zip("ADGN", (1, 51, 101, 151)) for n in range(start, start + 50)]
    return train, test


def test_split_is_stratified_and_reproducible():
    train, test = _official_ids()
    df = make_splits(train[::-1], test, 0.2, 17)                           # input order must not matter
    assert list(df.columns) == ["image_id", "category", "split"]
    assert df.split.value_counts().to_dict() == {"train": 480, "test": 200, "val": 120}
    assert (df[df.split == "val"].category.value_counts() == 30).all()
    assert df.image_id.is_unique
    assert df.equals(make_splits(train, test, 0.2, 17))
    assert not df.equals(make_splits(train, test, 0.2, 18))


def test_grouped_split_keeps_groups_together_and_excludes():
    train, test = _official_ids()
    groups = [("train_1_A", "train_2_A"), ("train_2_A", "train_3_A"),        # chained -> one group of 3
              ("train_151_D", "train_152_D"), ("train_301_G", "train_302_G")]
    groups += [(f"train_{n}_N", f"train_{n + 1}_N") for n in range(451, 520, 2)]   # 35 pairs in N
    exclude = {"train_447_G": "empty mask", "train_448_G": "empty mask"}
    for seed in range(20):
        df = make_splits(train, test, 0.2, seed, groups=groups, exclude=exclude).set_index("image_id")
        for g in [("train_1_A", "train_2_A", "train_3_A"), ("train_151_D", "train_152_D")] + groups[4:]:
            assert len({df.split[i] for i in g}) == 1                          # never separated
        assert df.split["train_447_G"] == df.split["train_448_G"] == "excluded"
        val = df[df.split == "val"].category.value_counts().to_dict()
        assert val == {"A": 30, "D": 30, "G": 30, "N": 30}                    # G: round(0.2 * 148) = 30
        assert (df.split == "train").sum() == 478 and (df.split == "test").sum() == 200
    cross = [("train_67_A", "train_175_D"), ("train_81_A", "train_273_D")]   # groups spanning categories
    for seed in range(20):
        df = make_splits(train, test, 0.2, seed, groups=groups + cross, exclude=exclude).set_index("image_id")
        assert all(df.split[a] == df.split[b] for a, b in cross)
        assert df[df.split == "val"].category.value_counts().to_dict() == {"A": 30, "D": 30, "G": 30, "N": 30}


def test_parse_image_id():
    p = parse_image_id("test_101_G")
    assert (p.split, p.number, p.category) == ("test", 101, "G")
    for bad in ["test_101", "val_1_A", "train_x_A", "train_1_Q"]:
        with pytest.raises(ValueError):
            parse_image_id(bad)


def test_pairwise_shift_iou():
    from src.data import pairwise_shift_iou
    g = 16
    a = np.zeros((g, g), bool); a[3:9, 2:5] = True; a[3:5, 2:10] = True      # an "L"-like blob
    b = np.roll(a, 1, axis=1)                                                  # same pattern, shifted 1 cell
    c = np.zeros((g, g), bool); c[10:14, 10:14] = True                         # different pattern
    e = np.zeros((g, g), bool)                                                 # empty
    m = pairwise_shift_iou(np.stack([a, b, c, e]), max_shift=2)
    assert np.allclose(np.diag(m)[:3], 1) and m[3, 3] == 0                     # empty never matches
    assert np.isclose(m[0, 1], 1) and np.allclose(m, m.T)                      # shift-tolerant, symmetric
    assert m[0, 2] < 0.2 and m[0, 3] == 0
    assert pairwise_shift_iou(np.stack([a, b]), max_shift=0)[0, 1] < 1         # no tolerance -> imperfect


def test_redaction_pixelates_only_outside_fov():
    from src.data import redact_outside_fov
    rng = np.random.RandomState(6)
    img = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    fov = np.zeros((256, 256), bool); fov[64:192, 64:192] = True
    out = redact_outside_fov(img, fov, margin=4, block=32)
    assert np.array_equal(out[64:192, 64:192], img[64:192, 64:192])           # FOV untouched
    assert np.array_equal(out[60:196, 100:150], img[60:196, 100:150])         # margin band (straight edge) untouched
    corner = out[:32, :32].reshape(-1, 3)
    assert (corner == corner[0]).all()                                         # one flat block outside
    assert not np.array_equal(out[:32, :32], img[:32, :32])


def test_marking_regions_ignore_specks():
    from src.data import marking_stats
    img = np.zeros((2048, 2048, 3), np.uint8)
    cv2.circle(img, (1024, 1024), 1000, (150, 60, 30), -1)                     # the eye
    fov = img.max(2) > 15
    for x in (0, 18, 36):                                                      # a 3-stroke "label", top-left
        img[20:95, x:x + 8] = 255
    img[1990:1993, 300:303] = 255                                              # 9-px speck, bottom-left
    img[2040:2042, 1900:1902] = 255                                            # 4-px speck, bottom-right
    s = marking_stats(img, fov, threshold=40, margin=20, min_px=20)
    assert s["marking_n_regions"] == 1 and s["marking_regions"] == "top-left"  # strokes merged into one region
    assert s["marking_px"] == 3 * 75 * 8 and s["speck_px"] == 9 + 4
    assert (s["marking_y0"], s["marking_y1"], s["marking_x0"], s["marking_x1"]) == (20, 94, 0, 43)   # box ignores specks
    assert (s["topleft_y1"], s["topleft_x1"]) == (94, 43)
    img[1900:1960, 100:150] = 255                                              # a second real region, bottom-left
    s2 = marking_stats(img, fov, threshold=40, margin=20, min_px=20)
    assert s2["marking_n_regions"] == 2 and s2["marking_regions"] == "bottom-left;top-left"
