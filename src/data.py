"""Data: parquet -> PNG extraction, checks, split, preprocessing, caching, augmentation.

Command line (heavy steps run as a SLURM job on the batch partition):

    python -m src.data all        --config configs/base.yaml   # every step below, in order
    python -m src.data extract    # parquet -> data/FIVES/{train,test}/{images,masks}/<image_id>.png
    python -m src.data check      # counts, names, shapes, channels, mask values, categories,
                                  # PNG-vs-parquet comparison, FOV / burned-in marking QA
    python -m src.data split      # data/splits/splits.csv: grouped, category-balanced train/val (seed 17),
                                  # exclusions, + 200 official test; groups in split_groups.csv
    python -m src.data pairs      # likely same-patient pairs from identical burned-in markings (counts only;
                                  # the text is never read, transcribed or saved)
    python -m src.data pairs_summary  # recount those pairs by split from marking_pairs.csv (no images read)
    python -m src.data integrity  # duplicates, identical/empty masks, ground truth outside the eye, near-duplicates
    python -m src.data preprocess # CLAHE at 2048 -> area resize -> data/cache/images_clahe_<res>/
    python -m src.data stats      # CNN mean/std on the TRAIN split only -> data/splits/norm_stats.json
    python -m src.data visual     # figures in results/checks/
    python -m src.data masks      # Phase 2: training targets (train + val, never test) resized with
                                  # data.mask_resize -> data/cache/masks_<method>_<res>.npz
    python -m src.data native     # Phase 5: CLAHE images at native 2048 (tiled SAM) ->
                                  # data/cache/images_clahe_2048/, checked against the 1024 cache

Training masks are resized with the method chosen in Phase 2 (``soft``: area average, no
threshold) and stored as float16 in one compressed .npz archive per resolution, keyed by
image_id. The loader uses them as is (never thresholded). Evaluation always uses the
original 2048 masks in data/FIVES/<split>/masks/.

Normalization has two separate cases and never both: CNNs use the train-split mean/std
(``normalize_cnn``); SAM gets raw 0-255 pixels and applies its own normalization
(``to_sam_input``).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils import PROJECT_ROOT, load_config, project_path

CATEGORIES = ("A", "D", "G", "N")
CATEGORY_NAMES = {"A": "AMD", "D": "diabetic retinopathy", "G": "glaucoma", "N": "normal"}
SPLITS = ("train", "test")          # official FIVES splits (parquet files)
CHECKS_DIR = PROJECT_ROOT / "results" / "checks"
PRIVATE_DIR = CHECKS_DIR / "private"     # git-ignored: per-image marking locations and review sheets
MARKING_LOCATION_COLS = ["marking_regions", "marking_y0", "marking_y1", "marking_x0", "marking_x1",
                         "topleft_y1", "topleft_x1"]


# =========================================================================== ids and paths
@dataclass(frozen=True)
class ImageId:
    split: str      # official split: train / test
    number: int
    category: str   # A / D / G / N


def parse_image_id(image_id: str) -> ImageId:
    """'train_100_A' -> ImageId('train', 100, 'A'). Raises on anything unexpected."""
    parts = image_id.split("_")
    if len(parts) != 3 or parts[0] not in SPLITS or not parts[1].isdigit() or parts[2] not in CATEGORIES:
        raise ValueError(f"unexpected image_id {image_id!r}")
    return ImageId(parts[0], int(parts[1]), parts[2])


def id_sort_key(image_id: str) -> tuple:
    p = parse_image_id(image_id)
    return (SPLITS.index(p.split), p.number)


def fives_dir(cfg: dict) -> Path:
    return project_path(cfg["paths"]["fives_dir"])


def parquet_files(cfg: dict) -> dict[str, list[Path]]:
    d = fives_dir(cfg) / "data"
    return {s: sorted(d.glob(f"{s}-*.parquet")) for s in SPLITS}


def raw_image_path(cfg: dict, image_id: str) -> Path:
    return fives_dir(cfg) / parse_image_id(image_id).split / "images" / f"{image_id}.png"


def raw_mask_path(cfg: dict, image_id: str) -> Path:
    return fives_dir(cfg) / parse_image_id(image_id).split / "masks" / f"{image_id}.png"


def cache_image_dir(cfg: dict, resolution: int) -> Path:
    return project_path(cfg["paths"]["cache_dir"]) / f"images_clahe_{resolution}"


def cache_resolutions(cfg: dict) -> list[int]:
    """512 and 1024 (CLAUDE.md), plus cnn_resolution / sam_resolution if they differ."""
    d = cfg["data"]
    return sorted({512, 1024, int(d["cnn_resolution"]), int(d["sam_resolution"])})


def mask_cache_path(cfg: dict, resolution: int) -> Path:
    """Training-target archive for one resolution (Phase 2), e.g. data/cache/masks_soft_512.npz."""
    method = cfg["data"].get("mask_resize")
    if not method:
        raise ValueError("data.mask_resize is not set in the config")
    return project_path(cfg["paths"]["cache_dir"]) / f"masks_{method}_{resolution}.npz"


def norm_stats_path(cfg: dict) -> Path:
    return project_path(cfg["paths"]["splits_csv"]).parent / "norm_stats.json"


def load_splits(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(project_path(cfg["paths"]["splits_csv"]), dtype=str)


# =========================================================================== image I/O
def read_rgb(path: Path) -> np.ndarray:
    """8-bit RGB, HxWx3."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {path}")
    os.replace(tmp, path)


def read_unchanged(path: Path) -> np.ndarray:
    """Exactly what the PNG stores (channel count, bit depth) - used by the checks."""
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    return a


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# =========================================================================== .npz archives
# Many arrays in one compressed .npz, keyed by image_id: the training-target cache and the
# saved probability maps use this to keep file counts low on $WORK (2 files instead of
# hundreds). A member is read on its own, without loading the others.
def write_npz_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Written to a temporary file and renamed, so a killed job never leaves a partial archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def npz_keys(path: Path) -> list[str]:
    with np.load(path) as z:
        return list(z.files)


def read_npz_member(path: Path, key: str) -> np.ndarray:
    with np.load(path) as z:
        return z[key]


# =========================================================================== preprocessing
def clahe_rgb(rgb: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    """CLAHE on the L channel of LAB; A and B are untouched; returns 8-bit RGB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(int(t) for t in tile_grid))
    lab[..., 0] = clahe.apply(np.ascontiguousarray(lab[..., 0]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def resize_area(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def clahe_native(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    """2048 RGB -> CLAHE at native resolution (the first step of every cached image)."""
    c = cfg["data"]["clahe"]
    assert rgb.shape[:2] == (cfg["data"]["native_resolution"],) * 2, rgb.shape
    return clahe_rgb(rgb, c["clip_limit"], c["tile_grid"])


def preprocess_rgb(rgb: np.ndarray, cfg: dict) -> dict[int, np.ndarray]:
    """2048 RGB -> CLAHE at native resolution -> area resize to every cache resolution."""
    enhanced = clahe_native(rgb, cfg)
    return {r: resize_area(enhanced, r) for r in cache_resolutions(cfg)}


# =========================================================================== split
def _groups_to_components(ids: list[str], groups) -> list[list[str]]:
    """Merge overlapping groups (union-find); every id ends up in exactly one component."""
    ix = {i: k for k, i in enumerate(ids)}
    edges = [(ix[a], ix[b]) for g in groups for a in g for b in g if a in ix and b in ix and a != b]
    multi = _components(len(ids), edges)
    seen = {k for c in multi for k in c}
    comps = [[ids[k] for k in c] for c in multi] + [[ids[k]] for k in range(len(ids)) if k not in seen]
    return sorted((sorted(c, key=id_sort_key) for c in comps), key=lambda c: id_sort_key(c[0]))


def make_splits(train_ids: list[str], test_ids: list[str], val_fraction: float, seed: int,
                groups=(), exclude: dict | None = None) -> pd.DataFrame:
    """Official train -> train / val, grouped and balanced by category; official test kept.

    * ``exclude`` ids get split 'excluded' (used for neither training nor validation).
    * ``groups`` (iterables of ids) are merged into components whose members always land
      on the same side. A component may span categories.
    * Val gets exactly round(val_fraction * n_c) images of each category c: all components
      are visited in one seeded random order (RandomState(seed)) and a component goes to
      val only if it fits the remaining val places of every category it contains.
    Input order does not matter.
    Columns: image_id, category, split (train / val / test / excluded)."""
    exclude = exclude or {}
    pool = sorted(set(train_ids) - set(exclude), key=id_sort_key)
    comps = _groups_to_components(pool, groups)
    cat_of = lambda i: parse_image_id(i).category
    target = {c: int(round(val_fraction * sum(cat_of(i) == c for i in pool))) for c in CATEGORIES}
    got = {c: 0 for c in CATEGORIES}
    val: set[str] = set()
    for k in np.random.RandomState(seed).permutation(len(comps)):
        need = {c: sum(cat_of(i) == c for i in comps[k]) for c in CATEGORIES}
        if all(got[c] + need[c] <= target[c] for c in CATEGORIES):
            val.update(comps[k])
            for c in CATEGORIES:
                got[c] += need[c]
    if got != target:
        raise RuntimeError(f"could only fill val {got} of {target}")
    label = {i: ("val" if i in val else "train") for i in pool}
    label.update({i: "excluded" for i in train_ids if i in exclude})
    label.update({i: "test" for i in test_ids})
    order = {"train": 0, "val": 1, "excluded": 2, "test": 3}
    ids = sorted(label, key=lambda i: (order[label[i]], id_sort_key(i)))
    return pd.DataFrame({"image_id": ids, "category": [parse_image_id(i).category for i in ids],
                         "split": [label[i] for i in ids]})


# =========================================================================== augmentation / normalization
def augment(image: np.ndarray, mask: np.ndarray, rng, hflip_p: float = 0.5,
            vflip_p: float = 0.5, rot90: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal flip, vertical flip and 90-degree rotation, applied identically
    to image (HxWxC) and mask (HxW). ``rng`` is ``np.random`` (seeded per DataLoader
    worker by utils.seed_worker) or a ``np.random.RandomState``. Together the three
    give each of the 8 square symmetries with equal probability."""
    if rng.random() < hflip_p:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < vflip_p:
        image, mask = image[::-1], mask[::-1]
    if rot90:
        k = int(rng.randint(4))
        image, mask = np.rot90(image, k, axes=(0, 1)), np.rot90(mask, k, axes=(0, 1))
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def normalize_cnn(image_uint8: np.ndarray, mean, std) -> np.ndarray:
    """CNN input: HxWx3 uint8 -> 3xHxW float32, (x/255 - mean) / std (train-split stats)."""
    x = image_uint8.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def to_sam_input(image_uint8: np.ndarray) -> np.ndarray:
    """SAM input: HxWx3 uint8 -> 3xHxW float32 in 0..255, NOT normalized here; SAM's
    own preprocessing applies its pixel_mean / pixel_std."""
    return np.ascontiguousarray(image_uint8.astype(np.float32).transpose(2, 0, 1))


def load_mask_target(path: Path, key: str | None = None) -> np.ndarray:
    """Training target in [0, 1], float32, HxW, exactly as cached: a member of a mask
    archive (``key`` = image_id), a single-mask .npz (key 'mask'), or a 0/255 PNG (divided
    by 255). Never thresholded, rounded or clipped; anything outside [0, 1] raises."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            m = z["mask" if key is None else key].astype(np.float32)
    else:
        m = read_unchanged(path).astype(np.float32) / 255.0
    if m.ndim != 2 or not np.isfinite(m).all() or m.min() < 0 or m.max() > 1:
        raise ValueError(f"{path}{'' if key is None else ':' + key}: target must be HxW in [0, 1], "
                         f"got shape {m.shape}, range [{m.min()}, {m.max()}]")
    return m


class FIVESDataset:
    """Cached, CLAHE-processed images (+ optional cached training masks).

    image_dir     data/cache/images_clahe_<res>
    mask_source   Phase 2 target cache: an .npz archive keyed by image_id (``mask_cache_path``),
                  or a folder of <image_id>.png / .npz files; None = images only.
                  Targets are returned exactly as stored (soft values kept).
    normalization "cnn" (needs norm_stats {"mean", "std"}) or "sam" - exactly one of them
    train         True -> random flips / 90-degree rotations (training split only)
    Returns dict(image=float32 3xHxW tensor, mask=float32 1xHxW tensor or absent, image_id=str).
    """

    def __init__(self, image_ids, *, image_dir: Path, mask_source: Path | None = None,
                 normalization: str, norm_stats: dict | None = None, train: bool = False,
                 augmentation: dict | None = None):
        if normalization not in ("cnn", "sam"):
            raise ValueError("normalization must be 'cnn' or 'sam'")
        if normalization == "cnn" and norm_stats is None:
            raise ValueError("CNN normalization needs the train-split norm_stats")
        self.ids = list(image_ids)
        self.image_dir, self.mask_source = Path(image_dir), (Path(mask_source) if mask_source else None)
        self.normalization, self.norm_stats, self.train = normalization, norm_stats, train
        self.aug = augmentation or {}

    def __len__(self) -> int:
        return len(self.ids)

    def _load_mask(self, image_id: str) -> np.ndarray:
        if self.mask_source.is_file():
            return load_mask_target(self.mask_source, key=image_id)
        for ext in (".png", ".npz"):
            p = self.mask_source / f"{image_id}{ext}"
            if p.exists():
                return load_mask_target(p)
        raise FileNotFoundError(f"no cached mask for {image_id} in {self.mask_source}")

    def __getitem__(self, index: int) -> dict:
        import torch

        image_id = self.ids[index]
        image = read_rgb(self.image_dir / f"{image_id}.png")
        mask = self._load_mask(image_id) if self.mask_source else None
        if mask is not None and mask.shape != image.shape[:2]:
            raise ValueError(f"{image_id}: mask {mask.shape} vs image {image.shape}")
        if self.train:
            if mask is None:
                raise ValueError("training needs masks")
            image, mask = augment(image, mask, np.random, self.aug.get("hflip_p", 0.5),
                                  self.aug.get("vflip_p", 0.5), self.aug.get("rot90", True))
        x = (normalize_cnn(image, self.norm_stats["mean"], self.norm_stats["std"])
             if self.normalization == "cnn" else to_sam_input(image))
        out = {"image": torch.from_numpy(x), "image_id": image_id}
        if mask is not None:
            out["mask"] = torch.from_numpy(np.ascontiguousarray(mask[None]))
        return out


def split_ids(cfg: dict, split: str) -> list[str]:
    """Image ids of one split of splits.csv (train / val / test), in file order."""
    sp = load_splits(cfg)
    ids = sp.loc[sp.split == split, "image_id"].tolist()
    if not ids:
        raise ValueError(f"no images in split {split!r}")
    return ids


def load_norm_stats(cfg: dict, resolution: int) -> dict:
    """Train-split mean / std of the cached CLAHE images at ``resolution`` (Phase 1)."""
    return json.loads(norm_stats_path(cfg).read_text())[str(int(resolution))]


def build_cnn_dataset(cfg: dict, ids, *, train: bool, with_masks: bool,
                      resolution: int | None = None) -> FIVESDataset:
    """CNN data at ``data.cnn_resolution`` (or ``resolution``, e.g. the 1024 probe): cached
    CLAHE images, train-split normalization, soft training targets from the Phase 2
    archive, random flips / 90-degree rotations only when ``train``."""
    res = int(resolution or cfg["data"]["cnn_resolution"])
    return FIVESDataset(ids, image_dir=cache_image_dir(cfg, res),
                        mask_source=mask_cache_path(cfg, res) if with_masks else None,
                        normalization="cnn", norm_stats=load_norm_stats(cfg, res), train=train,
                        augmentation=cfg["data"]["augmentation"] if train else None)


def tile_slices(native: int, tile: int, index: int) -> tuple[slice, slice]:
    """Row-major tile ``index`` of a native x native image cut into (native/tile)^2 tiles."""
    n = native // tile
    if n * tile != native or not 0 <= index < n * n:
        raise ValueError(f"bad tiling {native}/{tile}, index {index}")
    r, c = divmod(index, n)
    return slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile)


def stitch_tiles(tiles, n: int):
    """Inverse of the row-major tiling: a sequence/array of n*n tiles (..., h, w) -> (..., n*h, n*w).
    Works for NumPy arrays and torch tensors."""
    rows = [tiles[r * n:(r + 1) * n] for r in range(n)]
    if hasattr(tiles, "dim"):                                  # torch
        import torch
        return torch.cat([torch.cat(list(row), dim=-1) for row in rows], dim=-2)
    return np.concatenate([np.concatenate(list(row), axis=-1) for row in rows], axis=-2)


class SamTileDataset:
    """Tiled SAM (Phase 5, main fine-tuned SAM): native-resolution crops of the CLAHE images.

    Each 2048 image is cut into 2x2 non-overlapping 1024x1024 tiles (row-major index 0-3);
    one item = one tile, so an epoch covers every tile of every image once (the same image
    content per epoch as the CNNs). Image: 3x1024x1024 float32 in 0..255 (SAM normalizes
    it itself). Mask (when ``with_masks``): the matching crop of the ORIGINAL 2048 mask,
    binary 0/1 float32 (native resolution: nothing is resized). Training: random flips /
    90-degree rotations of the tile, image and mask together.
    """

    def __init__(self, cfg: dict, image_ids, *, train: bool, with_masks: bool):
        self.cfg, self.train, self.with_masks = cfg, train, with_masks
        self.native, self.tile = int(cfg["data"]["native_resolution"]), int(cfg["data"]["sam_resolution"])
        self.n = self.native // self.tile
        self.items = [(i, t) for i in image_ids for t in range(self.n * self.n)]
        self.image_dir = cache_image_dir(cfg, self.native)
        self.aug = cfg["data"]["augmentation"] if train else {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        import torch

        image_id, t = self.items[index]
        ys, xs = tile_slices(self.native, self.tile, t)
        image = read_rgb(self.image_dir / f"{image_id}.png")[ys, xs]
        mask = None
        if self.with_masks:
            mask = (read_unchanged(raw_mask_path(self.cfg, image_id)) > 0)[ys, xs].astype(np.float32)
        if self.train:
            image, mask = augment(image, mask, np.random, self.aug.get("hflip_p", 0.5),
                                  self.aug.get("vflip_p", 0.5), self.aug.get("rot90", True))
        out = {"image": torch.from_numpy(to_sam_input(np.ascontiguousarray(image))), "image_id": image_id, "tile": t}
        if mask is not None:
            out["mask"] = torch.from_numpy(np.ascontiguousarray(mask[None]))
        return out


# =========================================================================== FOV / marking QA
def estimate_fov(rgb: np.ndarray, threshold: int) -> np.ndarray:
    """Rough field of view for the QA only (the Phase 2 FOV mask is defined and checked
    separately): max-channel > threshold, opened to cut thin text strokes, largest
    connected component, holes filled."""
    from scipy import ndimage as ndi

    g = rgb.max(axis=2)
    fg = (g > threshold).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros(g.shape, bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ndi.binary_fill_holes(labels == largest)


def _region_name(cy: float, cx: float, size: int = 2048) -> str:
    v = "top" if cy < size / 3 else ("bottom" if cy > 2 * size / 3 else "middle")
    h = "left" if cx < size / 3 else ("right" if cx > 2 * size / 3 else "centre")
    return f"{v}-{h}"


def marking_stats(rgb: np.ndarray, fov: np.ndarray, threshold: int, margin: int, min_px: int,
                  merge_px: int = 41) -> dict:
    """Burned-in markings outside the (dilated) FOV, counted per label region: bright
    pixels (max(R,G,B) > threshold) are merged into regions with a merge_px closing (the
    strokes of one label), regions with >= min_px bright pixels are markings, smaller ones
    are specks (dust, noise). Location and box come from the marking regions only."""
    g = rgb.max(axis=2)
    near = cv2.dilate(fov.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
    outside = near == 0
    bright = outside & (g > threshold)
    out = {"outside_max": int(g[outside].max()) if outside.any() else 0,
           "outside_p999": float(np.percentile(g[outside], 99.9)) if outside.any() else 0.0}
    merged = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (merge_px, merge_px)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    px = np.bincount(lab[bright], minlength=n)                     # bright pixels per region
    keep = [k for k in range(1, n) if px[k] >= min_px]
    boxes = {}                                                     # from the bright pixels themselves (the
    for k in keep:                                                 # closing can pad a region to the image edge)
        ys, xs = np.nonzero(bright & (lab == k))
        boxes[k] = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    regions = {k: _region_name((b[0] + b[1]) / 2, (b[2] + b[3]) / 2) for k, b in boxes.items()}
    out.update(marking_px=int(px[keep].sum()), speck_px=int(bright.sum() - px[keep].sum()),
               marking_n_regions=len(keep), marking_regions=";".join(sorted(set(regions.values()))))
    if keep:
        out.update(marking_y0=min(b[0] for b in boxes.values()), marking_y1=max(b[1] for b in boxes.values()),
                   marking_x0=min(b[2] for b in boxes.values()), marking_x1=max(b[3] for b in boxes.values()))
    else:
        out.update(marking_y0=-1, marking_y1=-1, marking_x0=-1, marking_x1=-1)
    tl = [boxes[k] for k in keep if regions[k] == "top-left"]
    out["topleft_y1"] = max(b[1] for b in tl) if tl else -1
    out["topleft_x1"] = max(b[3] for b in tl) if tl else -1
    return out


def redact_outside_fov(rgb: np.ndarray, fov: np.ndarray, margin: int, block: int) -> np.ndarray:
    """For saved figures only: pixelate everything outside the (dilated) FOV in
    block x block squares, so burned-in text stays visible as a blob but is not legible."""
    near = cv2.dilate(fov.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
    h, w = rgb.shape[:2]
    pix = cv2.resize(cv2.resize(rgb, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA),
                     (w, h), interpolation=cv2.INTER_NEAREST)
    out = rgb.copy()
    out[near == 0] = pix[near == 0]
    return out


def _redacted(cfg: dict, rgb: np.ndarray, fov_source: np.ndarray | None = None) -> np.ndarray:
    """Redact ``rgb`` for a figure; the FOV is estimated from ``fov_source`` (default rgb)."""
    q = cfg["qa"]
    fov = estimate_fov(rgb if fov_source is None else fov_source, q["fov_threshold"])
    return redact_outside_fov(rgb, fov, q["fov_margin_px"], q["redact_block_px"])


# =========================================================================== workers
_CFG: dict | None = None


def _init_worker(cfg: dict) -> None:
    global _CFG
    _CFG = cfg
    cv2.setNumThreads(1)


def _check_one(image_id: str) -> dict:
    cfg, qa = _CFG, _CFG["qa"]
    ip, mp = raw_image_path(cfg, image_id), raw_mask_path(cfg, image_id)
    img, msk = read_unchanged(ip), read_unchanged(mp)
    row = {"image_id": image_id, "image_sha256_file": sha256_file(ip), "mask_sha256_file": sha256_file(mp),
           "img_h": img.shape[0], "img_w": img.shape[1], "img_channels": 1 if img.ndim == 2 else img.shape[2],
           "img_dtype": str(img.dtype), "mask_h": msk.shape[0], "mask_w": msk.shape[1],
           "mask_channels": 1 if msk.ndim == 2 else msk.shape[2], "mask_dtype": str(msk.dtype),
           "mask_values": " ".join(str(v) for v in np.unique(msk)[:10])}
    m2 = msk if msk.ndim == 2 else msk[..., 0]
    row["vessel_fraction"] = float((m2 > 0).mean())
    rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB) if img.ndim == 3 else np.repeat(img[..., None], 3, 2)
    fov = estimate_fov(rgb, qa["fov_threshold"])
    ys, xs = np.nonzero(fov)
    row.update(fov_fraction=float(fov.mean()), fov_mean_gray=float(rgb.max(axis=2)[fov].mean()) if fov.any() else 0.0,
               fov_y0=int(ys.min()) if fov.any() else -1, fov_y1=int(ys.max()) if fov.any() else -1,
               fov_x0=int(xs.min()) if fov.any() else -1, fov_x1=int(xs.max()) if fov.any() else -1)
    row.update(marking_stats(rgb, fov, qa["marking_threshold"], qa["fov_margin_px"], qa["min_marking_px"]))
    near = cv2.dilate(fov.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * qa["fov_margin_px"] + 1,) * 2))
    row["mask_px_outside_fov"] = int(((m2 > 0) & (near == 0)).sum())
    return row


def _preprocess_one(image_id: str) -> str:
    cfg = _CFG
    for res, arr in preprocess_rgb(read_rgb(raw_image_path(cfg, image_id)), cfg).items():
        write_rgb(cache_image_dir(cfg, res) / f"{image_id}.png", arr)
    return image_id


def _mask_targets_one(args) -> tuple:
    """Training targets of one image at every resolution: float16 array + float16 rounding
    error + target mean (equals the 2048 vessel fraction for the area methods)."""
    from src.resample import downsample_mask
    image_id, resolutions, method = args
    gt = read_unchanged(raw_mask_path(_CFG, image_id))
    if gt.ndim != 2:
        raise ValueError(f"{image_id}: mask has shape {gt.shape}")
    gt = gt > 0
    out = {}
    for r in resolutions:
        t32 = downsample_mask(gt, r, method)
        t16 = t32.astype(np.float16)
        out[r] = (t16, float(np.abs(t16.astype(np.float32) - t32).max()), float(t32.astype(np.float64).mean()))
    return image_id, float(gt.mean()), out


def _native_one(image_id: str) -> dict:
    """CLAHE at 2048 -> images_clahe_2048; check that its area resize to 1024 reproduces the
    Phase 1 cache exactly (same CLAHE, same pipeline)."""
    cfg = _CFG
    native = int(cfg["data"]["native_resolution"])
    enhanced = clahe_native(read_rgb(raw_image_path(cfg, image_id)), cfg)
    write_rgb(cache_image_dir(cfg, native) / f"{image_id}.png", enhanced)
    back = read_rgb(cache_image_dir(cfg, native) / f"{image_id}.png")
    ref = read_rgb(cache_image_dir(cfg, 1024) / f"{image_id}.png")
    return {"image_id": image_id, "roundtrip_equal": bool(np.array_equal(back, enhanced)),
            "matches_1024_cache": bool(np.array_equal(resize_area(enhanced, 1024), ref))}


def _sums_one(args) -> tuple:
    path, = args
    x = read_rgb(Path(path)).astype(np.float64) / 255.0
    return x.reshape(-1, 3).sum(0), (x.reshape(-1, 3) ** 2).sum(0), x.shape[0] * x.shape[1]


def reduce_channel_sums(parts) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-channel mean and (population) std over all pixels of all images, from the
    per-image (sum, sum of squares, n_pixels) returned by ``_sums_one``."""
    s = sum(p[0] for p in parts); s2 = sum(p[1] for p in parts); n = sum(p[2] for p in parts)
    mean = s / n
    return mean, np.sqrt(s2 / n - mean ** 2), int(n)


def _workers(n: int | None) -> int:
    return n or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def _pool(cfg: dict, workers: int | None) -> ProcessPoolExecutor:
    """Worker pool started with 'spawn', not the Linux default 'fork'. A forked child
    inherits the parent's OpenCV/BLAS thread-pool locks without the threads, and its
    first parallel call can hang forever (this hung job 19024116 in the pairs step,
    after the parent had used OpenCV with 16 threads to draw the check figures)."""
    import multiprocessing
    return ProcessPoolExecutor(_workers(workers), mp_context=multiprocessing.get_context("spawn"),
                               initializer=_init_worker, initargs=(cfg,))


# =========================================================================== commands
def cmd_extract(cfg: dict, workers: int | None = None) -> None:
    """Write the PNG bytes stored in the parquet files unchanged (no re-encoding)."""
    import pyarrow.parquet as pq

    rows = []
    for split, files in parquet_files(cfg).items():
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=20, columns=["image_id", "image", "mask"]):
                for r in batch.to_pylist():
                    iid = r["image_id"]
                    p = parse_image_id(iid)
                    if p.split != split:
                        raise ValueError(f"{iid} found in {f.name}")
                    for kind, out in (("image", raw_image_path(cfg, iid)), ("mask", raw_mask_path(cfg, iid))):
                        data = r[kind]["bytes"]
                        if data[:8] != b"\x89PNG\r\n\x1a\n":
                            raise ValueError(f"{iid} {kind}: not PNG bytes")
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if not (out.exists() and out.read_bytes() == data):
                            tmp = out.with_name(out.stem + ".tmp")
                            tmp.write_bytes(data)
                            os.replace(tmp, out)
                    rows.append({"image_id": iid, "official_split": split, "category": p.category,
                                 "parquet_file": f.name, "image_path_in_parquet": r["image"]["path"],
                                 "mask_path_in_parquet": r["mask"]["path"],
                                 "image_sha256_parquet": hashlib.sha256(r["image"]["bytes"]).hexdigest(),
                                 "mask_sha256_parquet": hashlib.sha256(r["mask"]["bytes"]).hexdigest(),
                                 "image_bytes": len(r["image"]["bytes"]), "mask_bytes": len(r["mask"]["bytes"])})
    man = pd.DataFrame(rows).sort_values("image_id", key=lambda s: s.map(id_sort_key))
    man.to_csv(fives_dir(cfg) / "manifest.csv", index=False)
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    man.to_csv(CHECKS_DIR / "extraction_manifest.csv", index=False)
    print(f"[extract] wrote {len(man)} image/mask pairs; manifest -> {CHECKS_DIR / 'extraction_manifest.csv'}")
    for split in SPLITS:
        for kind in ("images", "masks"):
            print(f"[extract] {split}/{kind}: {len(list((fives_dir(cfg) / split / kind).glob('*.png')))} PNG files")


def _pixel_compare(cfg: dict, ids: list[str]) -> list[dict]:
    """Decode the parquet bytes with PIL (what Hugging Face `datasets` uses) and the
    extracted file with OpenCV; compare pixel by pixel."""
    import pyarrow.parquet as pq
    from PIL import Image

    want, out = set(ids), []
    for split, files in parquet_files(cfg).items():
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=20, columns=["image_id", "image", "mask"]):
                for r in batch.to_pylist():
                    if r["image_id"] not in want:
                        continue
                    iid = r["image_id"]
                    pi = np.array(Image.open(io.BytesIO(r["image"]["bytes"])))
                    pm = np.array(Image.open(io.BytesIO(r["mask"]["bytes"])))
                    fi = read_rgb(raw_image_path(cfg, iid))
                    fm = read_unchanged(raw_mask_path(cfg, iid))
                    out.append({"image_id": iid, "image_equal": bool(pi.shape == fi.shape and np.array_equal(pi, fi)),
                                "mask_equal": bool(pm.shape == fm.shape and np.array_equal(pm, fm)),
                                "pil_image_mode": Image.open(io.BytesIO(r["image"]["bytes"])).mode,
                                "pil_mask_mode": Image.open(io.BytesIO(r["mask"]["bytes"])).mode})
    return sorted(out, key=lambda d: id_sort_key(d["image_id"]))


def cmd_check(cfg: dict, workers: int | None = None) -> None:
    root = fives_dir(cfg)
    man = pd.read_csv(root / "manifest.csv")
    summary: dict = {}
    print("=" * 70, "\n[check] 1. files per folder, image/mask name pairing")
    names = {}
    for split in SPLITS:
        im = sorted(p.name for p in (root / split / "images").glob("*.png"))
        mk = sorted(p.name for p in (root / split / "masks").glob("*.png"))
        names[split] = im
        summary[f"{split}_images"], summary[f"{split}_masks"] = len(im), len(mk)
        summary[f"{split}_unpaired"] = sorted(set(im) ^ set(mk))
        print(f"  {split}: {len(im)} images, {len(mk)} masks, unpaired: {summary[f'{split}_unpaired'] or 'none'}")
    overlap = set(names["train"]) & set(names["test"])
    summary["names_shared_by_train_and_test"] = sorted(overlap)
    print(f"  file names shared by train and test: {len(overlap)}")

    print("[check] 2. extracted files vs parquet: IDs, stored paths, SHA-256 of every file")
    file_ids = {n[:-4] for s in SPLITS for n in names[s]}
    summary["ids_match_parquet"] = file_ids == set(man["image_id"])
    paths_ok = ((man["image_path_in_parquet"] == man["image_id"] + ".png") &
                (man["mask_path_in_parquet"] == man["image_id"] + ".png")).all()
    summary["parquet_paths_equal_image_id"] = bool(paths_ok)
    print(f"  extracted IDs == parquet IDs: {summary['ids_match_parquet']} ({len(file_ids)} vs {len(man)}); "
          f"parquet path == <image_id>.png for every row: {bool(paths_ok)}")

    print(f"[check] 3. decoding all {len(man)} pairs with {_workers(workers)} workers (shapes, channels, mask values, FOV QA)")
    t0 = time.time()
    ids = sorted(man["image_id"], key=id_sort_key)
    with _pool(cfg, workers) as ex:
        qa = pd.DataFrame(list(ex.map(_check_one, ids, chunksize=4)))
    print(f"  done in {time.time() - t0:.0f}s")
    qa = man[["image_id", "official_split", "category", "image_sha256_parquet", "mask_sha256_parquet"]].merge(qa, on="image_id")
    qa["file_equals_parquet"] = ((qa["image_sha256_file"] == qa["image_sha256_parquet"]) &
                                 (qa["mask_sha256_file"] == qa["mask_sha256_parquet"]))
    summary["files_equal_parquet_sha256"] = int(qa["file_equals_parquet"].sum())
    print(f"  files byte-identical to parquet bytes (SHA-256): {summary['files_equal_parquet_sha256']}/{len(qa)}")
    dup = qa["image_sha256_parquet"].duplicated(keep=False)
    summary["duplicate_image_files"] = qa.loc[dup, "image_id"].tolist()
    print(f"  byte-identical duplicate images (any split): {dup.sum()}")
    for col in ["img_h", "img_w", "img_channels", "img_dtype", "mask_h", "mask_w", "mask_channels", "mask_dtype", "mask_values"]:
        vc = qa[col].value_counts().to_dict()
        summary[f"values_{col}"] = {str(k): int(v) for k, v in vc.items()}
        print(f"  {col}: {vc}")
    print(f"  vessel fraction per mask: min {qa.vessel_fraction.min():.4f}, median {qa.vessel_fraction.median():.4f}, "
          f"max {qa.vessel_fraction.max():.4f}")

    print("[check] 4. category letter and counts")
    qa["category_from_number"] = [
        CATEGORIES[(parse_image_id(i).number - 1) // (150 if s == "train" else 50)]
        for i, s in zip(qa["image_id"], qa["official_split"])]
    summary["category_letter_matches_number_block"] = bool((qa["category"] == qa["category_from_number"]).all())
    ct = pd.crosstab(qa["official_split"], qa["category"], margins=True)
    summary["category_counts"] = {s: {c: int(ct.loc[s, c]) for c in ct.columns} for s in ct.index}
    print(ct.to_string())
    print(f"  letter agrees with number block (train 1-150 A, 151-300 D, ...; test 1-50 A, ...): "
          f"{summary['category_letter_matches_number_block']}")

    print("[check] 5. pixel-by-pixel: 12 random pairs (seed 0), PIL decode of parquet bytes vs OpenCV decode of file")
    rng = np.random.RandomState(0)
    sample = sorted(rng.choice(ids, 12, replace=False).tolist(), key=id_sort_key)
    px = _pixel_compare(cfg, sample)
    for d in px:
        print(f"  {d['image_id']}: image equal {d['image_equal']}, mask equal {d['mask_equal']} "
              f"(PIL modes {d['pil_image_mode']}/{d['pil_mask_mode']})")
    summary["pixel_compare"] = px

    print("[check] 6. FOV and markings outside the eye circle")
    q = cfg["qa"]
    qa["has_marking"] = qa["marking_n_regions"] > 0
    qa["marking_regions"] = qa["marking_regions"].fillna("")
    qa["fov_touches_border"] = (qa.fov_y0 <= 0) | (qa.fov_x0 <= 0) | (qa.fov_y1 >= qa.img_h - 1) | (qa.fov_x1 >= qa.img_w - 1)
    summary["markings"] = {
        "rule": f"pixels with max(R,G,B) > {q['marking_threshold']} more than {q['fov_margin_px']} px outside the "
                f"estimated FOV, merged into label regions (41 px closing); a region with >= {q['min_marking_px']} "
                "such pixels is a marking, smaller ones are specks",
        "images_with_specks_only": int(((qa.speck_px > 0) & ~qa.has_marking).sum()),
        "n_flagged": int(qa.has_marking.sum()),
        "by_split_category": {f"{s}_{c}": int(v) for (s, c), v in qa.groupby(["official_split", "category"]).has_marking.sum().items()},
        "region_sets": {k: int(v) for k, v in qa.loc[qa.has_marking, "marking_regions"].value_counts().items()},
        "regions_per_image": {str(k): int(v) for k, v in qa.loc[qa.has_marking, "marking_n_regions"].value_counts().sort_index().items()},
        "test_101_G": qa.set_index("image_id").loc["test_101_G", ["marking_px"]].to_dict()
                      if "test_101_G" in set(qa.image_id) else None,
    }
    summary["fov"] = {"fraction_min": float(qa.fov_fraction.min()), "fraction_median": float(qa.fov_fraction.median()),
                      "fraction_max": float(qa.fov_fraction.max()), "touches_border": int(qa.fov_touches_border.sum()),
                      "mean_gray_min": float(qa.fov_mean_gray.min()), "mean_gray_median": float(qa.fov_mean_gray.median()),
                      "mask_px_outside_fov_images": int((qa.mask_px_outside_fov > 0).sum())}
    print(json.dumps({"markings": summary["markings"], "fov": summary["fov"]}, indent=1, default=str))
    print("  distribution of marking_px:", qa.marking_px.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_dict())

    qa = qa.drop(columns=["category_from_number"])
    write_qa(qa)
    with open(CHECKS_DIR / "phase1_checks.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    _marking_sheets(cfg, qa)
    print(f"[check] wrote {CHECKS_DIR / 'image_qa.csv'}, {PRIVATE_DIR / 'marking_locations.csv'} (not public) "
          f"and {CHECKS_DIR / 'phase1_checks.json'}")


def write_qa(qa: pd.DataFrame) -> None:
    """image_qa.csv without per-image marking locations (public); the locations go to the
    git-ignored results/checks/private/ (decision D8: no marking locations or crops are published)."""
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    qa.drop(columns=MARKING_LOCATION_COLS).to_csv(CHECKS_DIR / "image_qa.csv", index=False)
    qa[["image_id"] + MARKING_LOCATION_COLS].to_csv(PRIVATE_DIR / "marking_locations.csv", index=False)


def load_qa(with_locations: bool = False) -> pd.DataFrame:
    qa = pd.read_csv(CHECKS_DIR / "image_qa.csv")
    if with_locations:
        qa = qa.merge(pd.read_csv(PRIVATE_DIR / "marking_locations.csv"), on="image_id", how="left", validate="1:1")
    return qa


def _marking_sheets(cfg: dict, qa: pd.DataFrame) -> None:
    """Contact sheets for reviewing the marking detector. Markings may be fragments of
    patient names, so they are never shown legibly: everything outside the FOV is
    pixelated (redact_outside_fov); the detected marking box is outlined in magenta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q, size = cfg["qa"], 384
    rng = np.random.RandomState(0)

    def sheet(rows_df, title, fname, box):
        if not len(rows_df):
            return
        pick = rows_df.iloc[np.sort(rng.choice(len(rows_df), min(24, len(rows_df)), replace=False))]
        fig, axes = plt.subplots(3, 8, figsize=(17, 7), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, r) in zip(axes.flat, pick.iterrows()):
            rgb = read_rgb(raw_image_path(cfg, r.image_id))
            fov = estimate_fov(rgb, q["fov_threshold"])
            small = resize_area(redact_outside_fov(rgb, fov, q["fov_margin_px"], q["redact_block_px"]), size).copy()
            cnts, _ = cv2.findContours(cv2.resize(fov.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(small, cnts, -1, (0, 255, 255), 1)
            if box and r.marking_px:
                f = size / 2048
                cv2.rectangle(small, (int(r.marking_x0 * f) - 3, int(r.marking_y0 * f) - 3),
                              (int(r.marking_x1 * f) + 3, int(r.marking_y1 * f) + 3), (255, 0, 255), 1)
            ax.imshow(small)
            ax.set_title(f"{r.image_id} ({r.marking_px} px, max outside {r.outside_max})", fontsize=6)
        fig.suptitle(f"{title}: {len(pick)} of {len(rows_df)} (random, seed 0). Cyan: rough FOV estimate of this check; "
                     f"magenta: detected marking box. Outside the FOV the image is pixelated ({q['redact_block_px']} px "
                     "blocks) so text is not legible.", fontsize=8)
        PRIVATE_DIR.mkdir(parents=True, exist_ok=True)              # marking boxes: never published (D8)
        fig.tight_layout(); fig.savefig(PRIVATE_DIR / fname, dpi=110); plt.close(fig)

    order = qa.sort_values("image_id", key=lambda s: s.map(id_sort_key))
    sheet(order[order.has_marking], "Flagged (marking detected)", "markings_flagged_sample.png", True)
    sheet(order[~order.has_marking], "Not flagged (look for misses)", "markings_unflagged_sample.png", True)


SPLIT_ORDER = ["train", "val", "test", "excluded"]
SPLIT_COMBOS = ["train-train", "val-val", "test-test", "train-val", "train-test", "val-test"]


def split_combo(split_a: str, split_b: str) -> str:
    """'train-val' etc.; pairs with an excluded image give e.g. 'train-excluded'."""
    return "-".join(sorted([split_a, split_b], key=SPLIT_ORDER.index))


def marking_group_sizes(pairs: pd.DataFrame, t: float) -> dict[str, int]:
    """image_id -> size of its identical-marking group (pairs with IoU >= t, merged).
    Groups of exactly 2 are the 'isolated pairs' (likely same patient)."""
    sel = pairs[pairs.iou >= t]
    ids = sorted(set(sel.image_a) | set(sel.image_b))
    ix = {i: k for k, i in enumerate(ids)}
    comps = _components(len(ids), [(ix[a], ix[b]) for a, b in zip(sel.image_a, sel.image_b)])
    return {ids[i]: len(c) for c in comps for i in c}


def split_groups(cfg: dict) -> pd.DataFrame:
    """Groups that must not be separated by the train/val split (official train images):
    byte-identical duplicate images; image pairs with an identical burned-in marking AND
    consecutive numbers; and isolated identical-marking pairs (a marking group of exactly
    2 images over all 800). The last two are likely both eyes of one patient.
    Columns: image_a, image_b, reason."""
    qa = pd.read_csv(CHECKS_DIR / "image_qa.csv")
    pairs = pd.read_csv(CHECKS_DIR / "marking_pairs.csv")
    t = cfg["data"]["split"]["group_marking_iou"]
    rows = []
    for _, g in qa[qa.image_sha256_parquet.duplicated(keep=False)].groupby("image_sha256_parquet"):
        ids = sorted(g.image_id, key=id_sort_key)
        rows += [(ids[0], b, "identical image file") for b in ids[1:]]
    adj = pairs[(pairs.iou >= t) & pairs.adjacent_numbers]
    rows += [(a, b, f"identical marking (IoU >= {t}) + adjacent numbers") for a, b in zip(adj.image_a, adj.image_b)]
    gsize = marking_group_sizes(pairs, t)
    iso = pairs[(pairs.iou >= t) & (pairs.image_a.map(gsize) == 2)]
    rows += [(a, b, f"identical marking (IoU >= {t}), isolated pair") for a, b in zip(iso.image_a, iso.image_b)]
    df = pd.DataFrame(rows, columns=["image_a", "image_b", "reason"])
    return df[df.image_a.str.startswith("train_") & df.image_b.str.startswith("train_")]


def cmd_split(cfg: dict, workers: int | None = None) -> None:
    man = pd.read_csv(fives_dir(cfg) / "manifest.csv")
    qa = pd.read_csv(CHECKS_DIR / "image_qa.csv")
    s = cfg["data"]["split"]
    exclude = dict(s.get("exclude") or {})
    empty_train = set(qa.loc[(qa.vessel_fraction == 0) & (qa.official_split == "train"), "image_id"])
    if not empty_train <= set(exclude):
        raise ValueError(f"empty training masks not excluded in the config: {sorted(empty_train - set(exclude))}")
    gp = split_groups(cfg)
    train_ids = man.loc[man.official_split == "train", "image_id"].tolist()
    df = make_splits(train_ids, man.loc[man.official_split == "test", "image_id"].tolist(), s["val_fraction"],
                     s["seed"], groups=[(a, b) for a, b in zip(gp.image_a, gp.image_b)], exclude=exclude)
    out = project_path(cfg["paths"]["splits_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # group file (one row per grouped image) and the checks
    pool = [i for i in sorted(train_ids, key=id_sort_key) if i not in exclude]
    comps = [c for c in _groups_to_components(pool, [(a, b) for a, b in zip(gp.image_a, gp.image_b)]) if len(c) > 1]
    lab = df.set_index("image_id").split
    reasons = {}
    for a, b, r in gp.itertuples(index=False):
        for i in (a, b):
            reasons.setdefault(i, set()).add(r)
    grp = pd.DataFrame([{"image_id": i, "group": f"g{k:03d}", "group_size": len(c), "split": lab[i],
                         "reasons": "; ".join(sorted(reasons.get(i, [])))} for k, c in enumerate(comps) for i in c])
    grp.to_csv(out.parent / "split_groups.csv", index=False)
    crossing = [c for c in comps if len({lab[i] for i in c}) > 1]
    ct = pd.crosstab(df.split, df.category, margins=True)
    summary = {"splits_csv_sha256": sha256_file(out), "seed": s["seed"], "val_fraction": s["val_fraction"],
               "counts": {sp_: {c: int(ct.loc[sp_, c]) for c in ct.columns} for sp_ in ct.index},
               "excluded": exclude,
               "group_pairs": {r: int(v) for r, v in gp.reason.value_counts().items()},
               "groups": int(len(comps)), "group_sizes": {str(k): int(v) for k, v in
                                                          pd.Series([len(c) for c in comps]).value_counts().sort_index().items()},
               "grouped_images_by_split": {k: int(v) for k, v in grp.split.value_counts().items()},
               "groups_crossing_train_val": int(len(crossing))}
    with open(CHECKS_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[split] wrote {out} ({len(df)} rows), sha256 {summary['splits_csv_sha256'][:16]}; {out.parent / 'split_groups.csv'}")
    print(ct.to_string())
    print(json.dumps({k: summary[k] for k in ("excluded", "group_pairs", "groups", "group_sizes",
                                             "grouped_images_by_split", "groups_crossing_train_val")}, indent=1))
    if crossing:
        raise RuntimeError(f"{len(crossing)} groups cross train/val: {crossing[:3]}")


def _marking_pattern(args) -> np.ndarray:
    """Binary marking pattern in the top-left window, on a grid x grid array (kept in
    memory only; never written to disk)."""
    image_id, window = args
    cfg, q = _CFG, _CFG["qa"]
    rgb = read_rgb(raw_image_path(cfg, image_id))
    fov = estimate_fov(rgb, q["fov_threshold"])
    near = cv2.dilate(fov.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * q["fov_margin_px"] + 1,) * 2))
    b = ((rgb[:window, :window].max(2) > q["pairs"]["binarize_threshold"]) & (near[:window, :window] == 0)).astype(np.float32)
    g = q["pairs"]["grid"]
    return cv2.resize(b, (g, g), interpolation=cv2.INTER_AREA) > 0.25


def pairwise_shift_iou(patterns: np.ndarray, max_shift: int) -> np.ndarray:
    """patterns: N x G x G bool. Returns the N x N matrix of IoU between pattern pairs,
    maximized over integer shifts of up to ``max_shift`` cells (no wrap-around)."""
    n, g, _ = patterns.shape
    base = patterns.reshape(n, -1).astype(np.float32)
    area = base.sum(1)
    best = np.zeros((n, n), np.float32)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            s = np.zeros_like(patterns)
            s[:, max(dy, 0):g + min(dy, 0), max(dx, 0):g + min(dx, 0)] = \
                patterns[:, max(-dy, 0):g + min(-dy, 0), max(-dx, 0):g + min(-dx, 0)]
            s = s.reshape(n, -1).astype(np.float32)
            inter = s @ base.T
            union = s.sum(1)[:, None] + area[None, :] - inter
            best = np.maximum(best, np.where(union > 0, inter / np.maximum(union, 1), 0))
    return np.maximum(best, best.T)


def _components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def cmd_pairs(cfg: dict, workers: int | None = None) -> None:
    """Likely same-patient pairs: images whose top-left burned-in markings are identical
    (shift-tolerant IoU of binary patterns). Only counts, IDs and scores are saved; the
    text is never read, transcribed or stored. Caveats: an identical fragment can occur
    for different patients, and images without a marking cannot be matched."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = cfg["qa"]["pairs"]
    qa = load_qa(with_locations=True)
    tl = qa[qa.topleft_y1 >= 0].sort_values("image_id", key=lambda s: s.map(id_sort_key))
    ids = tl.image_id.tolist()
    window = int(min(512, np.ceil((max(tl.topleft_y1.max(), tl.topleft_x1.max()) + 16) / 64) * 64))
    print(f"[pairs] {len(ids)} images with a top-left marking ({int(qa.has_marking.sum())} with any marking, "
          f"{len(qa)} total); window {window}x{window} px -> {q['grid']}x{q['grid']} grid, shift +-{q['max_shift']}")
    with _pool(cfg, workers) as ex:
        pats = np.stack(list(ex.map(_marking_pattern, [(i, window) for i in ids], chunksize=4)))
    iou = pairwise_shift_iou(pats, int(q["max_shift"]))
    del pats
    iu = np.triu_indices(len(ids), 1)
    scores = iou[iu]

    parsed = [parse_image_id(i) for i in ids]
    rows = []
    for a, b in zip(*iu):
        s = float(iou[a, b])
        if s >= min(q["thresholds"]) - 0.2:
            pa, pb = parsed[a], parsed[b]
            rows.append({"image_a": ids[a], "image_b": ids[b], "iou": round(s, 4),
                         "same_category": pa.category == pb.category,
                         "adjacent_numbers": pa.split == pb.split and abs(pa.number - pb.number) == 1})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(CHECKS_DIR / "marking_pairs.csv", index=False)

    meta = {"method": f"binary pattern max(R,G,B) > {q['binarize_threshold']} outside the FOV in the top-left "
                      f"{window}x{window} px, on a {q['grid']}x{q['grid']} grid; IoU maximized over shifts of "
                      f"+-{q['max_shift']} cells (patterns kept in memory only)",
            "n_images_compared": len(ids), "n_images_without_top_left_marking": int(len(qa) - len(ids)),
            "n_pairs_compared": int(len(scores)),
            "iou_quantiles_all_pairs": {str(p): round(float(np.quantile(scores, p)), 4) for p in (0.5, 0.9, 0.99, 0.999)}}
    with open(CHECKS_DIR / "marking_pairs_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("[pairs] split counts: run the pairs_summary step (after split)")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(scores, bins=np.linspace(0, 1, 51), log=True, color="0.4")
    for t in q["thresholds"]:
        ax.axvline(t, ls="--", lw=0.8, color="tab:red")
    ax.set_xlabel("marking-pattern IoU (shift-tolerant)"); ax.set_ylabel("pairs (log scale)")
    ax.set_title(f"All {len(scores)} pairs of the {len(ids)} images with a top-left marking", fontsize=9)
    fig.tight_layout(); fig.savefig(CHECKS_DIR / "marking_pair_iou_hist.png", dpi=110); plt.close(fig)
    print(f"[pairs] wrote marking_pairs.csv (pairs with IoU >= {min(q['thresholds']) - 0.2:.1f}), "
          "same_patient_pairs.json, marking_pair_iou_hist.png")


def cmd_pairs_summary(cfg: dict, workers: int | None = None) -> None:
    """Counts of likely same-patient pairs by split, from marking_pairs.csv (IDs and
    scores only; no image is read). Split membership is taken from the current
    splits.csv. Three levels, for each IoU threshold:
      identical_marking  all pairs with IoU >= t. Upper bound: a fragment of a common
                         character also matches between different patients.
      isolated_pairs     pairs whose identical-marking group has exactly 2 images
                         ("likely same patient": mostly adjacent numbers, same category).
      adjacent_numbers   identical marking AND consecutive image numbers in the same
                         official split (strongest evidence; adjacency is undefined
                         across the official train/test split)."""
    q = cfg["qa"]["pairs"]
    pairs = pd.read_csv(CHECKS_DIR / "marking_pairs.csv")
    meta = json.loads((CHECKS_DIR / "marking_pairs_meta.json").read_text())
    split = load_splits(cfg).set_index("image_id")["split"]
    combos = SPLIT_COMBOS
    pairs["combo"] = [split_combo(split[a], split[b]) for a, b in zip(pairs.image_a, pairs.image_b)]

    def counts(sel: pd.DataFrame) -> dict:
        by = sel.combo.value_counts().to_dict()
        return {"n_pairs": int(len(sel)),
                "within_one_split": int(sum(by.get(c, 0) for c in combos[:3])),
                "across_splits": int(sum(by.get(c, 0) for c in combos[3:])),
                "by_split": {c: int(by.get(c, 0)) for c in combos},
                "involving_excluded": int(sum(v for c, v in by.items() if "excluded" in c)),
                "same_category": int(sel.same_category.sum()), "adjacent_numbers": int(sel.adjacent_numbers.sum()),
                "images_involved": int(len(set(sel.image_a) | set(sel.image_b)))}

    out = {**meta, "split_file": cfg["paths"]["splits_csv"], "levels": {
        "identical_marking": "all pairs with IoU >= t (upper bound; common characters also match across patients)",
        "isolated_pairs": "identical-marking groups of exactly 2 images (likely same patient)",
        "adjacent_numbers": "identical marking and consecutive numbers in the same official split"},
        "by_threshold": {}}
    for t in q["thresholds"]:
        sel = pairs[pairs.iou >= t]
        gsize = marking_group_sizes(pairs, t)
        n_groups = {k: v // k for k, v in pd.Series(list(gsize.values())).value_counts().items()}
        sizes = pd.Series(n_groups).sort_index()
        out["by_threshold"][str(t)] = {
            "identical_marking": {**counts(sel), "groups": int(sum(n_groups.values())),
                                  "group_sizes": {str(k): int(v) for k, v in sizes.items()}},
            "isolated_pairs": counts(sel[sel.image_a.map(gsize) == 2]),
            "adjacent_numbers": counts(sel[sel.adjacent_numbers])}
    with open(CHECKS_DIR / "same_patient_pairs.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["by_threshold"], indent=1))
    print(f"[pairs_summary] wrote {CHECKS_DIR / 'same_patient_pairs.json'}")


def _mask_signature(image_id: str) -> np.ndarray:
    """Coarse vessel-layout signature: 'any vessel' in 16x16 blocks -> 128x128, dilated by
    one block (tolerates small shifts). Masks only; nothing from the image is used."""
    m = read_unchanged(raw_mask_path(_CFG, image_id)) > 0
    s = m.reshape(128, 16, 128, 16).any(axis=(1, 3)).astype(np.uint8)
    return cv2.dilate(s, np.ones((3, 3), np.uint8)) > 0


def _gt_outside_fov(image_id: str) -> dict:
    """Where the ground truth lies outside the rough QA FOV (+margin): on the image
    border (annotation artifact) or in a dim part of the eye that the rough FOV missed."""
    cfg, q = _CFG, _CFG["qa"]
    rgb = read_rgb(raw_image_path(cfg, image_id))
    fov = estimate_fov(rgb, q["fov_threshold"])
    near = cv2.dilate(fov.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * q["fov_margin_px"] + 1,) * 2)) > 0
    gt_out = (read_unchanged(raw_mask_path(cfg, image_id)) > 0) & ~near
    ys, xs = np.nonzero(gt_out)
    edge = (ys < 2) | (ys > 2045) | (xs < 2) | (xs > 2045)
    g = rgb.max(axis=2)[gt_out]
    sides = [n for n, sel in (("top", ys < 2), ("bottom", ys > 2045), ("left", xs < 2), ("right", xs > 2045)) if sel.any()]
    return {"image_id": image_id, "gt_px_outside_fov": int(gt_out.sum()), "share_on_image_border": round(float(edge.mean()), 4),
            "border_sides": ";".join(sides), "image_gray_median_there": float(np.median(g)), "image_gray_max_there": int(g.max()),
            "kind": "border_line" if edge.mean() >= 0.95 else ("dim_eye_rim" if np.median(g) >= 8 else "other")}


def cmd_integrity(cfg: dict, workers: int | None = None) -> None:
    """Data-integrity checks that matter for leakage and labels (results/checks/integrity.json):
    byte-identical images (and how much their masks agree), identical masks on different
    images, empty masks, ground truth outside the eye, and near-duplicate images found by
    vessel-layout similarity of their masks (same eye photographed twice)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qa = pd.read_csv(CHECKS_DIR / "image_qa.csv")
    split = load_splits(cfg).set_index("image_id")["split"]
    combo = lambda a, b: split_combo(split[a], split[b])
    mask = lambda i: read_unchanged(raw_mask_path(cfg, i)) > 0
    out: dict = {}

    groups = []
    for _, g in qa[qa.image_sha256_parquet.duplicated(keep=False)].groupby("image_sha256_parquet"):
        ids = sorted(g.image_id, key=id_sort_key)
        ms = [mask(i) for i in ids]
        d = 2 * (ms[0] & ms[1]).sum() / max(ms[0].sum() + ms[1].sum(), 1) if len(ids) == 2 else None
        groups.append({"images": ids, "splits": [split[i] for i in ids], "masks_identical": bool(g.mask_sha256_parquet.nunique() == 1),
                       "mask_dice": None if d is None else round(float(d), 4)})
    out["identical_images"] = {"n_groups": len(groups), "n_images": int(sum(len(x["images"]) for x in groups)),
                               "groups_across_splits": int(sum(len(set(x["splits"])) > 1 for x in groups)), "groups": groups}

    groups = []
    for _, g in qa[qa.mask_sha256_parquet.duplicated(keep=False)].groupby("mask_sha256_parquet"):
        ids = sorted(g.image_id, key=id_sort_key)
        im = [read_rgb(raw_image_path(cfg, i)).astype(np.int16) for i in ids[:2]]
        groups.append({"images": ids, "splits": [split[i] for i in ids], "images_identical": bool(g.image_sha256_parquet.nunique() == 1),
                       "mean_abs_image_difference_0_255": round(float(np.abs(im[0] - im[1]).mean()), 2),
                       "vessel_fraction": round(float(g.vessel_fraction.iloc[0]), 4)})
    out["identical_masks_on_different_images"] = groups
    out["empty_masks"] = [{"image": i, "split": split[i]} for i in qa.loc[qa.vessel_fraction == 0, "image_id"]]

    with _pool(cfg, workers) as ex:
        gto = pd.DataFrame(list(ex.map(_gt_outside_fov, qa.loc[qa.mask_px_outside_fov > 0, "image_id"].tolist())))
        sigs = np.stack(list(ex.map(_mask_signature, qa.image_id.tolist(), chunksize=8)))
    gto["split"] = gto.image_id.map(split)
    gto.to_csv(CHECKS_DIR / "gt_outside_fov.csv", index=False)
    out["gt_outside_fov"] = {"n_images": int(len(gto)), "by_kind": {k: int(v) for k, v in gto.kind.value_counts().items()},
                             "images_by_kind": {k: g.image_id.tolist() for k, g in gto.groupby("kind")},
                             "px_by_kind": {k: int(v) for k, v in gto.groupby("kind").gt_px_outside_fov.sum().items()}}

    ids = qa.image_id.tolist()
    f = sigs.reshape(len(ids), -1).astype(np.float32)
    inter = f @ f.T
    area = f.sum(1)
    iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1)
    iu = np.triu_indices(len(ids), 1)
    vals = iou[iu]
    rows = [{"image_a": ids[a], "image_b": ids[b], "mask_layout_iou": round(float(iou[a, b]), 4), "splits": combo(ids[a], ids[b])}
            for a, b in zip(*iu) if iou[a, b] >= 0.6]
    sim = pd.DataFrame(rows, columns=["image_a", "image_b", "mask_layout_iou", "splits"]).sort_values("mask_layout_iou", ascending=False)
    sim.to_csv(CHECKS_DIR / "mask_similar_pairs.csv", index=False)
    combos = SPLIT_COMBOS + ["involving_excluded"]
    sim["c"] = ["involving_excluded" if "excluded" in x else x for x in sim.splits]
    out["near_duplicates_by_mask_layout"] = {
        "method": "masks as 'any vessel' in 16x16 blocks (128x128), dilated 1 block; IoU over all image pairs",
        "n_pairs": int(len(vals)), "iou_quantiles": {str(p): round(float(np.quantile(vals, p)), 4) for p in (0.5, 0.99, 0.999, 1.0)},
        "by_threshold": {str(t): {c: int(((sim.mask_layout_iou >= t) & (sim.c == c)).sum()) for c in combos}
                         for t in (0.7, 0.8, 0.9)}}
    with open(CHECKS_DIR / "integrity.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(vals, bins=np.linspace(0, 1, 51), log=True, color="0.4")
    ax.set_xlabel("vessel-layout IoU of the two masks (coarse, shift-tolerant)"); ax.set_ylabel("pairs (log scale)")
    ax.set_title(f"All {len(vals)} image pairs", fontsize=9)
    fig.tight_layout(); fig.savefig(CHECKS_DIR / "mask_layout_iou_hist.png", dpi=110); plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "gt_outside_fov"} | {"gt_outside_fov": {
        k: out["gt_outside_fov"][k] for k in ("n_images", "by_kind", "px_by_kind")}}, indent=1, default=str))
    print(f"[integrity] wrote integrity.json, gt_outside_fov.csv, mask_similar_pairs.csv, mask_layout_iou_hist.png")


def cmd_preprocess(cfg: dict, workers: int | None = None) -> None:
    man = pd.read_csv(fives_dir(cfg) / "manifest.csv")
    ids = sorted(man["image_id"], key=id_sort_key)
    res = cache_resolutions(cfg)
    t0 = time.time()
    with _pool(cfg, workers) as ex:
        done = list(ex.map(_preprocess_one, ids, chunksize=4))
    params = {"source": "data/FIVES/<split>/images/<image_id>.png (2048x2048 RGB)",
              "clahe": cfg["data"]["clahe"], "resize": cfg["data"]["image_resize"],
              "format": "8-bit RGB PNG, before normalization", "n_images": len(done),
              "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for r in res:
        d = cache_image_dir(cfg, r)
        with open(d / "_params.json", "w") as f:
            json.dump({**params, "resolution": r}, f, indent=1)
        files = sorted(d.glob("*.png"))
        a = read_rgb(files[0])
        print(f"[preprocess] {d}: {len(files)} files, e.g. {files[0].name} {a.shape} {a.dtype}")
    print(f"[preprocess] {len(done)} images in {time.time() - t0:.0f}s")


def cmd_stats(cfg: dict, workers: int | None = None) -> None:
    sp = load_splits(cfg)
    train_ids = sp.loc[sp.split == "train", "image_id"].tolist()
    assert train_ids and not set(train_ids) & set(sp.loc[sp.split != "train", "image_id"])
    print(f"[stats] {len(train_ids)} training images (split == 'train')")
    stats = {"computed_on": f"train split only ({len(train_ids)} images) of {cfg['paths']['splits_csv']} "
                            f"(sha256 {sha256_file(project_path(cfg['paths']['splits_csv']))[:16]})",
             "input": "CLAHE-processed cached PNGs, scaled to [0, 1], all pixels (background included)",
             "channel_order": "RGB"}
    for r in cache_resolutions(cfg):
        d = cache_image_dir(cfg, r)
        with _pool(cfg, workers) as ex:
            parts = list(ex.map(_sums_one, [(str(d / f"{i}.png"),) for i in train_ids], chunksize=8))
        mean, std, n = reduce_channel_sums(parts)
        stats[str(r)] = {"mean": [round(float(v), 6) for v in mean], "std": [round(float(v), 6) for v in std],
                         "n_pixels": int(n)}
        print(f"[stats] {r}: mean {stats[str(r)]['mean']} std {stats[str(r)]['std']}")
    with open(norm_stats_path(cfg), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"[stats] wrote {norm_stats_path(cfg)}")


def cmd_masks(cfg: dict, workers: int | None = None) -> None:
    """Phase 2 training-target cache: the train and val masks (never test, never excluded
    images), resized from the original 2048 masks with ``data.mask_resize`` to every cache
    resolution, float16, one compressed archive per resolution (``mask_cache_path``)."""
    method = cfg["data"]["mask_resize"]
    dtype = cfg["data"].get("mask_cache_dtype", "float16")
    if dtype != "float16":
        raise ValueError("mask cache is float16")
    sp = load_splits(cfg)
    ids = sp.loc[sp.split.isin(["train", "val"]), "image_id"].tolist()
    counts = sp.split.value_counts().to_dict()
    assert not set(ids) & set(sp.loc[sp.split.isin(["test", "excluded"]), "image_id"])
    res = cache_resolutions(cfg)
    print(f"[masks] method {method!r}, {dtype}, resolutions {res}; {len(ids)} images "
          f"(train {counts.get('train', 0)} + val {counts.get('val', 0)}; "
          f"not cached: test {counts.get('test', 0)}, excluded {counts.get('excluded', 0)})")
    t0 = time.time()
    with _pool(cfg, workers) as ex:
        results = list(ex.map(_mask_targets_one, [(i, res, method) for i in ids], chunksize=4))
    info = {"method": method, "dtype": dtype, "source": "data/FIVES/train/masks/<image_id>.png (2048, original)",
            "images": len(ids), "splits": {"train": counts.get("train", 0), "val": counts.get("val", 0)},
            "splits_csv_sha256": sha256_file(project_path(cfg["paths"]["splits_csv"]))[:16],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "resolutions": {}}
    gt_frac = np.array([r[1] for r in results])
    for r in res:
        arrays = {i: out[r][0] for i, _, out in results}
        err = max(out[r][1] for _, _, out in results)
        tmean = np.array([out[r][2] for _, _, out in results])
        path = mask_cache_path(cfg, r)
        write_npz_archive(path, arrays)
        # read back from disk and check every member
        keys = npz_keys(path)
        assert keys == ids, "archive keys differ from the train + val ids"
        vals, lo, hi = set(), 1.0, 0.0
        with np.load(path) as z:
            for i in keys:
                a = z[i]
                assert a.dtype == np.float16 and a.shape == (r, r), (i, a.dtype, a.shape)
                assert np.array_equal(a, arrays[i]), i
                lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
                if len(vals) < 1000:
                    vals |= set(np.unique(a).tolist())
        frac_soft = float(np.mean([((arrays[i] > 0) & (arrays[i] < 1)).mean() for i in ids]))
        info["resolutions"][str(r)] = {
            "file": path.name, "size_mb": round(path.stat().st_size / 1e6, 1), "shape": [r, r],
            "value_range": [lo, hi], "distinct_values": len(vals),
            "max_float16_rounding_error": err,
            "max_abs_target_mean_minus_2048_vessel_fraction": float(np.abs(tmean - gt_frac).max()),
            "mean_share_of_pixels_strictly_between_0_and_1": round(frac_soft, 5)}
        print(f"[masks] {path}: {len(keys)} members, {r}x{r} float16, range [{lo}, {hi}], "
              f"{len(vals)} distinct values, float16 rounding error {err}, "
              f"|target mean - 2048 vessel fraction| max {info['resolutions'][str(r)]['max_abs_target_mean_minus_2048_vessel_fraction']:.2e}, "
              f"{path.stat().st_size / 1e6:.1f} MB")
    with open(project_path(cfg["paths"]["cache_dir"]) / f"masks_{method}.json", "w") as f:
        json.dump(info, f, indent=1)
    with open(CHECKS_DIR / "mask_cache.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"[masks] {len(ids)} images in {time.time() - t0:.0f}s; wrote masks_{method}.json and results/checks/mask_cache.json")


def cmd_native(cfg: dict, workers: int | None = None) -> None:
    """Phase 5 (tiled SAM): CLAHE-processed images at native 2048, 8-bit RGB PNG, all 800 images
    (the test images are preprocessed like every other cache; nothing is learned from them)."""
    man = pd.read_csv(fives_dir(cfg) / "manifest.csv")
    ids = sorted(man["image_id"], key=id_sort_key)
    t0 = time.time()
    with _pool(cfg, workers) as ex:
        rows = list(ex.map(_native_one, ids, chunksize=4))
    df = pd.DataFrame(rows)
    d = cache_image_dir(cfg, int(cfg["data"]["native_resolution"]))
    info = {"source": "data/FIVES/<split>/images/<image_id>.png (2048x2048 RGB)", "clahe": cfg["data"]["clahe"],
            "format": "8-bit RGB PNG, CLAHE at native resolution, no resize, before normalization",
            "n_images": len(df), "all_roundtrip_equal": bool(df.roundtrip_equal.all()),
            "all_match_1024_cache_after_area_resize": bool(df.matches_1024_cache.all()),
            "mismatches": df.loc[~df.matches_1024_cache, "image_id"].tolist(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(d / "_params.json", "w") as f:
        json.dump(info, f, indent=1)
    with open(CHECKS_DIR / "native_cache.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"[native] {len(df)} images in {time.time() - t0:.0f}s -> {d}; round trip equal: {info['all_roundtrip_equal']}; "
          f"area resize to 1024 equals the Phase 1 cache: {info['all_match_1024_cache_after_area_resize']} "
          f"({len(info['mismatches'])} mismatches)")


def cmd_visual(cfg: dict, workers: int | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    sp = load_splits(cfg)
    image_id = np.random.RandomState(0).choice(sp.loc[sp.split == "train", "image_id"].tolist())
    rgb = read_rgb(raw_image_path(cfg, image_id))
    m = read_unchanged(raw_mask_path(cfg, image_id)) > 0
    c = cfg["data"]["clahe"]
    enh = clahe_rgb(rgb, c["clip_limit"], c["tile_grid"])          # the real pipeline: unredacted input
    show_rgb, show_enh = _redacted(cfg, rgb), _redacted(cfg, enh, fov_source=rgb)   # figures only
    # thin-vessel-rich 256 window (for the zooms): most skeleton pixels with local radius < 3
    skel = skeletonize(m)
    radius = ndi.distance_transform_edt(m)
    thin = (skel & (radius < 3)).astype(np.float64)
    win = 256
    ii = thin.cumsum(0).cumsum(1)
    best, by, bx = -1, 0, 0
    for y in range(0, 2048 - win + 1, 64):
        for x in range(0, 2048 - win + 1, 64):
            v = ii[y + win - 1, x + win - 1] - (ii[y - 1, x + win - 1] if y else 0) - (ii[y + win - 1, x - 1] if x else 0) + (ii[y - 1, x - 1] if x and y else 0)
            if v > best:
                best, by, bx = v, y, x
    sl = (slice(by, by + win), slice(bx, bx + win))

    # 1. before / after CLAHE
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(resize_area(show_rgb, 1024)); ax[0, 0].set_title(f"{image_id}: original (2048, shown at 1024)")
    ax[0, 1].imshow(resize_area(show_enh, 1024)); ax[0, 1].set_title(f"CLAHE on L (clip {c['clip_limit']}, tiles {c['tile_grid']}) at 2048")
    ax[1, 0].imshow(show_rgb[sl]); ax[1, 0].set_title(f"original, native crop y{by} x{bx}")
    ax[1, 1].imshow(show_enh[sl]); ax[1, 1].set_title("CLAHE, same crop")
    for a in ax.flat:
        a.axis("off")
    fig.suptitle("Outside the FOV the image is pixelated in this figure (burned-in text not legible); CLAHE itself runs on the unredacted image", fontsize=9)
    fig.tight_layout(); fig.savefig(CHECKS_DIR / "clahe_before_after.png", dpi=100); plt.close(fig)

    # 2. mask overlay
    ov = show_rgb.copy(); ov[m] = (0.4 * ov[m] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(resize_area(ov, 1024)); ax[0].set_title(f"{image_id}: ground truth (cyan) over the original image")
    ax[1].imshow(ov[sl]); ax[1].set_title(f"native crop y{by} x{bx}")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(CHECKS_DIR / "mask_overlay.png", dpi=100); plt.close(fig)

    # 3. mask downsampling, visual only (measured comparison is Phase 2)
    mf = m.astype(np.float32)
    panels = [("2048 ground truth", mf[sl])]
    for r in (1024, 512):
        f = 2048 // r
        csl = (slice(by // f, (by + win) // f), slice(bx // f, (bx + win) // f))
        near = cv2.resize(mf, (r, r), interpolation=cv2.INTER_NEAREST)[csl]
        area = cv2.resize(mf, (r, r), interpolation=cv2.INTER_AREA)[csl]
        panels += [(f"{r}, nearest-neighbour", np.kron(near, np.ones((f, f)))),
                   (f"{r}, area-average (soft, 0-1)", np.kron(area, np.ones((f, f))))]
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.6))
    for a, (t, p) in zip(ax, panels):
        a.imshow(p, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle(f"{image_id}: 256x256 native crop (y{by} x{bx}, chosen as the window with most thin-vessel skeleton "
                 "pixels) and the same region after downsampling; low-res pixels drawn as blocks", fontsize=10)
    fig.tight_layout(); fig.savefig(CHECKS_DIR / "mask_downsampling_crop.png", dpi=100); plt.close(fig)
    print(f"[visual] {image_id}, crop y{by} x{bx} ({int(best)} thin skeleton px); wrote clahe_before_after.png, "
          "mask_overlay.png, mask_downsampling_crop.png")


COMMANDS = {"extract": cmd_extract, "check": cmd_check, "pairs": cmd_pairs, "split": cmd_split,
            "pairs_summary": cmd_pairs_summary, "integrity": cmd_integrity,
            "preprocess": cmd_preprocess, "stats": cmd_stats, "visual": cmd_visual, "masks": cmd_masks,
            "native": cmd_native}


def main() -> None:
    ap = argparse.ArgumentParser(description="FIVES data preparation (Phase 1)")
    ap.add_argument("command", nargs="+", choices=[*COMMANDS, "all"],
                    help="one or more steps, run in the order given; 'all' = every step")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    cv2.setNumThreads(int(os.environ.get("OMP_NUM_THREADS", "1")))   # stay within the job's CPUs
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    for name in (list(COMMANDS) if "all" in a.command else a.command):
        t0 = time.time()
        print(f"\n######## {name} ########", flush=True)
        COMMANDS[name](cfg, a.workers)
        print(f"######## {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
