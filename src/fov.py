"""Field-of-view (FOV) mask: a circle fitted to the boundary of the eye.

Every metric is computed inside this mask, applied identically to the prediction and
the ground truth (decision after Phase 1; the ground truth is never edited). The
circle fit, unlike a plain threshold, also covers the dim rim of very dark images and
eyes that touch the image border.

Steps (``fit_fov_circle``):
  1. g = max(R, G, B), Gaussian-smoothed (sigma 1 px; a wider blur spreads bright edges
     outward past the low threshold and biases the circle toward the bright side: on a
     synthetic eye sigma 3 shifted the centre by 3.7 px, sigma 1 by 1.3 px).
  2. Background level = 99th percentile of g in the three image corners without burned-in
     text (top-right, bottom-left, bottom-right); threshold = background + ``delta``.
  3. Foreground = g > threshold, opened (15 px) to cut text strokes, largest connected
     component, holes filled.
  4. Boundary = outer contour of that component, minus points on the image frame (where
     the eye touches the border the contour runs along the frame, not along the eye).
  5. Circle = algebraic least-squares fit, refitted after dropping points whose distance
     to the circle deviates by more than ``max(3 px, 3 * 1.4826 * MAD)``; repeated until
     stable. This removes the camera's notch and arcs that are darker than the threshold.
  6. FOV mask = fitted disk (clipped to the image) UNION the thresholded eye region of
     step 3. The disk covers dim rims that fall below the threshold in very dark images;
     the region adds the camera's notch (a bright tab on the upper-right rim that contains
     image content and annotated vessels, but lies outside the circle). Burned-in text is
     not connected to the eye and stays outside.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class FovCircle:
    cx: float
    cy: float
    r: float
    rms_px: float          # RMS distance of the inlier boundary points to the circle
    n_points: int          # boundary points used for the fit (after removing frame points)
    inlier_frac: float
    threshold: float       # intensity threshold used for the foreground
    background: float      # background level (99th percentile of the text-free corners)
    touches_frame: bool    # the fitted circle extends beyond the image

    def as_dict(self) -> dict:
        return asdict(self)


def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle: x^2 + y^2 + D x + E y + F = 0."""
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    (d, e, f), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = -d / 2, -e / 2
    return float(cx), float(cy), float(np.sqrt(max(cx ** 2 + cy ** 2 - f, 0.0)))


def eye_region(rgb: np.ndarray, delta: float = 1.5, corner: int = 150, sigma: float = 1.0) -> tuple[np.ndarray, float, float]:
    """Thresholded eye region (steps 1-3) with the threshold and background level used."""
    h, w = rgb.shape[:2]
    g = cv2.GaussianBlur(rgb.max(axis=2), (0, 0), sigma).astype(np.float32)
    corners = np.concatenate([g[:corner, -corner:].ravel(), g[-corner:, :corner].ravel(),
                              g[-corner:, -corner:].ravel()])
    background = float(np.percentile(corners, 99))
    threshold = background + delta
    fg = (g > threshold).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        raise ValueError("no foreground found")
    return ndi.binary_fill_holes(lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))), threshold, background


def boundary_points(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outer contour points of the eye region, minus points on the image frame."""
    h, w = region.shape
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len)[:, 0, :].astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    keep = (x > 2) & (x < w - 3) & (y > 2) & (y < h - 3)
    return x[keep], y[keep]


def fit_fov_circle(rgb: np.ndarray, delta: float = 1.5, max_iter: int = 10,
                   region: tuple | None = None) -> tuple[FovCircle, np.ndarray]:
    """Robust circle fit (steps 4-5). Returns the circle and the inlier flag of each
    boundary point. ``region`` = a precomputed ``eye_region(rgb, delta)``."""
    h, w = rgb.shape[:2]
    reg, threshold, background = region if region is not None else eye_region(rgb, delta)
    x, y = boundary_points(reg)
    if len(x) < 50:
        raise ValueError(f"only {len(x)} boundary points")
    inl = np.ones(len(x), bool)
    for _ in range(max_iter):
        cx, cy, r = _fit_circle(x[inl], y[inl])
        res = np.hypot(x - cx, y - cy) - r
        mad = 1.4826 * np.median(np.abs(res[inl] - np.median(res[inl])))
        new = np.abs(res) <= max(3.0, 3 * mad)
        if np.array_equal(new, inl):
            break
        inl = new
    cx, cy, r = _fit_circle(x[inl], y[inl])
    res = np.hypot(x - cx, y - cy) - r
    circle = FovCircle(cx=cx, cy=cy, r=r, rms_px=float(np.sqrt(np.mean(res[inl] ** 2))), n_points=int(len(x)),
                       inlier_frac=float(inl.mean()), threshold=threshold, background=background,
                       touches_frame=bool(cx - r < 0 or cy - r < 0 or cx + r > w - 1 or cy + r > h - 1))
    return circle, inl


def circle_mask(circle: FovCircle | dict, shape: tuple[int, int] = (2048, 2048), scale: float = 1.0) -> np.ndarray:
    """Boolean FOV mask for an image of ``shape``. ``scale`` maps the circle (fitted at
    2048) to another resolution, e.g. 0.25 for 512."""
    c = circle if isinstance(circle, dict) else circle.as_dict()
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    # pixel centres: (x + 0.5) / scale - 0.5 in the 2048 frame
    return (((xx + 0.5) / scale - 0.5 - c["cx"]) ** 2 + ((yy + 0.5) / scale - 0.5 - c["cy"]) ** 2) <= c["r"] ** 2


def fov_mask(rgb: np.ndarray, delta: float = 1.5) -> tuple[np.ndarray, FovCircle]:
    """The evaluation FOV (step 6): fitted disk UNION thresholded eye region, at the
    image's resolution. Use the same mask for an image's prediction and ground truth."""
    reg = eye_region(rgb, delta)
    circle, _ = fit_fov_circle(rgb, delta, region=reg)
    return circle_mask(circle, rgb.shape[:2]) | reg[0], circle


# =========================================================================== CLI
# python -m src.fov examples --ids a b c ...   review figure (results/checks/fov_examples.png)
# python -m src.fov all                         every image: cached masks + table + summary (SLURM job)

def _image_stats(args) -> dict:
    """FOV for one image, with the checks used to judge it (masks only from files)."""
    image_id, cfg, cache = args
    from src.data import raw_image_path, raw_mask_path, read_rgb, read_unchanged
    cv2.setNumThreads(1)
    rgb = read_rgb(raw_image_path(cfg, image_id))
    reg = eye_region(rgb)
    circle, _ = fit_fov_circle(rgb, region=reg)
    circ = circle_mask(circle, rgb.shape[:2])
    fov = circ | reg[0]
    gt = read_unchanged(raw_mask_path(cfg, image_id)) > 0
    if cache:
        from src.data import write_rgb  # noqa: F401  (same atomic-write convention)
        out = cache / f"{image_id}.png"
        tmp = out.with_name(out.stem + ".tmp.png")
        cv2.imwrite(str(tmp), fov.astype(np.uint8) * 255)
        tmp.replace(out)
    return {"image_id": image_id, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in circle.as_dict().items()},
            "fov_fraction": round(float(fov.mean()), 4), "added_by_region_px": int((reg[0] & ~circ).sum()),
            "circle_not_in_region_px": int((circ & ~reg[0]).sum()), "gt_px": int(gt.sum()),
            "gt_px_outside_fov": int((gt & ~fov).sum()),
            "text_px_inside_fov": int(((rgb[:256, :256].max(2) > 40) & fov[:256, :256]).sum())}


def _display(rgb: np.ndarray, fov: np.ndarray, cfg: dict, size: int | None = None) -> tuple[np.ndarray, float]:
    """For figures only: pixelate outside the FOV (text not legible), then a linear gain
    so that dark images are visible."""
    from src.data import redact_outside_fov
    q = cfg["qa"]
    img = redact_outside_fov(rgb, fov, q["fov_margin_px"], q["redact_block_px"])
    gain = float(np.clip(200.0 / max(np.percentile(rgb.max(2)[fov], 99), 1), 1, 8))
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img, gain


def examples_figure(cfg: dict, rows: list[tuple[str, list[str]]], zooms: list[tuple[str, str, int, int]],
                    out_png, out_csv) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from src.data import raw_image_path, raw_mask_path, read_rgb, read_unchanged

    size, stats = 512, []
    ncol = max(len(ids) for _, ids in rows)
    fig, axes = plt.subplots(len(rows) + 1, ncol, figsize=(3.6 * ncol, 4.4 * (len(rows) + 1)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    cache = {}
    for r, (label, ids) in enumerate(rows):
        for c, i in enumerate(ids):
            rgb = read_rgb(raw_image_path(cfg, i)); fov, circle = fov_mask(rgb)
            gt = read_unchanged(raw_mask_path(cfg, i)) > 0
            cache[i] = (rgb, fov, circle, gt)
            img, gain = _display(rgb, fov, cfg, size)
            f = size / 2048
            cv2.circle(img, (int(round((circle.cx + 0.5) * f - 0.5)), int(round((circle.cy + 0.5) * f - 0.5))),
                       int(round(circle.r * f)), (255, 255, 0), 1)
            cnts, _ = cv2.findContours(cv2.resize(fov.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, (0, 255, 255), 1)
            out = int((gt & ~fov).sum())
            axes[r, c].imshow(img)
            axes[r, c].set_title(f"{label}: {i}\nr={circle.r:.0f} px, fit RMS {circle.rms_px:.1f} px, gain x{gain:.1f}\n"
                                 f"GT px outside FOV: {out}", fontsize=8)
            stats.append({"row": label, "image_id": i, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in circle.as_dict().items()},
                          "fov_fraction": round(float(fov.mean()), 4), "gt_px_outside_fov": out, "display_gain": round(gain, 2)})
    for c, (title, i, y, x) in enumerate(zooms):
        rgb, fov, circle, gt = cache[i] if i in cache else (lambda im: (im, *fov_mask(im), read_unchanged(raw_mask_path(cfg, i)) > 0))(read_rgb(raw_image_path(cfg, i)))
        img, _ = _display(rgb, fov, cfg)
        win = (slice(y, y + 256), slice(x, x + 256))
        # local gain: zooms show faint rims, so stretch to this crop's own range (display only)
        base = img[win].astype(np.float32) / max(_display(rgb, fov, cfg)[1], 1e-6)
        inside = fov[win]
        if inside.mean() < 0.05:                                   # mostly background: no stretch
            gain = 1.0
        else:
            gain = float(np.clip(200.0 / max(np.percentile(base.max(2)[inside], 99), 1), 1, 40))
        if gain > 4:                                               # very dim: show brightness only
            crop = np.repeat(np.clip(base.max(2) * gain, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        else:
            crop = np.clip(base * gain, 0, 255).astype(np.uint8)
        g = gt[win]
        crop[g & fov[win]] = (0.5 * crop[g & fov[win]] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)       # GT inside FOV
        crop[g & ~fov[win]] = (255, 0, 0)                                                                       # GT outside FOV
        cnts, _ = cv2.findContours(fov[win].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, cnts, -1, (0, 255, 255), 1)
        yy, xx = np.ogrid[y:y + 256, x:x + 256]
        ring = np.abs(np.hypot(xx - circle.cx, yy - circle.cy) - circle.r) < 0.7
        crop[ring] = (255, 255, 0)
        axes[-1, c].imshow(crop, interpolation="nearest")
        axes[-1, c].set_title(f"{title}\n{i}, native 256x256 at y{y} x{x}, gain x{gain:.1f}"
                              + (" (grey = max(R,G,B))" if gain > 4 else ""), fontsize=8)
    fig.suptitle("FOV = fitted circle (yellow) UNION thresholded eye region; cyan = final FOV outline. Bottom row: "
                 "ground truth inside FOV green, outside FOV red.\nDisplay only: outside the FOV pixelated (text not "
                 "legible); linear gain for dark images (zooms: gain from the crop itself).", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    pd.DataFrame(stats).to_csv(out_csv, index=False)


def cmd_all(cfg: dict, workers: int) -> None:
    import json
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd
    from src.data import CHECKS_DIR, fives_dir, id_sort_key
    from src.utils import project_path

    ids = sorted(pd.read_csv(fives_dir(cfg) / "manifest.csv").image_id, key=id_sort_key)
    cache = project_path(cfg["paths"]["cache_dir"]) / "fov_2048"
    cache.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        df = pd.DataFrame(list(ex.map(_image_stats, [(i, cfg, cache) for i in ids], chunksize=4)))
    gto = pd.read_csv(CHECKS_DIR / "gt_outside_fov.csv").set_index("image_id").kind
    df["phase1_gt_outside_kind"] = df.image_id.map(gto).fillna("")
    df.to_csv(CHECKS_DIR / "fov_all.csv", index=False)
    q = lambda col: {p: round(float(df[col].quantile(p)), 3) for p in (0, 0.01, 0.5, 0.99, 1)}
    bad = df[df.gt_px_outside_fov > 0]
    summary = {"n_images": int(len(df)), "cache": str(cache.relative_to(project_path("."))),
               "r_px": q("r"), "cx": q("cx"), "cy": q("cy"), "rms_px": q("rms_px"), "inlier_frac": q("inlier_frac"),
               "fov_fraction": q("fov_fraction"), "touches_frame": int(df.touches_frame.sum()),
               "added_by_region_px": q("added_by_region_px"), "circle_not_in_region_px": q("circle_not_in_region_px"),
               "images_with_gt_outside_fov": int(len(bad)), "gt_px_outside_fov_total": int(df.gt_px_outside_fov.sum()),
               "gt_outside_by_phase1_kind": {k or "new": int(v) for k, v in bad.phase1_gt_outside_kind.value_counts().items()},
               "images_with_text_inside_fov": int((df.text_px_inside_fov > 0).sum())}
    with open(CHECKS_DIR / "fov_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    worst = df.sort_values("rms_px", ascending=False).image_id.head(5).tolist() + \
            df.sort_values("inlier_frac").image_id.head(5).tolist()
    worst = list(dict.fromkeys(worst))[:10]
    examples_figure(cfg, [("highest fit RMS / lowest inlier share", worst[:5]), ("", worst[5:10])], [],
                    CHECKS_DIR / "fov_worst_fits.png", CHECKS_DIR / "fov_worst_fits.csv")
    print(f"[fov] {len(df)} masks in {cache}; wrote fov_all.csv, fov_summary.json, fov_worst_fits.png")


def main() -> None:
    import argparse
    import os

    from src.data import CHECKS_DIR
    from src.utils import load_config

    ap = argparse.ArgumentParser(description="FOV masks (circle fit + eye region)")
    ap.add_argument("command", choices=["examples", "all"])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--dark", nargs="*", default=[])
    ap.add_argument("--border", nargs="*", default=[])
    ap.add_argument("--zoom", nargs="*", default=[], help="title:image_id:y:x ...")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cv2.setNumThreads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    if a.command == "examples":
        zooms = [(t.replace("_", " "), i, int(y), int(x)) for t, i, y, x in (z.split(":") for z in a.zoom)]
        examples_figure(cfg, [("dark", a.dark), ("eye touches border", a.border)], zooms,
                        CHECKS_DIR / "fov_examples.png", CHECKS_DIR / "fov_examples.csv")
        print(f"[fov] wrote {CHECKS_DIR / 'fov_examples.png'} and fov_examples.csv")
    else:
        cmd_all(cfg, a.workers)


if __name__ == "__main__":
    main()
