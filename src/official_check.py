"""Post hoc check (decision D7, after the referee report): do the data-audit findings, made on the
Hugging Face copy (tyluan/FIVES), hold for the official FIVES release on figshare?

    python -m src.official_check --archive $WORK/FIVES_official/download/FIVES_official.rar
    (CPU job: jobs/cpu.sbatch src.official_check --archive ...)

The official release (figshare article 19688169, v1, one RAR5 archive, MD5 checked after download)
is extracted into its own empty folder; only the 1600 PNG files are extracted (no spreadsheet,
no Thumbs.db). Then, for every official file:
  1. map it to our image_id: <split>/Original/<n>_<C>.png -> <split>_<n>_<C> (image),
     <split>/Ground truth/<n>_<C>.png -> the same id (mask);
  2. SHA-256 of the file bytes, compared with the bytes stored in the Hugging Face parquet
     (results/checks/image_qa.csv: image_sha256_parquet, mask_sha256_parquet);
  3. if the bytes differ: decode both and compare the pixel arrays (official masks are stored with three
     identical channels, the copy's with one: compared after reducing to one channel);
  4. re-derive the audit findings directly from the official files: byte-identical image files
     (and whether their masks differ), empty masks, identical masks on different images, and the
     vessel-mask agreement of the same-eye train-test pair (pixel Dice of the two masks).
Fundus images are only hashed and compared as arrays; nothing is displayed, cropped or saved
from them. Output: results/checks/official_release_check.json and official_release_files.csv
(ids, hashes, equality flags only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

P7ZIP = "/util/opt/anaconda/deployed-conda-envs/packages/p7zip/envs/p7zip-16.02/bin/7z"   # HCC module p7zip/16.02
NAME = re.compile(r"(?P<split>train|test)/(?P<kind>Original|Ground truth)/(?P<num>\d+)_(?P<cat>[ADGN])\.png$")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(archive: Path, dest: Path) -> None:
    if dest.exists() and any(dest.rglob("*.png")):
        print(f"[official] already extracted in {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([P7ZIP, "x", str(archive), f"-o{dest}", "-ir!*.png", "-y", "-bd"], check=True,
                   stdout=subprocess.DEVNULL)


def main() -> None:
    import cv2

    from src.data import CHECKS_DIR, raw_image_path, raw_mask_path, read_unchanged
    from src.utils import environment_info, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--dest", default=None, help="extraction folder (default: <archive dir>/../extracted)")
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    archive = Path(a.archive)
    dest = Path(a.dest) if a.dest else archive.parent.parent / "extracted"
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    extract(archive, dest)

    rows = []
    for f in sorted(dest.rglob("*.png")):
        m = NAME.search(f.as_posix())
        if not m:
            raise ValueError(f"unexpected file in the official release: {f.relative_to(dest)}")
        rows.append({"image_id": f"{m['split']}_{int(m['num'])}_{m['cat']}",
                     "kind": "image" if m["kind"] == "Original" else "mask",
                     "official_path": f"{m['split']}/{m['kind']}/{f.name}", "path": f})
    off = pd.DataFrame(rows)
    counts = off.groupby(["kind"]).size().to_dict()
    print(f"[official] {len(off)} PNG files: {counts}")
    if off.duplicated(["image_id", "kind"]).any():
        raise ValueError("duplicate ids in the official release")

    qa = pd.read_csv(CHECKS_DIR / "image_qa.csv").set_index("image_id")
    off["sha256"] = [sha256(p) for p in off.path]
    off["in_hf_copy"] = off.image_id.isin(qa.index)
    hf_sha = {"image": qa.image_sha256_parquet.to_dict(), "mask": qa.mask_sha256_parquet.to_dict()}
    off["bytes_equal_hf"] = [hf_sha[k].get(i) == s for i, k, s in zip(off.image_id, off.kind, off.sha256)]
    pix, maxdiff, mask_rgb = [], [], []
    for i, k, p, same in zip(off.image_id, off.kind, off.path, off.bytes_equal_hf):
        if same or i not in qa.index:
            pix.append(bool(same)); maxdiff.append(0 if same else np.nan)
            continue
        ours = read_unchanged((raw_image_path if k == "image" else raw_mask_path)(cfg, i))
        theirs = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)          # same reader and channel order as ours
        if (k == "mask" and theirs is not None and theirs.ndim == 3 and ours.ndim == 2
                and all(np.array_equal(theirs[..., 0], theirs[..., c]) for c in range(1, 3))
                and (theirs.shape[2] == 3 or (theirs.shape[2] == 4 and bool((theirs[..., 3] == 255).all())))):
            mask_rgb.append((i, theirs.shape[2]))  # official masks: 3 identical colour channels (+ opaque alpha)
            theirs = theirs[..., 0]               # the copy stores one channel
        eq = theirs is not None and theirs.shape == ours.shape and bool(np.array_equal(theirs, ours))
        pix.append(eq)
        maxdiff.append(float(np.abs(theirs.astype(np.int32) - ours.astype(np.int32)).max())
                       if theirs is not None and theirs.shape == ours.shape else np.nan)
    off["pixels_equal_hf"], off["max_abs_pixel_diff"] = pix, maxdiff
    hf_ids = set(qa.index)
    missing_in_official = sorted(hf_ids - set(off.loc[off.kind == "image", "image_id"]))

    # ---- audit findings re-derived from the official files
    img, msk = off[off.kind == "image"].set_index("image_id"), off[off.kind == "mask"].set_index("image_id")
    dup_groups = [sorted(g.index.tolist()) for _, g in img.groupby("sha256") if len(g) > 1]
    mask_arr = lambda i: cv2.imread(str(msk.loc[i, "path"]), cv2.IMREAD_UNCHANGED)

    def dice(a, b):
        a, b = a > 0, b > 0
        s = int(a.sum() + b.sum())
        return 1.0 if s == 0 else 2.0 * int((a & b).sum()) / s

    dups = [{"images": g, "masks_byte_identical": msk.loc[g[0], "sha256"] == msk.loc[g[1], "sha256"],
             "mask_dice_full_image": round(dice(mask_arr(g[0]), mask_arr(g[1])), 4)} for g in dup_groups]
    empty, values = [], set()
    for i in msk.index:
        m = mask_arr(i)
        values |= set(np.unique(m).tolist())
        if not (m > 0).any():
            empty.append(i)
    mask_dup = [sorted(g.index.tolist()) for _, g in msk.groupby("sha256") if len(g) > 1]
    same_eye = ("train_174_D", "test_54_D")
    summary = {
        "source": "figshare article 19688169 v1 (doi:10.6084/m9.figshare.19688169.v1), one RAR5 archive",
        "archive_md5": md5, "archive_bytes": archive.stat().st_size,
        "n_png": int(len(off)), "by_kind": {k: int(v) for k, v in counts.items()},
        "ids_match_hf_copy": bool(off.in_hf_copy.all() and not missing_in_official),
        "hf_ids_missing_in_official": missing_in_official,
        "bytes_equal_hf": {k: int(g.bytes_equal_hf.sum()) for k, g in off.groupby("kind")},
        "pixels_equal_hf": {k: int(g.pixels_equal_hf.sum()) for k, g in off.groupby("kind")},
        "pixels_equal_hf_note": "masks compared after reducing the official masks (3 identical colour channels, "
                                "some with a fully opaque alpha channel) to one channel, as stored in the copy",
        "official_mask_channels": {"rgb": sum(c == 3 for _, c in mask_rgb), "rgba_opaque": sum(c == 4 for _, c in mask_rgb)},
        "findings_official": {
            "byte_identical_image_groups": dups,
            "n_byte_identical_image_groups": len(dups),
            "empty_masks": sorted(empty),
            "byte_identical_mask_groups": mask_dup,
            "mask_values": sorted(int(v) for v in values),
            "same_eye_pair": {"images": list(same_eye),
                              "mask_dice_full_image": round(dice(mask_arr(same_eye[0]), mask_arr(same_eye[1])), 4)},
        },
        "environment": {k: v for k, v in environment_info().items() if k in ("time", "host", "git", "slurm_job_id")},
    }
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    off.drop(columns=["path"]).to_csv(CHECKS_DIR / "official_release_files.csv", index=False)
    (CHECKS_DIR / "official_release_check.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "environment"}, indent=1, default=str))
    print(f"[official] wrote {CHECKS_DIR / 'official_release_check.json'} and official_release_files.csv")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
