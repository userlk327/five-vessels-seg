# Protocol

Every protocol decision, written down when it is made. One protocol applies to all
models (CLAUDE.md, Rule 4); any unavoidable difference between models is listed in
the "Differences between models" section.

Status: **complete (Phases 0–7, 2026-11-22).** Sections marked *(Phase N)* were written
during that phase. Results and their limitations: [`results/README.md`](../results/README.md).
Draft Methods and Results text for the report: [`docs/report_draft.md`](report_draft.md).

Contents: 1 compute · 2 reproducibility · 3 checkpointing · 4 mixed precision ·
5 data source · 6 split · 7 preprocessing · 8 training-mask resizing · 9 metrics ·
10 evaluation and thresholds · 11 training settings and runs · 12 differences between
models · 13 final test evaluation · 14 known limitations.

---

## 1. Compute environment (Phase 0)

- **Cluster:** HCC Swan. Code is edited over VS Code SSH; heavy work runs only as
  SLURM jobs (`jobs/*.sbatch`) or `srun` sessions, never on the login node.
- **GPU (approved in Phase 0):** one **NVIDIA L40S (48 GB)** per GPU job, on the
  group partition `ltarunl` (`--gres=gpu:1 --constraint=gpu_l40s`). If that node is
  busy, fall back to the shared `gpu` partition, still on L40S. `guest_gpu` is never
  used, because it is preemptible. CPU-only jobs (preprocessing, metrics) run on
  the `batch` partition.
  **Every model is trained, timed and measured on the same GPU type**, so the cost
  numbers (inference time, peak memory, training time) are comparable. A100, H100
  and H200 are also available. After the first U-Net run, the time per epoch is
  reported; if L40S is too slow, ALL models switch together to one faster type.
  The GPU name is recorded with every run (`run_info.jsonl`).
- **Software:** one conda env (Python 3.11) built from the pinned `requirements.txt`
  by `jobs/setup_env.sh`; the full resolved package list is saved in
  `requirements.lock.txt`, and every run also records all package versions.
  - PyTorch 2.14.1 with the **CUDA 12.6 build** (`+cu126`). Reason: the CUDA 13.0
    build is compiled only for sm_75 and newer (checked on Swan: `sm_75 sm_80 sm_86
    sm_90 sm_100 sm_120`), so it cannot run on V100 GPUs (sm_70). The cu126 build
    keeps one environment usable on both the L40S and V100 nodes. 2.14.1 is also
    the version the original notebook ran with.
  - **Verified on the L40S node** (`jobs/env_check.sbatch`, job 19023615):
    - Python 3.11.18; torch 2.14.1+cu126, CUDA 12.6, cuDNN 9.10.2; driver 580.178.04.
    - torch arch list: `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`. The L40S is
      sm_89 and runs the sm_86 code; same major version, so it is binary-compatible.
    - An fp16 autocast + GradScaler training step through conv/BatchNorm/ConvTranspose works.
  - The env is at `$WORK/fives-vessel-seg/env` (about 47.6k files, 5.2 GB). The
    conda and pip download caches were cleared after the install to keep file counts low.
- **Storage:**
  - `$HOME` (project folder: code, configs, results CSVs/figures, docs; backed up,
    20 GiB quota) is not used for job I/O.
  - `$WORK/fives-vessel-seg/` holds everything large (fast, not backed up, purged
    after ~6 months without access). The project reaches it through symlinks
    created by `jobs/setup_links.sh`: `outputs/`, `logs/` and `data/cache/`.
    The pip, torch, HF and matplotlib caches also live there (`jobs/env.sh`).
  - `data/FIVES` is a symlink to `$WORK/FIVES` (the Hugging Face parquet files).
  - Data, checkpoints and prediction maps are git-ignored.

## 2. Reproducibility (Phase 0)

- **Seeds:** `random`, NumPy and torch (CPU and all CUDA devices) are seeded from
  the run seed (`src.utils.set_seed`).
- **DataLoader:** at the start of every epoch the training loader's generator is
  reseeded with `epoch_seed(seed, epoch)`. Workers seed NumPy and `random` from
  their torch seed (`src.utils.seed_worker`). The shuffle order and the augmentation
  randomness of epoch *e* therefore depend only on (seed, *e*), and are the same
  whether or not the run was resumed.
- **cuDNN:** `deterministic=True`, `benchmark=False` (in `configs/base.yaml`).
  Limitation: some CUDA kernels (e.g. the backward pass of bilinear upsampling)
  are not deterministic, so two GPU runs with the same seed can still differ
  slightly. Seeds make runs repeatable in distribution, not bit-identical on the GPU.
- **Run record:** each run folder `outputs/<run_name>/seed<seed>/` stores the
  resolved `config.yaml`, and `run_info.jsonl` gets one line per session (first start
  and every resume). Each line holds the host, GPU, SLURM job id and restart count,
  the git commit and whether the working tree had uncommitted changes, and all
  package versions.
- **Version control** (set up 2026-11-21, before Phase 6). Runs up to and including
  Phase 5 record no commit, because the repository had none yet.
  - **Pre-test freeze commit:** `0ee73608db87db9e0b6f2ff46d29f2faa75fd1dc` (2026-11-21).
    It holds the code, configs, job scripts, this protocol, the shared split
    (`data/splits/splits.csv`, `split_groups.csv`), the CNN normalization statistics
    (`norm_stats.json`) and every Phase 1–5 check and validation result. It was made
    **before any test-set prediction** and pushed to the public repository
    `github.com/userlk327/five-vessels-seg` (branch `main`) the same day.
  - **Public repository:** `CLAUDE.md`, the rest of `data/` (images, masks, caches),
    `outputs/`, logs and checkpoints are never committed. No figure shows the burned-in
    markings unredacted. Result files store project-relative paths
    (`src.utils.portable_path`).

## 3. Checkpointing and resume (Phase 0)

- `last.pt` is written atomically (temporary file, then rename) at the end of
  every epoch. It holds:
  - the model, optimizer and GradScaler states;
  - the number of finished epochs, the best validation score and its epoch, and
    the early-stopping counter;
  - the RNG states of Python, NumPy, torch CPU and every CUDA device;
  - the best epoch's weights;
  - the seed, a hash of the resolved config, and a `finished` flag.
- `best.pt` holds only the best epoch's weights; it is exported from the same
  state. If a job dies between the two writes, the next start detects a stale or
  missing `best.pt` and re-exports it from `last.pt`. Log rows from unfinished
  epochs are dropped.
- **Resume:** `src/train.py` always continues from `last.pt` if it exists. So a job
  that hit its time limit is resumed by resubmitting the same command. In addition,
  `jobs/train.sbatch` asks SLURM for a signal 10 minutes before the time limit,
  stops training and requeues itself (at most `MAX_RESTARTS`, default 10). A
  finished run exits immediately. A resume with a different config or seed is
  refused.
- Checkpoints load with `torch.load(weights_only=True)`.
- **Verified** (`tests/test_checkpoint.py`, pinned env; 10/10 passed on the L40S
  node): a run interrupted at epoch 1 or 4 and resumed in a "fresh process"
  (different seed and initial weights) ends bit-identical to an uninterrupted run.
  - This covers the weights, the best epoch and score, `best.pt` and the per-epoch log.
  - DataLoader workers, dropout and GradScaler are in use.
  - It holds on CPU and on CUDA. The CUDA case loads `last.pt` with
    `map_location="cuda"` and uses the CUDA RNG for noise and dropout; deterministic
    algorithms are enabled for this test only.
  - Negative control: with the RNG restore disabled, the test fails.
  - The CUDA test found one bug: a NumPy RNG state loaded onto the GPU could not be
    restored. Fixed in `restore_rng_state`, which now moves every state to the CPU.
- **Early stopping:** an epoch counts as an improvement only if its validation score
  is strictly greater than the best so far (NaN never improves). Training stops when
  the number of consecutive epochs without improvement reaches `patience`, or at
  `max_epochs`.

## 4. Mixed precision (Phase 0)

- fp16 autocast with a GradScaler; the loss is computed in float32. fp16 rather than
  bf16, because V100 GPUs do not support bf16, so the same setting works on every
  GPU we might use.

## 5. Data source (Phase 1)

- **Source:** the Hugging Face mirror `tyluan/FIVES` (4 parquet files: 3 train, 1 test;
  columns `image`, `mask`, `image_id`), not the official figshare release. It is the
  single data source for every model; no other dataset is used.
- **Extraction** (`python -m src.data extract`, job 19024116): the PNG bytes stored
  in the parquet files are written unchanged (no re-encoding) to
  `data/FIVES/{train,test}/{images,masks}/<image_id>.png`.
- **Verified** (`results/checks/phase1_checks.json`, `image_qa.csv`,
  `extraction_manifest.csv`):
  - 600 train / 200 test; every image has a mask with the same name.
  - `image_id` is `<split>_<number>_<category>`, so names are unique across splits.
  - 800/800 files are byte-identical (SHA-256) to the parquet bytes. For 12 random
    pairs (seed 0), the PIL decode of the parquet bytes equals the OpenCV decode of
    the file, pixel by pixel.
  - All images are 2048×2048, 3-channel, uint8. All masks are 2048×2048, 1-channel,
    uint8, with values {0, 255}; 2 masks are all 0 (excluded, see §6).
  - The category letter agrees with the numbering blocks (train 1–150 A, 151–300 D,
    301–450 G, 451–600 N; test 1–50 A, 51–100 D, 101–150 G, 151–200 N). Counts:
    train 150 per category, test 50 per category.
- **Not used:** `Quality Assessment.xlsx` (not in the mirror; no image-quality breakdown).

## 6. Train / validation / test split (Phase 1)

- **Test:** the official 200 test images, unchanged.
- **Excluded from training and validation** (`data.split.exclude` in `configs/base.yaml`;
  label `excluded` in `splits.csv`):
  - `train_447_G` and `train_448_G`: both ground-truth masks are completely empty
    (all 0), and the two masks are byte-identical on near-identical images. An empty target
    would teach "no vessels" on a real fundus image. *Correction (2026-11-23, revision 2):*
    this entry first said "although the images show vessels". Contrast-stretched, both very
    dark images show the optic disc, but the vessels are barely discernible under the
    compression artefacts; the exclusion rests on the empty masks alone.
  - `train_174_D`: the same eye as `test_54_D` (practically identical vessel masks,
    vessel-layout IoU 0.997, `results/checks/mask_similar_pairs.csv`). It is removed
    from training so that no test image is also seen in training. The test set is
    unchanged.
- **Train / val (597 remaining official training images): grouped, category-balanced
  split, seed 17** (`src.data.make_splits`, `python -m src.data split`).
  - **Groups** (members always land on the same side; within official train):
    - byte-identical duplicate image files (5 pairs);
    - an identical burned-in marking (pattern IoU >= 0.8) *and* consecutive numbers
      (79 pairs);
    - isolated identical-marking pairs, i.e. a marking group of exactly 2 images over all
      800 (31 pairs; 4 of them span categories A and D).
    - The last two are likely both eyes of one patient. Overlapping pairs are merged:
      85 groups (79 of 2, 5 of 3, 1 of 4), 177 images. List:
      `data/splits/split_groups.csv`.
  - **Assignment:** val gets exactly round(0.2 × n) images of each category (30 each).
    All groups are visited in one seeded random order (`RandomState(17)`), and a group
    goes to val only if it fits the remaining val places of every category it contains.
    Groups may span categories.
  - **Result:** train 477 (A 120, D 119, G 118, N 120), val 120 (30 per category),
    excluded 3, test 200. **No group crosses train/val**: checked when the split is
    written, and again independently from `splits.csv` + `split_groups.csv`. Grouped
    images: 147 in train, 30 in val.
  - File: `data/splits/splits.csv` (columns image_id, category, split; SHA-256 starts
    `aee9e29af7d8f769`); summary in `results/checks/split_summary.json`. All models and
    all teammates use this file.
- **History:** this replaces two earlier splits.
  - The first was ungrouped (`train_test_split` with `stratify`, seed 17); two duplicate
    pairs crossed train/val (train_39_A/132_A, train_55_A/134_A).
  - The second grouped only duplicates and adjacent-numbered marking pairs, and kept
    train_174_D in training.

## 7. Image preprocessing and normalization (Phase 1)

- **CLAHE** on the L channel of LAB (OpenCV), at native 2048 resolution:
  clipLimit 2.0, 8×8 tile grid. A and B are untouched, and 3 channels are kept, so
  the same input works for the CNNs and SAM.
- **Resizing:** area interpolation to 512×512 and 1024×1024. The CNN resolution is
  the config value `data.cnn_resolution` (default 512, a practical default: it
  matches the notebook and fits GPU memory); SAM always uses 1024. A 2026 study
  (Amarnath R, arXiv:2604.02977) reported that on FIVES a U-Net's thin-vessel
  sensitivity was highest at moderate downsampling; its setup and width metric
  differ from ours, so it is context only, not evidence that 512 is best here.
- **Cache:** CLAHE-processed, resized images are stored as 8-bit RGB PNG, *before*
  normalization, in `data/cache/images_clahe_{512,1024}/` (800 each, with
  `_params.json`). The original 2048 masks are kept for evaluation.
- **Normalization, two separate cases, never both:**
  - CNNs: per-channel mean/std from the **477 final training images only**
    (`split == train`), computed on the cached CLAHE images scaled to [0, 1], all pixels
    (background included), and applied when loading (`data/splits/norm_stats.json`).
    At 512: mean (R, G, B) = (0.384131, 0.192080, 0.104345), std = (0.248743, 0.134776,
    0.068527).
  - SAM: raw 0–255 pixels; only SAM's built-in pixel normalization is applied.
- **Augmentation** (training only, identical for image and mask): horizontal flip
  (p = 0.5), vertical flip (p = 0.5), and a random 90° rotation (k ∈ {0, 1, 2, 3}).
  Together these give each of the 8 square symmetries with equal probability
  (tested in `tests/test_data.py`).
- **Masks for training:** soft targets (area average, no threshold), chosen in Phase 2
  (§8). The loader never binarizes them.
- **Figures and privacy:** some images carry burned-in text outside the eye, which may
  be fragments of patient names. Every figure saved from this project pixelates the
  area outside the field of view (64 px blocks), so the text is not legible. The text
  is never read, transcribed or stored.
## 8. Training-mask resizing (Phase 2)

- **Comparison** (`python -m src.evaluate maskresize`; `results/checks/mask_resize_*`):
  - **Sample:** 30 training masks, stratified A 8 / D 7 / G 8 / N 7, RandomState(0).
  - **Methods** (`src/resample.py`):
    - `nearest`: pixel-centre nearest neighbour, identical to the notebook's PIL resize;
    - `area_t50`: block mean ≥ 0.5;
    - `area_t25`: block mean ≥ 0.25;
    - `maxpool`: any vessel pixel in the block;
    - `soft`: block mean, no threshold.
  - **Reconstruction:** each target is upsampled to 2048 exactly as predictions will be
    (bilinear, half-pixel centres; OpenCV on CPU = torch on GPU to 1e-7), thresholded at
    0.5, and scored inside the FOV.
  - **Means over the 30 images** (Dice / precision / recall / thin centerline recall /
    clDice). Medium and thick centerline recall are 1.000 for every method at both sizes.

    | size | method | Dice | precision | recall | thin recall | clDice |
    |---|---|---|---|---|---|---|
    | 512 | nearest | 0.935 | 0.935 | 0.934 | 0.932 | 0.989 |
    | 512 | area_t50 | 0.940 | 0.928 | 0.953 | 0.956 | 0.993 |
    | 512 | area_t25 | 0.920 | 0.856 | 0.993 | 0.994 | 0.996 |
    | 512 | maxpool | 0.877 | 0.781 | 1.000 | 1.000 | 0.996 |
    | 512 | soft | 0.984 | 0.986 | 0.983 | 0.926 | 0.990 |
    | 512 | soft, threshold 0.25 (illustration) | 0.904 | 0.826 | 1.000 | 0.996 | 0.998 |
    | 1024 | nearest | 0.960 | 0.960 | 0.960 | 0.981 | 0.994 |
    | 1024 | area_t50 | 0.967 | 0.948 | 0.987 | 1.000 | 0.998 |
    | 1024 | area_t25 (= maxpool for 2×2 blocks) | 0.955 | 0.914 | 1.000 | 1.000 | 0.998 |
    | 1024 | soft | 0.995 | 0.995 | 0.995 | 0.988 | 0.997 |
    | 1024 | soft, threshold 0.25 (illustration) | 0.945 | 0.896 | 1.000 | 1.000 | 0.999 |

  - This measures what each training target keeps through the evaluation pipeline,
    not how a trained model behaves.
- **Chosen method (approved in Phase 2): `soft`**, i.e. area average with no threshold.
  It is used for every model: the CNNs at `cnn_resolution`, SAM at 1024. The binary
  alternative considered was `area_t50` (area average ≥ 0.5).
  - **Why soft:**
    - It is the only option that neither deletes thin vessels nor widens all vessels.
      - `area_t50` sets a 1-px vessel at 512 to 0, which teaches the model that such a
        vessel is background.
      - `area_t25` and `maxpool` keep thin vessels but thicken every vessel (precision
        0.856 and 0.781 at 512), which teaches the model to draw vessels too wide.
    - It keeps the most information: Dice 0.984 at 512, against 0.940 for `area_t50`.
    - A 1-px vessel is encoded as a low value (0.25 at 512). Whether it survives
      depends on the operating threshold, which is chosen on val (§10). In the
      comparison, thin recall was 0.926 at threshold 0.5 and 0.996 at 0.25.
  - **What this changes:** the training target is no longer a binary mask but a soft one
    in [0, 1].
  - **Storage:**
    - float16, in one compressed `.npz` archive per resolution, keyed by image_id:
      `data/cache/masks_soft_{512,1024}.npz`, built by `python -m src.data masks`. One
      archive instead of one file per image keeps file counts low on `$WORK`.
    - Only train and val images are cached (597). Test and excluded images never are.
    - Every block-mean value (k/16 at 512, k/4 at 1024) is exact in float16.
    - **Built by job 19029556** (`results/checks/mask_cache.json`), checked on read-back:
      - 597 members per archive (477 train + 120 val); test (200) and excluded (3)
        images are not in the cache.
      - Values lie in [0, 1]: 17 distinct values at 512 and 5 at 1024. The float16
        rounding error is 0.
      - Each target's mean equals its 2048 vessel fraction exactly.
      - Sizes are 12.0 MB at 512 and 19.4 MB at 1024.
      - Pixels strictly between 0 and 1: 4.2% at 512 and 1.4% at 1024 (mean per image).
  - **No binarization anywhere in the training path:**
    - `load_mask_target` returns the stored values and raises on anything outside [0, 1].
    - Augmentation only permutes pixels.
    - Both loss terms use the target as is (§11).
    - This is tested end to end in `tests/test_losses.py`: archive → dataset with
      augmentation → DataLoader → loss.
    - The original 2048 masks are binary and are used only for evaluation.
## 9. Metrics *(Phase 2)*

- **Decision after Phase 1:** the ground truth is never edited. Every metric is computed
  inside a FOV mask, applied identically to the prediction and the ground truth. Reason:
  19 masks contain annotation lines along the image border (outside the eye; 41,117 px
  in total), and these would distort the skeleton-based metrics (clDice, width-stratified
  centerline recall). The FOV mask must first be shown to be reliable. Plan: fit a circle
  to the eye boundary, and show it on the darkest images (e.g. train_373_G) and on images
  where the eye touches the image border. The rough threshold-based FOV used for the
  Phase 1 QA misses the dim rim of 5 dark images, so it is not the evaluation FOV.
- **Evaluation FOV** (`src/fov.py`; masks cached in `data/cache/fov_2048/`):
  1. g = max(R, G, B), Gaussian blur σ = 1 px.
  2. Threshold = the 99th percentile of g in the three corners without text, + 1.5.
  3. Foreground opened (15 px), largest component kept, holes filled (the "eye region").
  4. Boundary points of the region, minus points on the image frame.
  5. Least-squares circle, refitted after dropping outliers (|residual| > max(3 px,
     3 × robust SD)) until stable.
  6. **FOV = fitted disk (clipped to the image) ∪ eye region.**
  - The disk covers dim rims that fall below the threshold. The region adds the
    camera's notch (a bright tab at the upper-right rim that contains image content and
    annotated vessels). Burned-in text is not connected to the eye and stays outside.
  - σ = 1, not 3: a wider blur spreads bright edges outward past the low threshold. On
    a synthetic eye with one dim half, σ = 3 shifted the centre by 3.7 px; σ = 1 by
    1.3 px (`tests/test_fov.py`).
  - **Checked on the 5 dark images and 5 images where the eye touches the border**
    (`results/checks/fov_examples.png/.csv`): 0 ground-truth pixels outside the FOV in
    all 10; fit RMS 0.7–1.3 px. Zooms show the dim rim, the notch (with its vessels
    inside the FOV), an eye running off the frame, and a border-line annotation
    (outside the FOV).
  - **All 800 images** (job 19026279; `results/checks/fov_all.csv`, `fov_summary.json`):
    - radius 1003–1025 px; FOV covers 75.7–79.2% of the image; fit RMS median
      1.26 px (max 5.05); inlier share median 0.93 (min 0.61, train_353_G, the image
      with one very dim half).
    - 178 circles extend beyond the image frame. No burned-in text pixel lies inside
      any FOV.
    - **Ground truth outside the FOV:** 146 images.
      - 19 are the known border-line annotations and 1 is the stray blob in train_193_D.
      - The other 126 have 2,107 GT pixels in total outside (median 8 per image,
        max 161). In 99 of them, every such pixel is within 3 px of the FOV edge.
        Native-resolution zooms of the 4 farthest cases (8–14 px,
        `fov_gt_outside_zooms.png`) show annotated vessels drawn past the visible rim
        into the black background: annotation overshoot, not a FOV error.
        (`results/checks/fov_gt_outside_new.csv`)
    - Worst fits (highest RMS, lowest inlier share; `fov_worst_fits.png`) follow the eye
      correctly; their higher RMS comes from a wavy thresholded rim (faint glow), not a
      misplaced circle. No FOV was corrected by hand.
- **Metric definitions** (`src/metrics.py`; unit tests `tests/test_metrics.py`):
  - **Scope:** every metric is computed inside the FOV, on prediction & FOV and
    GT & FOV. Skeletons, boundaries and the vessel-radius map are computed on the full
    masks and then restricted to the FOV, so the FOV edge creates no artificial vessel
    ends or boundaries.
  - **Overlap:** Dice, IoU, precision, recall.
  - **Width-stratified centerline recall:** GT skeleton (`skimage` skeletonize); local
    radius = EDT of the GT at each skeleton pixel. The prediction is dilated by 1 px
    (3×3 square).
    - **Approved cut-offs (Phase 2): thin r < 3, medium 3 ≤ r ≤ 7, thick r > 7 px at
      2048.** r = 3 and r = 7 count as medium; the EDT takes discrete values, and 4.6% of
      skeleton pixels sit exactly at r = 3 and 2.4% at r = 7. For a straight vessel of
      width w the centre radius is (w + 1)/2, so thin ≈ w ≤ 4 px at 2048 (≤ 1 px at 512),
      and thick ≈ w ≥ 14 px.
    - **Class shares**, training ground truth (477 images, 10,692,024 skeleton pixels
      inside the FOV; `results/checks/width_radius_summary.json`,
      `width_radius_hist.png`): thin 10.3%, medium 49.9%, thick 39.8%. Per image, the
      thin share has median 9.1% (5th–95th percentile 5.6–17.7%, min 4.0%), so every
      training image has thin-class pixels. Thin share by category: A 12.3%, D 10.4%,
      G 10.9%, N 8.2%.
    - Alternatives considered (same file): 2/5 → 2.2 / 41.7 / 56.0%; 2.5/6 → 7.8 /
      45.0 / 47.2%; 3/6 → 10.3 / 42.5 / 47.2%; 4/8 → 22.2 / 45.8 / 32.0%. 3/7 was kept
      because it is the class of vessels that become ≤ 1 px at 512, which the project
      targets.
    - **Sensitivity check (Phase 6):** width-stratified recall is also reported with
      cut-offs 4 / 8 (`evaluation.width_sensitivity_radius_px`), from the same saved
      predictions, in a secondary table only; the main tables use 3 / 7.
  - **clDice:** as in CLAUDE.md.
  - **NSD:** a boundary-pixel approximation (counts boundary pixels instead of weighting
    by surface length). Boundary = mask minus its 3×3 erosion; τ = 2 px.
  - **Vessel density:** vessel pixels in the FOV / FOV pixels.
  - **Fractal dimension:** box counting with boxes of 2–512 px, slope of log N vs
    log(1/s).
  - **Empty sets:** a ratio with an empty denominator is 1 if both masks are empty,
    else 0. A width class absent from an image's GT gives NaN (skipped in averages, with
    n reported). An empty mask has NaN fractal dimension.
## 10. Evaluation and threshold selection *(Phase 2)*

- **Pipeline** (`python -m src.evaluate score --config <model> --seed <s> --split val`;
  tested end to end on synthetic data in `tests/test_evaluate.py`):
  1. **Probability maps:** float16 sigmoid outputs at the model's output size, in one
     compressed archive per run and split: `outputs/<run>/seed<seed>/<split>/probs.npz`,
     one member per image. The archive must contain exactly the split's image ids.
  2. **Upsampling:** bilinear (half-pixel centres) to 2048.
  3. **Threshold, per model and seed, on val only:**
     - Grid 0.05–0.95 in steps of 0.05; pick the highest mean Dice over the 120 val
       images. On an exact tie, take the lowest threshold.
     - Saved in `results/eval/<run>/seed<seed>/val_threshold.json`.
     - The test split reads that file; nothing is chosen on test.
     - Scoring the test split requires the flag `--final-test` (Phase 6).
  4. **Per-image metrics:** every metric of §9 at that threshold, at 2048, against the
     original GT inside the FOV. Output `{split}_metrics.csv`, one row per (model, seed,
     split, image): model, seed, split, category, image_id, threshold, prob_resolution,
     then every metric. That includes the 4 / 8 sensitivity columns, n_skel per width
     class, and density and fractal dimension for both prediction and GT.
  - **Threshold sweep** (`{split}_sweep.csv`):
    - Contents: Dice, IoU, precision, recall and width-stratified recall (3 / 7 and 4 / 8)
      for every image at every grid threshold.
    - It takes one pass: a 3×3 max filter of the probabilities replaces thresholding
      plus the 1-px dilation. The result is identical to full scoring (tested).
    - On val it is the basis of the threshold choice. On test it is for reporting only.
  - **Validation Dice during training (Phase 3)** will be computed the same way: GPU
    bilinear upsampling, fixed threshold 0.5, original masks, inside the FOV.
- **Phase 6 addition (user request):** a figure of thin-vessel recall against the
  threshold (0.05–0.95) for each model, from the saved val and test sweeps. The main
  tables still use the val-selected threshold.
- **Phase 6 threshold analysis: decided 2026-11-20, after Phase 4, before any test
  prediction exists.**
  - **Primary:** every run is scored on test at its **own val-selected threshold**, as
    originally planned. All main tables, statistics and the 2×2 ablation use this.
  - **Secondary:**
    1. all runs at a **fixed threshold of 0.5**, same metrics and ablation, in separate
       tables;
    2. the **thin-recall vs threshold figure** (0.05–0.95) for each model, from the saved
       val and test sweeps.
  - **Why:** in Phase 4 the val-selected thresholds differed between runs (0.45–0.60),
    and on val this alone was enough to reverse the sign of attention's thin-recall
    effect (+0.007 at the selected thresholds, −0.013 at 0.5).
  - The secondary analyses are reported next to the primary one, not instead of it.
- **Mask-resampling reference** (`python -m src.evaluate reference --split val`):
  - **Method:** each val ground truth is pushed through a model family's resolution path
    and scored with the same pipeline, including the same threshold rule:
    - `reference_soft_<cnn_resolution>`: 2048 → soft at `cnn_resolution` → float16 →
      bilinear to 2048 → threshold, as for CNN predictions;
    - `reference_soft_256`: 2048 → soft at 256 (SAM's decoder output) → bilinear to 1024
      (SAM's postprocessing) → float16 → bilinear to 2048 → threshold, as for SAM
      predictions.
  - **Interpretation:** this shows how much the resizing alone destroys. It is context
    for thin-vessel results, **not an upper bound** on what a model can reach.
  - **Val results** (job 19029556; `results/eval/reference_soft_{512,256}/val_*`):
    - The threshold rule picks 0.5 for both references.
    - Means over 120 images (sd in brackets):

    | reference | Dice | precision | recall | thin | medium | thick | clDice | NSD |
    |---|---|---|---|---|---|---|---|---|
    | soft at 512 (CNN path) | 0.985 (0.002) | 0.986 | 0.984 | 0.919 (0.028) | 1.000 | 1.000 | 0.991 | 0.996 |
    | soft at 256 → 1024 (SAM path) | 0.942 (0.014) | 0.967 | 0.918 | 0.377 (0.101) | 0.953 | 1.000 | 0.938 | 0.934 |

    - With the 4 / 8 cut-offs, thin recall is 0.963 (512) and 0.617 (256).
    - Thin recall at lower thresholds (from the sweep): 512 → 0.996 at 0.25; 256 → 0.934
      at 0.25 and 0.870 at 0.30. The Dice of the 256 reference at those thresholds falls
      to 0.83 and 0.86.
    - **Reading:** at the Dice-selected threshold, resizing alone loses about 8% of
      thin-vessel centreline at 512. On SAM's 256 decoder grid it loses about 62% of
      thin and 5% of medium centreline. Thin-vessel results for SAM must be read against
      this.

## 11. Training settings per model *(Phases 3-5)*

- **Loss**, shared by every trained model (`src/losses.py`, `configs/train_shared.yaml`):
  - **Terms:**
    - BCEWithLogitsLoss on the logits, averaged over every pixel of the batch;
    - soft Dice on the sigmoid outputs, per image: D = (2Σpt + 1) / (Σp + Σt + 1); the
      Dice term is the mean of (1 − D) over the batch.
    - Equal weights.
  - **Precision:** computed in float32, with autocast switched off inside the loss.
  - **Targets:** soft targets are used as is in both terms (§8; tested).
- **Property of this loss with soft targets** (measured before any training):
  - **How it was measured** (`python -m src.losses optimum`;
    `results/checks/loss_optimum_summary.json`, `loss_optimum_levels.csv`,
    `loss_optimum_sweep.csv`): free per-pixel logits, no network, were fitted to the
    512 soft targets of 4 training images (one per category) until converged (largest
    final gradient 2e-9).
  - **BCE alone:** reproduces the target exactly (|p − t| ≤ 1.6e-4).
  - **BCE + soft Dice:** pushes soft values toward 0 or 1. The soft-Dice gradient of a
    pixel is proportional to (D/2 − t), so pixels with t below about D/2 (≈ 0.46 here)
    are pushed down and pixels above it are pushed up.
  - **Optimum p for each target value t:**

    | t | 0.0625 | 0.125 | 0.25 | 0.375 | 0.5 | 0.75 |
    |---|---|---|---|---|---|---|
    | BCE + soft Dice | 0.011 | 0.024 | 0.073 | 0.198 | 0.620 | 0.944 |

  - **Effect on thin vessels:** a 1-px vessel (t = 0.25 at 512) is driven to about 0.07,
    not 0.25. That is still above the lowest grid threshold (0.05).
  - **At 2048** (mean over the 4 images):
    - the optimum map gives thin recall 0.958 at threshold 0.5;
    - its Dice-best threshold is 0.6, where thin recall is 0.922;
    - for comparison, the soft target itself gives 0.940 (its Dice-best threshold is 0.5).
  - **Caveat:** this describes the loss's optimum, not a trained model.
  - **Decision (after Phase 2): keep BCE + soft Dice as specified.** It is the loss in
    the approved proposal.
  - This is a **known effect of the loss**. It applies equally to every trained model
    (the four CNNs and fine-tuned SAM), so it does not favour any model in the
    comparison.
  - The Phase 6 figure of thin recall against the threshold shows how it plays out in
    the trained models.
- **Baseline U-Net (Phase 3):**
  - **Model:** the notebook's `UNET` class, unchanged (checked line by line against the
    notebook cell).
  - **Design:**
    - 4 levels down and 4 up, 64 → 1024 channels;
    - each block is two 3×3 conv + BatchNorm + ReLU;
    - 2×2 max-pool going down, 2×2 ConvTranspose going up;
    - a 1×1 output conv produces the logits.
  - **Size:** 31,037,633 parameters, all trainable.
- **Training procedure** (shared by the four CNNs; `src/train.py`,
  `configs/train_shared.yaml`):
  - **Inputs:** the cached CLAHE images at `cnn_resolution`, with train-split
    normalization. Targets are the soft masks (§8).
  - **Optimization:** batch 4, Adam with learning rate 1e-4, up to 150 epochs.
    - Every training image is used in every epoch (no batch dropped). With 477 images,
      the last batch of each epoch has 1 image.
    - The shuffle order and the augmentation depend only on (seed, epoch), so a resumed
      run sees the same data.
  - **Precision and determinism:** fp16 autocast with a GradScaler; the loss is computed
    in float32; cuDNN deterministic, benchmark off.
  - **Validation, every epoch:**
    - Loss: the same loss on the val soft targets.
    - Val Dice, as in §10: float16 probabilities, bilinear upsampling to 2048 on the GPU,
      threshold ≥ 0.5, Dice against the original mask inside the FOV, mean over the 120
      val images.
    - Check: on a CPU smoke run (8 train / 4 val images), the training-time val Dice was
      identical to the scoring pipeline's Dice at 0.5 for the same checkpoint (0.248482
      both).
  - **Early stopping:** patience 20 on val Dice. `best.pt` holds the best epoch.
  - **After training:** the same job predicts the val split (float16 archive) and scores
    it (§10). The test split is not predicted.
  - **Reported cost:**
    - training time = the sum of epoch times (train + val), excluding start-up;
    - peak GPU memory = the maximum over epochs of PyTorch's peak allocated memory
      (training and validation).

- **U-Net, seed 0, at 512** (job 19029603, 1× L40S; `results/train/unet/seed0/`,
  `results/eval/unet/seed0/`):
  - **Training:** stopped early after 141 epochs; the best epoch was 120 (val Dice at
    0.5: 0.9133).
  - **Time:** median 11.0 s per epoch (9.4 s training + 1.2 s validation); the first
    epoch took 18.3 s. Training time was 0.43 h.
  - **Memory:** peak GPU memory 3.36 GB allocated (3.99 GB reserved).
  - **Learning rate:** kept at 1e-4. The curves show no problem: train and val loss
    fall together, and the gap at the best epoch is 0.018.
  - **Consistency check:** the scoring pipeline's mean Dice at 0.5 equals the
    training-time val Dice of the best epoch (0.913278 in both).
- **1024 probe** (same job; `results/checks/probe_unet_r{512,1024}_b4.json`):
  - At 1024 with batch 4 (no reduction needed): 0.324 s per training step, 12.25 GB
    peak allocated (15.52 GB reserved).
  - At 512: 0.074 s per step, 3.36 GB.
- **Resolution pilot, 512 vs 1024 (Phase 3; user decision after the first U-Net run).**
  - **Rule, fixed before any 1024 result was seen:** switch all four CNNs to
    `cnn_resolution: 1024` only if, on val, both hold:
    - thin centreline recall (3 / 7 cut-offs) improves by **at least 0.05**;
    - Dice drops by **no more than 0.005**.

    Otherwise all four CNNs stay at 512.
  - **How it is applied:** U-Net, seed 0, at each run's own val-selected threshold, as
    means over the 120 val images (`results/eval/unet/seed0/` vs
    `results/eval/unet_r1024/seed0/`). A Dice increase satisfies the Dice condition.
    The test set is not used.
  - **Pilot setup:** `configs/unet_1024.yaml`, identical to `configs/unet.yaml` except
    `data.cnn_resolution: 1024`.
    - Batch size stays 4: the probe measured 12.25 GB at 1024.
    - Inputs are the cached CLAHE images at 1024 with their train-split normalization
      stats; targets are the cached soft masks at 1024.
    - Scored with the same pipeline, with the 1024 mask-resampling reference as context.
  - **Result** (job 19029728; 1024 reference: job 19029727; val, seed 0, each at its own
    val-selected threshold, 0.55 in both):

    | | Dice | thin | medium | thick | clDice | NSD | time / epoch | peak GPU memory |
    |---|---|---|---|---|---|---|---|---|
    | U-Net 512 | 0.9142 | 0.5548 | 0.9303 | 0.9866 | 0.9299 | 0.9021 | 11.0 s | 3.36 GB |
    | resampling reference 512 | 0.9848 | 0.9192 | 1.0000 | 1.0000 | 0.9906 | 0.9956 | | |
    | U-Net 1024 | 0.9240 | 0.5943 | 0.9282 | 0.9838 | 0.9313 | 0.9064 | 44.2 s | 12.25 GB |
    | resampling reference 1024 | 0.9953 | 0.9878 | 1.0000 | 1.0000 | 0.9969 | 0.9992 | | |

    - **Applying the rule:** thin recall changed by +0.0395 (the rule needs ≥ +0.05:
      **not met**). Dice changed by +0.0098 (the rule needs ≥ −0.005: met).
    - **Decision: all four CNNs stay at `cnn_resolution: 512`.**
    - The 1024 run (`unet_r1024`) is a documented pilot only and is not part of the
      model comparison.
- **All trained runs, as run** (`results/tables/val_runs.csv`, `val_sam_runs.csv`; val
  metrics at each run's own val-selected threshold, mean over the 120 val images). The
  overlapping-tile variant has no runs of its own: it reuses the three tiled-SAM models.

  | Model | Seed | Epochs run | Best epoch | Val threshold | s/epoch (median) | Training h | Val Dice | Val thin recall |
  |---|---|---|---|---|---|---|---|---|
  | U-Net | 0 | 141 | 120 | 0.55 | 11.0 | 0.43 | 0.9142 | 0.5548 |
  | U-Net | 1 | 117 | 96 | 0.45 | 11.1 | 0.36 | 0.9139 | 0.5809 |
  | U-Net | 2 | 69 | 48 | 0.55 | 11.1 | 0.21 | 0.9130 | 0.5369 |
  | Attention U-Net | 0 | 121 | 100 | 0.45 | 12.2 | 0.41 | 0.9140 | 0.5875 |
  | Attention U-Net | 1 | 117 | 96 | 0.45 | 12.3 | 0.40 | 0.9146 | 0.5642 |
  | Attention U-Net | 2 | 69 | 48 | 0.55 | 12.3 | 0.24 | 0.9129 | 0.5433 |
  | Multi-Scale U-Net | 0 | 96 | 75 | 0.60 | 16.9 | 0.45 | 0.9149 | 0.5528 |
  | Multi-Scale U-Net | 1 | 150 | 130 | 0.55 | 17.0 | 0.71 | 0.9145 | 0.5295 |
  | Multi-Scale U-Net | 2 | 106 | 85 | 0.55 | 16.9 | 0.50 | 0.9146 | 0.5220 |
  | Multi-Scale Attention U-Net | 0 | 82 | 61 | 0.45 | 18.1 | 0.41 | 0.9128 | 0.5381 |
  | Multi-Scale Attention U-Net | 1 | 105 | 84 | 0.45 | 18.3 | 0.53 | 0.9134 | 0.5532 |
  | Multi-Scale Attention U-Net | 2 | 106 | 85 | 0.55 | 18.3 | 0.54 | 0.9145 | 0.5323 |
  | SAM LoRA, tiled (main SAM) | 0 | 124 | 103 | 0.55 | 266.6 | 9.18 | 0.9053 | 0.5974 |
  | SAM LoRA, tiled (main SAM) | 1 | 150 | 149 | 0.45 | 267.1 | 11.18 | 0.9071 | 0.5580 |
  | SAM LoRA, tiled (main SAM) | 2 | 150 | 138 | 0.45 | 266.4 | 11.10 | 0.9063 | 0.6234 |
  | SAM LoRA, whole image | 0 | 150 | 143 | 0.50 | 68.7 | 2.86 | 0.9124 | 0.4889 |
  | SAM LoRA, whole image | 1 | 150 | 149 | 0.50 | 81.9 | 3.42 | 0.9124 | 0.4907 |
  | SAM LoRA, whole image | 2 | 150 | 145 | 0.50 | 83.3 | 3.47 | 0.9124 | 0.4834 |

  - Early stopping (patience 20) ended every CNN run except Multi-Scale U-Net seed 1,
    which reached the 150-epoch cap. U-Net and Attention U-Net seed 2 both stopped after
    69 epochs (best epoch 48).
  - Tiled SAM seeds 1 and 2 and all three whole-image SAM seeds reached the cap.
  - The cost table uses the seed-0 runs; later seeds partly ran next to other jobs or on
    another partition (§12, §13).

## 12. Differences between models *(Phases 3-5)*

- **Summary (Phase 7).** Everything not listed here is identical for every trained model:
  the split file, CLAHE preprocessing, augmentation (flips and 90° rotations, training
  only), loss (BCE + soft Dice, equal weights, float32), optimizer (Adam, learning rate
  1e-4), effective batch 4, early stopping (patience 20 on val Dice at 2048, threshold
  0.5, inside the FOV), at most 150 epochs, fp16 mixed precision, seeds 0–2, the
  threshold rule (chosen on val per run), and scoring at 2048 against the original masks
  inside the FOV. Every model was trained and timed on NVIDIA L40S GPUs.

  | | 4 CNNs | SAM LoRA, tiled (main SAM) | Overlapping-tile inference | SAM LoRA, whole image | SAM zero-shot, oracle prompts |
  |---|---|---|---|---|---|
  | Change vs U-Net | attention gates and/or multi-scale blocks (2×2 ablation) | different model (SAM ViT-B) | same models as tiled | different model (SAM ViT-B) | different model (SAM ViT-B) |
  | Network input | CLAHE image area-resized to `cnn_resolution` 512 | four native 1024×1024 crops of the 2048 CLAHE image | nine native 1024 crops, stride 512 | CLAHE image area-resized to 1024 | as whole image |
  | Input normalization | per-channel mean/std of the training split | SAM's own | SAM's own | SAM's own | SAM's own |
  | Training target | soft mask at 512 | binary mask crop at native resolution (no resizing) | – | soft mask at 1024 | none (no training) |
  | Loss computed on | 512 logits | 256 decoder logits ×4 → 1024, per tile | – | 256 decoder logits ×4 → 1024 | – |
  | Trainable parameters | all (31.0–33.8 M) | LoRA rank 4 (q, v) + mask decoder: 4.2 M of 93.9 M | – | as tiled | 0 |
  | Batch of 4 | 4 images | micro-batch 2 × accumulation 2 (tiles) | – | micro-batch 2 × accumulation 2 | – |
  | Optimizer steps per epoch | 120 | 477 (4 tiles per image) | – | 120 | – |
  | Prompts | – | none | none | none | 5 positive + 5 negative points from the image's own GT |
  | Saved probability map | 512 | 512 (4 decoder grids stitched) | 512 (tent-blended logits) | 1024 (from the 256 decoder grid) | 1024 (from 256) |
  | Mask-resampling reference (context) | `reference_soft_512` | `reference_soft_512` | `reference_soft_512` | `reference_soft_256` | `reference_soft_256` |
  | Reported | main tables | main tables | sensitivity row | main tables (secondary SAM) | separate table, never ranked |

  - The oracle experiment uses ground-truth information from every image it segments,
    test images included (the exception in CLAUDE.md Rule 3).
  - Whole-image SAM seeds 1 and 2 and the overlap inference for seeds 1 and 2 ran on the
    shared `gpu` partition (still L40S) instead of `ltarunl` (§13).

- **The 2×2 ablation, models 1–4** (`src/models/blocks.py`; Phase 4 design,
  **approved by the user**: paper-style attention gates, and the split-channel
  multi-scale block in all 9 blocks):
  - **Shared backbone:** all four keep the baseline backbone: same depth, channels,
    pooling, ConvTranspose upsampling and 1×1 output, with the same module names. With
    both switches off, the variant class is identical to the notebook `UNET`: same
    state-dict keys and shapes, same weights for the same seed, same outputs (tested).
  - **Attention gate** (Oktay et al., 2018; additive "grid" attention), on each of the 4
    skip connections:
    - Gating signal: the decoder features one level coarser. That is the bottleneck
      output for skip e4, then d4, d3 and d2.
    - Computation: α = sigmoid(ψ(ReLU(θ_x(x) + φ_g(g)))), where
      - θ_x is a 2×2 stride-2 conv without bias, which brings the skip to the gate's grid;
      - φ_g and ψ are 1×1 convs with bias;
      - the intermediate width is C_x / 2.
    - α is upsampled bilinearly to the skip's resolution, and the gated skip x·α
      replaces the plain skip in the concatenation.
    - The gates are created after the backbone, so with the same seed the Attention
      U-Net starts from the U-Net's backbone weights (tested).
    - The ×2 bilinear upsampling of α uses an equivalent written with slicing and
      concatenation only (`upsample2x_bilinear`). PyTorch's `interpolate` has a
      non-deterministic backward on CUDA, which would break bit-exact reproducibility and
      exact resume. Values and gradients are identical to `interpolate` (tested).
  - **Multi-scale block:** three parallel 3×3 convs with dilation 1, 2 and 4.
    - Their output channels are **split**, not tripled (64 → 22 + 21 + 21).
    - The outputs are concatenated, then BN → ReLU → 1×1 conv → BN → ReLU (fusion).
    - It replaces the **second** 3×3 conv of all 9 blocks: encoder 1–4, bottleneck,
      decoder 4–1. The first conv of each block is kept.
    - The split branches cost the same as one ordinary 3×3 conv, so the only additions
      are the 1×1 fusion and one BN per block.
  - **Size and compute** (`results/tables/model_params.csv`; MACs of all conv layers for
    one 512×512 image):

    | model | parameters | vs U-Net | attention gates | multi-scale fusion + BN | GMACs at 512 | vs U-Net |
    |---|---|---|---|---|---|---|
    | U-Net | 31,037,633 | – | – | – | 192.67 | – |
    | Attention U-Net | 32,083,077 | +3.37% | 1,045,444 | – | 195.89 | +1.7% |
    | Multi-Scale U-Net | 32,788,417 | +5.64% | – | 1,756,672 | 202.33 | +5.0% |
    | Multi-Scale Attention U-Net | 33,833,861 | +9.01% | 1,045,444 | 1,756,672 | 205.56 | +6.7% |

  - **Training:** each model is trained with exactly the Phase 3 config, at
    `cnn_resolution` 512, with seeds 0, 1 and 2 (U-Net seed 0 already exists).
  - **Schedule (approved):**
    - The three seed-0 runs go one after another, alone on the GPU node, so their
      training times are comparable to U-Net seed 0, which also ran alone. The cost
      table uses the seed-0 runs.
    - Then the 8 seed-1/2 runs (including U-Net seeds 1 and 2) go 4 at a time, or
      2 at a time if other users have jobs waiting on the `ltarunl` partition. Their
      training times are reported but may include contention between jobs.
    - **As run:** 2 at a time (on submission, another user's 2 guest jobs held 2 of
      the node's 4 L40S GPUs). All 11 jobs completed (19031870–19031880, see
      `results/checks/phase4_jobs.txt`).
      - Median epoch times hardly differ between runs that had the GPU node to
        themselves and runs that shared it: U-Net 11.01 s alone vs 11.06 s shared. So
        contention was negligible.
      - Multi-Scale U-Net seed 1 reached max_epochs (150; best epoch 130). Every other
        run stopped early.
  - **Phase 4 val results** (exploratory; the confirmatory analysis is on test in
    Phase 6): `results/tables/val_runs.csv`, `val_model_level.csv`,
    `val_ablation_effects.csv`.
    - Sensitivity check: the same analysis with every run at a fixed threshold of 0.5
      (`*_fixed_t0.5.csv`), because the val-selected thresholds differ between runs
      (0.45–0.60), and thin recall depends strongly on the threshold.

- **SAM resolution: plan for Phase 5** (decided after Phase 2):
  - **The problem:** whole-image SAM takes the 2048 image at 1024, and its mask decoder
    outputs 256×256. The mask-resampling reference (§10) shows that this 256 grid loses
    about 62% of thin-vessel centreline even with perfect predictions: thin recall 0.377
    at the Dice-selected threshold, against 0.919 on the CNNs' 512 grid.
  - **Two fine-tuned SAM variants (5b) will be built:**
    1. **Whole image:** the CLAHE image at 1024, decoder output 256, probabilities saved
       at 1024 (SAM's postprocessing).
    2. **Tiled:** each 2048 image is cut into 2×2 non-overlapping native-resolution
       crops of 1024×1024, each fed to SAM at 1024. Each crop's 256 decoder output
       covers a quarter of the image, so the combined output grid is 512×512, the same
       as the CNNs (`cnn_resolution` = 512).
  - **To settle in Phase 5:**
    - The training target per crop: the native 1024 crop of the 2048 mask is binary,
      because there is no downsampling. Alternatively, the loss could be computed on the
      decoder's grid.
    - Building native-resolution CLAHE crops: the Phase 1 cache only holds 512 and 1024.
    - The tiled path's own mask-resampling reference.
    - Seams between crops.
  - **If the CNNs move to `cnn_resolution` 1024** (resolution pilot, §11), a
    resolution-matched tiled SAM would need **4×4** tiling: 512×512 native crops, each
    upsampled to SAM's 1024 input, giving a combined 1024 output grid. The 2×2 tiling
    above would give only 512. This is decided in Phase 5.
  - **Which variant is the main fine-tuned SAM** is decided at Phase 5. Both are
    reported, and any difference from the CNN protocol is recorded here.
- **SAM (Phase 5; user decisions after Phase 4):**
  - **Package and checkpoint:**
    - `segment-anything==1.0` (already pinned in `requirements.txt` since Phase 0), with
      `torchvision==0.29.1+cu126`.
    - ViT-B checkpoint `sam_vit_b_01ec64.pth` from Meta's official URL:
      - SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`;
      - MD5 `01ec64d29a2fca3f0661936605ae66f8` (its prefix is the `01ec64` in the file
        name);
      - loads strictly into the ViT-B model; 93,735,472 parameters.
  - **Seeds:** one seed (0) during Phase 5. After the Phase 6 test evaluation, tiled SAM
    and whole-image SAM were trained with seeds 1 and 2 as well, and the overlapping-tile
    inference was run on each tiled seed (frozen configs, unchanged; §13). The oracle
    experiment has no training and keeps its single prompt seed.
  - **Input:** the same CLAHE-processed images as the CNNs, with only SAM's own pixel
    normalization.
  - **5a, oracle-prompted zero-shot** (`src/sam_oracle.py`, `configs/sam_oracle.yaml`;
    label "zero-shot, oracle point prompts"; separate table, never ranked against the
    automatic models):
    - **Usage:** standard SAM on whole images. `SamPredictor` on the CLAHE image at
      1024, `multimask_output=False`; logits at 1024 → sigmoid → float16 probability
      map at 1024.
    - **Prompts,** from each image's own 2048 GT and FOV:
      - K positives drawn uniformly from the GT skeleton;
      - K negatives drawn uniformly from FOV background at least 10 px (at 2048) from
        any vessel;
      - one permutation per image, so the prompt sets are nested across K;
      - RNG seeded by (seed 0, image_id).
    - **K** ∈ {5, 10, 20}. Each K gets its own val-selected threshold; the chosen K has
      the highest mean val Dice.
  - **5b, LoRA fine-tuning** (`src/models/sam_lora.py`, `src/sam_train.py`):
    - **Trainable parts:**
      - LoRA, rank 4, on the q and v parts of the fused `qkv` in all 12 image-encoder
        blocks: A is kaiming-uniform and B is zero, as in SAMed.
      - The mask decoder.
      - Total: 4,205,796 trainable of 93,882,928 parameters (LoRA 147,456; decoder
        4,058,340).
    - **Frozen:** the image encoder and the prompt encoder.
    - **No prompts:** the empty sparse embedding and the learned no-mask embedding.
    - **Gradient path:** the encoder and decoder are called directly, with gradients on,
      because `Sam.forward` runs under `no_grad`. The decoder is called once per sample,
      because it is written for one image embedding per call.
    - **Learning rate (confirmed before training): Adam, 1e-4, identical to the CNNs.**
      It is not tuned for SAM and not overridden in the SAM configs (checked in the
      resolved configs). The only training difference from the CNNs is how the batch
      of 4 is formed (micro-batch 2 × accumulation 2).
    - **Checkpoint selection for tiled SAM (confirmed):**
      - Val Dice is computed per **whole image**, not per tile. Each val batch is one
        image's 4 tiles in order (checked at run time and tested).
      - Their 256 logit grids are stitched into one 512 grid → sigmoid → float16 →
        bilinear to 2048 on the GPU → ≥ 0.5 → Dice against the original 2048 mask
        inside the FOV. That is the same function as for the CNNs (`val_dice_batch`),
        averaged over the 120 val images.
      - Only the val loss (not used for selection) is per tile.
    - **Settings shared with the CNNs:** loss, optimizer (Adam, learning rate 1e-4),
      early stopping (patience 20 on val Dice at 2048, threshold 0.5, inside the FOV),
      maximum 150 epochs, AMP and augmentation.
    - **Effective batch:** 4 = micro-batch × gradient accumulation. The micro-batch is
      set by the memory probe; each optimizer step averages over its samples.
    - **Upsampling:** 256 → 1024 (SAM's bilinear postprocessing) uses the deterministic
      integer-factor bilinear upsampling, identical to `F.interpolate` (tested).
    - **Main variant: tiled** (`configs/sam_lora_tiled.yaml`):
      - **Tiles:** each 2048 image is cut into 2×2 non-overlapping native 1024×1024
        tiles, taken from CLAHE images cached at native resolution
        (`python -m src.data native`). That cache is checked against the 1024 cache:
        area-resizing it must reproduce the 1024 cache exactly.
      - **Samples:** one sample = one tile. One epoch = every tile of every training
        image once: the same image content per epoch as the CNNs, but 477 optimizer
        steps instead of 120.
      - **Augmentation:** applied per tile (flips and 90° rotations).
      - **Loss:** the 256 logits upsampled ×4 to 1024, against the native binary mask
        tile. At native resolution nothing is resized, so the soft-target method does
        not apply.
      - **Output:** the four 256 logit grids are stitched into one 512×512 grid →
        sigmoid → probability map at 512. That is the same grid and evaluation path as
        the CNNs, so the CNN mask-resampling reference (`reference_soft_512`) applies to
        it unchanged.
    - **Secondary variant: whole image** (`configs/sam_lora_whole.yaml`):
      - the CLAHE image at 1024, loss against the cached soft 1024 masks;
      - logits upsampled to 1024 → sigmoid → probability map at 1024;
      - decoder grid 256, so its reference is `reference_soft_256`.
    - **Gradient check before training:** 3 optimizer steps on one fixed batch, with
      the training setup (`python -m src.sam_train gradcheck`). Evidence goes to
      `results/checks/sam_gradcheck_*.json`.
- **Phase 5 checks** (job 19042367 for the native cache, job 19042368 for the rest; 1×
  L40S):
  - **Native CLAHE cache:** all 800 images written. Area-resizing each to 1024
    reproduces the Phase 1 cache exactly, with 0 mismatches
    (`results/checks/native_cache.json`).
  - **Gradient check** (`results/checks/sam_gradcheck_sam_lora_{tiled,whole}.json`;
    same result for both variants):
    - **Skipped steps:** the GradScaler skipped the first iterations (inf/NaN gradients
      at its initial scale of 65,536), so 3 steps were applied in 7 iterations. That is
      standard AMP behaviour.
    - **First applied step:** LoRA-A gradients are exactly 0 (B = 0) and all 24 LoRA-B
      tensors have non-zero gradients.
    - **From the second applied step:** all 24 LoRA-A tensors have non-zero gradients.
    - **Mask decoder:** 96 tensors have non-zero gradients.
      - The hypernetwork MLPs of mask tokens 1–3 (18 tensors) get exactly zero
        gradients, because only mask 0 is used (`multimask_output=False`).
      - The IoU head (6 tensors) gets none, because it is not in the loss.
    - **After 3 steps:** every LoRA tensor and the 96 used decoder tensors changed, and
      exactly those 24 decoder tensors did not.
    - **Frozen parts:** the image encoder (177 tensors) and the prompt encoder (16)
      received no gradients and are bit-identical.
  - **Memory and speed probe** (`results/checks/sam_probe_*.json`):
    - Per-sample cost is the same for both variants: about 0.128 s per training sample
      and about 0.05 s per val sample at micro-batch 1, 2 or 4.
    - Peak memory: 8.1 / 15.7 / 30.7 GB allocated at micro-batch 1 / 2 / 4.
    - **Chosen: micro-batch 2 × accumulation 2 = effective batch 4.** It is not faster
      than 4, but leaves memory headroom.
    - **Estimated time per epoch:** tiled 265 s (1,908 tiles + 480 val tiles), whole
      image 67 s (477 + 120 images). At the 150-epoch maximum: 11.1 h and 2.8 h.
  - **5a oracle, val** (`results/eval/sam_oracle_k{5,10,20}/seed0/`,
    `results/eval/sam_oracle_choice.json`):
    - **Chosen: K = 5, threshold 0.20** (mean Dice 0.181; K = 10: 0.180; K = 20: 0.180).
    - **What SAM does:** it segments essentially the whole fundus. Predicted vessel
      density is 0.95–0.97 of the FOV, against 0.098 in the GT.
      - Precision is 0.10.
      - Thin, medium and thick recall are all ≈ 0.98 for the same reason: everything is
        predicted.
      - For comparison, predicting the whole FOV would give Dice 0.177.
    - **Prompt placement is correct** (`results/checks/sam_oracle_prompt_check.json`):
      - 100% of positives lie on GT vessels and 100% of negatives on background.
      - SAM's probability at the negative points is 0.67–0.72 on average, above 0.5 in
        85–97% of images. So SAM overrides the negative prompts; this is not a
        coordinate error.
- **Phase 5 training** (seed 0, 1× L40S on c0903; tiled first, then whole image as a
  dependent job; `results/checks/phase5_train_jobs.txt`). Each job then predicted and
  scored val only; no test predictions.
  - **Tiled** (job 19042606, the main fine-tuned SAM): early stopping after 124 epochs,
    best epoch 103 (val Dice at 0.5: 0.9038).
    - Time: 266.6 s/epoch (241.5 s train + 24.3 s val), 9.18 h in total.
    - Peak GPU memory: 15.66 GB allocated (20.85 GB reserved).
  - **Whole image** (job 19042607): reached max_epochs (150), best epoch 143 (val Dice
    at 0.5: 0.9124). Still improving slowly at the cap: the best val Dice rose by about
    0.001 per 20 epochs over the last 50 epochs. The same 150-epoch cap applies to every
    model (Multi-Scale U-Net seed 1 also reached it).
    - Time: 68.7 s/epoch (61.6 s train + 6.2 s val), 2.86 h in total.
    - Peak GPU memory: 15.66 GB allocated (17.85 GB reserved).
  - **Val results** (`results/tables/val_sam_runs.csv`, `val_sam_vs_cnn_paired.csv`,
    `val_sam_oracle.csv`; `python -m src.analysis sam`; exploratory, one seed):

    | model (val-selected threshold) | Dice | thin | medium | thick | clDice | NSD |
    |---|---|---|---|---|---|---|
    | SAM LoRA, tiled (0.55) | 0.905 | 0.597 | 0.932 | 0.987 | 0.926 | 0.889 |
    | SAM LoRA, whole image (0.50) | 0.912 | 0.489 | 0.914 | 0.985 | 0.922 | 0.881 |
    | U-Net, 3 seeds (mean) | 0.914 | 0.558 | 0.929 | 0.987 | 0.930 | 0.900 |
    | reference_soft_512 (context) | 0.985 | 0.919 | 1.000 | 1.000 | 0.991 | 0.996 |
    | reference_soft_256 (context) | 0.942 | 0.377 | 0.953 | 1.000 | 0.938 | 0.934 |

    - Paired per image vs U-Net (U-Net averaged over its 3 seeds), mean difference [95%
      bootstrap CI]:
      - tiled: Dice −0.008 [−0.010, −0.007], thin recall +0.040 [+0.029, +0.050]
        (higher in 77% of images), NSD −0.011 [−0.014, −0.009];
      - whole image: Dice −0.001 [−0.003, +0.000], thin recall −0.069 [−0.082, −0.056],
        NSD −0.019 [−0.021, −0.017].
    - Whole-image SAM's thin recall (0.489) is above its resampling reference (0.377).
      The reference is not an upper bound: it bilinearly upsamples block-averaged
      *probabilities*, whereas the model's 256 logits are upsampled in *logit* space
      and are not constrained to block averages, so they can keep thin lines sharper.
- **Tile-seam check of tiled SAM** (val only, after the pre-test freeze, before any SAM test
  prediction; `python -m src.analysis seam`; `results/checks/seam_check_summary.json`):
  - **Why:** the 2×2 tiles meet at x = 1024 and y = 1024 and are stitched without overlap.
  - **Measures:**
    1. **Discontinuity:** the mean probability jump across the seam on the model's own
       output grid, divided by the median jump between other adjacent columns and rows
       (about 1 means no seam effect).
    2. **Accuracy near the seam:** Dice and centreline recall in bands of half-width 16,
       32 and 64 px around the seam lines. All models are somewhat worse along the
       image's central lines (optic disc, macula), so a seam band alone is confounded.
       The comparison is therefore a difference-in-differences: (tiled − control) in the
       seam band minus (tiled − control) in bands around the tile midlines (x, y = 512,
       1536). The controls are models without seams: U-Net (seeds averaged per image) and
       whole-image SAM. 95% bootstrap CIs over the 120 val images.
  - **Result for the frozen tiled SAM: a real but local artifact.**
    - Jump ratio: 1.43 (U-Net 0.99, whole-image SAM 1.06); tiled − U-Net +0.44
      [+0.34, +0.56].
    - Difference-in-differences vs U-Net at ±16 / ±32 / ±64 px:
      - Dice −0.029 [−0.034, −0.024] / −0.021 / −0.013;
      - thin recall −0.064 [−0.104, −0.023] / −0.044 / −0.026.
    - The ±16 px band is about 3% of the image, so the effect on whole-image scores is
      small.
  - **User decision (2026-11-21): "frozen main + overlap sensitivity".** The frozen tiled
    SAM stays the main SAM and goes to test unchanged. An inference-only
    overlapping-tile variant is added as a secondary (sensitivity) result, reported in
    its own row.
- **Overlapping-tile sensitivity variant** (`src/sam_overlap.py`, run name
  `sam_lora_tiled_overlap`):
  - **Same model:** the trained tiled SAM (`sam_lora_tiled/seed0/best.pt`); no training
    of its own. Its training cost is the tiled run's.
  - **Inference:**
    - 3×3 tiles of 1024 native px at stride 512 (the 4 main tiles are among them).
    - Each tile's 256 logits are placed on the same 512 grid as the main variant.
    - Overlaps are averaged with a separable tent weight per tile,
      w(i) = 1 − |(i + 0.5)/128 − 1|. With 50% overlap the weights sum to 1 inside the
      image, so no tile edge produces a hard seam (tested).
    - Sigmoid of the blended logits → probability map at 512, scored like every other
      model.
  - **Threshold:** chosen on val by the same rule (0.55).
  - **Val results:**
    - Dice 0.9079, thin recall 0.603, clDice 0.929, NSD 0.894. The frozen tiled SAM
      had 0.9053, 0.597, 0.926 and 0.889.
    - Seam check: jump ratio 1.00 (− U-Net: +0.01 [−0.02, +0.04]). Every
      difference-in-differences CI includes 0 (Dice at ±16 px: −0.001 [−0.004, +0.003]).
    - So the blending removes the seam effect.
  - **Test:** predicted once (part C), only after these val checks.

## 13. Final test evaluation *(Phase 6)*

- **Order of events (2026-11-21):** pre-test freeze commit pushed (§2) → val seam check
  → test jobs. Job ids: `results/checks/phase6_jobs.txt`.
  - **Part A (job 19054745):** the 12 CNN runs (4 models × seeds 0–2), whole-image SAM,
    oracle-prompted SAM (K = 5, chosen on val) and the two mask-resampling references
    (`reference_soft_512`, `reference_soft_256`) on the test ground truth.
  - **Part B (job 19054751):** the frozen tiled SAM on test. Then the overlap variant on
    val only, the seam-check rerun and the SAM val table.
  - **Part C (job 19054772):** the overlap variant on test.
  - **Not predicted on test:** the 1024 pilot (`unet_r1024`), which stays a pilot.
- **Tiled SAM, seeds 1 and 2 (2026-11-21/22; user decision after the first Phase 6
  report):** jobs 19055109 and 19055110, in parallel on two free L40S GPUs (nobody
  preempted). Config unchanged from the freeze (same config hash).
  - Each job trains, predicts and scores val, then predicts the test split once and
    scores it at that seed's own val-selected threshold (`TEST_AFTER=1` in
    `jobs/train.sbatch`; submitted by `jobs/submit_sam_seeds.sh`). Code at commit
    `a0102fb`.
  - Both seeds ran to the 150-epoch cap. Best epochs: seed 1 at 149 (the last epoch),
    seed 2 at 138; seed 0 had stopped early at 124 (best 103). Best val Dice at 0.5:
    0.9038 / 0.9045 / 0.9047 for seeds 0 / 1 / 2. Val-selected thresholds: 0.55 / 0.45 /
    0.45.
  - Then job 19055111 (`afterok` both) reran `src.analysis sam` and
    `src.report --final-test`, so every Phase 6 table and figure now uses tiled SAM as a
    3-seed model, with the same rules as the CNNs.
  - At that point the overlapping-tile variant and whole-image SAM were still seed 0
    only; both were brought to 3 seeds next.
- **Overlap variant and whole-image SAM, seeds 1 and 2 (2026-11-22; user decision):**
  submitted by `jobs/submit_overlap_seeds.sh` (`PARTITION=gpu WHOLE=1`). Job ids:
  `results/checks/phase6_jobs.txt`.
  - **Partition:** every L40S on c0903 was held by other users' `guest_gpu` jobs that an
    `ltarunl` job would have preempted, so these jobs ran on the shared `gpu` partition,
    still on L40S GPUs (`--constraint=gpu_l40s`). Nobody was preempted.
  - **Overlap inference, seeds 1 and 2** (no training; each seed's own tiled `best.pt`):
    one job predicted and scored val (each seed's own threshold: 0.45 / 0.55 for seeds
    1 / 2; seed 0: 0.55), then a second job, only if the first succeeded, predicted the
    test split once per seed and scored it at that threshold. So a failed val step could
    not use up a seed's single test prediction.
  - **Whole-image SAM, seeds 1 and 2:** trained in parallel with the frozen config
    (same config hash as seed 0), then val, then test once (`TEST_AFTER=1`). Like seed 0,
    both ran to the 150-epoch cap. Best epochs 143 / 149 / 145 for seeds 0 / 1 / 2; best
    val Dice at 0.5: 0.9124 for all three (to 4 decimals). Val-selected threshold 0.5
    for all three.
    - Epochs took about 82 s for seeds 1 and 2 (shared `gpu` nodes) against 69 s for
      seed 0 (c0903). This affects training time only. The cost table keeps the seed-0
      training times, which were all measured on c0903.
  - Then the CPU job reran `src.analysis sam` and `src.report --final-test`: every
    automatic model, including all three fine-tuned SAM rows, now has 3 seeds.
  - **Provenance:** these jobs record commit `a0102fb` with uncommitted changes. The
    uncommitted changes were only `src/report.py` (figure labels and seed bands),
    `tests/test_report.py`, `jobs/submit_overlap_seeds.sh`, this document and
    `results/checks/phase6_jobs.txt`. The prediction and scoring code was unchanged.
  - After the report job, the threshold figure was re-plotted once on the login node
    from the saved sweeps (overlap line dashed, because it lies on the tiled line); its
    CSV was byte-identical.
- **Each run predicts the test split once.** `src.predict`, `src.sam_oracle` and
  `src.sam_overlap` require `--final-test` and refuse to overwrite an existing test
  archive (tested).
  - There is no test-time augmentation and no post-processing.
  - Each run is scored at its own val-selected threshold, and the full 0.05–0.95 sweep
    is saved for the secondary analysis.
- **Provenance:** every prediction and score file records the git commit current when
  it was written.
  - The test files record `ee677cd`, `6f1752e` or `f8a76bb`, because commits were made
    while part A was running.
  - The prediction and scoring code is identical in all three: every module those steps
    import (`src.predict`, `src.evaluate`, `src.sam_oracle`, `src.sam_train`, `src.data`,
    `src.metrics`, `src.resample`, `src.utils`, `src.models`, `configs/`).
  - Between those commits, the only code changes were new modules (`src/sam_overlap.py`,
    `src/benchmark.py`, `src/report.py`), `src/analysis.py` and `jobs/submit_phase6.sh`.
    The prediction and scoring steps do not import any of them.
  - The "uncommitted changes" flag was set only by regenerated result files and this
    document.
- **Analysis** (`python -m src.report --final-test`; CLAUDE.md Phase 6):
  - **Seeds:**
    - Model-level numbers: each metric per seed (mean over the 200 images), then mean ±
      sd across seeds.
    - Image-level spread: sd and median over the seed-averaged per-image values.
    - Per-image analyses (Wilcoxon, ablation bootstrap) average each image's metric over
      the model's seeds first.
  - **Statistics:** a paired Wilcoxon signed-rank test (two-sided) per metric, each
    automatic model vs U-Net. Holm correction is applied **within each metric, across
    all models compared with U-Net**, including the overlap sensitivity variant
    (conservative).
  - **Secondary analysis:** every run at a fixed threshold of 0.5, for the
    threshold-dependent metrics (from the sweeps), as decided in §10.
  - **Oracle-prompted SAM:** its own table, never tested against or ranked with the
    automatic models.
  - **Qualitative figures:** 8 test images, 2 per category, drawn with
    `RandomState(0)` from each category's sorted ids.
    - Panels: the image, pixelated outside the FOV; the ground truth; and one error map
      per seed-0 automatic model at its threshold (TP grey, FP orange, FN blue). An error
      map also shows the prediction, since TP ∪ FP = prediction.
    - Each image also gets a native 256×256 crop: the window with the most thin-class
      skeleton pixels, chosen automatically.
- **How to word the SAM results (3 seeds; wording for tiled SAM agreed with the user):**
  - **Thin-vessel recall:** a modest average gain over U-Net (+0.027; mean over images
    of the seed-averaged per-image difference), **within training-seed variability, not
    a robust improvement.**
    - Per-seed test thin recall: SAM 0.574 / 0.537 / 0.598; U-Net 0.544 / 0.565 /
      0.520. Seed sd: 0.031 for SAM, 0.022 for U-Net.
    - SAM seed 1 is below two of the three U-Net seeds.
    - The image-level Wilcoxon test treats the 200 images as the sample, so its small
      p-value does not capture training-seed variability.
  - **Dice:** the deficit is consistent across seeds. Every SAM seed (0.8854–0.8869) is
    below every U-Net seed (0.8910–0.8934); mean difference −0.0065.
  - **Checkpoint selection by val Dice does not control thin recall.** The three SAM
    seeds have near-identical best val Dice (0.9038 / 0.9045 / 0.9047 at threshold 0.5),
    yet their val thin recall at the selected thresholds is 0.597 / 0.558 / 0.623.
  - Across thresholds (`thin_recall_vs_threshold.csv`, test): the 3-seed tiled-SAM mean
    is above the best CNN at all 19 thresholds (+0.015 to +0.049), but its seed range
    overlaps U-Net's seed range at 18 of the 19.
  - **Overlapping-tile variant (3 seeds):** at the same threshold its thin recall equals
    tiled SAM's within 0.004 (test: −0.002 to +0.000; val: +0.001 to +0.004). The
    larger gain seen with seed 0 alone was a seed effect, not an effect of overlapping
    tiles.
    - Removing the seams raises Dice for every seed (0.8883 / 0.8885 / 0.8868 vs tiled
      0.8856 / 0.8869 / 0.8854), but it stays below U-Net (−0.0046).
    - At the val-selected thresholds its thin-recall gain over U-Net is +0.014, smaller
      than tiled SAM's, because val Dice chose a higher threshold for overlap seed 2
      (0.55 vs 0.45).
  - **Whole-image SAM (3 seeds):** the thin-recall deficit is consistent across seeds.
    Every seed (0.4755–0.4812) is below every U-Net seed (0.5204–0.5652); mean difference
    −0.064. Dice is also below every U-Net seed (0.8902–0.8904 vs 0.8910–0.8934).
    Whole-image SAM varies very little between seeds (thin-recall seed sd 0.003).
  - Source files: `results/eval/<run>/seed<s>/{val,test}_metrics.csv`,
    `results/train/<run>/seed<s>/summary.json`, `results/tables/test_main.csv`,
    `test_wilcoxon.csv`, `thin_recall_vs_threshold.csv`.
- **How to word the 2×2 ablation result (wording agreed with the user, parallel to the
  SAM wording):**
  - **Multi-scale:** a modest average decrease in thin recall (−0.021 at val-selected
    thresholds, −0.013 at 0.5), **consistent in direction across threshold rules but
    overlapping training-seed variability.** Per-seed test thin recall: Multi-Scale U-Net
    0.533 / 0.517 / 0.511; U-Net 0.544 / 0.565 / 0.520.
  - **Attention:** **no consistent effect on thin recall** (+0.005 at val-selected
    thresholds, −0.015 at 0.5).
  - **Dice:** **no meaningful effect for either** (main effects within ±0.0006).
  - Source files: `results/tables/test_ablation_effects.csv`,
    `test_ablation_effects_fixed_t0.5.csv`, `results/eval/<run>/seed<s>/test_metrics.csv`.
- **Post hoc descriptive analysis, added during manuscript writing (2026-11-22; user
  decision): agreement between two available annotations of the same image.**
  - Uses labels only: no model predictions, and no earlier result changes.
  - FIVES has 6 byte-identical image files with two different masks (5 in the official
    training set, 1 in test; all category A; data audit, §14). The dataset does not say who
    drew the masks, so this is not called inter-rater agreement.
  - Each mask is scored against the other with the evaluation metrics and settings
    (2048, inside the FOV, `src.metrics.compute_all`), in both directions; each pair's
    value is the mean of the two directions. n = 6, so the result is descriptive only.
  - `python -m src.agreement` → `results/tables/annotation_agreement.csv`,
    `annotation_agreement_directions.csv`, `results/checks/annotation_agreement_info.json`.
- **Cost table** (`python -m src.benchmark`, one GPU job on the training GPU type, after
  all test predictions):
  - **Inference time:** batch 1 (one image per call) over the 200 test images, after 10
    warm-up calls, with `torch.cuda.synchronize()` around each call.
    - Timed span: from the input on the GPU to the probability map at the model's output
      grid, including tiling, stitching or blending and the sigmoid.
    - Disk I/O and CPU preprocessing are excluded.
    - Outputs are discarded, so this is not a second test prediction.
  - **Other columns:** peak GPU memory at inference, FLOPs per image (PyTorch
    `FlopCounterMode`), and training time and peak training memory from the seed-0 run
    summaries.

## 14. Known limitations

- **Patients across splits.** 573 patients, 800 images, no patient IDs, so patient-level
  separation cannot be guaranteed.
  - Within the official training images, the grouped split keeps the strongest evidence
    together (§6).
  - Remaining evidence (burned-in markings compared as binary patterns, never read;
    `results/checks/same_patient_pairs.json`, IoU >= 0.8, final split):
    - Identical markings: 1,515 pairs (508 train–test, 109 val–test, 215 train–val).
      This is an upper bound, because fragments of common characters also match across
      different patients (groups of up to 18 images).
    - Isolated identical pairs (groups of exactly 2, "likely same patient"): 70. Of
      these, 49 are within one split, 13 are train–test, 8 are val–test and **0 are
      train–val**.
    - Identical marking + adjacent numbers: 108 pairs, none across splits (this criterion
      cannot detect train–test pairs).
    - 226 of 800 images have no top-left marking and cannot be checked this way.
- **Near-duplicates across train/test.** `train_174_D` and `test_54_D` are the same eye
  (vessel-layout IoU 0.997). `train_174_D` is therefore excluded from training (§6);
  no near-duplicate pair (layout IoU >= 0.8) remains between train/val and test.
- **Duplicates inside test.** `test_32_A` and `test_48_A` are byte-identical images with
  two different masks (Dice 0.993 between the masks). `test_22_A` and `test_23_A` share a
  byte-identical mask on two near-identical images. Each pair is counted as two test
  images, so the 200 test images are not fully independent.
- **Double annotations in training data.** 5 more byte-identical image pairs (all
  category A, official train) carry two different masks (Dice 0.878–0.982). They are
  kept, grouped on the same side of the split (`results/checks/integrity.json`).
- **Burned-in text.** 574 of 800 images carry text outside the eye, in the top-left
  corner. It is never read, transcribed or stored; figures pixelate it.
- **Data source.** The Hugging Face mirror of FIVES (`tyluan/FIVES`), not the official
  figshare release.
- **Loss with soft targets.** BCE + soft Dice, as in the approved proposal. At the loss's
  own optimum, the soft Dice term pushes soft target values toward 0 or 1: a thin
  vessel's 0.25 at 512 ends up at about 0.07 (§11). The effect is the same for every
  trained model.
- **SAM's output grid.** Whole-image SAM's 256×256 decoder output loses about 62% of
  thin-vessel centreline even with perfect predictions (§10). The main fine-tuned SAM is
  therefore the tiled variant (2×2 native crops, 512 combined grid, as for the CNNs); the
  whole-image variant is reported as a secondary run (§12).
- **Tile seams in the main tiled SAM.** The frozen 2×2 non-overlapping tiling leaves a
  probability discontinuity at the seams, which locally lowers Dice and thin recall within
  roughly ±64 px of x, y = 1024 (§12). The overlapping-tile sensitivity variant (same
  model) removes it on val and is reported next to the main result.
- **Seeds.** Every automatic model has three training seeds (the overlap variant reuses
  the three tiled-SAM models). Three seeds give only a rough estimate of seed-to-seed
  spread. Thin-vessel recall varies by seed much more than Dice does, for SAM as for the
  CNNs (§13), and image-level tests do not capture this.
- **Tiled SAM seeds 1 and 2 reached the 150-epoch cap**; seed 1's best epoch was the last
  one, so it may still have been improving slowly.
- **Whole-image SAM reached the 150-epoch cap** with all three seeds (best epochs 143,
  149, 145), so it may still have been improving slowly (§12, §13).
- **Image-level statistics.** The Wilcoxon tests and the ablation bootstrap CIs resample
  the 200 test images (after averaging each image over seeds). They do not include
  training-seed variability, which for thin-vessel recall (seed sd 0.003–0.031) is
  similar in size to most between-model differences.
- **Threshold dependence.** Thin-vessel recall depends strongly on the operating
  threshold, and val-Dice threshold selection moves it. On test, the attention main
  effect on thin recall is +0.005 at the val-selected thresholds and −0.015 at a fixed
  0.5 (`test_ablation_effects*.csv`). Both analyses are reported.
- **Checkpoint selection by val Dice** does not control thin-vessel recall: runs with
  near-identical best val Dice differ in val thin recall by up to 0.065 (tiled SAM, §13).
- **Evaluation region.** CLAUDE.md asks for overlap metrics over all pixels; every metric
  here is computed inside the evaluation FOV instead (decided after Phase 1, §9), because
  of annotation lines outside the eye. Prediction and ground truth always share the FOV.
- **Boundary metric.** NSD is a boundary-pixel approximation of normalized surface Dice
  (§9).
- **Image-derived vessel measures.** Vessel density and fractal dimension are computed
  from the masks for comparison only. Their clinical validity is not tested here.
- **Oracle-prompted SAM** uses prompts taken from each image's ground truth, so it is not
  comparable to the automatic models.
- **Resolution.** The CNNs and tiled SAM output 512×512 maps (`cnn_resolution` 512);
  whole-image SAM takes a 1024 input and outputs a 256 grid. The 512 resampling reference
  keeps 0.916 of test thin-vessel centreline and the 256 reference 0.400 (context only,
  not upper bounds).
- **Cost numbers** come from one benchmark job on one L40S (seed-0 checkpoints) and from
  the seed-0 training runs.
- **Single dataset.** All results are on FIVES; there is no external validation.

## 15. Post hoc analyses after the referee report *(manuscript revision, 2026-11-23)*
All four were decided by the user after the internal referee report (decisions D2, D4, D7, D9),
**after the main test evaluation**. None predicts anything again, selects anything on the test
set or changes an existing result; each is labelled post hoc in the manuscript.

- **D2: precision-type and operating-point checks** (`src/posthoc.py score`, then `tables`;
  `tests/test_posthoc.py`). The saved probability maps (val and test, 7 configurations × 3
  seeds) are re-scored at 2048 inside the FOV:
  - fine threshold sweep 0.01–0.99 (step 0.01), same metrics as the main sweep
    (`<split>_sweep_fine.csv`; checked against the 0.05-grid sweep);
  - **width-stratified centreline precision**: the mirror of the recall. Skeleton of the
    binary prediction inside the FOV; each predicted skeleton pixel classed by the
    prediction's own EDT radius (thin < 3, medium 3–7, thick > 7 px); precision = share of a
    class's predicted skeleton pixels within 1 px (3 × 3 dilation) of a GT vessel pixel. A
    class with no predicted skeleton pixels in an image is NaN for that image;
  - **clDice and NSD at the fixed 0.5** (the main analysis has them only at the selected
    thresholds);
  - **spur-pruned thin class**: terminal skeleton branches (end point at one end, junction
    at the other) whose pixel count is ≤ 1.5 × the largest EDT radius along the branch + 2 px
    are removed (one pass) before width classing, for the GT skeleton (recall) and the
    predicted skeleton (precision). A first rule (≤ R + 2 px) was replaced before any real
    data were scored because synthetic medial-axis spurs reach R + bump height (unit test);
  - **matched operating point**: per run, the fine-grid threshold whose mean **val**
    precision is closest to a target (ties: lower threshold), applied to test. Primary
    target: the U-Net's mean val precision at its val-selected thresholds (mean over its 3
    seeds); secondary targets 0.90 and 0.95.
- **D4: other resampling schemes** (`src/posthoc.py tables` → `posthoc_stats.csv`), for each
  model vs the U-Net and the 2 × 2 effects, at the selected thresholds, at 0.5 and at matched
  precision; 10,000 replicates, percentile 95% intervals:
  - image bootstrap as in the main analysis (reproduces its intervals; checked);
  - cluster bootstrap over possibly same-patient test clusters, built with the split rules
    on test–test pairs plus identical masks (`results/checks/test_groups.csv`, ids only);
  - seed × cluster bootstrap: clusters as above and, in the same replicate, each model's
    three seeds resampled with replacement (jointly for the four CNNs, whose seed s is
    coupled; independently for SAM vs the U-Net);
  - seed-paired values: the statistic from seed s of every model, s = 0, 1, 2.
- **D7: official release** (`src/official_check.py`). The figshare release (article
  19688169 v1, one RAR5 archive, MD5 789c80dd5376a82063e27fa49192bac9 as listed by
  figshare) was downloaded to `$WORK/FIVES_official/`, its 1600 PNG files were extracted
  (p7zip 16.02, HCC module) and compared with the Hugging Face copy by SHA-256 and, where
  the bytes differed, by decoded pixels; the audit findings were re-derived from the
  official files. Images were only hashed and compared as arrays.
- **D9: audit figure panel B** gets a contrast-stretched inset (green channel, 1st–99.5th
  percentile) of a 768 × 768 window at the FOV centre, checked to lie inside the FOV.
- **Results (2026-11-23; jobs 19075869, 19076386):** every re-scored value at the selected
  thresholds and every 0.05-grid sweep value equals the main analysis (max difference 0;
  `results/checks/posthoc_consistency_{val,test}.json`), and the image bootstrap reproduces the
  main ablation intervals (max difference 2.5e-17). Matched precision (target 0.9256): tiled
  SAM − U-Net thin recall +0.0036 (selected thresholds: +0.0267); multi-scale effect −0.0156;
  whole-image SAM −0.0675. Thin-vessel precision 0.711–0.768 for all models. Pruning removes
  6.7% of the GT thin centreline. 170 test clusters (30 of two images). Official release: 800/800
  images byte-identical, 800/800 masks pixel-identical after channel reduction (790 RGB, 10 RGBA
  with alpha 255); findings unchanged. Full numbers: `results/tables/posthoc_*.csv`,
  `posthoc_info.json`.
- **Not run (user decision):** D1 (resolution-matched CNN control; the one-seed 1024 pilot
  is discussed as suggestive evidence only), D3 (attention-coefficient statistics), D5
  (extended SAM training, learning-rate check), D6 (training-target sensitivity). They are
  limitations and future work in the manuscript.
- **D8 (privacy, repository):** per-image marking locations (`image_qa.csv` columns
  `marking_regions`, `marking_y0/y1/x0/x1`, `topleft_y1/x1`) and the marking review sheets
  moved to the git-ignored `results/checks/private/` (`src/data.py`: `write_qa`, `load_qa`);
  `phase1_checks.json` keeps only aggregate region counts. The public repository keeps image
  ids and pattern-similarity scores (`split_groups.csv`, `marking_pairs.csv`,
  `same_patient_pairs.json`, `test_groups.csv`), needed to reproduce the grouped split.
