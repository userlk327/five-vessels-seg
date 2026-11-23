# Results

Binary segmentation of retinal blood vessels in the FIVES fundus images (Jin et al.,
2022), with a focus on thin, low-contrast vessels. Four CNNs (a U-Net baseline and a 2×2
ablation of attention gates and multi-scale blocks) are compared with the Segment
Anything Model (SAM ViT-B), fine-tuned with LoRA and, separately, prompted zero-shot with
oracle points.

Every number below comes from a file in [`tables/`](tables/) or `eval/` (named under
each table); the tables were generated from those files by script and the text was checked
against them. Nothing is estimated. How each number was produced is in
[`docs/protocol.md`](../docs/protocol.md); a draft of the report's Methods and Results
text is in [`docs/report_draft.md`](../docs/report_draft.md).

**Contents:** [key findings](#key-findings) · [how to read the tables](#how-to-read-the-tables) ·
[1 main table](#1-main-table-automatic-models) · [2 vessel width](#2-width-stratified-centreline-recall) ·
[3 disease categories](#3-disease-categories) · [4 ablation](#4-the-22-ablation-cnns) ·
[5 SAM vs U-Net](#5-fine-tuned-sam-vs-the-u-net-baseline) · [6 vessel measures](#6-image-derived-vessel-measures-optional-extra) ·
[7 cost](#7-computational-cost) · [8 oracle SAM and references](#8-oracle-prompted-sam-separate-and-mask-resampling-references) ·
[9 fixed threshold 0.5](#9-secondary-analysis-every-run-at-a-fixed-threshold-of-05) ·
[figures](#figures) · [limitations](#limitations) · [files](#files)

## Key findings

All on the 200 test images, each run at its own threshold chosen on validation, 3
training seeds per model.

- **Overall overlap is close for every automatic model.** Dice ranges from 0.8860 to
  0.8930. The four CNNs lie within 0.0008 of each other (0.8922–0.8930); the three
  fine-tuned SAM variants are lower (0.8860–0.8903).
- **Thin vessels are the hard part for every model.** Thin-vessel centreline recall is
  0.48–0.57, against 0.87–0.89 for medium and about 0.98 for thick vessels.
- **2×2 ablation (CNNs):**
  - *Multi-scale blocks:* **a modest average decrease in thin recall (−0.021 at
    val-selected thresholds, −0.013 at 0.5), consistent in direction across threshold
    rules but overlapping training-seed variability.** Per-seed thin recall:
    Multi-Scale U-Net 0.511–0.533, U-Net 0.520–0.565.
  - *Attention gates:* **no consistent effect on thin recall** (+0.005 at val-selected
    thresholds, −0.015 at 0.5).
  - *Dice:* **no meaningful effect for either** (main effects within ±0.0006).
  - Details in [section 4](#4-the-22-ablation-cnns).
- **Fine-tuned SAM (tiled, main SAM):**
  - Thin-vessel recall: **a modest average gain of +0.027 over U-Net, within
    training-seed variability; not a robust improvement.**
  - Dice: **the deficit (−0.0065) is consistent across seeds.**
  - **Checkpoint selection by validation Dice does not control thin-vessel recall.**
  - Details in [section 5](#5-fine-tuned-sam-vs-the-u-net-baseline).
- **Whole-image SAM** (256 × 256 decoder output) loses thin vessels consistently:
  −0.064 thin recall vs U-Net, in every seed.
- **Cost:** tiled SAM takes 190 ms per image at inference against 4.4 ms for U-Net
  (43×), and 9.2 h of training against 0.43 h (seed-0 runs, one L40S).
- **Disease categories:** glaucoma images have the lowest Dice for every model
  (0.847–0.854); normal images have the lowest thin-vessel recall for every model.
- **Oracle-prompted SAM** (zero-shot, ground-truth point prompts; not comparable to the
  automatic models) labels most of the field of view as vessel: Dice 0.17.

## How to read the tables

- **Scoring:** every metric is computed per image at 2048 × 2048 against the original
  ground truth, inside the field-of-view (FOV) mask. Probability maps are upsampled
  bilinearly to 2048, then thresholded at the run's own threshold, chosen on the 120
  validation images (grid 0.05–0.95, highest mean Dice).
- **Seeds:** "mean ± sd" is the mean across the **3 training seeds** of each seed's mean
  over the 200 test images, ± the sd across seeds. 3 seeds × 200 images are still 200
  test images.
- **p-values** (in brackets after the mean): two-sided paired Wilcoxon signed-rank test
  against U-Net on per-image values (each image averaged over the model's seeds first,
  n = 200), Holm-corrected within each metric across the six comparisons with U-Net.
  They measure image-level differences, not training-seed variability.
- **Width classes:** thin, medium and thick are the ground-truth skeleton pixels with a
  local radius of < 3, 3–7 and > 7 px at 2048. Recall = share of those skeleton pixels
  found by the prediction (1-px tolerance).
- **Overlapping tiles** is an inference-only sensitivity variant of the tiled SAM (same
  three trained models; 3 × 3 overlapping tiles with blended logits, which removes the
  tile seams). It is not a separate model.
- **Oracle-prompted SAM** is never ranked against the automatic models and appears only
  in section 8.

## 1. Main table (automatic models)

Mean ± sd across 3 seeds (Holm-adjusted p vs U-Net). Source: `tables/test_main.csv`,
`tables/test_wilcoxon.csv`.

| Model | Dice | IoU | Precision | Recall | clDice | NSD |
|---|---|---|---|---|---|---|
| U-Net (baseline) | 0.8925 ± 0.0013 | 0.8140 ± 0.0018 | 0.9111 ± 0.0123 | 0.8803 ± 0.0126 | 0.9083 ± 0.0021 | 0.8580 ± 0.0026 |
| Attention U-Net | 0.8929 ± 0.0015 (0.248) | 0.8146 ± 0.0021 (0.262) | 0.9082 ± 0.0099 (6.1e-14) | 0.8841 ± 0.0121 (2.6e-19) | 0.9094 ± 0.0021 (1.3e-4) | 0.8590 ± 0.0024 (0.026) |
| Multi-Scale U-Net | 0.8930 ± 0.0016 (0.002) | 0.8150 ± 0.0019 (0.001) | 0.9206 ± 0.0038 (1.8e-22) | 0.8722 ± 0.0058 (1.4e-19) | 0.9071 ± 0.0022 (1.6e-4) | 0.8584 ± 0.0015 (0.652) |
| Multi-Scale Attention U-Net | 0.8922 ± 0.0011 (0.258) | 0.8135 ± 0.0018 (0.280) | 0.9080 ± 0.0103 (4.1e-5) | 0.8822 ± 0.0073 (0.007) | 0.9085 ± 0.0002 (0.900) | 0.8559 ± 0.0041 (7.7e-4) |
| SAM LoRA, tiled (main SAM) | 0.8860 ± 0.0008 (1.7e-23) | 0.8029 ± 0.0014 (1.9e-24) | 0.8990 ± 0.0091 (6.8e-24) | 0.8775 ± 0.0086 (5.7e-5) | 0.9062 ± 0.0010 (3.9e-6) | 0.8480 ± 0.0011 (2.9e-19) |
| SAM LoRA, tiled, overlapping tiles | 0.8879 ± 0.0009 (6.8e-18) | 0.8062 ± 0.0013 (5.8e-19) | 0.9086 ± 0.0062 (0.001) | 0.8737 ± 0.0076 (1.5e-11) | 0.9079 ± 0.0011 (0.317) | 0.8534 ± 0.0008 (1.6e-8) |
| SAM LoRA, whole image | 0.8903 ± 0.0001 (2.7e-7) | 0.8096 ± 0.0001 (2.5e-7) | 0.9082 ± 0.0027 (0.676) | 0.8764 ± 0.0026 (5.7e-5) | 0.9006 ± 0.0006 (5.9e-19) | 0.8387 ± 0.0012 (4.4e-27) |

**Image-level spread:** sd over images / median over images, of the seed-averaged
per-image values. Source: `tables/test_main.csv`.

| Model | Dice | IoU | Precision | Recall | clDice | NSD | Thin | Medium | Thick |
|---|---|---|---|---|---|---|---|---|---|
| U-Net (baseline) | 0.088 / 0.9167 | 0.106 / 0.8462 | 0.048 / 0.9272 | 0.104 / 0.9058 | 0.090 / 0.9349 | 0.129 / 0.8979 | 0.135 / 0.5551 | 0.133 / 0.9361 | 0.082 / 0.9978 |
| Attention U-Net | 0.086 / 0.9158 | 0.105 / 0.8447 | 0.045 / 0.9215 | 0.104 / 0.9096 | 0.090 / 0.9338 | 0.128 / 0.8989 | 0.138 / 0.5586 | 0.133 / 0.9384 | 0.081 / 0.9979 |
| Multi-Scale U-Net | 0.089 / 0.9168 | 0.107 / 0.8464 | 0.050 / 0.9369 | 0.102 / 0.8982 | 0.091 / 0.9325 | 0.128 / 0.8988 | 0.133 / 0.5231 | 0.133 / 0.9335 | 0.086 / 0.9978 |
| Multi-Scale Attention U-Net | 0.087 / 0.9159 | 0.106 / 0.8448 | 0.050 / 0.9250 | 0.103 / 0.9088 | 0.089 / 0.9337 | 0.130 / 0.8978 | 0.132 / 0.5295 | 0.133 / 0.9325 | 0.082 / 0.9976 |
| SAM LoRA, tiled (main SAM) | 0.084 / 0.9084 | 0.102 / 0.8321 | 0.053 / 0.9153 | 0.100 / 0.9021 | 0.087 / 0.9296 | 0.128 / 0.8879 | 0.141 / 0.5746 | 0.132 / 0.9317 | 0.076 / 0.9966 |
| SAM LoRA, tiled, overlapping tiles | 0.087 / 0.9108 | 0.103 / 0.8362 | 0.046 / 0.9233 | 0.102 / 0.9004 | 0.090 / 0.9322 | 0.129 / 0.8927 | 0.145 / 0.5790 | 0.135 / 0.9330 | 0.083 / 0.9973 |
| SAM LoRA, whole image | 0.081 / 0.9141 | 0.103 / 0.8418 | 0.059 / 0.9282 | 0.096 / 0.8984 | 0.081 / 0.9254 | 0.129 / 0.8814 | 0.129 / 0.4665 | 0.132 / 0.9139 | 0.073 / 0.9974 |

**Per-image difference from U-Net:** mean of (model − U-Net) over the 200 images (share
of images where the model is higher). Source: `tables/test_wilcoxon.csv`.

| Model − U-Net | Dice | Thin | clDice | NSD |
|---|---|---|---|---|
| Attention U-Net | +0.0005 (56%) | +0.0035 (60%) | +0.0010 (66%) | +0.0010 (57%) |
| Multi-Scale U-Net | +0.0005 (64%) | −0.0228 (24%) | −0.0012 (36%) | +0.0005 (48%) |
| Multi-Scale Attention U-Net | −0.0002 (48%) | −0.0161 (27%) | +0.0002 (50%) | −0.0021 (40%) |
| SAM LoRA, tiled (main SAM) | −0.0065 (10%) | +0.0267 (68%) | −0.0021 (32%) | −0.0100 (17%) |
| SAM LoRA, tiled, overlapping tiles | −0.0046 (12%) | +0.0139 (62%) | −0.0004 (42%) | −0.0046 (32%) |
| SAM LoRA, whole image | −0.0022 (35%) | −0.0644 (18%) | −0.0078 (13%) | −0.0193 (6%) |

## 2. Width-stratified centreline recall

Mean ± sd across 3 seeds (Holm-adjusted p vs U-Net). Main cut-offs: radius 3 / 7 px.
Source: `tables/test_width.csv`, `tables/test_wilcoxon.csv`.

| Model | Thin | Medium | Thick |
|---|---|---|---|
| U-Net (baseline) | 0.5430 ± 0.0224 | 0.8921 ± 0.0056 | 0.9786 ± 0.0013 |
| Attention U-Net | 0.5465 ± 0.0230 (0.006) | 0.8943 ± 0.0047 (1.2e-4) | 0.9793 ± 0.0004 (0.008) |
| Multi-Scale U-Net | 0.5202 ± 0.0111 (7.2e-15) | 0.8903 ± 0.0011 (3.7e-6) | 0.9780 ± 0.0020 (0.777) |
| Multi-Scale Attention U-Net | 0.5269 ± 0.0058 (9.7e-10) | 0.8884 ± 0.0021 (1.3e-10) | 0.9778 ± 0.0011 (0.038) |
| SAM LoRA, tiled (main SAM) | 0.5697 ± 0.0308 (8.9e-8) | 0.8894 ± 0.0058 (0.109) | 0.9784 ± 0.0013 (0.002) |
| SAM LoRA, tiled, overlapping tiles | 0.5569 ± 0.0203 (0.004) | 0.8894 ± 0.0053 (0.194) | 0.9779 ± 0.0022 (0.023) |
| SAM LoRA, whole image | 0.4786 ± 0.0029 (1.8e-19) | 0.8746 ± 0.0018 (1.2e-20) | 0.9797 ± 0.0010 (0.228) |

**Sensitivity check, cut-offs 4 / 8 px** (same predictions; secondary only):

| Model | Thin (4/8) | Medium (4/8) | Thick (4/8) |
|---|---|---|---|
| U-Net (baseline) | 0.6806 ± 0.0165 | 0.9286 ± 0.0037 | 0.9842 ± 0.0011 |
| Attention U-Net | 0.6829 ± 0.0153 (7.1e-4) | 0.9304 ± 0.0029 (0.005) | 0.9848 ± 0.0003 (0.034) |
| Multi-Scale U-Net | 0.6667 ± 0.0077 (3.3e-14) | 0.9281 ± 0.0009 (0.020) | 0.9834 ± 0.0022 (0.783) |
| Multi-Scale Attention U-Net | 0.6699 ± 0.0035 (1.5e-10) | 0.9262 ± 0.0017 (3.2e-7) | 0.9836 ± 0.0012 (0.783) |
| SAM LoRA, tiled (main SAM) | 0.6918 ± 0.0210 (0.004) | 0.9265 ± 0.0034 (0.025) | 0.9838 ± 0.0014 (0.003) |
| SAM LoRA, tiled, overlapping tiles | 0.6843 ± 0.0159 (0.159) | 0.9271 ± 0.0034 (0.054) | 0.9832 ± 0.0027 (0.020) |
| SAM LoRA, whole image | 0.6287 ± 0.0023 (6.3e-22) | 0.9209 ± 0.0012 (2.7e-12) | 0.9851 ± 0.0010 (0.783) |

The mask-resampling references in section 8 show how much thin centreline the output
grids themselves keep (context, not upper bounds).

## 3. Disease categories

Mean ± sd across 3 seeds, per category (descriptive; no tests between categories were
planned). Source: `tables/test_disease.csv` (all metrics).

**Dice**

| Model | AMD (A, n=50) | DR (D, n=50) | Glaucoma (G, n=50) | Normal (N, n=50) |
|---|---|---|---|---|
| U-Net (baseline) | 0.9165 ± 0.0017 | 0.8954 ± 0.0018 | 0.8531 ± 0.0014 | 0.9050 ± 0.0005 |
| Attention U-Net | 0.9167 ± 0.0011 | 0.8956 ± 0.0026 | 0.8535 ± 0.0017 | 0.9059 ± 0.0007 |
| Multi-Scale U-Net | 0.9175 ± 0.0006 | 0.8957 ± 0.0011 | 0.8534 ± 0.0038 | 0.9053 ± 0.0015 |
| Multi-Scale Attention U-Net | 0.9167 ± 0.0010 | 0.8950 ± 0.0009 | 0.8531 ± 0.0017 | 0.9041 ± 0.0020 |
| SAM LoRA, tiled (main SAM) | 0.9100 ± 0.0010 | 0.8895 ± 0.0008 | 0.8467 ± 0.0007 | 0.8977 ± 0.0011 |
| SAM LoRA, tiled, overlapping tiles | 0.9115 ± 0.0003 | 0.8922 ± 0.0007 | 0.8488 ± 0.0014 | 0.8989 ± 0.0017 |
| SAM LoRA, whole image | 0.9156 ± 0.0004 | 0.8940 ± 0.0002 | 0.8517 ± 0.0008 | 0.8999 ± 0.0002 |

**Thin-vessel centreline recall**

| Model | AMD (A, n=50) | DR (D, n=50) | Glaucoma (G, n=50) | Normal (N, n=50) |
|---|---|---|---|---|
| U-Net (baseline) | 0.5498 ± 0.0295 | 0.5426 ± 0.0208 | 0.5722 ± 0.0165 | 0.5075 ± 0.0293 |
| Attention U-Net | 0.5492 ± 0.0318 | 0.5469 ± 0.0307 | 0.5685 ± 0.0208 | 0.5215 ± 0.0181 |
| Multi-Scale U-Net | 0.5293 ± 0.0103 | 0.5268 ± 0.0209 | 0.5555 ± 0.0034 | 0.4691 ± 0.0151 |
| Multi-Scale Attention U-Net | 0.5329 ± 0.0087 | 0.5306 ± 0.0078 | 0.5665 ± 0.0136 | 0.4776 ± 0.0245 |
| SAM LoRA, tiled (main SAM) | 0.5987 ± 0.0297 | 0.5667 ± 0.0331 | 0.5963 ± 0.0274 | 0.5172 ± 0.0338 |
| SAM LoRA, tiled, overlapping tiles | 0.5918 ± 0.0156 | 0.5520 ± 0.0254 | 0.5814 ± 0.0220 | 0.5026 ± 0.0188 |
| SAM LoRA, whole image | 0.4897 ± 0.0066 | 0.4875 ± 0.0018 | 0.5175 ± 0.0043 | 0.4197 ± 0.0028 |

## 4. The 2×2 ablation (CNNs)

Per image, for each metric (U-Net, Att = Attention U-Net, MS = Multi-Scale U-Net,
MS-Att = Multi-Scale Attention U-Net; each image averaged over seeds first):

- attention main effect = mean of [(Att − U-Net), (MS-Att − MS)];
- multi-scale main effect = mean of [(MS − U-Net), (MS-Att − Att)];
- interaction = (MS-Att − MS) − (Att − U-Net).

Mean over the 200 images [95% bootstrap CI, resampling images]. Primary: each run at
its val-selected threshold; secondary: every run at 0.5 (clDice and NSD exist only at
the selected thresholds). Source: `tables/test_ablation_effects.csv`,
`tables/test_ablation_effects_fixed_t0.5.csv` (all metrics).

| Threshold | Metric | Attention main effect | Multi-scale main effect | Interaction |
|---|---|---|---|---|
| val-selected | Dice | −0.0001 [−0.0005, +0.0003] | −0.0001 [−0.0008, +0.0005] | −0.0012 [−0.0019, −0.0005] |
| val-selected | Thin | +0.0051 [+0.0023, +0.0079] | −0.0212 [−0.0260, −0.0163] | +0.0032 [−0.0022, +0.0089] |
| val-selected | clDice | +0.0012 [+0.0007, +0.0017] | −0.0011 [−0.0018, −0.0003] | +0.0004 [−0.0008, +0.0015] |
| val-selected | NSD | −0.0008 [−0.0014, −0.0002] | −0.0013 [−0.0022, −0.0004] | −0.0035 [−0.0048, −0.0023] |
| val-selected | Precision | −0.0078 [−0.0084, −0.0072] | +0.0046 [+0.0030, +0.0062] | −0.0098 [−0.0109, −0.0087] |
| val-selected | Recall | +0.0069 [+0.0063, +0.0075] | −0.0050 [−0.0065, −0.0034] | +0.0062 [+0.0048, +0.0076] |
| 0.5 | Dice | +0.0001 [−0.0003, +0.0004] | −0.0006 [−0.0013, −0.0000] | −0.0006 [−0.0014, +0.0002] |
| 0.5 | Thin | −0.0152 [−0.0180, −0.0124] | −0.0131 [−0.0177, −0.0082] | −0.0144 [−0.0198, −0.0087] |
| 0.5 | Precision | +0.0082 [+0.0076, +0.0087] | −0.0027 [−0.0043, −0.0012] | +0.0046 [+0.0034, +0.0058] |
| 0.5 | Recall | −0.0075 [−0.0081, −0.0069] | +0.0010 [−0.0005, +0.0025] | −0.0060 [−0.0075, −0.0045] |

- **Multi-scale blocks: a modest average decrease in thin recall (−0.021 at
  val-selected thresholds, −0.013 at 0.5), consistent in direction across threshold rules
  but overlapping training-seed variability.**
  - Per-seed test thin recall: Multi-Scale U-Net 0.533 / 0.517 / 0.511; U-Net 0.544 /
    0.565 / 0.520 (`eval/<run>/seed<s>/test_metrics.csv`).
- **Attention gates: no consistent effect on thin recall.** The effect is +0.005 at the
  val-selected thresholds and −0.015 at 0.5. The val-selected thresholds differ between
  runs (0.45–0.60), and attention shifts the balance between precision and recall.
- **Dice: no meaningful effect for either module** (main effects within ±0.0006;
  interaction −0.0012).
- The CIs resample images only. Training-seed variability (thin recall seed sd
  0.006–0.023 for the CNNs) is not included.

## 5. Fine-tuned SAM vs the U-Net baseline

- **Thin-vessel recall (tiled SAM): a modest average gain of +0.027 over U-Net, within
  training-seed variability. It is not a robust improvement.**
  - Per-seed test thin recall: SAM 0.574 / 0.537 / 0.598; U-Net 0.544 / 0.565 / 0.520.
    SAM seed 1 is below two of the three U-Net seeds.
  - Across thresholds 0.05–0.95 (`figures/thin_recall_vs_threshold.png`,
    `tables/thin_recall_vs_threshold.csv`), the 3-seed SAM mean is above the best CNN at
    every threshold (+0.015 to +0.049), but SAM's seed range overlaps U-Net's seed range
    at 18 of the 19 thresholds.
  - The Wilcoxon test (p = 8.9e-8) treats the 200 images as the sample, so it does not
    capture training-seed variability.
- **Dice (tiled SAM): the deficit is consistent across seeds.** Every SAM seed
  (0.8854–0.8869) is below every U-Net seed (0.8910–0.8934); mean difference −0.0065.
- **Checkpoint selection by validation Dice does not control thin-vessel recall.** The
  three tiled-SAM seeds have near-identical best validation Dice (0.9038 / 0.9045 /
  0.9047 at threshold 0.5), yet their validation thin recall is 0.597 / 0.558 / 0.623.
- **Overlapping-tile inference** (same three models, seams removed by blending):
  - At the same threshold its thin recall equals tiled SAM's within 0.004. A larger
    thin-recall gain seen earlier with seed 0 alone was a seed effect.
  - It raises Dice for every seed (0.8883 / 0.8885 / 0.8868 vs 0.8856 / 0.8869 /
    0.8854), but Dice stays below U-Net (−0.0046).
  - At its val-selected thresholds its thin-recall gain over U-Net is +0.014, smaller
    than the tiled SAM's, because validation Dice chose a higher threshold for seed 2
    (0.55 vs 0.45).
- **Whole-image SAM:** the thin-recall deficit is consistent across seeds. Every seed
  (0.4755–0.4812) is below every U-Net seed (0.5204–0.5652); mean difference −0.064. Its
  256 × 256 decoder grid alone keeps only 0.400 of thin centreline (section 8).

Per-seed sources: `eval/<run>/seed<s>/{val,test}_metrics.csv`,
`train/<run>/seed<s>/summary.json`.

## 6. Image-derived vessel measures (optional extra)

Vessel density (vessel pixels / FOV pixels) and fractal dimension (FD; box counting),
predicted vs ground truth, same FOV for both. Per-image values averaged over seeds;
MAD = mean absolute difference; bias = mean (predicted − GT) with 95% Bland–Altman
limits of agreement; r = Pearson correlation. n = 200. Source:
`tables/test_vessel_measures.csv`; plot `figures/test_bland_altman.png`. These measures
are not clinically validated here.

| Model | Density: pred vs GT | MAD | Bias [95% LoA] | r | FD: pred vs GT | MAD | Bias [95% LoA] | r |
|---|---|---|---|---|---|---|---|---|
| U-Net (baseline) | 0.0912 vs 0.0934 | 0.0037 | −0.0021 [−0.0110, +0.0067] | 0.987 | 1.4727 vs 1.4904 | 0.0182 | −0.0177 [−0.0700, +0.0346] | 0.930 |
| Attention U-Net | 0.0921 vs 0.0934 | 0.0036 | −0.0013 [−0.0104, +0.0077] | 0.988 | 1.4742 vs 1.4904 | 0.0168 | −0.0161 [−0.0688, +0.0366] | 0.932 |
| Multi-Scale U-Net | 0.0892 vs 0.0934 | 0.0046 | −0.0042 [−0.0129, +0.0044] | 0.986 | 1.4683 vs 1.4904 | 0.0226 | −0.0221 [−0.0782, +0.0341] | 0.913 |
| Multi-Scale Attention U-Net | 0.0915 vs 0.0934 | 0.0034 | −0.0019 [−0.0109, +0.0070] | 0.986 | 1.4740 vs 1.4904 | 0.0170 | −0.0164 [−0.0622, +0.0295] | 0.935 |
| SAM LoRA, tiled (main SAM) | 0.0918 vs 0.0934 | 0.0033 | −0.0016 [−0.0101, +0.0070] | 0.988 | 1.4749 vs 1.4904 | 0.0162 | −0.0155 [−0.0545, +0.0235] | 0.948 |
| SAM LoRA, tiled, overlapping tiles | 0.0906 vs 0.0934 | 0.0038 | −0.0028 [−0.0117, +0.0061] | 0.986 | 1.4721 vs 1.4904 | 0.0188 | −0.0182 [−0.0640, +0.0275] | 0.938 |
| SAM LoRA, whole image | 0.0905 vs 0.0934 | 0.0039 | −0.0029 [−0.0117, +0.0060] | 0.986 | 1.4749 vs 1.4904 | 0.0160 | −0.0155 [−0.0450, +0.0140] | 0.961 |

## 7. Computational cost

One L40S GPU. Inference: batch 1 over the 200 test images after 10 warm-up calls, from
the input on the GPU to the probability map (tiling, stitching or blending included),
seed-0 checkpoints; FLOPs from PyTorch's `FlopCounterMode`. Training time and memory:
seed-0 runs. Source: `tables/cost.csv`, `tables/model_params.csv`.

| Model | Parameters | Trainable | Inference ms/image | GFLOPs/image | Inference peak GPU (GB) | Training time (seed 0) | Training peak GPU (GB) |
|---|---|---|---|---|---|---|---|
| U-Net (baseline) | 31,037,633 | 31,037,633 | 4.4 ± 0.1 | 385 | 0.49 | 0.43 h | 3.36 |
| Attention U-Net | 32,083,077 | 32,083,077 | 5.3 ± 0.1 | 392 | 0.50 | 0.41 h | 3.41 |
| Multi-Scale U-Net | 32,788,417 | 32,788,417 | 7.8 ± 0.0 | 405 | 0.50 | 0.45 h | 4.33 |
| Multi-Scale Attention U-Net | 33,833,861 | 33,833,861 | 8.4 ± 0.0 | 411 | 0.51 | 0.41 h | 4.38 |
| SAM LoRA, tiled (main SAM) | 93,882,928 | 4,205,796 | 190.2 ± 0.3 | 3,908 | 5.62 | 9.18 h | 15.66 |
| SAM LoRA, tiled, overlapping tiles | 93,882,928 | 4,205,796 | 428.7 ± 0.4 | 8,793 | 4.39 | 9.18 h (tiled run) | 15.66 |
| SAM LoRA, whole image | 93,882,928 | 4,205,796 | 48.0 ± 0.1 | 977 | 1.68 | 2.86 h | 15.66 |
| SAM zero-shot, oracle prompts (K=5) | 93,735,472 | 0 | 87.5 ± 0.2 | 976 | 2.91 | – | – |

Training times of all seeds are in `docs/protocol.md` §11. Later SAM seeds took longer
(tiled 11.1–11.2 h, whole image 3.4–3.5 h), partly on busier nodes.

## 8. Oracle-prompted SAM (separate) and mask-resampling references

- **Oracle-prompted SAM ("zero-shot, oracle point prompts")** gets 5 positive and 5
  negative points sampled from each image's own ground truth (test images included).
  It is a separate experiment and is not comparable to the automatic models.
  - K = 5 and threshold 0.20 were chosen on validation.
  - It labels most of the FOV as vessel: predicted density 0.950 against 0.093 in the
    ground truth (`tables/test_sam_oracle.csv`).
- **Mask-resampling references** push each test ground truth through a model family's
  output grid and back to 2048 (soft downsampling, bilinear upsampling, same threshold
  rule). They show how much the resizing alone loses. They are **context, not upper
  bounds**.

Means over the 200 test images. Source: `tables/test_sam_oracle.csv`,
`tables/test_references.csv`.

| Row | Dice | Precision | Recall | Thin | Medium | Thick | clDice | NSD |
|---|---|---|---|---|---|---|---|---|
| SAM zero-shot, oracle point prompts (K = 5) | 0.1735 | 0.0960 | 0.9750 | 0.9759 | 0.9798 | 0.9769 | 0.1463 | 0.0183 |
| reference_soft_512 (CNN and tiled-SAM output grid) | 0.9846 | 0.9856 | 0.9835 | 0.9159 | 1.0000 | 1.0000 | 0.9904 | 0.9955 |
| reference_soft_256 (whole-image SAM decoder grid) | 0.9415 | 0.9669 | 0.9176 | 0.4003 | 0.9555 | 1.0000 | 0.9405 | 0.9363 |

## 9. Secondary analysis: every run at a fixed threshold of 0.5

Mean ± sd across 3 seeds (Holm-adjusted p vs U-Net). Threshold-dependent metrics only
(from the saved threshold sweeps). Source: `tables/test_main_fixed_t0.5.csv`,
`tables/test_wilcoxon_fixed_t0.5.csv` (all metrics).

| Model | Dice | Precision | Recall | Thin | Medium |
|---|---|---|---|---|---|
| U-Net (baseline) | 0.8918 ± 0.0015 | 0.9057 ± 0.0033 | 0.8838 ± 0.0019 | 0.5496 ± 0.0103 | 0.8933 ± 0.0039 |
| Attention U-Net | 0.8922 ± 0.0012 (0.074) | 0.9116 ± 0.0056 (1.4e-28) | 0.8793 ± 0.0028 (6.8e-24) | 0.5417 ± 0.0100 (2.9e-4) | 0.8935 ± 0.0029 (0.936) |
| Multi-Scale U-Net | 0.8914 ± 0.0023 (0.003) | 0.9007 ± 0.0066 (1.3e-12) | 0.8878 ± 0.0028 (5.0e-9) | 0.5438 ± 0.0219 (0.008) | 0.8940 ± 0.0024 (0.618) |
| Multi-Scale Attention U-Net | 0.8912 ± 0.0012 (0.100) | 0.9112 ± 0.0048 (2.6e-12) | 0.8772 ± 0.0069 (1.0e-16) | 0.5214 ± 0.0200 (4.2e-19) | 0.8874 ± 0.0053 (4.4e-17) |
| SAM LoRA, tiled (main SAM) | 0.8842 ± 0.0003 (3.3e-25) | 0.9012 ± 0.0093 (2.7e-5) | 0.8719 ± 0.0078 (3.3e-22) | 0.5645 ± 0.0397 (0.008) | 0.8884 ± 0.0082 (9.8e-4) |
| SAM LoRA, tiled, overlapping tiles | 0.8863 ± 0.0005 (1.8e-20) | 0.9028 ± 0.0094 (1.7e-4) | 0.8760 ± 0.0080 (3.0e-14) | 0.5632 ± 0.0404 (0.008) | 0.8905 ± 0.0086 (0.407) |
| SAM LoRA, whole image | 0.8903 ± 0.0001 (0.002) | 0.9082 ± 0.0027 (7.0e-5) | 0.8764 ± 0.0026 (2.5e-11) | 0.4786 ± 0.0029 (2.5e-21) | 0.8746 ± 0.0018 (3.8e-22) |

## Figures

In [`figures/`](figures/):

- `test_main_metrics.png`: Dice, clDice, NSD and thin recall per model (bar = 3-seed
  mean, whisker = sd across seeds).
- `test_width_recall.png`: thin, medium and thick centreline recall per model.
- `test_disease.png`: Dice and thin recall per disease category.
- `thin_recall_vs_threshold.png`: thin recall against the threshold (0.05–0.95), val and
  test; line = mean over seeds, band = range across seeds; CNNs and SAM variants in
  separate rows.
- `test_bland_altman.png`: vessel density and fractal dimension, predicted vs ground
  truth.
- `test_qualitative_<image_id>.png` (8 images, 2 per category, drawn at random with a
  fixed seed): the image (pixelated outside the FOV), the ground truth, and an error map
  per model (seed 0; grey true positive, orange false positive, blue false negative),
  plus a native-resolution 256 × 256 crop chosen automatically as the window with the
  most thin-class skeleton pixels.
- `val_dice_curves_cnns.png`: validation Dice per epoch for the CNN runs.

Also: training curves per run in `train/<run>/seed<s>/curves.png`, and the Phase 1–5
check figures in `checks/` (CLAHE before/after, mask overlay, mask resizing crops, FOV
examples, vessel-radius histogram).

## Limitations

- **Seeds:** 3 training seeds per model. Image-level statistics (Wilcoxon tests,
  bootstrap CIs) do not capture training-seed variability, which for thin-vessel recall
  (seed sd 0.003–0.031) is similar in size to most between-model differences.
- **Input and output resolution:** the CNNs work at `cnn_resolution` = 512 (input and
  output). SAM takes a 1024 input and its decoder outputs 256 × 256. Tiled SAM covers
  the image with 4 native 1024 crops, so its combined output is 512, as for the CNNs;
  whole-image SAM's output is 256.
- **Oracle-prompted SAM** uses prompts derived from each image's ground truth and is not
  comparable to the automatic models.
- **Training-mask resizing:** CNN and whole-image SAM targets are soft masks (area
  average, no threshold) at 512 and 1024. Pushed back to 2048, the 512 soft mask keeps
  0.916 of test thin-vessel centreline at threshold 0.5, and SAM's 256 grid 0.400 (the
  mask-resampling references). Tiled SAM trains on native binary crops.
- **Threshold dependence:** thin-vessel recall depends strongly on the threshold, and
  validation-Dice selection moves it; the attention effect changes sign between the
  val-selected thresholds and a fixed 0.5. Checkpoint selection by validation Dice does
  not control thin-vessel recall.
- **Vessel density and fractal dimension** are image-derived measures, not clinically
  validated.
- **Boundary metric:** NSD is a boundary-pixel approximation of normalized surface Dice
  (boundary pixels are counted, not weighted by surface length; tolerance 2 px).
- **Evaluation region:** every metric is computed inside a FOV mask (fitted to the eye),
  not over all pixels, because some ground-truth masks contain annotation lines outside
  the eye.
- **Patients across splits:** FIVES has 800 images from 573 patients and no patient IDs,
  so the same patient may appear in different splits. Likely same-patient pairs within
  the official training images were kept on one side of the train/val split; one
  training image that is the same eye as a test image was excluded from training.
- **Test set:** two test pairs are duplicates or near-duplicates, so the 200 test images
  are not fully independent.
- **Data source:** the Hugging Face mirror of FIVES (`tyluan/FIVES`), not the official
  figshare release. A post hoc check (2026-11-23, `src/official_check.py`,
  `results/checks/official_release_check.json`) found the mirror identical to the release:
  800/800 images byte-identical, 800/800 masks pixel-identical (the release stores three
  identical colour channels, ten masks with an opaque alpha channel). No other dataset was
  used, so there is no external validation.
- **Post hoc analyses after the referee report** (`docs/protocol.md` §15,
  `results/tables/posthoc_*.csv`): added after the main test evaluation, no new predictions.
  At matched validation precision the tiled-SAM thin-recall advantage shrinks from +0.027 to
  +0.004; the multi-scale decrease persists (−0.016).
- **Training length:** tiled SAM seeds 1 and 2, all three whole-image SAM seeds and
  Multi-Scale U-Net seed 1 reached the 150-epoch cap.
- **Tile seams:** the main tiled SAM stitches four non-overlapping tiles, which leaves a
  local seam artifact; the overlapping-tile variant removes it (section 5).

The full list, with evidence, is in `docs/protocol.md` §14.

## Files

- `tables/test_*.csv`: all Phase 6 tables (per-seed means, sd, medians, tests, CIs).
- `tables/val_*.csv`: validation results per run (exploratory).
- `eval/<run>/seed<s>/`: per-image metrics (`*_metrics.csv`), threshold sweeps
  (`*_sweep.csv`), the val-selected threshold and run summaries.
- `train/<run>/seed<s>/`: training logs, curves and summaries.
- `checks/`: Phase 1–5 data, metric and pipeline checks, and the job records.
