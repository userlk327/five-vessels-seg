# Draft: Methods and Results

Draft text for the course report. Every number comes from the result files; the full
tables are in [`results/README.md`](../results/README.md) and every protocol detail is in
[`protocol.md`](protocol.md). Check the reference details before submission.

## Methods

### Data

We used FIVES (Jin et al., 2022), a public set of 800 colour fundus photographs
(2048 × 2048 pixels) with manual annotations of all retinal blood vessels. The images
come from 573 patients and fall into four groups of 200: age-related macular
degeneration (AMD), diabetic retinopathy (DR), glaucoma and normal. We used the
official split of 600 training and 200 test images (50 test images per group). The
images were taken from a copy of FIVES on Hugging Face (`tyluan/FIVES`). We checked the
image counts, sizes and file names, and that all masks are binary. No other dataset was
used.

### Training and validation split

Three training images were left out:

- two whose masks are completely empty although the images show vessels;
- one that shows the same eye as a test image, so that no test image was also seen in
  training.

The remaining 597 training images were split into 477 for training and 120 for
validation (30 per disease group, random seed 17). FIVES gives no patient IDs, so we
kept likely duplicate or same-patient images on the same side of this split. These
were byte-identical files and images with an identical burned-in marking. The markings
were compared only as image patterns and never read. The test set was not changed. The
validation set was used for every choice: checkpoints, operating thresholds and the
prompt setting of the oracle experiment. The test set was used once per trained model,
at the end.

### Preprocessing

Contrast was enhanced with CLAHE (Zuiderveld, 1994) on the lightness channel of the LAB
colour space, at full resolution (clip limit 2.0, 8 × 8 tiles). Images were then
reduced by area averaging to 512 × 512 for the CNNs and to 1024 × 1024 for SAM. CNN
inputs were normalized with the per-channel mean and standard deviation of the training
images; SAM used its own built-in normalization. During training, images and masks were
randomly flipped horizontally and vertically and rotated by multiples of 90°.

To make training masks at 512 and 1024, we averaged the vessel pixels in each block
without thresholding, which gives "soft" masks with values between 0 and 1. On 30
training masks we compared this with thresholded and nearest-neighbour downsampling:

- thresholding at 0.5 deletes vessels one pixel wide at 512;
- lower thresholds keep them but make every vessel too wide;
- soft masks did neither, and kept the most information.

### Models

All four CNNs share one backbone, the U-Net (Ronneberger et al., 2015) from the
project's starting notebook. It has four down-sampling and four up-sampling levels with
64 to 1024 channels (31.0 million parameters).

- **Attention U-Net** adds additive attention gates (Oktay et al., 2018) to the four skip
  connections.
- **Multi-Scale U-Net** replaces the second 3 × 3 convolution of each of the nine
  convolution blocks with three parallel 3 × 3 convolutions with dilation 1, 2 and 4.
  Their outputs are concatenated and fused by a 1 × 1 convolution.
- **Multi-Scale Attention U-Net** has both changes.

These four models form a 2 × 2 design (attention: no/yes; multi-scale: no/yes), which
lets us estimate the effect of each module and their interaction. The added modules
increase the parameter count by 3.4%, 5.6% and 9.0%.

We fine-tuned the **Segment Anything Model** (SAM, ViT-B; Kirillov et al., 2023) with
low-rank adaptation (LoRA; Hu et al., 2022), following SAMed (Zhang and Liu, 2023):

- rank-4 LoRA layers were added to the query and value projections of all 12
  image-encoder blocks;
- these layers and the mask decoder were trained (4.2 million of 93.9 million
  parameters), and the rest of SAM was frozen;
- no prompts were given, so SAM worked fully automatically, like the CNNs.

Because SAM's mask decoder outputs only 256 × 256 pixels, we built two versions:

- **tiled (main SAM):** each image is cut into four 1024 × 1024 crops at full
  resolution. Together they give a 512 × 512 output, the same as the CNNs;
- **whole image:** the full image is reduced to 1024 × 1024, giving a 256 × 256 output.

On validation, the tiled version showed a small artifact where the four crops meet. We
therefore also report an **overlapping-tile** version of the same trained models: nine
crops that overlap by half, with the overlapping predictions blended smoothly.

As a separate experiment, we ran SAM **zero-shot with oracle point prompts**. For each
image, 5 points on vessels and 5 points on background were taken from that image's own
ground truth. Because it uses the ground truth, this experiment is reported separately
and not compared with the automatic models.

### Training

All trained models used the same procedure:

- loss: binary cross-entropy plus soft Dice, with equal weights;
- the Adam optimizer with learning rate 1e-4 and batches of 4 images (for SAM, two
  steps of 2 accumulated into one update of 4);
- up to 150 epochs, with mixed precision;
- training stopped when the validation Dice had not improved for 20 epochs, and the
  best epoch was kept.

The validation Dice was computed at full resolution (2048 × 2048) at threshold 0.5.
Each model was trained three times, with random seeds 0, 1 and 2, on one NVIDIA L40S
GPU per run.

### Evaluation

Each model's probability map was enlarged to 2048 × 2048 by bilinear interpolation and
thresholded. The threshold was chosen per run on the validation set (0.05 to 0.95 in
steps of 0.05, highest mean Dice) and then applied to the test set. All design choices
were fixed, and the code was committed, before any test prediction was made. There was
no test-time augmentation or post-processing.

Every metric was computed per image against the original 2048 × 2048 ground truth,
inside a mask of the eye's field of view:

- **overlap:** Dice, IoU, precision and recall;
- **connectivity:** clDice (Shit et al., 2021);
- **boundary accuracy:** normalized surface Dice with a 2-pixel tolerance (NSD; Nikolov
  et al., 2021), computed from boundary pixels;
- **thin vessels:** centreline recall by vessel width. Vessel centrelines in the ground
  truth were grouped by local radius (thin < 3, medium 3–7, thick > 7 pixels at 2048),
  and recall is the share of each group's centreline that the prediction finds, within
  1 pixel;
- **image-derived measures (optional):** vessel density and fractal dimension
  (box counting), predicted against ground truth.

To show how much the output grids themselves lose, each ground truth was also reduced
to 512 and to 256 the way the training masks were, then enlarged back and scored. These
reference values are context, not upper bounds.

**Statistics.** For each model we computed each metric's mean over the 200 test images
for each seed, then the mean and standard deviation across the three seeds. Each model
was compared with the U-Net by a paired Wilcoxon (1945) signed-rank test on per-image values,
after averaging each image over seeds. We used Holm's (1979) correction within each
metric. For the 2 × 2 design we computed per image:

- the main effect of attention: the mean of its two effects, with and without
  multi-scale blocks;
- the main effect of the multi-scale blocks, in the same way;
- their interaction.

Each came with a 95% bootstrap confidence interval over images. Because thin-vessel
recall depends strongly on the threshold, we repeated the analysis with every model at
a fixed threshold of 0.5. We also measured:

- inference time per image (batch of 1, after 10 warm-up runs);
- floating-point operations, peak GPU memory and training time.

## Results

### Overall segmentation

All fully automatic models reached a similar overall overlap with the ground truth
(Table 1).

- The four CNNs had a mean Dice of 0.8922–0.8930, with a standard deviation across
  seeds of 0.0011–0.0016.
- The fine-tuned SAM versions were lower: 0.8860 (tiled), 0.8879 (overlapping tiles)
  and 0.8903 (whole image).
- The same pattern held for NSD: CNNs 0.8559–0.8590, SAM 0.8387–0.8534.

**Table 1.** Test set (200 images), mean ± standard deviation across 3 training seeds.
Each model at its threshold chosen on validation.

| Model | Dice | clDice | NSD | Thin | Medium | Thick |
|---|---|---|---|---|---|---|
| U-Net | 0.8925 ± 0.0013 | 0.9083 ± 0.0021 | 0.8580 ± 0.0026 | 0.5430 ± 0.0224 | 0.8921 ± 0.0056 | 0.9786 ± 0.0013 |
| Attention U-Net | 0.8929 ± 0.0015 | 0.9094 ± 0.0021 | 0.8590 ± 0.0024 | 0.5465 ± 0.0230 | 0.8943 ± 0.0047 | 0.9793 ± 0.0004 |
| Multi-Scale U-Net | 0.8930 ± 0.0016 | 0.9071 ± 0.0022 | 0.8584 ± 0.0015 | 0.5202 ± 0.0111 | 0.8903 ± 0.0011 | 0.9780 ± 0.0020 |
| Multi-Scale Attention U-Net | 0.8922 ± 0.0011 | 0.9085 ± 0.0002 | 0.8559 ± 0.0041 | 0.5269 ± 0.0058 | 0.8884 ± 0.0021 | 0.9778 ± 0.0011 |
| SAM, tiled (main) | 0.8860 ± 0.0008 | 0.9062 ± 0.0010 | 0.8480 ± 0.0011 | 0.5697 ± 0.0308 | 0.8894 ± 0.0058 | 0.9784 ± 0.0013 |
| SAM, overlapping tiles | 0.8879 ± 0.0009 | 0.9079 ± 0.0011 | 0.8534 ± 0.0008 | 0.5569 ± 0.0203 | 0.8894 ± 0.0053 | 0.9779 ± 0.0022 |
| SAM, whole image | 0.8903 ± 0.0001 | 0.9006 ± 0.0006 | 0.8387 ± 0.0012 | 0.4786 ± 0.0029 | 0.8746 ± 0.0018 | 0.9797 ± 0.0010 |

Thin, medium, thick: centreline recall by vessel width.

### Thin vessels

Thin vessels were the hardest part for every model.

- **Thin-vessel recall:** 0.48–0.57.
- **Medium and thick vessels:** 0.87–0.89 and about 0.98.
- **What the 512 grid loses:** reducing the ground truth itself to 512 × 512 and
  enlarging it back kept 0.916 of the thin-vessel centreline.
- **What the 256 grid loses:** at 256 × 256, the whole-image SAM's output size, it
  kept 0.400.

### Attention and multi-scale modules (2 × 2 design)

- **Multi-scale blocks:** a modest average decrease in thin-vessel recall, consistent in
  direction across threshold rules but overlapping the variation between training
  seeds.
  - Main effect −0.021 (95% CI −0.026 to −0.016) at the thresholds chosen on
    validation, and −0.013 (−0.018 to −0.008) at a fixed threshold of 0.5.
  - Across seeds, thin-vessel recall ranged from 0.511 to 0.533 for the Multi-Scale
    U-Net and from 0.520 to 0.565 for the U-Net.
- **Attention gates:** no consistent effect on thin-vessel recall. The effect was +0.005
  (+0.002 to +0.008) at the thresholds chosen on validation and −0.015 (−0.018 to
  −0.012) at 0.5. Attention shifted the balance between precision and recall, so the
  direction depended on the threshold rule.
- **Dice:** no meaningful effect for either module (main effects within ±0.0006;
  interaction −0.0012).
- **Caveat:** these confidence intervals resample images, not training runs.

### Fine-tuned SAM compared with the U-Net

- **Thin vessels (tiled SAM):** a modest average gain over the U-Net (+0.027, higher in
  68% of test images), but within the variation between training seeds. It is not a
  robust improvement.
  - The three SAM seeds reached 0.574, 0.537 and 0.598; the three U-Net seeds reached
    0.544, 0.565 and 0.520.
  - Over all thresholds from 0.05 to 0.95, the SAM seed-average was above the best CNN
    at every threshold, but the range of the SAM seeds overlapped the range of the
    U-Net seeds at 18 of the 19 thresholds.
- **Dice (tiled SAM):** consistently lower. Every SAM seed (0.8854–0.8869) was below
  every U-Net seed (0.8910–0.8934); mean difference −0.0065.
- **Checkpoint selection:** choosing the checkpoint by validation Dice did not control
  thin-vessel recall. The three SAM seeds had almost the same best validation Dice
  (0.9038–0.9047), but their validation thin-vessel recall ranged from 0.558 to 0.623.
- **Overlapping tiles:** at the same threshold, they gave the same thin-vessel recall
  as the tiled version (within 0.004). Dice rose slightly in every seed (to 0.8879 on
  average) but stayed below the U-Net.
- **Whole-image SAM:** lost thin vessels consistently. Every seed (0.476–0.481) was
  below every U-Net seed (0.520–0.565); mean difference −0.064.

### Disease groups

- **Glaucoma images** had the lowest Dice for every model (0.847–0.854, against
  0.910–0.917 for AMD images).
- **Normal images** had the lowest thin-vessel recall for every model.

We did not test differences between groups.

### Image-derived vessel measures

All automatic models slightly underestimated:

- **vessel density:** mean difference −0.001 to −0.004 against a ground-truth mean of
  0.093; correlation with the ground truth 0.986–0.988;
- **fractal dimension:** −0.015 to −0.022 against 1.490; correlation 0.913–0.961.

These measures are not clinically validated here.

### Computational cost

**Table 2.** Cost per model. One L40S GPU, seed-0 runs; inference time per image at
batch 1.

| Model | Parameters (trainable) | Inference time per image | Operations per image | Training time |
|---|---|---|---|---|
| U-Net | 31.0 M (all) | 4.4 ms | 385 GFLOPs | 0.43 h |
| Attention U-Net | 32.1 M (all) | 5.3 ms | 392 GFLOPs | 0.41 h |
| Multi-Scale U-Net | 32.8 M (all) | 7.8 ms | 405 GFLOPs | 0.45 h |
| Multi-Scale Attention U-Net | 33.8 M (all) | 8.4 ms | 411 GFLOPs | 0.41 h |
| SAM, tiled (main) | 93.9 M (4.2 M) | 190 ms | 3,908 GFLOPs | 9.18 h |
| SAM, overlapping tiles | 93.9 M (4.2 M) | 429 ms | 8,793 GFLOPs | (same models as tiled) |
| SAM, whole image | 93.9 M (4.2 M) | 48 ms | 977 GFLOPs | 2.86 h |

The tiled SAM needed about 43 times the U-Net's inference time and 21 times its training
time.

### Oracle-prompted SAM

With five vessel and five background points taken from each image's ground truth,
zero-shot SAM labelled most of the field of view as vessel. Predicted vessel density was
0.950 against 0.093 in the ground truth, giving a Dice of 0.17 and a recall of 0.975.
Because its prompts come from the ground truth, this result is not compared with the
automatic models.

## Limitations to state in the report

- **Seeds:** three training seeds per model. Image-level tests do not capture
  training-seed variability, which is as large as most differences between models for
  thin vessels.
- **Thresholds:** thin-vessel recall depends strongly on the threshold.
- **Resolution:** the CNNs worked at 512 × 512. SAM's decoder outputs 256 × 256, and the
  tiled SAM's combined output is 512 × 512.
- **Patients:** the same patient may appear in both training and test images, because
  FIVES has no patient IDs.
- **Data:** a single dataset (a Hugging Face copy of FIVES), with no external validation.
- **Metric definitions:** NSD was approximated from boundary pixels. Vessel density and
  fractal dimension are not clinically validated.

## References

Checked on 2026-11-22 against Crossref (entries with a DOI) and the arXiv API (entries
with an arXiv ID); all authors are listed in publication order. Exceptions: Holm (1979)
has no DOI and the JSTOR ID was not checked against a registry; LoRA's venue (ICLR 2022)
is not in its arXiv record; the editor and volume of *Graphics Gems IV* are not in its
Crossref record.

```
Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian
  Journal of Statistics, 6(2), 65–70. JSTOR: 4615733 (no DOI).

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W.
  (2022). LoRA: Low-rank adaptation of large language models. International Conference
  on Learning Representations (ICLR 2022). arXiv:2106.09685.

Jin, K., Huang, X., Zhou, J., Li, Y., Yan, Y., Sun, Y., Zhang, Q., Wang, Y., & Ye, J.
  (2022). FIVES: A fundus image dataset for artificial intelligence based vessel
  segmentation. Scientific Data, 9(1), 475. https://doi.org/10.1038/s41597-022-01564-3

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T.,
  Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment
  anything. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV)
  (pp. 3992–4003). https://doi.org/10.1109/ICCV51070.2023.00371 (arXiv:2304.02643)

Nikolov, S., Blackwell, S., Zverovitch, A., Mendes, R., Livne, M., De Fauw, J., Patel,
  Y., Meyer, C., Askham, H., Romera-Paredes, B., Kelly, C., Karthikesalingam, A., Chu,
  C., Carnell, D., Boon, C., D'Souza, D., Moinuddin, S. A., Garie, B., McQuinlan, Y.,
  Ireland, S., Hampton, K., Fuller, K., Montgomery, H., Rees, G., Suleyman, M., Back,
  T., Hughes, C. O., Ledsam, J. R., & Ronneberger, O. (2021). Clinically applicable
  segmentation of head and neck anatomy for radiotherapy: Deep learning algorithm
  development and validation study. Journal of Medical Internet Research, 23(7),
  e26151. https://doi.org/10.2196/26151

Oktay, O., Schlemper, J., Le Folgoc, L., Lee, M., Heinrich, M., Misawa, K., Mori, K.,
  McDonagh, S., Hammerla, N. Y., Kainz, B., Glocker, B., & Rueckert, D. (2018).
  Attention U-Net: Learning where to look for the pancreas. Medical Imaging with Deep
  Learning (MIDL 2018). arXiv:1804.03999

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for
  biomedical image segmentation. In Medical Image Computing and Computer-Assisted
  Intervention – MICCAI 2015, Lecture Notes in Computer Science (pp. 234–241).
  Springer. https://doi.org/10.1007/978-3-319-24574-4_28 (arXiv:1505.04597)

Shit, S., Paetzold, J. C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim,
  J. P. W., Bauer, U., & Menze, B. H. (2021). clDice – a novel topology-preserving loss
  function for tubular structure segmentation. In 2021 IEEE/CVF Conference on Computer
  Vision and Pattern Recognition (CVPR) (pp. 16555–16564).
  https://doi.org/10.1109/CVPR46437.2021.01629 (arXiv:2003.07311)

Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin,
  1(6), 80–83. https://doi.org/10.2307/3001968

Zhang, K., & Liu, D. (2023). Customized segment anything model for medical image
  segmentation (technical report). arXiv:2304.13785

Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In
  P. S. Heckbert (Ed.), Graphics Gems IV (pp. 474–485). Academic Press.
  https://doi.org/10.1016/B978-0-12-336156-1.50061-6
```
