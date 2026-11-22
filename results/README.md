# Results

Retinal vessel segmentation on FIVES (test set: 200 images, 50 per disease category).
Every number below is copied from a file in `results/tables/` or `results/eval/`. The
full protocol is in [`docs/protocol.md`](../docs/protocol.md). The complete version of
this file (all tables, the figure list and the limitations) is written in Phase 7.
This section records the Phase 6 headline findings and how to word them.

## Phase 6: key findings (test set)

Each run is scored at 2048 × 2048 against the original masks, at its own threshold
chosen on validation. Model-level values are the mean ± sd **across 3 training seeds**
(each seed: the mean over the 200 test images). Source: `tables/test_main.csv`.

| Model | Dice | Thin-vessel recall | clDice | NSD |
|---|---|---|---|---|
| U-Net (baseline) | 0.8925 ± 0.0013 | 0.5430 ± 0.0224 | 0.9083 ± 0.0021 | 0.8580 ± 0.0026 |
| Attention U-Net | 0.8929 ± 0.0015 | 0.5465 ± 0.0230 | 0.9094 ± 0.0021 | 0.8590 ± 0.0024 |
| Multi-Scale U-Net | 0.8930 ± 0.0016 | 0.5202 ± 0.0111 | 0.9071 ± 0.0022 | 0.8584 ± 0.0015 |
| Multi-Scale Attention U-Net | 0.8922 ± 0.0011 | 0.5269 ± 0.0058 | 0.9085 ± 0.0002 | 0.8559 ± 0.0041 |
| SAM LoRA, tiled (main SAM) | 0.8860 ± 0.0008 | 0.5697 ± 0.0308 | 0.9062 ± 0.0010 | 0.8480 ± 0.0011 |
| SAM LoRA, tiled, overlapping tiles (sensitivity) | 0.8879 ± 0.0009 | 0.5569 ± 0.0203 | 0.9079 ± 0.0011 | 0.8534 ± 0.0008 |
| SAM LoRA, whole image | 0.8903 ± 0.0001 | 0.4786 ± 0.0029 | 0.9006 ± 0.0006 | 0.8387 ± 0.0012 |

Oracle-prompted SAM ("zero-shot, oracle point prompts") uses prompts taken from each
image's ground truth. It is reported separately in `tables/test_sam_oracle.csv` and is
never ranked against the models above.

### Fine-tuned SAM vs the U-Net baseline

- **Thin-vessel recall (tiled SAM): a modest average gain of +0.027 over U-Net, within
  training-seed variability. It is not a robust improvement.**
  - Per-seed test thin recall: SAM 0.574 / 0.537 / 0.598; U-Net 0.544 / 0.565 / 0.520.
    SAM seed 1 is below two of the three U-Net seeds.
  - Across thresholds 0.05–0.95 (`figures/thin_recall_vs_threshold.png`), the 3-seed
    SAM mean is above the best CNN at every threshold, but SAM's seed range overlaps
    U-Net's seed range at 18 of the 19 thresholds.
  - The paired Wilcoxon test (`tables/test_wilcoxon.csv`) treats the 200 images as the
    sample, so its small p-value does not capture training-seed variability.
- **Dice (tiled SAM): the deficit is consistent across seeds.** Every SAM seed
  (0.8854–0.8869) is below every U-Net seed (0.8910–0.8934); mean difference −0.0065.
- **Checkpoint selection by validation Dice does not control thin-vessel recall.** The
  three tiled-SAM seeds have near-identical best validation Dice (0.9038 / 0.9045 /
  0.9047 at threshold 0.5), yet their validation thin recall is 0.597 / 0.558 / 0.623.
- **Overlapping-tile inference** (same three models, seams removed by blending):
  - At the same threshold its thin recall equals tiled SAM's within 0.004. The larger
    thin-recall gain seen earlier with seed 0 alone was a seed effect.
  - It raises Dice slightly for every seed, but Dice stays below U-Net (−0.0046).
- **Whole-image SAM** (256 × 256 decoder output): the thin-recall deficit is consistent
  across seeds. Every seed (0.4755–0.4812) is below every U-Net seed (0.5204–0.5652);
  mean difference −0.064. Context, not an upper bound: downsampling the test ground
  truth to 256 and back keeps 0.400 of thin-vessel centreline
  (`tables/test_references.csv`).

Source files for the per-seed values: `eval/<run>/seed<s>/{val,test}_metrics.csv` and
`train/<run>/seed<s>/summary.json`.
