#!/bin/bash
# Phase 6, step 1: predict the test split ONCE per trained run, then score it at that run's
# own val-selected threshold (the full 0.05-0.95 sweep is saved too, for the secondary
# fixed-0.5 analysis and the thin-recall-vs-threshold figure). Run from the project root.
#
#   bash jobs/submit_phase6.sh          part A: the 12 CNN runs (4 models x seeds 0-2),
#                                       whole-image SAM, oracle-prompted SAM (K chosen on
#                                       val) and the two mask-resampling references on test
#   PART=tiled bash jobs/submit_phase6.sh
#                                       part B: tiled SAM as frozen (main SAM; user decision after
#                                       the val tile-seam check), then the overlapping-tile
#                                       sensitivity variant on VAL only (src.sam_overlap: threshold
#                                       chosen on val), the seam check rerun and the SAM val table
#   PART=overlap_test bash jobs/submit_phase6.sh
#                                       part C: the overlapping-tile variant on test, once (only
#                                       after its val results have been checked)
#
# One GPU job per part (1x L40S, the training GPU type), parts run in sequence inside it;
# a failing part is reported and does not stop the others. Every command passes --final-test,
# and src.predict / src.sam_oracle refuse to overwrite an existing test archive, so a
# resubmitted job can never predict the test split twice. The 1024 pilot (unet_r1024) is
# not predicted on test (it stays a pilot; user decision in Phase 3).
set -euo pipefail
PART="${PART:-main}"
S=$(env | grep -oE '^SLURM_[A-Z_]+' | sed 's/^/-u /' || true)
W=(--workers 8)
args=()
add() { [ ${#args[@]} -gt 0 ] && args+=("::"); args+=("$@"); }
run_pair() {   # <config> <seed>: predict test once, then score it
    add src.predict --config "$1" --seed "$2" --split test --final-test
    add src.evaluate score --config "$1" --seed "$2" --split test --final-test "${W[@]}"
}
case "$PART" in
    main)
        for m in unet attention_unet multiscale_unet ms_attention_unet; do
            for s in 0 1 2; do run_pair "configs/$m.yaml" "$s"; done
        done
        run_pair configs/sam_lora_whole.yaml 0
        add src.sam_oracle --split test --final-test "${W[@]}"
        add src.evaluate reference --config configs/base.yaml --split test --final-test "${W[@]}"
        TIME=03:00:00 ;;
    tiled)
        run_pair configs/sam_lora_tiled.yaml 0
        add src.sam_overlap --split val "${W[@]}"
        add src.analysis seam "${W[@]}"
        add src.analysis sam
        TIME=01:00:00 ;;
    overlap_test)
        add src.sam_overlap --split test --final-test "${W[@]}"
        TIME=00:30:00 ;;
    *) echo "PART must be main, tiled or overlap_test" >&2; exit 1 ;;
esac
env $S sbatch --job-name="fives-phase6-$PART" --time="$TIME" jobs/gpu.sbatch "${args[@]}"
