#!/bin/bash
# Create the $WORK folders and the project symlinks that point to them.
# Safe to re-run: existing correct links are kept; a real folder is never replaced.
#
#   project/outputs     -> $PROJECT_WORK/outputs      checkpoints, probability maps, run logs
#   project/logs        -> $PROJECT_WORK/logs         SLURM stdout/stderr
#   project/data/cache  -> $PROJECT_WORK/data_cache   CLAHE-processed / resized image and mask caches
#   project/data/FIVES  -> $FIVES_DIR (only if FIVES_DIR is given and the link is missing)
#
# Usage (from the project root):  bash jobs/setup_links.sh
#        FIVES_DIR=/path/to/FIVES bash jobs/setup_links.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WORK:?WORK is not set}"
PROJECT_WORK="${PROJECT_WORK:-$WORK/fives-vessel-seg}"

link() {  # link <target in $WORK> <link path in project>
    local target="$1" path="$2"
    mkdir -p "$target"
    if [ -L "$path" ]; then
        if [ "$(readlink -f "$path")" = "$(readlink -f "$target")" ]; then
            echo "ok      $path -> $(readlink "$path")"
        else
            echo "WARNING $path already links to $(readlink "$path"), not $target; left unchanged" >&2
        fi
    elif [ -e "$path" ]; then
        echo "WARNING $path exists and is not a symlink; left unchanged (move its contents to $target by hand)" >&2
    else
        mkdir -p "$(dirname "$path")"
        ln -s "$target" "$path"
        echo "created $path -> $target"
    fi
}

link "$PROJECT_WORK/outputs"    "$PROJECT_ROOT/outputs"
link "$PROJECT_WORK/logs"       "$PROJECT_ROOT/logs"
link "$PROJECT_WORK/data_cache" "$PROJECT_ROOT/data/cache"
mkdir -p "$PROJECT_WORK/sw_cache" "$PROJECT_ROOT/data/splits"

if [ -n "${FIVES_DIR:-}" ] && [ ! -e "$PROJECT_ROOT/data/FIVES" ]; then
    ln -s "$FIVES_DIR" "$PROJECT_ROOT/data/FIVES"
    echo "created data/FIVES -> $FIVES_DIR"
fi
if [ ! -e "$PROJECT_ROOT/data/FIVES" ]; then
    echo "WARNING data/FIVES is missing; re-run with FIVES_DIR=/path/to/FIVES" >&2
fi
