#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/TGRS_MLUDA-2024
PY=$ROOT/.venv/bin/python
CACHE=/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_fullft_cache.npz
SPAT=/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_full48_fullft_cache.npz
OUT=/nas1/zhangzj26/HyperSIGMA_adapted/clean_iterator_5seed
LOG=$OUT/logs
mkdir -p "$LOG"
run() {
  local version="$1" seed="$2" extra="$3" lambda="$4"
  local log="$LOG/${version}_${seed}.log"
  CUDA_VISIBLE_DEVICES=4 "$PY" -u "$ROOT/MLUDA_hu_fspec_rel_kd.py" \
    --device cuda:0 --epochs 100 --seed "$seed" --lambda-rel "$lambda" \
    --schedule anneal --cache "$CACHE" --output "$OUT/$version" $extra \
    > "$log" 2>&1
}
for seed in 1370 1417 1418; do run baseline "$seed" "" 0; done
for seed in 1370 1417 1418; do run fspec "$seed" "" 0.1; done
for seed in 1370 1417 1418; do run fspat "$seed" "--spatial-only --spat-cache $SPAT" 0.1; done
touch "$OUT/remaining_done"
