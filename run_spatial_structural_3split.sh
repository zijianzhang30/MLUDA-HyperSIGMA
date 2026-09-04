#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/TGRS_MLUDA-2024
PY="$ROOT/.venv/bin/python"
OUT=/nas1/zhangzj26/HyperSIGMA_adapted/spatial_structural_3split
mkdir -p "$OUT/logs"
for split in 1174 1703 2141; do
  CUDA_VISIBLE_DEVICES=7 "$PY" -u "$ROOT/MLUDA_hu_fspec_rel_kd.py" --device cuda:0 --epochs 100 \
    --split-seed "$split" --optimization-seed 1174 --lambda-rel 0.1 --schedule anneal \
    --spatial-only --spatial-structural --spat-relation-cache /nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_relation7_full48_fullft_cache.npz \
    --output "$OUT" > "$OUT/logs/split_${split}.log" 2>&1
done
