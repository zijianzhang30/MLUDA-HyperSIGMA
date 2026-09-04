#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangzj26/TGRS_MLUDA-2024
PY="$ROOT/.venv/bin/python"
OUT=/nas1/zhangzj26/HyperSIGMA_adapted/spatial_structural_10split
REL=/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_relation7_full48_fullft_cache.npz
mkdir -p "$OUT/logs"

# The three-split sanity runs already exist; this extension only fills the
# seven missing splits.  Optimization seed is fixed at 1174, while split seed
# identifies the source train/validation partition.
for split in 1370 1417 1418 1546 1801 1907 2029; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -u "$ROOT/MLUDA_hu_fspec_rel_kd.py" \
    --device cuda:0 --epochs 100 --split-seed "$split" --optimization-seed 1174 \
    --lambda-rel 0.1 --schedule anneal --spatial-only --spatial-structural \
    --spat-relation-cache "$REL" --output "$OUT" \
    > "$OUT/logs/split_${split}.log" 2>&1
done
