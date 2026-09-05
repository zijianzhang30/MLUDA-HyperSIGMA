#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/TGRS_MLUDA-2024
PY="$ROOT/.venv/bin/python"
OUT=/nas1/zhangzj26/HyperSIGMA_adapted/structural_gradient_diagnostic
REL=/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_relation7_full48_fullft_cache.npz
GPU=${1:?gpu}; SPLIT=${2:?split}
mkdir -p "$OUT/logs"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/MLUDA_hu_fspec_rel_kd.py" \
  --device cuda:0 --epochs 100 --split-seed "$SPLIT" --optimization-seed 1174 \
  --lambda-rel 0.1 --schedule anneal --spatial-only --spatial-structural \
  --spat-relation-cache "$REL" --diagnostic-epochs 1 5 10 20 30 40 60 80 100 \
  --output "$OUT" > "$OUT/logs/split_${SPLIT}.log" 2>&1
