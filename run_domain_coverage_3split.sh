#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/TGRS_MLUDA-2024
PY="$ROOT/.venv/bin/python"
BASE=/nas1/zhangzj26/HyperSIGMA_adapted/domain_coverage
SRCREL=/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_relation7_full48_fullft_cache.npz
TGTREL=/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_target_relation7_full48_fullft_cache.npz
MODE=${1:?target_only or source_target}
GPU=${2:?gpu id}
mkdir -p "$BASE/$MODE/logs"
for split in 1370 1703 1418; do
  if [[ "$MODE" == target_only ]]; then
    EXTRA="--target-structural --target-spat-relation-cache $TGTREL"
  else
    EXTRA="--source-target-structural --spat-relation-cache $SRCREL --target-spat-relation-cache $TGTREL"
  fi
  if [[ -f "$BASE/$MODE/lambda_0.1_anneal/split_${split}_opt_1174_best.pth" ]]; then
    echo "skip existing split $split"; continue
  fi
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/MLUDA_hu_fspec_rel_kd.py" \
    --device cuda:0 --epochs 100 --split-seed "$split" --optimization-seed 1174 \
    --lambda-rel 0.1 --schedule anneal --spatial-only --spatial-structural $EXTRA \
    --output "$BASE/$MODE" > "$BASE/$MODE/logs/split_${split}.log" 2>&1
done
