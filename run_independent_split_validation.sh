#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/TGRS_MLUDA-2024
PY="$ROOT/.venv/bin/python"
OUT=/nas1/zhangzj26/HyperSIGMA_adapted/independent_split_validation
GPU_ID="${1:?GPU id required}"
WORKER_ID="${2:?worker id required}"
SPLITS=(1703 1801 1907 2029 2141)
mkdir -p "$OUT/logs"
run_one() {
  local method="$1" split="$2" lam="$3"
  local stem="split_${split}_opt_1174" dir="$OUT/$method/lambda_${lam}_anneal"
  [[ -f "$dir/${stem}_history.json" ]] && return
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u "$ROOT/MLUDA_hu_fspec_rel_kd.py" \
    --device cuda:0 --epochs 100 --split-seed "$split" --optimization-seed 1174 \
    --lambda-rel "$lam" --schedule anneal --spatial-only \
    --spat-cache /nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_full48_fullft_cache.npz \
    --output "$OUT/$method" > "$OUT/logs/${method}_${stem}.log" 2>&1
}
for ((i=WORKER_ID; i<${#SPLITS[@]}; i+=2)); do
  split=${SPLITS[$i]}
  run_one baseline "$split" 0
  run_one fspat "$split" 0.1
done
touch "$OUT/worker_${WORKER_ID}_done"
