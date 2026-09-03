#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangzj26/TGRS_MLUDA-2024
PY="$ROOT/.venv/bin/python"
OUT=/nas1/zhangzj26/HyperSIGMA_adapted/seed_disentanglement
SPAT=/nas1/zhangzj26/HyperSIGMA_adapted/hypersigma_fspat_full48_fullft_cache.npz
WORKER_ID="${1:?worker id is required}"
GPU_ID="${2:?physical GPU id is required}"
WORKERS="${3:-4}"
SEEDS=(1174 1370 1417 1418 1546)
PAIRS=()

for opt_seed in "${SEEDS[@]}"; do
  PAIRS+=("1174:${opt_seed}")
done
for split_seed in "${SEEDS[@]}"; do
  if [[ "$split_seed" != 1174 ]]; then
    PAIRS+=("${split_seed}:1174")
  fi
done

mkdir -p "$OUT/logs"

run_one() {
  local method="$1" split_seed="$2" opt_seed="$3" lambda_rel="$4"
  local log="$OUT/logs/${method}_split_${split_seed}_opt_${opt_seed}.log"
  if [[ -f "$OUT/$method/lambda_${lambda_rel}_anneal/split_${split_seed}_opt_${opt_seed}_history.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u "$ROOT/MLUDA_hu_fspec_rel_kd.py" \
    --device cuda:0 --epochs 100 \
    --split-seed "$split_seed" --optimization-seed "$opt_seed" \
    --lambda-rel "$lambda_rel" --schedule anneal \
    --spatial-only --spat-cache "$SPAT" --output "$OUT/$method" \
    > "$log" 2>&1
}

for ((idx=WORKER_ID; idx<${#PAIRS[@]}; idx+=WORKERS)); do
  IFS=: read -r split_seed opt_seed <<< "${PAIRS[$idx]}"
  # Matched control: spatial projection/forward are identical, but lambda=0.
  run_one baseline "$split_seed" "$opt_seed" 0
  run_one fspat "$split_seed" "$opt_seed" 0.1
done

touch "$OUT/worker_${WORKER_ID}_done"
