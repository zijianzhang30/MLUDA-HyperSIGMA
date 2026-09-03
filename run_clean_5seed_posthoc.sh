#!/usr/bin/env bash
set -euo pipefail
BASE=/nas1/zhangzj26/HyperSIGMA_adapted/clean_iterator_5seed
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
CACHE=/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_fullft_cache.npz
mkdir -p "$BASE/posthoc_logs"
run_one() {
  local ver="$1" seed="$2" lam="$3"
  "$PY" -u /home/zhangzj26/TGRS_MLUDA-2024/eval_mluDA_fspec_proto_kd.py \
    --device cpu --cache "$CACHE" \
    --checkpoint "$BASE/$ver/lambda_${lam}_anneal/seed_${seed}_best.pth" \
    > "$BASE/posthoc_logs/${ver}_${seed}.log" 2>&1
}
run_one baseline 1370 0
run_one baseline 1417 0
run_one baseline 1418 0
run_one fspec 1370 0.1
run_one fspec 1417 0.1
run_one fspec 1418 0.1
run_one fspat 1370 0.1
run_one fspat 1417 0.1
run_one fspat 1418 0.1
touch "$BASE/posthoc_done"
