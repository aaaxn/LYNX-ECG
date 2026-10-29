#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PTBXL_DIR="graph_datasets/ptbxl"
SUFFIX="bp0p5-40hz-o4__resample-300hz__norm-p95med__natural__leads-0-1-2-3-4-5-6-7-8-9-10-11.h5"
STRICT="${PTBXL_DIR}/ptbxl__turning_points__40hz-original-mgap80ms__500hz__code6strict__${SUFFIX}"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <ckpt> [ckpt ...]" >&2
    exit 2
fi
ckpts=("$@")

OUT_DIR="reports/ptbxl_code6_eval"
mkdir -p "$OUT_DIR"

fail=0
for ckpt in "${ckpts[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "[eval] MISSING checkpoint: $ckpt" >&2
        fail=1
        continue
    fi
    tag="${ckpt#results/}"
    tag="${tag%.pt}"
    tag="${tag%/best_f1_val}"
    tag="${tag//\//_}"

    out="${OUT_DIR}/${tag}__code6strict.json"
    echo ""
    echo "=== eval: ${ckpt}  ×  code6strict ==="
    if ! uv run evaluate \
        --ckpt "$ckpt" \
        --dataset "$STRICT" \
        --threshold 0.5 \
        --output "$out"; then
        echo "[eval] FAILED (see error above): ${ckpt} × code6strict" >&2
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "[eval] at least one evaluation failed or was incompatible; see errors above." >&2
    exit 1
fi
echo ""
echo "[eval] all evaluations complete; JSON written under ${OUT_DIR}/"
