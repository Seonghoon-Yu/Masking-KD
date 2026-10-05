#!/bin/bash
# Evaluate a model on the 7 benchmarks with greedy decoding (up to 4096 new tokens).
# Usage: bash scripts/eval.sh <model path or HF id> [output dir]
set -e

MODEL=${1:?"Usage: bash scripts/eval.sh <model path or HF id> [output dir]"}
OUTPUT_DIR=${2:-"./eval_outputs/$(echo "$MODEL" | sed 's#^\./##; s#/#_#g')"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DISABLE_VERSION_CHECK=1
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BENCHMARKS=(
    hiyouga_geometry3k
    AI4Math_MathVista
    We-Math_We-Math
    PAPO_MMK12
    AI4Math_MathVerse
    lscpku_LogicVista
    MMMU_MMMU_Pro
)

for BENCHMARK in "${BENCHMARKS[@]}"; do
    python evaluation/infer.py \
        --model "$MODEL" \
        --dataset "$BENCHMARK" \
        --output "$OUTPUT_DIR/$BENCHMARK.jsonl" \
        --tensor-parallel-size "$NUM_GPUS" \
        --max-new-tokens 4096 \
        --temperature 0
done

python evaluation/score.py --pred-dir "$OUTPUT_DIR"
