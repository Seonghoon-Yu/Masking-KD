#!/bin/bash
# 1) Greedy teacher rollouts on ViRL39K with Qwen3-VL-8B-Thinking (up to 4096 new tokens).
# 2) Keep only the rollouts whose final answer is correct.
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TEACHER_MODEL="Qwen/Qwen3-VL-8B-Thinking"
ROLLOUT_DIR="./data/rollouts/qwen3_vl_8b_thinking"

python generate_rollouts.py \
    --model "$TEACHER_MODEL" \
    --data-path "PAPOGalaxy/PAPO_ViRL39K_train" \
    --output-dir "$ROLLOUT_DIR" \
    --max-new-tokens 4096

python filter_rollouts.py \
    --input-dir "$ROLLOUT_DIR" \
    --output-dir "${ROLLOUT_DIR}_correct"
