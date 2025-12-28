#!/usr/bin/env bash
# Experiment: Compare different base models

set -euo pipefail

MODELS=(
    "meta-llama/Llama-3.2-1B"
    "google/gemma-3-1b-pt"
    "Qwen/Qwen2.5-1.5B"
)

LORA_RANK=16
BATCH_SIZE=32
MILESTONE_TOKENS=1000000
NUM_MILESTONES=3

for model in "${MODELS[@]}"; do
    model_slug=$(echo ${model} | tr '/:' '__')

    echo "=========================================="
    echo "Testing model: ${model}"
    echo "=========================================="

    OUTPUT_DIR="outputs/model_comparison/${model_slug}"

    MODEL_NAME="${model}" \
    LORA_RANK=${LORA_RANK} \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    MILESTONE_TOKENS=${MILESTONE_TOKENS} \
    NUM_MILESTONES=${NUM_MILESTONES} \
    bash scripts/run_milestone_loop.sh

    echo "✅ Completed ${model}"
    echo ""
done

echo "🎉 Model comparison complete!"
