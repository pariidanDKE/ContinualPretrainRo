#!/usr/bin/env bash
# Experiment: Test different LoRA ranks

set -euo pipefail

MODEL="meta-llama/Llama-3.2-1B"
LORA_RANKS=(8 16 32 64)
BATCH_SIZE=32
MILESTONE_TOKENS=1000000
NUM_MILESTONES=3

for rank in "${LORA_RANKS[@]}"; do
    echo "=========================================="
    echo "Testing LoRA rank: ${rank}"
    echo "=========================================="

    OUTPUT_DIR="outputs/lora_sweep/rank${rank}"

    # Override the main loop script variables
    MODEL_NAME="${MODEL}" \
    LORA_RANK=${rank} \
    LORA_ALPHA=$((rank * 2)) \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    MILESTONE_TOKENS=${MILESTONE_TOKENS} \
    NUM_MILESTONES=${NUM_MILESTONES} \
    bash scripts/run_milestone_loop.sh

    echo "✅ Completed rank ${rank}"
    echo ""
done

echo "🎉 LoRA rank sweep complete!"
