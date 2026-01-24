#!/usr/bin/env bash
# Experiment: Learning rate ablations for Llama3-IT
    
set -euo pipefail

# Model configuration
MODEL="meta-llama/Llama-3.2-1B-Instruct"
LORA_RANK=128
LORA_ALPHA=256
BATCH_SIZE=32
MILESTONE_TOKENS=25000000  # 25M tokens per milestone
NUM_MILESTONES=4           # Total: 100M tokens
USE_PACKING=true

MILESTONE_TOKENS=5000000  # 25M tokens per milestone
NUM_MILESTONES=5           # Total: 100M tokens

# Learning rate configurations to test
# Format: "layer_lr,embedding_lr,experiment_name,wandb_run_name"
LR_CONFIGS=(
    "6e-5,2e-5,fixloss_test9_tokenstats_noval_gradacum1,test_run"

)

for config in "${LR_CONFIGS[@]}"; do
    IFS=',' read -r LAYER_LR EMBEDDING_LR EXPERIMENT_NAME WANDB_RUN_NAME <<< "$config"

    echo "=========================================="
    echo "Testing LR config: ${EXPERIMENT_NAME}"
    echo "Layer LR: ${LAYER_LR}"
    echo "Embedding LR: ${EMBEDDING_LR}"
    echo "WandB Run Name: ${WANDB_RUN_NAME}"
    echo "=========================================="

    OUTPUT_DIR="outputs/lr_ablation/${EXPERIMENT_NAME}"

    # Run milestone loop with learning rate overrides
    MODEL_NAME="${MODEL}" \
    LORA_RANK=${LORA_RANK} \
    LORA_ALPHA=${LORA_ALPHA} \
    BATCH_SIZE=${BATCH_SIZE} \
    MILESTONE_TOKENS=${MILESTONE_TOKENS} \
    NUM_MILESTONES=${NUM_MILESTONES} \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    LEARNING_RATE="${LAYER_LR}" \
    EMBEDDING_LEARNING_RATE="${EMBEDDING_LR}" \
    USE_PACKING="${USE_PACKING}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
    bash scripts/run_milestone_loop.sh

    echo "✅ Completed ${EXPERIMENT_NAME}"
    echo ""
done

echo "🎉 Learning rate ablation complete!"
