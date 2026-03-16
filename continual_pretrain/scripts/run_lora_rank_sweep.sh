#!/bin/bash

# LoRA Rank Sweep Script
# Compares training performance across different LoRA ranks (16, 32, 64, 128)
# Each run trains for 100M tokens with evaluations at 25M, 50M, 75M, and 100M token milestones
#
# All runs use:
# - WITH embeddings (embed_tokens + lm_head in LoRA target modules)
# - Main learning rate: 1e-4
# - Embedding learning rate: 2e-5
# - LoRA alpha = LoRA rank

set -e  # Exit on error

# Navigate to the continual_pretrain directory
cd "$(dirname "$0")/.."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
TOKENS_PER_MILESTONE=25000000  # 25M tokens per milestone
NUM_MILESTONES=5               # 4 milestones = 100M tokens total

LORA_RANKS=(128 256 8)      # LoRA ranks to test

LEARNING_RATE=1e-4
EMBEDDING_LR=2e-5
BATCH_SIZE=16
GRAD_ACCUM=8

# 80/20 data mix configuration (80% Romanian / 20% English edu)
TOTAL_SAMPLES=170000

# Generate timestamp for this sweep
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_NAME="lora_rank_sweep_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting LoRA Rank Sweep${NC}"
echo -e "${BLUE}Sweep ID: ${SWEEP_NAME}${NC}"
echo -e "${BLUE}Total tokens per run: $((TOKENS_PER_MILESTONE * NUM_MILESTONES))${NC}"
echo -e "${BLUE}Milestones: ${NUM_MILESTONES} (every ${TOKENS_PER_MILESTONE} tokens)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run a single training experiment
run_experiment() {
    local run_id=$1
    local lora_rank=$2

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Run ${run_id}/4${NC}"
    echo -e "${GREEN}LoRA Rank: ${lora_rank} (Alpha: ${lora_rank})${NC}"
    echo -e "${GREEN}Main LR: ${LEARNING_RATE}, Embedding LR: ${EMBEDDING_LR}${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Custom name for this run
    CUSTOM_NAME="rank${lora_rank}_lr${LEARNING_RATE}_elr${EMBEDDING_LR}"

    # Run the training
    python train_milestones.py \
        model.lora.r=${lora_rank} \
        model.lora.lora_alpha=${lora_rank} \
        training_args.learning_rate=${LEARNING_RATE} \
        training_args.embedding_learning_rate=${EMBEDDING_LR} \
        training_args.per_device_train_batch_size=${BATCH_SIZE} \
        training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
        milestone.tokens_per_milestone=${TOKENS_PER_MILESTONE} \
        milestone.num_milestones=${NUM_MILESTONES} \
        data_builder.enabled=true \
        data_builder.total_sample_size=${TOTAL_SAMPLES} \
        "data_builder.proportions=[0.8,0.2]" \
        milestone.run_benchmarks=true \
        wandb.custom_run_name="${CUSTOM_NAME}" \
        wandb.group="${SWEEP_NAME}"

    echo -e "${GREEN}Run ${run_id}/4 completed successfully!${NC}"
    echo ""
}

# Run experiments for each LoRA rank
for i in "${!LORA_RANKS[@]}"; do
    run_id=$((i + 1))
    rank=${LORA_RANKS[$i]}
    run_experiment $run_id $rank
done

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}All 4 experiments completed!${NC}"
echo -e "${YELLOW}Sweep ID: ${SWEEP_NAME}${NC}"
echo -e "${YELLOW}Check WandB group '${SWEEP_NAME}' for comparison${NC}"