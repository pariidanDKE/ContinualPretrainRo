#!/bin/bash

# Embedding Layer Comparison Script
# Compares 3 strategies for handling embed_tokens/lm_head layers
# Each run trains for 125M tokens with evaluations at 25M intervals
#
# Run 1 (COMPLETED): Full weights - modules_to_save, embedding_learning_rate=2e-5
# Run 2 (COMPLETED): Frozen - embeddings not in target_modules

set -e  # Exit on error

# Navigate to the continual_pretrain directory
cd "$(dirname "$0")/.."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
TOKENS_PER_MILESTONE=50000000  # 25M tokens per milestone
NUM_MILESTONES=5              # 4 milestones = 100M tokens total
LORA_RANK=64                   # LoRA rank
LORA_ALPHA=64                  # LoRA alpha (typically equal to rank)
LEARNING_RATE=1e-4             # Main learning rate
EMBEDDING_LR=2e-5              # Embedding learning rate (only for Run 1)
BATCH_SIZE=32                  # Per-device batch size
GRAD_ACCUM=4                   # Gradient accumulation steps

# Generate timestamp for this comparison
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_NAME="embed_comparison_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Embedding Layer Comparison${NC}"
echo -e "${BLUE}Comparison ID: ${SWEEP_NAME}${NC}"
echo -e "${BLUE}Total tokens per run: $((TOKENS_PER_MILESTONE * NUM_MILESTONES))${NC}"
echo -e "${BLUE}Milestones: ${NUM_MILESTONES} (every ${TOKENS_PER_MILESTONE} tokens)${NC}"
echo -e "${BLUE}LoRA Rank: ${LORA_RANK}, Alpha: ${LORA_ALPHA}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run a single training experiment
run_experiment() {
    local run_id=$1
    local custom_name=$2
    shift 2
    local extra_args=("$@")

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Run ${run_id}/2: ${custom_name}${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Run the training with common parameters + experiment-specific overrides
    python train_milestones.py \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        training_args.learning_rate=${LEARNING_RATE} \
        training_args.per_device_train_batch_size=${BATCH_SIZE} \
        training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
        milestone.tokens_per_milestone=${TOKENS_PER_MILESTONE} \
        milestone.num_milestones=${NUM_MILESTONES} \
        wandb.custom_run_name="${custom_name}" \
        wandb.group="${SWEEP_NAME}" \
        training_args.packing=false \
        data_builder.enabled=false \
        "${extra_args[@]}"

    echo -e "${GREEN}Run ${run_id}/2 completed successfully!${NC}"
    echo ""
}

# #Run 1: WITH embedding layers (embed_tokens + lm_head) using separate embedding_learning_rate
echo -e "${BLUE}Run 1: Training WITH embed_tokens/lm_head layers${NC}"
echo -e "${BLUE}  - embed_tokens and lm_head included in LoRA target_modules${NC}"
echo -e "${BLUE}  - embedding_learning_rate: ${EMBEDDING_LR}${NC}"
echo -e "${BLUE}  - Main LR: ${LEARNING_RATE}${NC}"
echo ""
run_experiment 1 "with_embeddings_r${LORA_RANK}" \
    training_args.embedding_learning_rate=${EMBEDDING_LR} \
    'model.lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,embed_tokens,lm_head]'

#Run 2: WITHOUT embedding layers (only transformer projections)
echo -e "${BLUE}Run 2: Training WITHOUT embed_tokens/lm_head layers${NC}"
echo -e "${BLUE}  - Only transformer layers in target_modules${NC}"
echo -e "${BLUE}  - Main LR: ${LEARNING_RATE} (no separate embedding LR)${NC}"
echo ""
run_experiment 2 "no_embeddings_r${LORA_RANK}" \
    'model.lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' 

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Experiment completed!${NC}"
echo -e "${YELLOW}Comparison ID: ${SWEEP_NAME}${NC}"
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Results Summary (2 configurations):${NC}"
echo -e "${YELLOW}  Run 1: Full weights embeddings (COMPLETED)${NC}"
echo -e "${YELLOW}    - modules_to_save, Embedding LR: ${EMBEDDING_LR}${NC}"
echo -e "${YELLOW}  Run 2: Frozen embeddings (COMPLETED)${NC}"
echo -e "${YELLOW}    - No embeddings trained${NC}"
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Check WandB group '${SWEEP_NAME}' for comparison${NC}"
echo -e "${YELLOW}========================================${NC}"
