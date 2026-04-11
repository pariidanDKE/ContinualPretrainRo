#!/bin/bash

# Embedding LR Sweep Script
# Tests three embedding learning rates relative to main LR (1e-4):
#   - ELR = 5e-5  (LR / 2)
#   - ELR = 2e-5  (LR / 5)
#   - ELR = 1e-5  (LR / 10)
#
# All runs use:
# - 80/20 data mix (80% Romanian / 20% English) — final config
# - WITH embeddings (embed_tokens + lm_head)
# - r64, alpha=64 (Pareto winner from LoRA rank sweep)
# - No packing
# - 125M tokens (5 milestones x 25M)

set -e  # Exit on error

# Navigate to the continual_pretrain directory
cd "$(dirname "$0")/.."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

#Configuration
TOKENS_PER_MILESTONE=25000000  # 25M tokens per milestone
NUM_MILESTONES=5               # 5 milestones = 125M tokens total

LORA_RANK=64
LORA_ALPHA=64
LEARNING_RATE=1e-4
BATCH_SIZE=32
GRAD_ACCUM=4

#EMBEDDING_LRS=(1e-4 5e-5 2e-5)  # LR/1, LR/2, LR/5
EMBEDDING_LRS=(5e-5)  



# 80/20 data mix configuration
TOTAL_SAMPLES=200000

# Generate timestamp for this sweep
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_NAME="embed_lr_sweep_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Embedding LR Sweep${NC}"
echo -e "${BLUE}Sweep ID: ${SWEEP_NAME}${NC}"
echo -e "${BLUE}Main LR: ${LEARNING_RATE}${NC}"
echo -e "${BLUE}ELRs to test: ${EMBEDDING_LRS[*]}${NC}"
echo -e "${BLUE}Total tokens per run: $((TOKENS_PER_MILESTONE * NUM_MILESTONES))${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

run_experiment() {
    local run_id=$1
    local elr=$2

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Run ${run_id}/3${NC}"
    echo -e "${GREEN}Embedding LR: ${elr} (Main LR: ${LEARNING_RATE})${NC}"
    echo -e "${GREEN}========================================${NC}"

    CUSTOM_NAME="cpt-llama-elr${elr}_lr${LEARNING_RATE}-ms${NUM_MILESTONES}-125M"

    python train_milestones.py \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        training_args.learning_rate=${LEARNING_RATE} \
        training_args.embedding_learning_rate=${elr} \
        training_args.per_device_train_batch_size=${BATCH_SIZE} \
        training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
        training_args.packing=false \
        milestone.tokens_per_milestone=${TOKENS_PER_MILESTONE} \
        milestone.num_milestones=${NUM_MILESTONES} \
        data_builder.enabled=true \
        data_builder.total_sample_size=${TOTAL_SAMPLES} \
        "data_builder.proportions=[0.8,0.2]" \
        'model.lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,embed_tokens,lm_head]' \
        wandb.custom_run_name="${CUSTOM_NAME}" \
        wandb.group="${SWEEP_NAME}"

    echo -e "${GREEN}Run ${run_id}/3 completed successfully!${NC}"
    echo ""
}

for i in "${!EMBEDDING_LRS[@]}"; do
    run_id=$((i + 1))
    elr=${EMBEDDING_LRS[$i]}
    run_experiment $run_id $elr
done

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}All 3 experiments completed!${NC}"
echo -e "${YELLOW}Sweep ID: ${SWEEP_NAME}${NC}"
echo -e "${YELLOW}Configs:${NC}"
echo -e "${YELLOW}  Run 1: ELR=5e-5 (LR/2)${NC}"
echo -e "${YELLOW}  Run 2: ELR=2e-5 (LR/5)${NC}"
echo -e "${YELLOW}  Run 3: ELR=1e-5 (LR/10)${NC}"
echo -e "${YELLOW}Check WandB group '${SWEEP_NAME}' for comparison${NC}"
echo -e "${YELLOW}========================================${NC}"
