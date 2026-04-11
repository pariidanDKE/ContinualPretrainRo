#!/bin/bash

# Gradient Accumulation sweep script
# Tests GA in {1, 2, 8} with fixed LR=1e-4, ELR=2e-5 (LR/5)
# GA=4 baseline already exists: run daywaxrs (cpt-llama-elr2e-5_lr1e-4-ms5-125M)
# Goal: isolate effect of optimizer step frequency on convergence within a fixed token budget

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
EMBEDDING_LR=2e-5  # LR/5 — fixed, matches GA=4 baseline run
BATCH_SIZE=32

GRAD_ACCUM_VALUES=(1 2 8)  # GA=4 baseline already exists (run daywaxrs)

# 80/20 data mix configuration
TOTAL_SAMPLES=200000

# Generate timestamp for this sweep
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_NAME="hyperparam_sweep_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Gradient Accumulation Sweep${NC}"
echo -e "${BLUE}Sweep ID: ${SWEEP_NAME}${NC}"
echo -e "${BLUE}Total tokens per run: $((TOKENS_PER_MILESTONE * NUM_MILESTONES))${NC}"
echo -e "${BLUE}Milestones: ${NUM_MILESTONES} (every ${TOKENS_PER_MILESTONE} tokens)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run a single training experiment
run_experiment() {
    local run_id=$1
    local lr=$2
    local grad_accum=$3

    echo -e "${GREEN}========================================${NC}"
    local num_runs=${#GRAD_ACCUM_VALUES[@]}
    echo -e "${GREEN}Run ${run_id}/${num_runs}${NC}"
    echo -e "${GREEN}Learning Rate: ${lr} (fixed)${NC}"
    echo -e "${GREEN}Gradient Accumulation Steps: ${grad_accum}${NC}"
    echo -e "${GREEN}========================================${NC}"


    # Create a concise custom identifier (will be inserted into auto-generated name)
    # Format: {cpt/sft}-{model}-{custom_name}-ms{milestones}-{tokens}M
    CUSTOM_NAME="lr${lr}_elr${EMBEDDING_LR}_ga${grad_accum}"

    # Run the training
    python train_milestones.py \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        'model.lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,embed_tokens,lm_head]' \
        training_args.learning_rate=${lr} \
        training_args.embedding_learning_rate=${EMBEDDING_LR} \
        training_args.per_device_train_batch_size=${BATCH_SIZE} \
        training_args.gradient_accumulation_steps=${grad_accum} \
        training_args.packing=false \
        milestone.tokens_per_milestone=${TOKENS_PER_MILESTONE} \
        milestone.num_milestones=${NUM_MILESTONES} \
        data_builder.enabled=true \
        data_builder.total_sample_size=${TOTAL_SAMPLES} \
        "data_builder.proportions=[0.8,0.2]" \
        wandb.custom_run_name="${CUSTOM_NAME}" \
        wandb.group="${SWEEP_NAME}"

    echo -e "${GREEN}Run ${run_id} completed successfully!${NC}"
    echo ""
}

for i in "${!GRAD_ACCUM_VALUES[@]}"; do
    run_experiment $((i + 1)) ${LEARNING_RATE} ${GRAD_ACCUM_VALUES[$i]}
done

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}All ${#GRAD_ACCUM_VALUES[@]} experiments completed!${NC}"
echo -e "${YELLOW}Sweep ID: ${SWEEP_NAME}${NC}"
echo -e "${YELLOW}Check WandB for results and comparisons${NC}"
echo -e "${YELLOW}========================================${NC}"
