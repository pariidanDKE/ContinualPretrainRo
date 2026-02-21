#!/bin/bash

# Hyperparameter sweep script
# Runs 3 sequential training experiments with different learning rates and gradient accumulation
# Each run trains for 100M tokens with evaluations at 25M, 50M, 75M, and 100M token milestones

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

# Generate timestamp for this sweep
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_NAME="hyperparam_sweep_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Hyperparameter Sweep${NC}"
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
    echo -e "${GREEN}Run ${run_id}/3${NC}"
    echo -e "${GREEN}Learning Rate: ${lr}${NC}"
    echo -e "${GREEN}Gradient Accumulation Steps: ${grad_accum}${NC}"
    echo -e "${GREEN}========================================${NC}"


    # Calculate embedding learning rate as main_lr / 5
    local embedding_lr=$(awk "BEGIN {print $lr/5}")

    # Create a concise custom identifier (will be inserted into auto-generated name)
    # Format: {cpt/sft}-{model}-{custom_name}-ms{milestones}-{tokens}M
    CUSTOM_NAME="lr${lr}_elr${embedding_lr}_ga${grad_accum}"

    # Run the training
    python train_milestones.py \
        training_args.learning_rate=${lr} \
        training_args.embedding_learning_rate=${embedding_lr} \
        training_args.gradient_accumulation_steps=${grad_accum} \
        milestone.tokens_per_milestone=${TOKENS_PER_MILESTONE} \
        milestone.num_milestones=${NUM_MILESTONES} \
        wandb.custom_run_name="${CUSTOM_NAME}" \
        wandb.group="${SWEEP_NAME}"

    echo -e "${GREEN}Run ${run_id}/3 completed successfully!${NC}"
    echo ""
}
# Run 2: lr=3e-4, grad_accum=4
#run_experiment 1 1e-4 4

# Run 2: lr=3e-4, grad_accum=4
run_experiment 1 3e-4 4

# Run 3: lr=5e-4, grad_accum=4
run_experiment 2 5e-4 4

# Run 2: lr=3e-4, grad_accum=4
run_experiment 3 1e-3 4



echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}All 3 experiments completed!${NC}"
echo -e "${YELLOW}Sweep ID: ${SWEEP_NAME}${NC}"
echo -e "${YELLOW}Check WandB for results and comparisons${NC}"
echo -e "${YELLOW}========================================${NC}"
