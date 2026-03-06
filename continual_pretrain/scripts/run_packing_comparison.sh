#!/bin/bash

# Packing Comparison Script
# Compares packing=true vs packing=false on a 100% Romanian (fineweb2-ro-llm) dataset.
# All other hyperparams match the 100/0 baseline in run_data_mixing_comparison.sh.
#
# Run 1: packing=true  — sequences packed into full context windows (faster, denser batches)
# Run 2: packing=false — each sample padded to max_length (standard training)

set -e

cd "$(dirname "$0")/.."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Configuration (mirrors run_data_mixing_comparison.sh 100% Romanian baseline) ──
TOKENS_PER_MILESTONE=25000000
NUM_MILESTONES=5
LORA_RANK=64
LORA_ALPHA=64
LEARNING_RATE=1e-4
EMBEDDING_LR=2e-5
BATCH_SIZE=32
GRAD_ACCUM=4

# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_NAME="packing_cmp_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Packing Comparison${NC}"
echo -e "${BLUE}Comparison ID: ${SWEEP_NAME}${NC}"
echo -e "${BLUE}Dataset: 100% fineweb2-ro-llm (pure Romanian)${NC}"
echo -e "${BLUE}Total tokens per run: $((TOKENS_PER_MILESTONE * NUM_MILESTONES))${NC}"
echo ""

run_experiment() {
    local run_id=$1
    local custom_name=$2
    local packing=$3

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Run ${run_id}/2: ${custom_name} (packing=${packing})${NC}"
    echo -e "${GREEN}========================================${NC}"

    python train_milestones.py \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        training_args.learning_rate=${LEARNING_RATE} \
        training_args.embedding_learning_rate=${EMBEDDING_LR} \
        training_args.per_device_train_batch_size=${BATCH_SIZE} \
        training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
        training_args.packing=${packing} \
        milestone.tokens_per_milestone=${TOKENS_PER_MILESTONE} \
        milestone.num_milestones=${NUM_MILESTONES} \
        milestone.do_evaluate=true \
        validation_cfg.return_validation=true \
        data_builder.enabled=false \
        wandb.custom_run_name="${custom_name}" \
        wandb.group="${SWEEP_NAME}"

    echo -e "${GREEN}Run ${run_id}/2 completed!${NC}"
    echo ""
}

# Run 1: 100% Romanian + packing enabled
echo -e "${BLUE}Run 1: packing=true — packed sequences, denser GPU utilisation${NC}"
echo ""
run_experiment 1 "ro100_packing_on" true

# Run 2: 100% Romanian + packing disabled
echo -e "${BLUE}Run 2: packing=false — standard padded batches${NC}"
echo ""
run_experiment 2 "ro100_packing_off" false

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Packing Comparison Complete!${NC}"
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Results Summary (2 configurations):${NC}"
echo -e "${YELLOW}  Run 1: packing=true  — packed sequences${NC}"
echo -e "${YELLOW}  Run 2: packing=false — padded sequences${NC}"
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Check WandB group '${SWEEP_NAME}' for comparison${NC}"
echo -e "${YELLOW}========================================${NC}"
