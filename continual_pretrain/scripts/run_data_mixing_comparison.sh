#!/bin/bash

# Data Mixing Comparison Script
# Compares 2 mixing configs: pure Romanian vs 80/20 Romanian/English
# 250M tokens across 10 milestones for stronger downstream task signal
#
# Run 1: 100/0  - pure Romanian (data_builder disabled, single dataset)
# Run 2:  80/20 - 80% Romanian + 20% English edu
#
# NOTE: proportions are only applied when total_sample_size is set (see TOTAL_SAMPLES below).
# Without it, mixing ratio = raw dataset sizes, ignoring proportion values entirely.
# NOTE: for a cleaner 100/0 baseline, larger fineweb-edu sample needed (see danp27/fineweb-edu-200k-sample).

set -e

cd "$(dirname "$0")/.."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Configuration ─────────────────────────────────────────────────────────────
# TOKENS_PER_MILESTONE=25000000   # 25M tokens per milestone
# NUM_MILESTONES=10               # 10 milestones = 250M tokens total
TOKENS_PER_MILESTONE=2000000   # 25M tokens per milestone
NUM_MILESTONES=1               # 10 milestones = 250M tokens total
LORA_RANK=64
LORA_ALPHA=64
LEARNING_RATE=1e-4
EMBEDDING_LR=2e-5
BATCH_SIZE=32
GRAD_ACCUM=4

# Total samples drawn from all datasets combined (controls actual mixing ratios).
# Proportions are applied to this budget. Set higher if you have more data.
TOTAL_SAMPLES=600000

# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_NAME="data_mix_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Data Mixing Comparison${NC}"
echo -e "${BLUE}Comparison ID: ${SWEEP_NAME}${NC}"
echo -e "${BLUE}Total tokens per run: $((TOKENS_PER_MILESTONE * NUM_MILESTONES))${NC}"
echo ""

run_experiment() {
    local run_id=$1
    local custom_name=$2
    shift 2
    local extra_args=("$@")

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Run ${run_id}/2: ${custom_name}${NC}"
    echo -e "${GREEN}========================================${NC}"

    python train_milestones.py \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        training_args.learning_rate=${LEARNING_RATE} \
        training_args.embedding_learning_rate=${EMBEDDING_LR} \
        training_args.per_device_train_batch_size=${BATCH_SIZE} \
        training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
        milestone.tokens_per_milestone=${TOKENS_PER_MILESTONE} \
        milestone.num_milestones=${NUM_MILESTONES} \
        milestone.do_evaluate=true \
        milestone.run_benchmarks=false \
        validation_cfg.return_validation=true \
        wandb.custom_run_name="${custom_name}" \
        wandb.group="${SWEEP_NAME}" \
        "${extra_args[@]}"

    echo -e "${GREEN}Run ${run_id}/2 completed!${NC}"
    echo ""
}

# # Run 1: 100% Romanian — disable data_builder, use single dataset path
# # (proportion=0.0 is falsy in Python and skips truncation, so we avoid it entirely)
# echo -e "${BLUE}Run 1: 100% fineweb-ro / 0% fineweb-edu (pure Romanian baseline)${NC}"
# echo -e "${BLUE}  - data_builder disabled, falls back to dataset config${NC}"
# echo ""
# run_experiment 1 "mix_100_0_250M" \
#     data_builder.enabled=false

# Run 2: 80% Romanian / 20% English edu
echo -e "${BLUE}Run 2: 80% fineweb-ro / 20% fineweb-edu${NC}"
echo -e "${BLUE}  - total_sample_size: ${TOTAL_SAMPLES}${NC}"
echo ""
run_experiment 2 "mix_80_20_250M" \
    data_builder.enabled=true \
    data_builder.total_sample_size=${TOTAL_SAMPLES} \
    "data_builder.proportions=[0.8,0.2]"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Data Mixing Comparison Complete!${NC}"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Results Summary (2 configurations):${NC}"
echo -e "${YELLOW}  Run 1: 100/0  - pure Romanian CPT (baseline)${NC}"
echo -e "${YELLOW}  Run 2:  80/20 - Romanian + English edu (Pareto candidate)${NC}"
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Check WandB group '${SWEEP_NAME}' for comparison${NC}"
echo -e "${YELLOW}========================================${NC}"
