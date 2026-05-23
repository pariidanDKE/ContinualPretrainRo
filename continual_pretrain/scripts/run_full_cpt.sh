#!/bin/bash

# Full CPT Run — Llama-3.2-1B Romanian Continual Pretraining
# Final config from hyperparameter sweeps:
#   - LoRA rank: 64, alpha: 64, rslora: true
#   - LR: 1e-4, ELR: 2e-5 (LR/5)
#   - GA: 8, batch size: 16 (effective BS=128)
#   - Data mix: 80% Romanian / 20% English
#   - Packing: off
#   - Scheduler: warmup_stable_decay
# 8 milestones x 300M = 2.4B total token budget

set -e

cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Config ────────────────────────────────────────────────
TOKENS_PER_MILESTONE=300000000   # 300M tokens per milestone
NUM_MILESTONES=8                 # 8 milestones = 2.4B total budget

LORA_RANK=64
LORA_ALPHA=64
LEARNING_RATE=1e-4
EMBEDDING_LR=2e-5
# Desktop-safe profile: lower micro-batch to free VRAM while keeping
# the same effective batch size (4 * 32 = 128, previously 8 * 16 = 128).
BATCH_SIZE=4
GRAD_ACCUM=32

TOTAL_SAMPLES=2500000            # 2M RO + 500k EN (strict 80/20)

# Helps reduce allocator fragmentation during long runs near memory limits.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="full_cpt_llama_1b_${TIMESTAMP}"
# ─────────────────────────────────────────────────────────

EVAL_ON_START=true

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Full CPT Run — Llama-3.2-1B${NC}"
echo -e "${BLUE}Run name: ${RUN_NAME}${NC}"
echo -e "${BLUE}Total tokens: $((TOKENS_PER_MILESTONE * NUM_MILESTONES / 1000000))M (${NUM_MILESTONES} milestones x $((TOKENS_PER_MILESTONE / 1000000))M)${NC}"
echo -e "${BLUE}LR=${LEARNING_RATE}, ELR=${EMBEDDING_LR}, r=${LORA_RANK}, GA=${GRAD_ACCUM}, BS=${BATCH_SIZE}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

python train_milestones.py \
    model.lora.r=${LORA_RANK} \
    model.lora.lora_alpha=${LORA_ALPHA} \
    'model.lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,embed_tokens,lm_head]' \
    training_args.learning_rate=${LEARNING_RATE} \
    training_args.embedding_learning_rate=${EMBEDDING_LR} \
    training_args.per_device_train_batch_size=${BATCH_SIZE} \
    training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
    training_args.packing=false \
    training_args.eval_on_start=${EVAL_ON_START} \
    training_args.max_steps=30000 \
    training_args.save_total_limit=11 \
    milestone.tokens_per_milestone=${TOKENS_PER_MILESTONE} \
    milestone.num_milestones=${NUM_MILESTONES} \
    data_builder.enabled=true \
    data_builder.total_sample_size=${TOTAL_SAMPLES} \
    "data_builder.proportions=[0.8,0.2]" \
    "data_builder.datasets.0.name=./data/fineweb2_ro_score4.parquet" \
    seed=42 \
    milestone.run_benchmarks=false \
    milestone.do_evaluate=true \
    wandb.custom_run_name="${RUN_NAME}" \
    wandb.group="${RUN_NAME}"


    

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Full CPT run completed!${NC}"
echo -e "${YELLOW}Run: ${RUN_NAME}${NC}"
echo -e "${YELLOW}Check WandB for milestone-by-milestone results${NC}"
echo -e "${YELLOW}========================================${NC}"
