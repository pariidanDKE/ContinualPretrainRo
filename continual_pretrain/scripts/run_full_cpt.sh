#!/bin/bash

# Full CPT Run — Llama-3.2-1B Romanian Continual Pretraining — Epoch 2
# Final config from hyperparameter sweeps:
#   - LoRA rank: 64, alpha: 64, rslora: true
#   - LR: 1e-4, ELR: 2e-5 (LR/5)
#   - GA: 4, batch size: 32
#   - Data mix: 80% Romanian / 20% English
#   - Packing: off
#   - Scheduler: warmup_stable_decay
# Resuming from step 9017 (950,004,117 tokens seen) — end of stable phase, before decay
# 20 milestones x 100M = 2B total token budget; ~1B new tokens for epoch 2
# resume_tokens_seen places WSD scheduler in stable phase immediately (skips warmup)

set -e

cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Config ────────────────────────────────────────────────
TOKENS_PER_MILESTONE=100000000   # 100M tokens per milestone
NUM_MILESTONES=20                # 20 milestones = 2B total budget (epoch 2 adds ~1B)

LORA_RANK=64
LORA_ALPHA=64
LEARNING_RATE=1e-4
EMBEDDING_LR=2e-5
BATCH_SIZE=16
GRAD_ACCUM=8

TOTAL_SAMPLES=1000000            # 800k RO + 200k EN — epoch 2, offset 0 (full dataset restart)
RESUME_CHECKPOINT="/home/dan-parii/Documents/ContinualPretrainRo/continual_pretrain/outputs/cpt/full_cpt_llama_1b_20260401_003748/cpt-llama-full_cpt_llama_1b_20260401_003748-ms10-1B/20260401_0038-iruygely/checkpoint-9017"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="full_cpt_llama_1b_${TIMESTAMP}"
# ─────────────────────────────────────────────────────────

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
    training_args.max_steps=30000 \
    training_args.save_total_limit=11 \
    milestone.tokens_per_milestone=${TOKENS_PER_MILESTONE} \
    milestone.num_milestones=${NUM_MILESTONES} \
    data_builder.enabled=true \
    data_builder.total_sample_size=${TOTAL_SAMPLES} \
    "data_builder.proportions=[0.8,0.2]" \
    seed=42 \
    milestone.resume_from_checkpoint="${RESUME_CHECKPOINT}" \
    milestone.resume_tokens_seen=950004117 \
    milestone.run_benchmarks=true \
    milestone.do_evaluate=true \
    training_args.ignore_data_skip=true \
    validation_cfg.val_dataset_path="/home/dan-parii/Documents/ContinualPretrainRo/continual_pretrain/data/val_dataset" \
    wandb.custom_run_name="${RUN_NAME}" \
    wandb.group="${RUN_NAME}"


    

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Full CPT run completed!${NC}"
echo -e "${YELLOW}Run: ${RUN_NAME}${NC}"
echo -e "${YELLOW}Check WandB for milestone-by-milestone results${NC}"
echo -e "${YELLOW}========================================${NC}"
