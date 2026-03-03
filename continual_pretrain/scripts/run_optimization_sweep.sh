#!/bin/bash

# Optimization Stack Sweep — Loss Comparison (bs=1 only)
# Runs all 9 optimization configs at bs=1 ga=32 to show per-config loss curves.
# Only bs=1 is used: the throughput sweep (opt1) already shows why other BS configs
# behave differently, making the bs=1 loss comparison self-contained.
#
# Stack order (each config adds one optimization):
#   1. fp32            — baseline, no optimizations
#   2. +bf16           — mixed precision
#   3. +GC             — gradient checkpointing
#   4. +FA2            — Flash Attention 2
#   5. +paged_opt      — 8-bit paged AdamW optimizer
#   6. +lora_r64       — LoRA (r=64, no quantization)
#   7. +qlora          — 4-bit quantization (qLoRA)
#   8. +unsloth        — Unsloth kernels
#   9. +packing        — token packing (throughput, not memory)
#
# OOM configs are skipped gracefully; the sweep continues.
# Each run: 20M tokens, 1 milestone, warmup_ratio=0.1 (2M tokens), no eval.

set -e

cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

HF_MODEL="meta-llama/Llama-3.2-1B"
UNSLOTH_MODEL="unsloth/Llama-3.2-1B"
TOKENS=50000000
MILESTONES=1
WARMUP_RATIO=0.1
LORA_RANK=64
LORA_ALPHA=64
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

BS_GA_PAIRS=("1 32")
SWEEP_NAME="opt2_loss_sweep_bs${BS}_ga${GA}_${TIMESTAMP}"

# Globals updated each loop iteration: BS, GA, MODEL, SWEEP_NAME, RESULTS
ALL_RESULTS=()

run_config() {
    local name=$1
    shift
    echo -e "${GREEN}----------------------------------------${NC}"
    echo -e "${GREEN}Config: ${name}  [bs=${BS} ga=${GA} eff_bs=$((BS * GA))]${NC}"
    echo -e "${GREEN}----------------------------------------${NC}"

    python train_milestones.py \
        model.builder.model_name=${MODEL} \
        model.builder.packing=false \
        milestone.tokens_per_milestone=${TOKENS} \
        milestone.num_milestones=${MILESTONES} \
        milestone.do_evaluate=false \
        validation_cfg.return_validation=false \
        training_args.per_device_train_batch_size=${BS} \
        training_args.gradient_accumulation_steps=${GA} \
        training_args.warmup_ratio=${WARMUP_RATIO} \
        training_args.save_strategy=no \
        wandb.enabled=true \
        wandb.group="${SWEEP_NAME}" \
        wandb.custom_run_name="${name}" \
        "$@"

    echo -e "${GREEN}Config '${name}' done.${NC}"
    echo ""
}

try_config() {
    local name=$1
    shift
    if run_config "$name" "$@"; then
        RESULTS+=("${GREEN}✓ ${name}${NC}")
    else
        echo -e "${RED}Config '${name}' failed (OOM or error) — skipping.${NC}"
        echo ""
        RESULTS+=("${RED}✗ ${name} (OOM/failed)${NC}")
    fi
}

run_all_configs() {
    local prefix="bs${BS}_ga${GA}"

    # ── Configs 1-7: standard HF model ──────────────────────────────────────
    MODEL="${HF_MODEL}"

    # # # Config 1: fp32 — baseline, expected to OOM
    # export USE_UNSLOTH=0
    # try_config "${prefix}_1_fp32" \
    #     training_args.bf16=false \
    #     training_args.fp16=false \
    #     training_args.gradient_checkpointing=false \
    #     training_args.optim=adamw_torch \
    #     training_args.embedding_learning_rate=null \
    #     model.builder.bnb_4bit_compute_dtype=fp32 \
    #     model.builder.use_flash_attention=false \
    #     model.builder.use_lora=false \
    #     model.builder.load_in_4bit=false \
    #     model.builder.use_unsloth=false

    # # Config 2: +bf16 — halves activation memory, still likely OOM without GC
    # export USE_UNSLOTH=0
    # try_config "${prefix}_2_bf16" \
    #     training_args.bf16=true \
    #     training_args.fp16=false \
    #     training_args.gradient_checkpointing=false \
    #     training_args.optim=adamw_torch \
    #     training_args.embedding_learning_rate=null \
    #     model.builder.bnb_4bit_compute_dtype=bfloat16 \
    #     model.builder.use_flash_attention=false \
    #     model.builder.use_lora=false \
    #     model.builder.load_in_4bit=false \
    #     model.builder.use_unsloth=false

    # Config 3: +GC — trades compute for memory via activation recomputation
    export USE_UNSLOTH=0
    try_config "${prefix}_3_bf16_gc" \
        training_args.bf16=true \
        training_args.fp16=false \
        training_args.gradient_checkpointing=true \
        training_args.optim=adamw_torch \
        training_args.embedding_learning_rate=null \
        model.builder.bnb_4bit_compute_dtype=bfloat16 \
        model.builder.use_flash_attention=false \
        model.builder.use_lora=false \
        model.builder.load_in_4bit=false \
        model.builder.use_unsloth=false

    # Config 4: +FA2 — O(n) memory attention, first stable config at bs=1 seq=2048
    export USE_UNSLOTH=0
    try_config "${prefix}_4_bf16_gc_fa2" \
        training_args.bf16=true \
        training_args.fp16=false \
        training_args.gradient_checkpointing=true \
        training_args.optim=adamw_torch \
        training_args.embedding_learning_rate=null \
        model.builder.bnb_4bit_compute_dtype=bfloat16 \
        model.builder.use_flash_attention=true \
        model.builder.use_lora=false \
        model.builder.load_in_4bit=false \
        model.builder.use_unsloth=false

    # Config 5: +paged_opt — 8-bit optimizer, most impactful before LoRA (~1B params)
    export USE_UNSLOTH=0
    try_config "${prefix}_5_bf16_gc_fa2_paged" \
        training_args.bf16=true \
        training_args.fp16=false \
        training_args.gradient_checkpointing=true \
        training_args.optim=paged_adamw_8bit \
        training_args.embedding_learning_rate=null \
        model.builder.bnb_4bit_compute_dtype=bfloat16 \
        model.builder.use_flash_attention=true \
        model.builder.use_lora=false \
        model.builder.load_in_4bit=false \
        model.builder.use_unsloth=false

    # Config 6: +LoRA r64 — trainable params drop from ~1B to ~5-10M
    export USE_UNSLOTH=0
    try_config "${prefix}_6_bf16_gc_fa2_paged_lora64" \
        training_args.bf16=true \
        training_args.fp16=false \
        training_args.gradient_checkpointing=true \
        training_args.optim=paged_adamw_8bit \
        training_args.embedding_learning_rate=null \
        model.builder.bnb_4bit_compute_dtype=bfloat16 \
        model.builder.use_flash_attention=true \
        model.builder.use_lora=true \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        model.builder.load_in_4bit=false \
        model.builder.use_unsloth=false

    # Config 7: +qLoRA — base model weights go 4-bit (NF4)
    export USE_UNSLOTH=0
    
    #NOTE: PUT THIS BACK IN
    #training_args.optim=paged_adamw_8bit \
    try_config "${prefix}_7_bf16_gc_fa2_paged_qlora64" \
        training_args.bf16=true \
        training_args.fp16=false \
        training_args.gradient_checkpointing=true \
        training_args.embedding_learning_rate=null \
        model.builder.bnb_4bit_compute_dtype=bfloat16 \
        model.builder.use_flash_attention=true \
        model.builder.use_lora=true \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        model.builder.load_in_4bit=true \
        model.builder.use_unsloth=false

    # # ── Configs 8-9: switch to unsloth-optimized weights ────────────────────
    UNSLOTH_MODEL="unsloth/Llama-3.2-1B"

    MODEL="${UNSLOTH_MODEL}"
    #        training_args.optim=paged_adamw_8bit \
    # Config 8: +Unsloth — fused kernels + custom CUDA ops on top of qLoRA
    export USE_UNSLOTH=1
    try_config "${prefix}_8_bf16_gc_fa2_paged_qlora64_unsloth" \
        training_args.bf16=true \
        training_args.fp16=false \
        training_args.gradient_checkpointing=true \
        training_args.embedding_learning_rate=null \
        model.builder.bnb_4bit_compute_dtype=bfloat16 \
        model.builder.use_flash_attention=true \
        model.builder.use_lora=true \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        model.builder.load_in_4bit=true \
        model.builder.use_unsloth=true

#     #Config 9: +packing — maximizes GPU utilization, improves throughput not peak memory
    # export USE_UNSLOTH=1
    # try_config "${prefix}_9_full_stack_packing" \
    #     training_args.bf16=true \
    #     training_args.fp16=false \
    #     training_args.gradient_checkpointing=true \
    #     training_args.optim=paged_adamw_8bit \
    #     training_args.embedding_learning_rate=2e-5 \
    #     model.builder.bnb_4bit_compute_dtype=bfloat16 \
    #     model.builder.use_flash_attention=true \
    #     model.builder.use_lora=true \
    #     model.lora.r=${LORA_RANK} \
    #     model.lora.lora_alpha=${LORA_ALPHA} \
    #     model.builder.load_in_4bit=true \
    #     model.builder.prepare_kbit_training=true \
    #     model.builder.use_unsloth=true \
    #     model.builder.packing=true \
    #     training_args.packing=true
  }

# Smoothed training loss from the 6 configurations which did not OOM on batch size. The run has batch size =1 and was made over 20M tokens, to show signs of stability.

# ── Outer loop over BS/GA pairs ──────────────────────────────────────────────
for pair in "${BS_GA_PAIRS[@]}"; do
    BS=$(echo "$pair" | awk '{print $1}')
    GA=$(echo "$pair" | awk '{print $2}')
    SWEEP_NAME="opt4_sweep_bs${BS}_ga${GA}_${TIMESTAMP}"
    RESULTS=()

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Batch Config: bs=${BS}  ga=${GA}  (eff_bs=$((BS * GA)))${NC}"
    echo -e "${BLUE}WandB group: ${SWEEP_NAME}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    run_all_configs

    echo -e "${YELLOW}Results — bs=${BS} ga=${GA}:${NC}"
    for result in "${RESULTS[@]}"; do
        echo -e "  ${result}"
    done
    echo ""

    ALL_RESULTS+=("${YELLOW}bs=${BS} ga=${GA}:${NC}")
    ALL_RESULTS+=("${RESULTS[@]}")
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Full sweep complete. Timestamp: ${TIMESTAMP}${NC}"
echo -e "${YELLOW}WandB groups: opt2_loss_sweep_bs*_${TIMESTAMP}${NC}"
echo -e "${YELLOW}----------------------------------------${NC}"
for result in "${ALL_RESULTS[@]}"; do
    echo -e "  ${result}"
done
echo -e "${YELLOW}========================================${NC}"
