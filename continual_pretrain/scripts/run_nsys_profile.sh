#!/bin/bash
# nsys kernel profiling — configs 3-9 × BS/GA pairs
#
# Mirrors the config progression in run_optimization_sweep.sh but wraps each
# run with nsys to capture steady-state kernel activity.
#
# nsys strategy:
#   --delay  90    skip model load + optional CUDA graph compilation
#   --duration 90  capture ~90s of steady-state training per config
#
# Configs 3-7 use the HF model (no Unsloth compilation overhead).
# Configs 8-9 use the Unsloth model (CUDA graph compile adds ~30-60s).
# A uniform 90s delay is conservative enough to clear both cases.
#
# Output layout:
#   outputs/nsys_profiles/bs{BS}_ga{GA}/{config_name}.nsys-rep
#   outputs/nsys_profiles/bs{BS}_ga{GA}/{config_name}.sqlite
#
# Usage:
#   chmod +x scripts/run_nsys_profile.sh
#   ./scripts/run_nsys_profile.sh
#   python diagnostics_debugging/nsys_kernel_analysis.py

cd "$(dirname "$0")/.."

HF_MODEL="meta-llama/Llama-3.2-1B"
UNSLOTH_MODEL="unsloth/Llama-3.2-1B"
LORA_RANK=64
LORA_ALPHA=64

NSYS_DELAY=90
NSYS_DURATION=90
RUN_TS=$(date +%Y%m%d_%H%M)

BS_GA_PAIRS=("1 32" "8 4")

GREEN='\033[0;32m'
BLUE='\033[0;34m'   
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ── Per-run profiler ──────────────────────────────────────────────────────────
profile_config() {
    local name=$1
    shift
    local out_dir="outputs/nsys_profiles/batch_size_comparison/bs${BS}_ga${GA}_${RUN_TS}"
    local out="${out_dir}/${name}"
    mkdir -p "$out_dir"

    echo -e "${BLUE}────────────────────────────────────────${NC}"
    echo -e "${BLUE}Profiling: ${name}  [bs=${BS} ga=${GA}]${NC}"
    echo -e "${BLUE}────────────────────────────────────────${NC}"

    NVTX_PROFILE=1 nsys profile \
        --trace=cuda,nvtx \
        --output="${out}" \
        --force-overwrite=true \
        --delay=${NSYS_DELAY} \
        --duration=${NSYS_DURATION} \
        python train_milestones.py \
            model.builder.packing=false \
            milestone.tokens_per_milestone=1000000 \
            milestone.num_milestones=1 \
            milestone.do_evaluate=false \
            validation_cfg.return_validation=false \
            training_args.per_device_train_batch_size=${BS} \
            training_args.gradient_accumulation_steps=${GA} \
            training_args.save_strategy=no \
            wandb.enabled=false \
            "$@"

    echo -e "${GREEN}Exporting SQLite: ${out}.sqlite${NC}"
    nsys export \
        --type sqlite \
        --output "${out}.sqlite" \
        --force-overwrite=true \
        "${out}.nsys-rep"

    echo -e "${GREEN}Done: ${name}${NC}"
    echo ""
}

try_profile() {
    local name=$1
    shift
    if profile_config "$name" "$@"; then
        RESULTS+=("${GREEN}✓ ${name}${NC}")
    else
        echo -e "${RED}Config '${name}' failed (OOM or error) — skipping.${NC}"
        RESULTS+=("${RED}✗ ${name} (OOM/failed)${NC}")
    fi
}

# ── Config definitions (mirrors run_optimization_sweep.sh) ───────────────────
run_all_configs() {

    # # # Config 3: bf16 + gradient checkpointing
    # export USE_UNSLOTH=0
    # try_profile "3_bf16_gc" \
    #     model.builder.model_name=${HF_MODEL} \
    #     training_args.bf16=true \
    #     training_args.gradient_checkpointing=true \
    #     training_args.optim=adamw_torch \
    #     training_args.embedding_learning_rate=null \
    #     model.builder.bnb_4bit_compute_dtype=bfloat16 \
    #     model.builder.use_flash_attention=false \
    #     model.builder.use_lora=false \
    #     model.builder.load_in_4bit=false \
    #     model.builder.prepare_kbit_training=false \
    #     model.builder.use_unsloth=false

    # # # Config 4: +FA2
    # export USE_UNSLOTH=0
    # try_profile "4_bf16_gc_fa2" \
    #     model.builder.model_name=${HF_MODEL} \
    #     training_args.bf16=true \
    #     training_args.gradient_checkpointing=true \
    #     training_args.optim=adamw_torch \
    #     training_args.embedding_learning_rate=null \
    #     model.builder.bnb_4bit_compute_dtype=bfloat16 \
    #     model.builder.use_flash_attention=true \
    #     model.builder.use_lora=false \
    #     model.builder.load_in_4bit=false \
    #     model.builder.prepare_kbit_training=false \
    #     model.builder.use_unsloth=false

    # Config 5: +paged optimizer
    # export USE_UNSLOTH=0
    # try_profile "5_bf16_gc_fa2_paged" \
    #     model.builder.model_name=${HF_MODEL} \
    #     training_args.bf16=true \
    #     training_args.gradient_checkpointing=true \
    #     training_args.optim=paged_adamw_8bit \
    #     training_args.embedding_learning_rate=null \
    #     model.builder.bnb_4bit_compute_dtype=bfloat16 \
    #     model.builder.use_flash_attention=true \
    #     model.builder.use_lora=false \
    #     model.builder.load_in_4bit=false \
    #     model.builder.prepare_kbit_training=false \
    #     model.builder.use_unsloth=false

    # Config 6: +LoRA r64
    # export USE_UNSLOTH=0
    # try_profile "6_bf16_gc_fa2_paged_lora64" \
    #     model.builder.model_name=${HF_MODEL} \
    #     training_args.bf16=true \
    #     training_args.gradient_checkpointing=true \
    #     training_args.optim=paged_adamw_8bit \
    #     training_args.embedding_learning_rate=null \
    #     model.builder.bnb_4bit_compute_dtype=bfloat16 \
    #     model.builder.use_flash_attention=true \
    #     model.builder.use_lora=true \
    #     model.lora.r=${LORA_RANK} \
    #     model.lora.lora_alpha=${LORA_ALPHA} \
    #     model.builder.load_in_4bit=false \
    #     model.builder.prepare_kbit_training=false \
    #     model.builder.use_unsloth=false

    # Config 7: +qLoRA (4-bit base)
    # export USE_UNSLOTH=0
    # try_profile "7_bf16_gc_fa2_paged_qlora64" \
    #     model.builder.model_name=${HF_MODEL} \
    #     training_args.bf16=true \
    #     training_args.gradient_checkpointing=true \
    #     training_args.optim=paged_adamw_8bit \
    #     training_args.embedding_learning_rate=null \
    #     model.builder.bnb_4bit_compute_dtype=bfloat16 \
    #     model.builder.use_flash_attention=true \
    #     model.builder.use_lora=true \
    #     model.lora.r=${LORA_RANK} \
    #     model.lora.lora_alpha=${LORA_ALPHA} \
    #     model.builder.load_in_4bit=true \
    #     model.builder.prepare_kbit_training=true \
    #     model.builder.use_unsloth=false

    # Config 8: +Unsloth kernels
    export USE_UNSLOTH=1
    try_profile "8_bf16_gc_fa2_paged_qlora64_unsloth" \
        model.builder.model_name=${UNSLOTH_MODEL} \
        training_args.bf16=true \
        training_args.gradient_checkpointing=true \
        training_args.optim=paged_adamw_8bit \
        training_args.embedding_learning_rate=2e-5 \
        model.builder.bnb_4bit_compute_dtype=bfloat16 \
        model.builder.use_flash_attention=true \
        model.builder.use_lora=true \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        model.builder.load_in_4bit=true \
        model.builder.use_unsloth=true

    # # Config 9: +packing
    # export USE_UNSLOTH=1
    # try_profile "9_full_stack_packing" \
    #     model.builder.model_name=${UNSLOTH_MODEL} \
    #     training_args.bf16=true \
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

# ── Outer loop over BS/GA pairs ───────────────────────────────────────────────
ALL_RESULTS=()

for pair in "${BS_GA_PAIRS[@]}"; do
    BS=$(echo "$pair" | awk '{print $1}')
    GA=$(echo "$pair" | awk '{print $2}')
    RESULTS=()

    echo -e "${YELLOW}════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Batch Config: bs=${BS}  ga=${GA}  (eff_bs=$((BS * GA)))${NC}"
    echo -e "${YELLOW}Output: outputs/nsys_profiles/bs${BS}_ga${GA}/${NC}"
    echo -e "${YELLOW}════════════════════════════════════════${NC}"
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

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo -e "${YELLOW}All profiles complete.${NC}"
echo -e "${YELLOW}════════════════════════════════════════${NC}"
for result in "${ALL_RESULTS[@]}"; do
    echo -e "  ${result}"
done
echo ""
echo -e "${GREEN}SQLite files in: outputs/nsys_profiles/${NC}"
echo -e "${GREEN}Run analysis:${NC}"
echo -e "${GREEN}  python diagnostics_debugging/nsys_kernel_analysis.py${NC}"
