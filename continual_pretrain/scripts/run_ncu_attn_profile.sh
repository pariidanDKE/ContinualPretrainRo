#!/bin/bash
# ncu attention kernel profiling — FA2 vs eager attention comparison
#
# Complements run_ncu_profile.sh (GEMM sweep) and run_nsys_profile.sh.
# This script goes *inside* the attention kernels specifically, using NVTX
# range filtering to isolate the attention forward pass without relying on
# fragile --launch-skip heuristics.
#
# Two paths profiled:
#
#   FA2  — flash_fwd_kernel (HeadDim=128, BF16, causal)
#          NVTX range : "attn_scores"
#          Bottleneck : SRAM tiling throughput, register pressure, occupancy
#          Key metrics: DRAM bytes (expected ~small), SMEM transactions,
#                       L2 hit rate, warp occupancy
#
#   Eager — cutlass.*gemm.*bf16 (QK^T and score×V) + softmax_warp_forward
#           NVTX range : "attn_scores/eager"
#           Bottleneck : HBM bandwidth (full NxN attention matrix materialised)
#           Key metrics: DRAM bytes (expected large), L2 hit rate, GEMM tile
#                        efficiency, softmax warp utilisation
#
# Kernel names are extracted from:
#   diagnostics_debugging/profiling_querries/fa2_kernels.json
#   diagnostics_debugging/profiling_querries/eager_kernels.json
#
# NCU strategy:
#   --nvtx-include  restrict profiling to the labelled NVTX attention region
#                   (much more reliable than --launch-skip for scoped regions)
#   --launch-count  profile N kernel launches per matched range (default 5)
#                   — enough for stable averages; FA2 hits this quickly since
#                     it is a single fused kernel per attention call
#   --kernel-name   further narrow to the specific attention kernels so that
#                   unrelated elementwise bookkeeping is excluded from the
#                   ncu-rep and does not inflate replay time
#   --set full      collect the complete hardware counter set;  FA2 and
#                   softmax are short-running, so the replay overhead is
#                   acceptable here
#
# Metric groups collected (beyond --set full):
#   ATTENTION_MEM   — SRAM load/store wavefronts (FA2 tiling cost vs eager)
#   DRAM_DETAIL     — per-read/write HBM bytes (materialised NxN cost in eager)
#   OCCUPANCY       — theoretical vs achieved occupancy (register pressure in FA2)
#
# Output layout:
#   outputs/ncu_attn_profiles/{mode}_{timestamp}/
#     {name}.ncu-rep   — Nsight Compute GUI report
#     {name}.csv       — machine-readable for analysis notebooks
#
# Usage:
#   chmod +x scripts/run_ncu_attn_profile.sh
#   ./scripts/run_ncu_attn_profile.sh                 # profiles both FA2 and eager
#   MODE=fa2   ./scripts/run_ncu_attn_profile.sh      # FA2 only
#   MODE=eager ./scripts/run_ncu_attn_profile.sh      # eager only
#
# Overrides (env vars):
#   NCU_LAUNCH_COUNT      kernels to capture per range (default 5)
#   NCU_BS / NCU_GA       batch size / gradient accumulation (default 1 / 32)
#   NCU_SEQ_LEN           sequence length to pass via pack_length (default 2048)
#   NCU_MODEL             model name (default unsloth/Llama-3.2-1B)

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Configurable defaults ──────────────────────────────────────────────────────
MODE="${MODE:-both}"                             # fa2 | eager | both
MODEL="meta-llama/Llama-3.2-1B"

BS="${NCU_BS:-1}"
GA="${NCU_GA:-32}"
SEQ_LEN="${NCU_SEQ_LEN:-2048}"
LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-10}"

# ── Kernel regexes (demangled substrings from the profiling JSON files) ────────
# FA2: single fused forward kernel; HeadDim and layout encoded in template args
FA2_KERNEL_REGEX="flash_fwd_kernel"

# Eager: two CUTLASS GEMMs (QK^T is tn layout, score×V is nn layout),
#        the fused softmax warp kernel, and elementwise kernels observed
#        inside the attn_scores/eager NVTX region with significant duration
#        (100-141 µs each, comparable to the GEMMs — see eager_kernels.json).
#        Demangled names suggest scaling (MulFunctor), bias/mask (CUDAFunctor_add),
#        and BF16 cast (direct_copy_kernel_cuda), but ncu dram__bytes metrics
#        are needed to confirm which tensors they operate on.
EAGER_KERNEL_REGEX="Kernel2|softmax_warp_forward"



# ── NVTX range names (must match the NVTX annotations in the training code) ───
FA2_NVTX_RANGE="attn_scores/"
EAGER_NVTX_RANGE="attn_scores_eager/"

# ── Metrics ────────────────────────────────────────────────────────────────────
# COMPUTE — SM utilisation and tensor-core pipe
METRICS_COMPUTE=(
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
    "sm__warps_active.avg.pct_of_peak_sustained_active"
    "smsp__warps_eligible.avg.pct_of_peak_sustained_active"
)

# MEMORY — HBM bandwidth is the main differentiator between FA2 and eager
# FA2 avoids writing the full NxN attention matrix to DRAM; eager writes it.
METRICS_DRAM=(
    "dram__bytes_read.sum"
    "dram__bytes_write.sum"
    "lts__t_hit_rate.pct"
    "l1tex__t_sector_hit_rate.pct"
    "lts__t_bytes.sum"
)

# SHARED MEMORY — FA2 tiles Q/K/V blocks into SRAM; this quantifies that cost
METRICS_SMEM=(
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"
    "l1tex__average_t_sectors_per_request_mem_shared_op_ld.ratio"
)

# OCCUPANCY — FA2 is register-heavy; high register usage limits warps per SM
METRICS_OCCUPANCY=(
    "sm__maximum_warps_per_active_cycle_pct"
    "sm__warps_launched.sum"
    "launch__registers_per_thread"
    "launch__shared_mem_per_block_static"
)

# ROOFLINE — arithmetic intensity: HMMA instructions / DRAM bytes
METRICS_ROOFLINE=(
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
    "gpu__time_duration.sum"
)

ALL_METRICS=$(IFS=,; echo "${METRICS_COMPUTE[*]},${METRICS_DRAM[*]},${METRICS_SMEM[*]},${METRICS_OCCUPANCY[*]},${METRICS_ROOFLINE[*]}")

RUN_TS=$(date +%Y%m%d_%H%M)

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ── Shared training launch args ────────────────────────────────────────────────
# Minimal run: 1 milestone, no eval, no checkpointing, WandB off.
# Gradient checkpointing disabled to keep the attention NVTX region clean
# (GC replays the forward pass, doubling attention kernel launches).
common_train_args=(
    milestone.tokens_per_milestone=500000
    milestone.num_milestones=1
    milestone.do_evaluate=false
    validation_cfg.return_validation=false
    training_args.per_device_train_batch_size="${BS}"
    training_args.gradient_accumulation_steps="${GA}"
    training_args.gradient_checkpointing=false
    training_args.save_strategy=no
    training_args.bf16=true
    training_args.fp16=false
    training_args.optim=adamw_torch
    training_args.embedding_learning_rate=null
    wandb.enabled=false
    model.builder.model_name="${MODEL}"
    model.builder.packing=false
    model.builder.use_lora=true
    model.builder.load_in_4bit=true
    model.builder.prepare_kbit_training=false
    model.builder.bnb_4bit_compute_dtype=bfloat16
    model.builder.use_unsloth=false
)

# ── Core profiler function ─────────────────────────────────────────────────────
# Args: <profile_name> <nvtx_range> <kernel_regex> [extra hydra overrides...]
profile_attn() {
    local name=$1
    local nvtx_range=$2
    local kernel_regex=$3
    shift 3

    local out_dir="outputs/ncu_attn_profiles/${name}_${RUN_TS}"
    local out_base="${out_dir}/${name}"
    local out_rep="${out_base}.ncu-rep"
    local out_csv="${out_base}.csv"
    mkdir -p "$out_dir"

    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}NCU Attention Profile: ${name}${NC}"
    echo -e "${BLUE}  nvtx range  : ${nvtx_range}${NC}"
    echo -e "${BLUE}  kernel regex: ${kernel_regex}${NC}"
    echo -e "${BLUE}  launches    : ${LAUNCH_COUNT}${NC}"
    echo -e "${BLUE}  bs=${BS}  ga=${GA}  seq=${SEQ_LEN}${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"

    # Disable CUDA graphs: graph replay hides individual kernel launches
    # from NCU. Eager dispatch required for per-kernel counter collection.
    export UNSLOTH_DISABLE_CUDA_GRAPH=1

    ncu \
        --target-processes all \
        --kernel-name "regex:${kernel_regex}" \
        --kernel-name-base demangled \
        --nvtx \
        --nvtx-include "${nvtx_range}" \
        --launch-count "${LAUNCH_COUNT}" \
        --metrics "${ALL_METRICS}" \
        --force-overwrite \
        -o "${out_base}" \
        python train_milestones.py \
            "${common_train_args[@]}" \
            "$@"


    echo -e "${GREEN}Exporting CSV: ${out_csv}${NC}"
    ncu \
        -i "${out_rep}" \
        --csv \
        --metrics "${ALL_METRICS}" \
        > "${out_csv}"

    echo -e "${GREEN}Done: ${name}${NC}"
    echo -e "${GREEN}  Report : ${out_rep}${NC}"
    echo -e "${GREEN}  CSV    : ${out_csv}${NC}"
    echo ""
}

try_profile_attn() {
    local name=$1
    shift
    if profile_attn "$name" "$@"; then
        RESULTS+=("${GREEN}✓ ${name}${NC}")
    else
        echo -e "${RED}Profile '${name}' failed — skipping.${NC}"
        RESULTS+=("${RED}✗ ${name} (failed)${NC}")
    fi
}

# ── Profile runners ────────────────────────────────────────────────────────────
run_fa2() {
    try_profile_attn \
        "fa2_bs${BS}_seq${SEQ_LEN}" \
        "${FA2_NVTX_RANGE}" \
        "${FA2_KERNEL_REGEX}" \
        model.builder.use_flash_attention=true
}

run_eager() {
    try_profile_attn \
        "eager_bs${BS}_seq${SEQ_LEN}" \
        "${EAGER_NVTX_RANGE}" \
        "${EAGER_KERNEL_REGEX}" \
        model.builder.use_flash_attention=false \
        model.builder.use_unsloth=false
}

# ── Main ───────────────────────────────────────────────────────────────────────
RESULTS=()
export USE_UNSLOTH=0
export NVTX_PROFILE=1 

echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo -e "${YELLOW}NCU Attention Profiler${NC}"
echo -e "${YELLOW}  model   : ${MODEL}${NC}"
echo -e "${YELLOW}  mode    : ${MODE}${NC}"
echo -e "${YELLOW}  bs=${BS}  ga=${GA}  seq=${SEQ_LEN}${NC}"
echo -e "${YELLOW}  output  : outputs/ncu_attn_profiles/${NC}"
echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo ""

case "$MODE" in
    fa2)    run_fa2 ;;
    eager)  run_eager ;;
    both)   run_fa2; run_eager ;;
    *)
        echo -e "${RED}Unknown MODE='${MODE}'. Use: fa2 | eager | both${NC}"
        exit 1
        ;;
esac

echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo -e "${YELLOW}All attention profiles complete.${NC}"
echo -e "${YELLOW}════════════════════════════════════════${NC}"
for result in "${RESULTS[@]}"; do
    echo -e "  ${result}"
done
echo ""
echo -e "${GREEN}.ncu-rep : open with Nsight Compute GUI${NC}"
echo -e "${GREEN}CSVs in  : outputs/ncu_attn_profiles/${NC}"
echo ""
echo -e "${GREEN}Key comparison queries:${NC}"
echo -e "${GREEN}  DRAM bytes FA2 vs eager  — dram__bytes_read.sum (FA2 should be ~seq_len×d_head×N vs N² for eager)${NC}"
echo -e "${GREEN}  SMEM cost  FA2 only      — l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum${NC}"
echo -e "${GREEN}  Occupancy limit          — launch__registers_per_thread (FA2 often 255, limits warps/SM)${NC}"
echo ""
echo -e "${GREEN}Override examples:${NC}"
echo -e "${GREEN}  MODE=fa2 NCU_BS=1 NCU_SEQ_LEN=4096 ./scripts/run_ncu_attn_profile.sh${NC}"
echo -e "${GREEN}  MODE=eager NCU_LAUNCH_COUNT=10 ./scripts/run_ncu_attn_profile.sh${NC}"
