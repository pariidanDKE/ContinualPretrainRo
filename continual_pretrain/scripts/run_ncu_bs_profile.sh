#!/bin/bash
# ncu kernel profiling — per-kernel hardware counter analysis
#
# Complements run_nsys_profile.sh: where nsys gives a system-level timeline,
# ncu goes *inside* a single kernel to measure occupancy, memory bandwidth,
# tensor-core throughput, and roofline bottlenecks.
#
# Current analysis: Config 8 (full Unsloth stack) × {BS=1, BS=32}
#   Config 8 = qLoRA r64 + paged AdamW 8-bit + FA2 + GC + Unsloth fused kernels
#   Goal: quantify how batch size affects GEMM tile efficiency, tensor-core
#         pipe utilisation, and HBM bandwidth per token.
#
# NCU strategy:
#   --launch-skip   skip the first N matched kernel launches (clears model
#                   load + Unsloth CUDA graph compilation, ~200 launches)
#   --launch-count  profile exactly N steady-state launches (10 gives stable
#                   averages without excessive replay overhead)
#   --kernel-name   restrict to BF16 GEMM kernels only; profiling every kernel
#                   would make the run ~100× slower
#   --metrics       explicit counter set (3 replay passes total vs ~10 for
#                   --set full)
#
# Metric groups collected:
#   COMPUTE  — SM throughput, tensor-core HMMA pipe, warp occupancy, scheduler
#   MEMORY   — HBM read/write bytes, L2 hit rate, L1 hit rate, L2 traffic
#   ROOFLINE — HMMA/FMA instruction counts + kernel wall time (arithmetic
#              intensity proxy: HMMA_instrs / dram_bytes)
#
# Output layout (mirrors nsys):
#   outputs/ncu_profiles/bs{BS}_ga{GA}_{timestamp}/
#     {config_name}.ncu-rep   — open with Nsight Compute GUI
#     {config_name}.csv       — machine-readable for analysis scripts
#
# Usage:
#   chmod +x scripts/run_ncu_profile.sh
#   ./scripts/run_ncu_profile.sh
#   python diagnostics_debugging/ncu_kernel_analysis.py
#
# Override kernel / skip / count without editing this file:
#   NCU_KERNEL_REGEX='cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8' \
#   NCU_LAUNCH_SKIP=200 NCU_LAUNCH_COUNT=5 \
#   ./scripts/run_ncu_profile.sh

cd "$(dirname "$0")/.."

# ── Model (Config 8 uses the Unsloth-optimised weights) ───────────────────────
UNSLOTH_MODEL="unsloth/Llama-3.2-1B"
LORA_RANK=64
LORA_ALPHA=64

# ── BS/GA pairs to compare (effective batch size stays constant at 32) ─────────
BS_GA_PAIRS=("1 32" "32 1")
BS_GA_PAIRS=("8 4" "1 32")

# ── Target kernel ──────────────────────────────────────────────────────────────
# CUTLASS kernels are C++ templates: the tile name lives in the template arg,
# not the function name. --kernel-name-base demangled exposes it as a substring.
# The exact tile name is known from the NCU "Available Kernels" list.
# Broad regex: matches all bf16 CUTLASS GEMM tiles regardless of tile size.
# BS=1  → 256x128 tile; BS=8 → 128x128 / 128x256 / 256x64 tiles (CUTLASS
# picks the tile heuristically based on problem shape, so a fixed tile name
# only works for one batch size).
KERNEL_REGEX="${NCU_KERNEL_REGEX:-cutlass_80_tensorop_bf16_s16816gemm}"

#cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>

# Start with LAUNCH_SKIP=0 to confirm the kernel regex matches at all.
# Once you see captures, bump to ~100-200 to skip warmup/compilation.
LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-10}"
LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-10}"

# ── Metrics ────────────────────────────────────────────────────────────────────
# COMPUTE GROUP
METRICS_COMPUTE=(
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"                        # overall SM busy %
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"  # tensor-core HMMA pipe
    "sm__warps_active.avg.pct_of_peak_sustained_active"                       # warp occupancy
    "smsp__warps_eligible.avg.pct_of_peak_sustained_active"                   # scheduler pressure
)

# MEMORY GROUP
METRICS_MEMORY=(
    "dram__bytes_read.sum"          # HBM bytes read per kernel invocation
    "dram__bytes_write.sum"         # HBM bytes written per kernel invocation
    "lts__t_hit_rate.pct"           # L2 hit rate
    "l1tex__t_sector_hit_rate.pct"  # L1 hit rate
    "lts__t_bytes.sum"              # total L2 traffic (reads + writes)
)

# ROOFLINE GROUP (arithmetic intensity proxy)
METRICS_ROOFLINE=(
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"  # BF16/FP16 HMMA instructions
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"  # FP32 FMA instructions
    "gpu__time_duration.sum"                               # kernel wall time (ns)
)

# SM DUTY CYCLE — active/elapsed ratio shows how long the SM was doing useful
# work vs idle between waves. Key for BS comparison: BS=32 keeps the SM busy
# across ~98 waves; BS=1 finishes 4 waves then sits idle until next kernel launch.
METRICS_DUTY_CYCLE=(
    "sm__active_cycles.sum"   # cycles with at least one warp running on the SM
    "sm__elapsed_cycles.sum"  # total wall-clock cycles (including inter-wave idle)
)

# Flatten to comma-separated string for --metrics
ALL_METRICS=$(IFS=,; echo "${METRICS_COMPUTE[*]},${METRICS_MEMORY[*]},${METRICS_ROOFLINE[*]},${METRICS_DUTY_CYCLE[*]}")

RUN_TS=$(date +%Y%m%d_%H%M)

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ── Per-run profiler ───────────────────────────────────────────────────────────
profile_config() {
    local name=$1
    shift
    local out_dir="outputs/ncu_profiles/bs${BS}_ga${GA}_${RUN_TS}"
    local out_base="${out_dir}/${name}"
    local out_rep="${out_dir}/${name}.ncu-rep"
    local out_csv="${out_dir}/${name}.csv"
    mkdir -p "$out_dir"

    echo -e "${BLUE}────────────────────────────────────────${NC}"
    echo -e "${BLUE}NCU profiling: ${name}${NC}"
    echo -e "${BLUE}  kernel     : ${KERNEL_REGEX}${NC}"
    echo -e "${BLUE}  skip/count : ${LAUNCH_SKIP} / ${LAUNCH_COUNT}${NC}"
    echo -e "${BLUE}  bs=${BS}  ga=${GA}  (eff_bs=$((BS * GA)))${NC}"
    echo -e "${BLUE}────────────────────────────────────────${NC}"

    # ── .ncu-rep (Nsight Compute GUI report) ──────────────────────────────────
    # Disable Unsloth CUDA graphs: graph replay hides individual kernel launches
    # from NCU. Eager dispatch is required for per-kernel counter collection.
    export USE_UNSLOTH=1
    export UNSLOTH_DISABLE_CUDA_GRAPH=1
    ncu \
        --target-processes all \
        --kernel-name "regex:${KERNEL_REGEX}" \
        --kernel-name-base demangled \
        --nvtx \
        --nvtx-include "mlp_layer5/" \
        --launch-skip "${LAUNCH_SKIP}" \
        --launch-count "${LAUNCH_COUNT}" \
        --metrics "${ALL_METRICS}" \
        --force-overwrite \
        -o "${out_base}" \
        python train_milestones.py \
            milestone.tokens_per_milestone=500000 \
            milestone.num_milestones=1 \
            milestone.do_evaluate=false \
            validation_cfg.return_validation=false \
            training_args.per_device_train_batch_size=${BS} \
            training_args.gradient_accumulation_steps=${GA} \
            training_args.save_strategy=no \
            wandb.enabled=false \
            model.builder.model_name=${UNSLOTH_MODEL} \
            training_args.bf16=true \
            training_args.fp16=false \
            training_args.gradient_checkpointing=false \
            training_args.optim=paged_adamw_8bit \
            training_args.embedding_learning_rate=2e-5 \
            model.builder.bnb_4bit_compute_dtype=bfloat16 \
            model.builder.use_flash_attention=true \
            model.builder.use_lora=true \
            model.lora.r=${LORA_RANK} \
            model.lora.lora_alpha=${LORA_ALPHA} \
            model.builder.load_in_4bit=true \
            model.builder.use_unsloth=true \
            model.builder.packing=true \
            training_args.packing=true
            "$@"

    # ── CSV export (for analysis scripts) ────────────────────────────────────
    if [ ! -f "${out_rep}" ]; then
        echo -e "${RED}No .ncu-rep produced — kernel regex '${KERNEL_REGEX}' matched nothing.${NC}"
        echo -e "${RED}Try: NCU_KERNEL_REGEX='' to capture all kernels and discover names.${NC}"
        return 1
    fi

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

try_profile() {
    local name=$1
    shift
    if profile_config "$name" "$@"; then
        RESULTS+=("${GREEN}✓ ${name}${NC}")
    else
        echo -e "${RED}Config '${name}' failed — skipping.${NC}"
        RESULTS+=("${RED}✗ ${name} (failed)${NC}")
    fi
}

# ── Config definitions ─────────────────────────────────────────────────────────
# Config 8: full Unsloth stack (qLoRA r64 + paged AdamW + FA2 + GC + Unsloth)
# Mirrors run_optimization_sweep.sh Config 8 exactly.
run_all_configs() {
    local prefix="bs${BS}_ga${GA}"

    try_profile "${prefix}_8_bf16_gc_fa2_paged_qlora64_unsloth"
    # Uncomment to add comparisons within the same BS/GA pair, e.g. without Unsloth:
    # try_profile "${prefix}_7_bf16_gc_fa2_paged_qlora64_no_unsloth" \
    #     model.builder.use_unsloth=false \
    #     model.builder.model_name=meta-llama/Llama-3.2-1B
}

# ── Outer loop over BS/GA pairs ────────────────────────────────────────────────
ALL_RESULTS=()
export NVTX_PROFILE=1

echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo -e "${YELLOW}NCU Kernel Profiler — Config 8 × {BS=1, BS=32}${NC}"
echo -e "${YELLOW}  model   : ${UNSLOTH_MODEL}${NC}"
echo -e "${YELLOW}  kernel  : ${KERNEL_REGEX}${NC}"
echo -e "${YELLOW}  skip    : ${LAUNCH_SKIP}   count: ${LAUNCH_COUNT}${NC}"
echo -e "${YELLOW}════════════════════════════════════════${NC}"

for pair in "${BS_GA_PAIRS[@]}"; do
    BS=$(echo "$pair" | awk '{print $1}')
    GA=$(echo "$pair" | awk '{print $2}')
    RESULTS=()

    echo -e "${YELLOW}════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Batch Config: bs=${BS}  ga=${GA}  (eff_bs=$((BS * GA)))${NC}"
    echo -e "${YELLOW}Output: outputs/ncu_profiles/no_packing/bs${BS}_ga${GA}_${RUN_TS}/${NC}"
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

# ── Summary ────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}════════════════════════════════════════${NC}"
echo -e "${YELLOW}All NCU profiles complete.${NC}"
echo -e "${YELLOW}════════════════════════════════════════${NC}"
for result in "${ALL_RESULTS[@]}"; do
    echo -e "  ${result}"
done
echo ""
echo -e "${GREEN}.ncu-rep files: open with Nsight Compute GUI${NC}"
echo -e "${GREEN}CSV files in : outputs/ncu_profiles/${NC}"
echo -e "${GREEN}Run analysis :${NC}"
echo -e "${GREEN}  python diagnostics_debugging/ncu_kernel_analysis.py${NC}"
echo ""
echo -e "${GREEN}To use an exact kernel name (from your nsys timeline):${NC}"
echo -e "${GREEN}  NCU_KERNEL_REGEX='cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8' \\${NC}"
echo -e "${GREEN}  NCU_LAUNCH_SKIP=200 NCU_LAUNCH_COUNT=5 ./scripts/run_ncu_profile.sh${NC}"
