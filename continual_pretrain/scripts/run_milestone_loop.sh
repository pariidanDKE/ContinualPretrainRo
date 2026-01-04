#!/usr/bin/env bash
# Orchestrate milestone training: train → merge → eval → resume

set -euo pipefail

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
LORA_RANK=16
LORA_ALPHA=32
BATCH_SIZE=32
GRAD_ACCUM=2
SAMPLE_SIZE=null
MAX_LENGTH=2048
USE_UNSLOTH=true

MILESTONE_TOKENS=10000000  # 10M tokens per milestone
NUM_MILESTONES=10
DO_EVALUATE=true

OUTPUT_DIR="outputs/milestone_$(echo ${MODEL_NAME} | tr '/:' '__')_r${LORA_RANK}"
MERGED_DIR="${OUTPUT_DIR}/merged"
PREBUILT_DATASET_PATH="data/mixed_milestone_dataset"

# =============================================================================
# WANDB CONFIGURATION
# =============================================================================

# Generate unique run ID for entire milestone session (all milestones log to same run)
WANDB_RUN_ID="${WANDB_RUN_ID:-$(python3 -c 'import wandb; print(wandb.util.generate_id())')}"
export WANDB_RUN_ID
export WANDB_RESUME="allow"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

calc_steps() {
    local tokens=$1
    echo $((tokens / (BATCH_SIZE * GRAD_ACCUM * MAX_LENGTH )))
}

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# =============================================================================
# BUILD PREBUILT DATASET
# =============================================================================

echo "=========================================="
echo "STEP 0: Building Mixed Milestone Dataset"
echo "=========================================="
echo "This will create a mixed dataset and tag each example with its milestone number."
echo "Dataset will be saved to: ${PREBUILT_DATASET_PATH}"
echo ""

log "Building prebuilt dataset..."
python build_mixed_dataset.py \
    model.builder.model_name="${MODEL_NAME}" \
    output_dir="${PREBUILT_DATASET_PATH}" \
    milestone.num_milestones=${NUM_MILESTONES} \
    milestone.tokens_per_milestone=${MILESTONE_TOKENS} \
    ${SAMPLE_SIZE:+dataset.sample_size=${SAMPLE_SIZE}}

log "✅ Prebuilt dataset created"
echo ""

# =============================================================================
# MAIN LOOP
# =============================================================================

echo "Milestone Training: ${NUM_MILESTONES} x ${MILESTONE_TOKENS} tokens"
echo "Model: ${MODEL_NAME}, LoRA r=${LORA_RANK}, BS=${BATCH_SIZE}"
echo "WandB Run ID: ${WANDB_RUN_ID}"
echo ""

mkdir -p "${MERGED_DIR}"

RESUME_CHECKPOINT=""
TOTAL_CALCULATED_TOKENS=$(( MILESTONE_TOKENS * NUM_MILESTONES ))

for i in $(seq 1 ${NUM_MILESTONES}); do
    CURRENT_MILESTONE=$((i - 1))  # Convert to 0-indexed
    TARGET_TOKENS=$((MILESTONE_TOKENS * i))
    STEPS=$(calc_steps ${TARGET_TOKENS})
    CHECKPOINT="${OUTPUT_DIR}/checkpoint-${STEPS}"
    MERGED="${MERGED_DIR}/checkpoint-${STEPS}"

    echo "=========================================="
    echo "MILESTONE ${i}/${NUM_MILESTONES} (index ${CURRENT_MILESTONE}) - ${TARGET_TOKENS} tokens"
    echo "=========================================="

    # Train (merges LoRA automatically at the end)
    log "Training milestone ${CURRENT_MILESTONE} to ${TARGET_TOKENS} tokens..."
    python train_milestone_segment.py \
        model.builder.model_name="${MODEL_NAME}" \
        model.builder.use_unsloth="${USE_UNSLOTH}" \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        training_args.per_device_train_batch_size=${BATCH_SIZE} \
        training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
        training_args.output_dir="${OUTPUT_DIR}" \
        training_args.max_length=${MAX_LENGTH} \
        milestone.target_tokens=${TARGET_TOKENS} \
        milestone.total_training_tokens=${TOTAL_CALCULATED_TOKENS} \
        milestone.merged_output_dir="${MERGED_DIR}" \
        milestone.do_evaluate=${DO_EVALUATE} \
        milestone.prebuilt_dataset_path="${PREBUILT_DATASET_PATH}" \
        milestone.current_milestone=${CURRENT_MILESTONE} \
        wandb.enabled=true \
        ${RESUME_CHECKPOINT:+milestone.resume_from_checkpoint=${RESUME_CHECKPOINT}}

    # Evaluate (if enabled)
    if [ "${DO_EVALUATE}" = "true" ]; then
        log "Evaluating..."
        python evaluate.py \
            --config-path=configs \
            --config-name=evaluate_ro \
            model_path="${MERGED}"
    else
        log "Skipping evaluation (DO_EVALUATE=false)"
    fi

    RESUME_CHECKPOINT="${CHECKPOINT}"
    log "✅ Milestone ${i} complete"
    echo ""
done

echo ""
echo "=========================================="
echo "All milestones completed!"
echo "=========================================="
echo "Final checkpoint: ${RESUME_CHECKPOINT}"
echo ""

# =============================================================================
# CLEANUP
# =============================================================================

log "Cleaning up prebuilt dataset..."
if [ -d "${PREBUILT_DATASET_PATH}" ]; then
    rm -rf "${PREBUILT_DATASET_PATH}"
    log "✅ Removed ${PREBUILT_DATASET_PATH}"
else
    log "⚠️  Prebuilt dataset not found (already deleted?)"
fi

echo ""
echo "🎉 Training complete!"