#!/usr/bin/env bash
# Orchestrate milestone training: train → merge → eval → resume

set -euo pipefail

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

MODEL_NAME="meta-llama/Llama-3.2-1B"
LORA_RANK=16
LORA_ALPHA=32
BATCH_SIZE=32
GRAD_ACCUM=2
SAMPLE_SIZE=10000
MAX_LENGTH=2048

MILESTONE_TOKENS=1000000  # 1M tokens per milestone
NUM_MILESTONES=3

OUTPUT_DIR="outputs/milestone_$(echo ${MODEL_NAME} | tr '/:' '__')_r${LORA_RANK}"
MERGED_DIR="${OUTPUT_DIR}/merged"

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
# MAIN LOOP
# =============================================================================

echo "Milestone Training: ${NUM_MILESTONES} x ${MILESTONE_TOKENS} tokens"
echo "Model: ${MODEL_NAME}, LoRA r=${LORA_RANK}, BS=${BATCH_SIZE}"
echo ""

mkdir -p "${MERGED_DIR}"

RESUME_CHECKPOINT=""

for i in $(seq 1 ${NUM_MILESTONES}); do
    TARGET_TOKENS=$((MILESTONE_TOKENS * i))
    STEPS=$(calc_steps ${TARGET_TOKENS})
    CHECKPOINT="${OUTPUT_DIR}/checkpoint-${STEPS}"
    MERGED="${MERGED_DIR}/checkpoint-${STEPS}"

    echo "=========================================="
    echo "MILESTONE ${i}/${NUM_MILESTONES} - ${TARGET_TOKENS} tokens"
    echo "=========================================="

    # Train
    log "Training to ${TARGET_TOKENS} tokens..."
    python train_milestone_segment.py \
        model.builder.model_name="${MODEL_NAME}" \
        model.lora.r=${LORA_RANK} \
        model.lora.lora_alpha=${LORA_ALPHA} \
        dataset.sample_size=${SAMPLE_SIZE} \
        training_args.per_device_train_batch_size=${BATCH_SIZE} \
        training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
        training_args.output_dir="${OUTPUT_DIR}" \
        training_args.max_length=${MAX_LENGTH} \
        milestone.target_tokens=${TARGET_TOKENS} \
        ${RESUME_CHECKPOINT:+milestone.resume_from_checkpoint=${RESUME_CHECKPOINT}}

    # Merge LoRA
    log "Merging LoRA weights..."
    python merge_lora_checkpoint.py "${CHECKPOINT}" "${MERGED}"

    # Evaluate
    log "Evaluating..."
    python evaluate.py \
        --config-path=configs \
        --config-name=evaluate_ro \
        model_path="${MERGED}"

    RESUME_CHECKPOINT="${CHECKPOINT}"
    log "✅ Milestone ${i} complete"
    echo ""
done

echo "🎉 All milestones completed!"
echo "Final checkpoint: ${RESUME_CHECKPOINT}"