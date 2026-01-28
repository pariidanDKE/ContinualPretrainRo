#!/usr/bin/env bash
# Orchestrate milestone training: train → merge → eval → resume

set -euo pipefail

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Model and LoRA configuration (can be overridden via environment variables)
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-1B-Instruct"}
LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-256}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-1}
SAMPLE_SIZE=${SAMPLE_SIZE:-null}
MAX_LENGTH=${MAX_LENGTH:-2048}
USE_UNSLOTH=${USE_UNSLOTH:-true}
USE_PACKING=${USE_PACKING:-true}

# Learning rate configuration
LEARNING_RATE=${LEARNING_RATE:-6e-5}
EMBEDDING_LEARNING_RATE=${EMBEDDING_LEARNING_RATE:-2e-5}

# Data builder configuration (optional override for dataset mixing experiments)
DATASET_CONFIG_NAME=${DATASET_CONFIG_NAME:-"build_dataset"}


# Milestone configuration
MILESTONE_TOKENS=${MILESTONE_TOKENS:-25000000}
NUM_MILESTONES=${NUM_MILESTONES:-4}
DO_EVALUATE=${DO_EVALUATE:-true}

# Output directories
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/test_runs/"}
MERGED_DIR="${OUTPUT_DIR}/merged"
PREBUILT_DATASET_PATH="data/mixed_milestone_dataset"

# =============================================================================
# WANDB CONFIGURATION
# =============================================================================

# Generate unique run ID for entire milestone session (all milestones log to same run)
WANDB_RUN_ID="${WANDB_RUN_ID:-$(python3 -c 'import wandb; print(wandb.util.generate_id())')}"
export WANDB_RUN_ID
export WANDB_RESUME="allow"

WANDB_GROUP=${WANDB_GROUP:-"fix_loss"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"test_run"}

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
    --config-name="${DATASET_CONFIG_NAME}" \
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
TOTAL_CALCULATED_TOKENS=$(( MILESTONE_TOKENS * NUM_MILESTONES)) 


# Initial baseline eval
if [ "${DO_EVALUATE}" = "true" ]; then
    log "Evaluating..."
    python evaluate.py \
        --config-path=configs \
        --config-name=evaluate_ro \
        model_path="${MODEL_NAME}"
else
    log "Skipping evaluation (DO_EVALUATE=false)"
fi


for i in $(seq 1 ${NUM_MILESTONES}); do
    CURRENT_MILESTONE=$((i - 1))  # Convert to 0-indexed
    TARGET_TOKENS=$((MILESTONE_TOKENS * i))
    STEPS=$(calc_steps ${TARGET_TOKENS})
    CHECKPOINT="${OUTPUT_DIR}/checkpoint-${STEPS}"
    MERGED="${MERGED_DIR}/checkpoint-${STEPS}"

    echo "=========================================="
    echo "MILESTONE ${i}/${NUM_MILESTONES} (index ${CURRENT_MILESTONE}) - ${TARGET_TOKENS} tokens"
    echo "=========================================="

    #Train (merges LoRA automatically at the end)
    log "Training milestone ${CURRENT_MILESTONE} to ${TARGET_TOKENS} tokens..."
    python train_milestone_segment.py \
        model.builder.model_name="${MODEL_NAME}" \
        model.builder.use_unsloth="${USE_UNSLOTH}" \
        data_collator.packing="${USE_PACKING}" \
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
        wandb.group="${WANDB_GROUP}" \
        ${LEARNING_RATE:+training_args.learning_rate=${LEARNING_RATE}} \
        ${EMBEDDING_LEARNING_RATE:+training_args.embedding_learning_rate=${EMBEDDING_LEARNING_RATE}} \
        ${WANDB_RUN_NAME:+wandb.run_name=${WANDB_RUN_NAME}} \
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