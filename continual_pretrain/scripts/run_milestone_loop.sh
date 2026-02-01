#!/usr/bin/env bash
# Orchestrate milestone training: train → merge → eval → resume

set -euo pipefail

# Source helper functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/milestone_helpers.sh"

# =============================================================================
# WANDB CONFIGURATION
# =============================================================================

# Generate unique run ID for entire milestone session (all milestones log to same run)
WANDB_RUN_ID="${WANDB_RUN_ID:-$(python3 -c 'import wandb; print(wandb.util.generate_id())')}"
export WANDB_RUN_ID
export WANDB_RESUME="allow"

# WandB group name (examples: TestCPT, TestSFT, ActualCPT, ActualSFT)
WANDB_GROUP=${WANDB_GROUP:-"TestCPT"}

# Optional custom experiment name (will be included in run name if set)
CUSTOM_NAME=${CUSTOM_NAME:-"CPTSignalEstablishment_maxgrad1_lr1e4"}
export CUSTOM_NAME

# Run name will be auto-generated in Python with format:
# {run_type}-{model}-{custom_name}-ms{num_milestones}-{tokens}M
# Example: cpt-llama-myexp-ms10-900M or sft-qwen-ms5-100M
# Note: WANDB_RUN_NAME can still override this if explicitly set
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}
export WANDB_RUN_NAME


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Model and LoRA configuration (can be overridden via environment variables)
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-1B"}
LORA_RANK=${LORA_RAAK:-128}
LORA_ALPHA=${LORA_LPHA:-256}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-4}
SAMPLE_SIZE=${SAMPLE_SIZE:-null}
MAX_LENGTH=${MAX_LENGTH:-4096}
USE_UNSLOTH=${USE_UNSLOTH:-true}
USE_PACKING=${USE_PACKING:-true}
USE_SFT=${USE_SFT:-false}  # false = CPT mode (raw text), true = SFT mode (Romanian tags → chat tokens)

TEXT_COLUMN=${TEXT_COLUMN:-"formatted_text"}

# Learning rate configuration
LEARNING_RATE=${LEARNING_RATE:-1e-4}
EMBEDDING_LEARNING_RATE=${EMBEDDING_LEARNING_RATE:-2e-2}

# Data builder configuration (optional override for dataset mixing experiments)
DATASET_CONFIG_NAME=${DATASET_CONFIG_NAME:-"build_dataset"}

# =============================================================================
# MILESTONE CONFIGURATION
# =============================================================================

# Milestone configuration
MILESTONE_TOKENS=${MILESTONE_TOKENS:-80000000}
NUM_MILESTONES=${NUM_MILESTONES:-10}
DO_EVALUATE=${DO_EVALUATE:-false}


# Export for Python access (used in run name generation)
export MILESTONE_TOKENS
export NUM_MILESTONES

# Generate run name if not already set
if [ -z "$WANDB_RUN_NAME" ]; then
    WANDB_RUN_NAME=$(generate_run_name "$MODEL_NAME" "$USE_SFT")
    export WANDB_RUN_NAME
    echo "Generated run name: ${WANDB_RUN_NAME}"
fi

# Output directories (include run_name for organization)
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"outputs/test_runs"}
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${WANDB_RUN_NAME}"
MERGED_DIR="${OUTPUT_DIR}/merged"
PREBUILT_DATASET_PATH="data/mixed_milestone_dataset"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${MERGED_DIR}"

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
# python build_mixed_dataset.py \
#     --config-name="${DATASET_CONFIG_NAME}" \
#     model.builder.model_name="${MODEL_NAME}" \
#     output_dir="${PREBUILT_DATASET_PATH}" \
#     milestone.num_milestones=${NUM_MILESTONES} \
#     milestone.tokens_per_milestone=${MILESTONE_TOKENS} \
#     data_builder.max_length=${MAX_LENGTH} \
#     dataset.text_field="text" \
#     ${SAMPLE_SIZE:+dataset.sample_size=${SAMPLE_SIZE}}

log "✅ Prebuilt dataset created"
echo ""

# =============================================================================
# MAIN LOOP
# =============================================================================

echo "Milestone Training: ${NUM_MILESTONES} x ${MILESTONE_TOKENS} tokens"
echo "Model: ${MODEL_NAME}, LoRA r=${LORA_RANK}, BS=${BATCH_SIZE}"
echo "WandB Run ID: ${WANDB_RUN_ID}"
echo "Run Name: ${WANDB_RUN_NAME}"
echo ""

RESUME_CHECKPOINT=""
TOTAL_CALCULATED_TOKENS=$(( MILESTONE_TOKENS * NUM_MILESTONES)) 


# Initial baseline eval (at step 0)
if [ "${DO_EVALUATE}" = "true" ]; then
    log "Evaluating baseline at step 0..."
    python evaluate.py \
        --config-path=configs \
        --config-name=evaluate_ro \
        model_path="${MODEL_NAME}" \
        use_sft="${USE_SFT}" \
        global_step=0
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
        dataset.use_sft=${USE_SFT} \
        dataset.text_field="text" \
        wandb.group="${WANDB_GROUP}" \
        ${LEARNING_RATE:+training_args.learning_rate=${LEARNING_RATE}} \
        ${EMBEDDING_LEARNING_RATE:+training_args.embedding_learning_rate=${EMBEDDING_LEARNING_RATE}} \
        ${WANDB_RUN_NAME:+wandb.run_name=${WANDB_RUN_NAME}} \
        ${RESUME_CHECKPOINT:+milestone.resume_from_checkpoint=${RESUME_CHECKPOINT}}

    # Evaluate (if enabled)
    if [ "${DO_EVALUATE}" = "true" ]; then
        log "Evaluating at step ${STEPS}..."
        python evaluate.py \
            --config-path=configs \
            --config-name=evaluate_ro \
            model_path="${MERGED}" \
            use_sft="${USE_SFT}" \
            global_step="${STEPS}"
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