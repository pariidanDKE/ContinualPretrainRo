#!/usr/bin/env bash
# Continuous milestone training: build dataset → train all milestones in one process

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
PACKING_STRATEGY=${PACKING_STRATEGY:-"bfd"}  # "bfd" or "wrapped"

# Learning rate configuration
LEARNING_RATE=${LEARNING_RATE:-6e-5}
EMBEDDING_LEARNING_RATE=${EMBEDDING_LEARNING_RATE:-2e-5}

# Data builder configuration (optional override for dataset mixing experiments)
DATASET_CONFIG_NAME=${DATASET_CONFIG_NAME:-"build_dataset"}

# Milestone configuration
MILESTONE_TOKENS=${MILESTONE_TOKENS:-5000000}
NUM_MILESTONES=${NUM_MILESTONES:-5}

# Output directories
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/continuous_milestone_training"}
PREBUILT_DATASET_PATH="data/mixed_milestone_dataset"

# =============================================================================
# WANDB CONFIGURATION
# =============================================================================

# Generate unique run ID for entire training session
WANDB_RUN_ID="${WANDB_RUN_ID:-$(python3 -c 'import wandb; print(wandb.util.generate_id())')}"
export WANDB_RUN_ID
export WANDB_RESUME="allow"

WANDB_GROUP=${WANDB_GROUP:-"continuous_milestone"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"continuous_run"}
WANDB_PROJECT=${WANDB_PROJECT:-"RoLLM"}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# =============================================================================
# STEP 1: BUILD MIXED MILESTONE DATASET
# =============================================================================

echo "=========================================="
echo "STEP 1: Building Mixed Milestone Dataset"
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
# STEP 2: CONTINUOUS TRAINING ACROSS ALL MILESTONES
# =============================================================================

echo "=========================================="
echo "STEP 2: Continuous Milestone Training"
echo "=========================================="
echo "Training configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  LoRA rank: ${LORA_RANK}, alpha: ${LORA_ALPHA}"
echo "  Batch size: ${BATCH_SIZE}, grad accum: ${GRAD_ACCUM}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Packing: ${USE_PACKING}, strategy: ${PACKING_STRATEGY}"
echo "  Milestones: ${NUM_MILESTONES} x ${MILESTONE_TOKENS} tokens"
echo "  Total tokens: $((MILESTONE_TOKENS * NUM_MILESTONES))"
echo "  WandB Run ID: ${WANDB_RUN_ID}"
echo ""

log "Starting continuous training..."
python train_continuous_milestones.py \
    model.builder.model_name="${MODEL_NAME}" \
    model.builder.use_unsloth="${USE_UNSLOTH}" \
    model.lora.r=${LORA_RANK} \
    model.lora.lora_alpha=${LORA_ALPHA} \
    training_args.per_device_train_batch_size=${BATCH_SIZE} \
    training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
    training_args.output_dir="${OUTPUT_DIR}" \
    training_args.max_length=${MAX_LENGTH} \
    data_collator.packing="${USE_PACKING}" \
    data_collator.packing_strategy="${PACKING_STRATEGY}" \
    milestone.prebuilt_dataset_path="${PREBUILT_DATASET_PATH}" \
    wandb.group="${WANDB_GROUP}" \
    wandb.project="${WANDB_PROJECT}" \
    ${LEARNING_RATE:+training_args.learning_rate=${LEARNING_RATE}} \
    ${EMBEDDING_LEARNING_RATE:+training_args.embedding_learning_rate=${EMBEDDING_LEARNING_RATE}} \
    ${WANDB_RUN_NAME:+wandb.run_name=${WANDB_RUN_NAME}}

log "✅ Continuous training complete"
echo ""

# =============================================================================
# CLEANUP (OPTIONAL)
# =============================================================================

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Final checkpoint: ${OUTPUT_DIR}"
echo "Prebuilt dataset: ${PREBUILT_DATASET_PATH} (preserved for reuse)"
echo ""
echo "🎉 All milestones trained in a single session!"
