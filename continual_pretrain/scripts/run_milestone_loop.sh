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
CUSTOM_NAME=${CUSTOM_NAME:-"CPTTestSpikyLoss_maxgrad1_lr1e4_ga4"}
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
MODEL_NAME=${MODEL_NAME:-"unsloth/Llama-3.2-1B"} # 
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-4}
SAMPLE_SIZE=${SAMPLE_SIZE:-null}
MAX_LENGTH=${MAX_LENGTH:-4096}
USE_UNSLOTH=${USE_UNSLOTH:-true}
USE_PACKING=${USE_PACKING:-true}
USE_SFT=${USE_SFT:-false}  # false = CPT mode (raw text), true = SFT mode (Romanian tags → chat tokens)

TEXT_COLUMN=${TEXT_COLUMN:-"formatted_text"}
DELETE_CHECKPOINT=${DELETE_CHECKPOINT:-true}  # If true, delete merged checkpoint after each eval to save disk

# Learning rate configuration
LEARNING_RATE=${LEARNING_RATE:-1e-4}
EMBEDDING_LEARNING_RATE=${EMBEDDING_LEARNING_RATE:-2e-2}

# Data builder configuration (optional override for dataset mixing experiments)
DATASET_CONFIG_NAME=${DATASET_CONFIG_NAME:-"build_dataset"}

# =============================================================================
# MILESTONE CONFIGURATION
# =============================================================================

# Milestone configuration
# MILESTONE_TOKENS=${MILESTONE_TOKENS:-80000000}
# NUM_MILESTONES=${NUM_MILESTONES:-10}
DO_EVALUATE=${DO_EVALUATE:-false}

MILESTONE_TOKENS=${MILESTONE_TOKENS:-5000000}
NUM_MILESTONES=${NUM_MILESTONES:-20}

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

# COMMENTED OUT: Using existing dataset for isolated milestone test
# echo "=========================================="
# echo "STEP 0: Building Mixed Milestone Dataset"
# echo "=========================================="
# echo "This will create a mixed dataset and tag each example with its milestone number."
# echo "Dataset will be saved to: ${PREBUILT_DATASET_PATH}"
# echo ""
#
# log "Building prebuilt dataset..."
# python build_mixed_dataset.py \
#     --config-name="${DATASET_CONFIG_NAME}" \
#     model.builder.model_name="${MODEL_NAME}" \
#     output_dir="${PREBUILT_DATASET_PATH}" \
#     milestone.num_milestones=${NUM_MILESTONES} \
#     milestone.tokens_per_milestone=${MILESTONE_TOKENS} \
#     data_builder.max_length=${MAX_LENGTH} \
#     training_args.per_device_train_batch_size=${BATCH_SIZE} \
#     training_args.gradient_accumulation_steps=${GRAD_ACCUM} \
#     dataset.text_field="text" \
#     ${SAMPLE_SIZE:+dataset.sample_size=${SAMPLE_SIZE}}
#
# log "✅ Prebuilt dataset created"
# echo ""

echo "=========================================="
echo "ISOLATED MILESTONE TEST"
echo "=========================================="
echo "Using existing dataset at: ${PREBUILT_DATASET_PATH}"
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

# =============================================================================
# EVALUATION WANDB SETUP
# =============================================================================

# Check if we should use separate WandB run for evaluation
USE_SEPARATE_WANDB=$(python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('configs/evaluate_ro.yaml'); print(cfg.get('use_separate_wandb', False))")

# Generate a single eval run ID to be used for ALL evaluations (if separate wandb is enabled)
if [ "${USE_SEPARATE_WANDB}" = "True" ]; then
    EVAL_WANDB_RUN_ID=$(python3 -c 'import wandb; print(wandb.util.generate_id())')
    EVAL_WANDB_RUN_NAME="${WANDB_RUN_NAME}-eval"
    log "Evaluation will use separate WandB run: ${EVAL_WANDB_RUN_NAME} (ID: ${EVAL_WANDB_RUN_ID})"
fi

# COMMENTED OUT: Skip baseline eval for isolated test
# Initial baseline eval (at step 0)
# if [ "${DO_EVALUATE}" = "true" ]; then
#     log "Evaluating baseline at step 0..."
#     ...
# fi


# MODIFIED: Run only milestone 3 (loads shard 2) to test in isolation
TEST_MILESTONE=3  # The failing milestone

for i in $(seq ${TEST_MILESTONE} ${TEST_MILESTONE}); do
    CURRENT_MILESTONE=$((i - 1))  # Convert to 0-indexed
    TARGET_TOKENS=$((MILESTONE_TOKENS * 1))
    echo "==================== FIXED TARGET TOKENS TO 1 MILESTONE CHANGE BACK AFTER DEBUG ==================== "
    STEPS=$(calc_steps ${TARGET_TOKENS})
    CHECKPOINT="${OUTPUT_DIR}/checkpoint-${STEPS}"
    MERGED="${MERGED_DIR}/checkpoint-${STEPS}"

    echo "=========================================="
    echo "TESTING MILESTONE ${i} (shard index ${CURRENT_MILESTONE}) - ${TARGET_TOKENS} tokens"
    echo "Running from SCRATCH (no resume) to test shard capacity"
    echo "=========================================="

    #Train WITHOUT resuming from previous checkpoint (start fresh)
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
        ${WANDB_RUN_NAME:+wandb.run_name=${WANDB_RUN_NAME}}

    # COMMENTED OUT: Skip evaluation for isolated test
    # if [ "${DO_EVALUATE}" = "true" ]; then
    #     log "Evaluating at step ${STEPS}..."
    #     ...
    # fi

    log "✅ Milestone ${i} test complete"
    echo ""
done

echo ""
echo "=========================================="
echo "Isolated milestone test complete!"
echo "=========================================="
echo ""

# COMMENTED OUT: Keep dataset for further testing
# =============================================================================
# CLEANUP
# =============================================================================
# log "Cleaning up prebuilt dataset..."
# if [ -d "${PREBUILT_DATASET_PATH}" ]; then
#     rm -rf "${PREBUILT_DATASET_PATH}"
#     log "✅ Removed ${PREBUILT_DATASET_PATH}"
# else
#     log "⚠️  Prebuilt dataset not found (already deleted?)"
# fi

echo ""
echo "🎉 Test complete! Dataset preserved at ${PREBUILT_DATASET_PATH}"