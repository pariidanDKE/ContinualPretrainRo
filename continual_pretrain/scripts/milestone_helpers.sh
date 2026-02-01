#!/usr/bin/env bash
# Helper functions for milestone training loop

calc_steps() {
    local tokens=$1
    echo $((tokens / (BATCH_SIZE * GRAD_ACCUM * MAX_LENGTH )))
}

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Generate run name from model name
# Extracts model family (llama/qwen/gemma/etc) from MODEL_NAME
generate_run_name() {
    local model_name="$1"
    local use_sft="$2"

    # Extract base model name (part after last /)
    local model_base=$(basename "$model_name" | tr '[:upper:]' '[:lower:]')

    # Determine model family
    local model_family=""
    for family in llama qwen gemma mistral phi; do
        if [[ "$model_base" == *"$family"* ]]; then
            model_family="$family"
            break
        fi
    done

    # If no family match, use first part of model name
    if [ -z "$model_family" ]; then
        model_family=$(echo "$model_base" | cut -d'-' -f1 | cut -d'.' -f1)
    fi

    # Build run name parts
    local run_type="cpt"
    [ "$use_sft" = "true" ] && run_type="sft"

    local parts="${run_type}-${model_family}"

    # Add custom name if set
    if [ -n "$CUSTOM_NAME" ]; then
        parts="${parts}-${CUSTOM_NAME}"
    fi

    # Add milestone count
    parts="${parts}-ms${NUM_MILESTONES}"

    # Add token amount
    local total_tokens=$((MILESTONE_TOKENS * NUM_MILESTONES))
    if [ $total_tokens -ge 1000000000 ]; then
        parts="${parts}-$((total_tokens / 1000000000))B"
    elif [ $total_tokens -ge 1000000 ]; then
        parts="${parts}-$((total_tokens / 1000000))M"
    else
        parts="${parts}-$((total_tokens / 1000))K"
    fi

    echo "$parts"
}
