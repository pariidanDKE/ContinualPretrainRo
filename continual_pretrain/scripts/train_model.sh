#!/usr/bin/env bash
# Simple driver for continual_pretrain/train_model.py with convenient overrides.
# Mirrors scripts/evaluate_model.sh structure so you can launch multiple runs
# with different dataset sample sizes, model names, and batch configuration.

set -euo pipefail

# Customize these lists as needed
MODEL_NAMES=(
  "meta-llama/Llama-3.2-1B"
)

SAMPLE_SIZES=(
  5000   # quick smoke-test subset; set to null (without quotes) for full data
)

BATCH_SIZES=( # SOMETHING IS BROKE HERE THIS DOES NOT ACTUALLY AFFECT ANYTHING
  32
)

GRAD_ACC_STEPS=2
use_unsloth=true

for model_name in "${MODEL_NAMES[@]}"; do
  for sample_size in "${SAMPLE_SIZES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
      echo "======================================"
      echo "Training model: ${model_name}"
      echo "Sample size:    ${sample_size}"
      echo "Batch size:     ${batch_size}"
      echo "======================================"

      run_slug=$(echo "${model_name}" | tr '/:' '__')
      output_dir="outputs/train_${run_slug}_bs${batch_size}_s${sample_size}"

      python train_model.py \
        model.builder.model_name="${model_name}" \
        dataset.sample_size="${sample_size}" \
        training_args.per_device_train_batch_size=${batch_size} \
        training_args.gradient_accumulation_steps=${GRAD_ACC_STEPS} \
        training_args.output_dir="${output_dir}"  \
        model.builder.use_unsloth=${use_unsloth}

      echo "✅ Finished training for ${model_name} (sample_size=${sample_size}, batch=${batch_size})"
      echo
    done
  done
done

echo "🎯 All training runs completed!"
