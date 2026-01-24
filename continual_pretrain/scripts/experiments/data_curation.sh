#!/usr/bin/env bash
# Experiment: Data curation mix ablations
# Tests different combinations of datasets while keeping norobots, dolly, oasst fixed

set -euo pipefail

# Model configuration
MODEL="meta-llama/Llama-3.2-1B-Instruct"
LORA_RANK=128
LORA_ALPHA=256
BATCH_SIZE=32
MILESTONE_TOKENS=25000000  # 25M tokens per milestone
NUM_MILESTONES=4        # Total: 200M tokens

# Data mix configurations to test
# Fixed datasets (always included): norobots (2.4%), dolly (2.2%), oasst (2.4%) = 7% total
# Variable datasets: remaining 93% split 50/50 between two datasets (46.5% each)
# Target: 200M tokens total (13.9M fixed + 186.1M variable)

# Experiment 1: Camel + Orca
DATASET_MIX_1='[
  {name: "danp27/norobots_sft", split: train, proportion: 0.024, sample_size: null},
  {name: "./data/formatted_data/dolly", split: train, proportion: 0.022, sample_size: null},
  {name: "./data/formatted_data/oasst", split: train, proportion: 0.024, sample_size: null},
  {name: "./data/formatted_data/camel", split: train, proportion: 0.465, sample_size: 110000},
  {name: "./data/formatted_data/orca", split: train, proportion: 0.465, sample_size: 110000}
]'

# Experiment 2: Magpie + Orca
DATASET_MIX_2='[
  {name: "danp27/norobots_sft", split: train, proportion: 0.024, sample_size: null},
  {name: "./data/formatted_data/dolly", split: train, proportion: 0.022, sample_size: null},
  {name: "./data/formatted_data/oasst", split: train, proportion: 0.024, sample_size: null},
  {name: "./data/formatted_data/magpie", split: train, proportion: 0.465, sample_size: 110000},
  {name: "./data/formatted_data/orca", split: train, proportion: 0.465, sample_size: 110000}
]'

# Experiment 3: Camel + Magpie
DATASET_MIX_3='[
  {name: "danp27/norobots_sft", split: train, proportion: 0.024, sample_size: null},
  {name: "./data/formatted_data/dolly", split: train, proportion: 0.022, sample_size: null},
  {name: "./data/formatted_data/oasst", split: train, proportion: 0.024, sample_size: null},
  {name: "./data/formatted_data/camel", split: train, proportion: 0.465, sample_size: 110000},
  {name: "./data/formatted_data/magpie", split: train, proportion: 0.465, sample_size: 110000}
]'

# Array of experiments: "dataset_config|experiment_name|description"
EXPERIMENTS=(
    "${DATASET_MIX_1}|camel_orca_50_50|Camel (46.5%) + Orca (46.5%)"
    "${DATASET_MIX_2}|magpie_orca_50_50|Magpie (46.5%) + Orca (46.5%)"
    "${DATASET_MIX_3}|camel_magpie_50_50|Camel (46.5%) + Magpie (46.5%)"
)

for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r DATASET_CONFIG EXPERIMENT_NAME DESCRIPTION <<< "$experiment"

    echo "=========================================="
    echo "Testing data mix: ${EXPERIMENT_NAME}"
    echo "=========================================="
    echo "Configuration:"
    echo "  Fixed: norobots (2.4%) + dolly (2.2%) + oasst (2.4%) = 7%"
    echo "  Variable: ${DESCRIPTION} = 93%"
    echo "  Total: 100M tokens (4 milestones × 25M tokens)"
    echo ""

    # Create config file for this experiment
    mkdir -p configs/experiments
    CONFIG_FILE="configs/experiments/${EXPERIMENT_NAME}_dataset.yaml"

    cat > "${CONFIG_FILE}" << EOF
# Auto-generated config for experiment: ${EXPERIMENT_NAME}

defaults:
  - /build_dataset
  - _self_

# Override dataset configuration for this experiment
data_builder:
  enabled: true
  datasets: ${DATASET_CONFIG}
EOF

    echo "📝 Created config: ${CONFIG_FILE}"

    OUTPUT_DIR="outputs/data_mix/${EXPERIMENT_NAME}"

    # Run milestone loop with custom dataset config
    MODEL_NAME="${MODEL}" \
    LORA_RANK=${LORA_RANK} \
    LORA_ALPHA=${LORA_ALPHA} \
    BATCH_SIZE=${BATCH_SIZE} \
    MILESTONE_TOKENS=${MILESTONE_TOKENS} \
    NUM_MILESTONES=${NUM_MILESTONES} \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DATASET_CONFIG_NAME="experiments/${EXPERIMENT_NAME}_dataset" \
    bash scripts/run_milestone_loop.sh

    echo "✅ Completed ${EXPERIMENT_NAME}"
    echo ""
done

echo "🎉 Data curation experiments complete!"
