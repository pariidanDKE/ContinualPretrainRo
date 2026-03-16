#!/bin/bash

CKPT_100_0="outputs/cpt/data_mix_20260310_212432/cpt-llama-mix_100_0_250M-ms10-250M/20260310_2125-mbou88kb/checkpoint-2351"
CKPT_80_20="outputs/cpt/data_mix_20260310_212432/cpt-llama-mix_80_20_250M-ms10-250M/20260311_0948-9e8qrox1/checkpoint-2385"

FINEWEB_RO="OpenLLM-Ro/fineweb2-ro-llm"
FINEWEB_EN="danp27/fineweb-edu-500k-sample"
RO_OFFSET=300000   # 240k used in training + 10k buffer
EN_OFFSET=80000    # 60k used in training + 5k buffer
EVAL_SAMPLES=10000

run_eval() {
    local label=$1
    local ckpt=$2
    local dataset=$3
    local offset=$4

    echo "=== $label ==="
    if [ -z "$ckpt" ]; then
        python run_evaluation.py \
            "benchmark_evaluation_cfg.tasks_to_run=[]" \
            dataset.name="$dataset" \
            dataset.sample_size=$EVAL_SAMPLES \
            dataset.sample_offset=$offset \
            training_args.per_device_eval_batch_size=4
    else
        python run_evaluation.py \
            +evaluate.checkpoint_path="$ckpt" \
            "benchmark_evaluation_cfg.tasks_to_run=[]" \
            dataset.name="$dataset" \
            dataset.sample_size=$EVAL_SAMPLES \
            dataset.sample_offset=$offset \
            training_args.per_device_eval_batch_size=4
    fi
}

# FineWeb-RO (unseen)
run_eval "base    | FineWeb-RO" ""          "$FINEWEB_RO" $RO_OFFSET
run_eval "100% RO | FineWeb-RO" "$CKPT_100_0" "$FINEWEB_RO" $RO_OFFSET
run_eval "80/20   | FineWeb-RO" "$CKPT_80_20" "$FINEWEB_RO" $RO_OFFSET

# FineWeb-EN (unseen)
run_eval "base    | FineWeb-EN" ""          "$FINEWEB_EN" $EN_OFFSET
run_eval "100% RO | FineWeb-EN" "$CKPT_100_0" "$FINEWEB_EN" $EN_OFFSET
run_eval "80/20   | FineWeb-EN" "$CKPT_80_20" "$FINEWEB_EN" $EN_OFFSET
