#!/bin/bash
# Evaluate full-CPT checkpoints on all benchmarks.
# ALL_TASKS (11): _ro_winogrande, _ro_arc_challenge, arc_challenge, winogrande,
#   ro_wiki, wikitext, _ro_hellaswag, _ro_mmlu, _ro_grammar, _ro_belebele, _ro_gsm8k
#
# Equally-spaced selection (~4757 steps apart) across all 3 training runs.
# checkpoint-0 uses the base model directly (LoRA B=0 at init, so it's equivalent).

ALL_TASKS="[_ro_winogrande,_ro_arc_challenge,arc_challenge,winogrande,ro_wiki,wikitext,_ro_hellaswag,_ro_mmlu,_ro_grammar,_ro_belebele,_ro_gsm8k]"

set -e
cd "$(dirname "$0")/.."   # run from continual_pretrain/

BASE_MODEL="unsloth/llama-3.2-1b-unsloth-bnb-4bit"

RUN_A="outputs/cpt/full_cpt_llama_1b_20260328_174407/cpt-llama-full_cpt_llama_1b_20260328_174407-ms10-1B/20260328_1744-7owjds6f"
RUN_B="outputs/cpt/full_cpt_llama_1b_20260401_003748/cpt-llama-full_cpt_llama_1b_20260401_003748-ms10-1B/20260401_0038-iruygely"
RUN_C="outputs/cpt/full_cpt_llama_1b_20260407_212443/cpt-llama-full_cpt_llama_1b_20260407_212443-ms20-2B/20260407_2125-nfzxnkkj"

CHECKPOINTS=(
    "$RUN_A/checkpoint-4756"
    "$RUN_B/checkpoint-9017"
    "$RUN_C/checkpoint-13774"
    "$RUN_C/checkpoint-18539"
    "$RUN_C/checkpoint-22344"
)

run_eval_base() {
    echo ""
    echo "========================================"
    echo "=== base model (checkpoint-0) ==="
    echo "========================================"

    python run_evaluation.py \
        model.builder.model_name="$BASE_MODEL" \
        dataset.sample_size=null \
        benchmark_evaluation_cfg.eval_batch_size=4 \
        "benchmark_evaluation_cfg.tasks_to_run=$ALL_TASKS"
}

run_eval() {
    local label=$1
    local ckpt=$2

    echo ""
    echo "========================================"
    echo "=== $label ==="
    echo "========================================"

    python run_evaluation.py \
        +evaluate.checkpoint_path="$ckpt" \
        dataset.sample_size=null \
        benchmark_evaluation_cfg.eval_batch_size=4 \
        "benchmark_evaluation_cfg.tasks_to_run=$ALL_TASKS"
}

run_eval_base
for ckpt in "${CHECKPOINTS[@]}"; do
    step=$(basename "$ckpt")
    run_eval "$step" "$ckpt"
done

echo ""
echo "All evaluations complete. Results in eval_results/"
