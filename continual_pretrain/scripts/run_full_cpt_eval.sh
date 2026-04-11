#!/bin/bash
# Evaluate full-CPT checkpoints on all benchmarks.
# ALL_TASKS (11): _ro_winogrande, _ro_arc_challenge, arc_challenge, winogrande,
#   ro_wiki, wikitext, _ro_hellaswag, _ro_mmlu, _ro_grammar, _ro_belebele, _ro_gsm8k
# NEW_TASKS (3): _ro_mmlu, _ro_grammar, _ro_belebele

ALL_TASKS="[_ro_winogrande,_ro_arc_challenge,arc_challenge,winogrande,ro_wiki,wikitext,_ro_hellaswag,_ro_mmlu,_ro_grammar,_ro_belebele,_ro_gsm8k]"

set -e
cd "$(dirname "$0")/.."   # run from continual_pretrain/

RUN="outputs/cpt/full_cpt_llama_1b_20260407_212443/cpt-llama-full_cpt_llama_1b_20260407_212443-ms20-2B/20260407_2125-nfzxnkkj"

CHECKPOINTS=(
    "$RUN/checkpoint-13774"
    "$RUN/checkpoint-14727"
    "$RUN/checkpoint-15677"
    "$RUN/checkpoint-16633"
    "$RUN/checkpoint-16830"
    "$RUN/checkpoint-17589"
    "$RUN/checkpoint-18058"
    "$RUN/checkpoint-18539"
    "$RUN/checkpoint-19491"
    "$RUN/checkpoint-20441"
    "$RUN/checkpoint-21391"
    "$RUN/checkpoint-22344"
)

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

for ckpt in "${CHECKPOINTS[@]}"; do
    step=$(basename "$ckpt")
    run_eval "20260407 | $step" "$ckpt"
done

echo ""
echo "All evaluations complete. Results in eval_results/"
