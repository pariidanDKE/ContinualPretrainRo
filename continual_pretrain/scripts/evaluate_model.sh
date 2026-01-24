#!/usr/bin/env bash
# This script runs tests on preliminary language understanding,
# providing an early signal of which model understands Romanian better.
export eval_batch_size=32
export use_mps=False

# All models to evaluate
model_paths=(
    "google/gemma-3-1b-pt"
    "google/gemma-3-1b-it"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-1.5B-Instruct"
)
# All Romanian and English tasks
tasks_to_run=(
    "_ro_arc_challenge"
    "arc_challenge"
    "wikitext"
    "ro_wiki"
    "_ro_winogrande"
    "_ro_belebele"
    "winogrande"
)
tasks_list="[$(IFS=,; echo "${tasks_to_run[*]}")]"

for model_path in "${model_paths[@]}"; do
    echo "======================================"
    echo "Running Romanian evaluations for model: ${model_path}"
    echo "======================================"
    
    # Conditionally set apply_chat_template based on the model type
    if [[ "$model_path" == *"-it" ]] || [[ "$model_path" == *"Instruct"* ]]; then
        apply_chat_template=true
    else
        apply_chat_template=true
    fi

    python evaluate.py \
        eval_batch_size=${eval_batch_size} \
        model_path=${model_path} \
        use_mps=${use_mps} \
        apply_chat_template=${apply_chat_template} \
        tasks_to_run=${tasks_list}

    echo "✅ Finished Evaluation for ${model_path}"
    echo
done

echo "🎯 All evaluations completed!"



