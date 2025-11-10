#!/usr/bin/env bash
# This script runs tests on preliminary language understanding,
# providing an early signal of which model understands Romanian better.

export eval_batch_size=8
export use_mps=True
#tasks_to_run=("ro_wiki")
# Convert bash array to Hydra list format: [task1,task2,task3]
tasks_list="[$(IFS=,; echo "${tasks_to_run[*]}")]"

# Define the models to test
model_paths=("google/gemma-3-1b-it" "google/gemma-3-1b-pt")
#model_paths=("meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-1B-Instruct")


for model_path in "${model_paths[@]}"; do
    echo "======================================"
    echo "Running Romanian evaluations for model: ${model_path}"
    echo "======================================"
    
    # Conditionally set apply_chat_template based on the model type
    if [[ "$model_path" == *"-it" ]] || [[ "$model_path" == *"Instruct"* ]]; then
        apply_chat_template=true
    else
        apply_chat_template=false
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