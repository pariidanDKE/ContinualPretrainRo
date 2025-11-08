#!/usr/bin/env bash
# This script runs tests on preliminary language understanding,
# providing an early signal of which model understands Romanian better.


export eval_batch_size=16
export model_path="google/gemma-3-1b-it"
# Choose which tasks to run (must match names in evaluation_tasks in the YAML)
#ro_tasks=(arc_challenge)

# Convert the bash array to a Hydra-compatible list string
ro_tasks_list=$(printf "[%s]" "$(IFS=,; echo "${ro_tasks[*]}")")

echo "Running Romanian evaluations with tasks: ${ro_tasks_list}"

python evaluate.py --config-name=evaluate_ro.yaml \
    eval_batch_size=${eval_batch_size} \
    tasks_to_run="${ro_tasks_list}" \
    model_path=${model_path}

echo "✅ Finalized Evaluation"
