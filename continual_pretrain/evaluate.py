import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from lm_eval import evaluator
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import torch
import wandb
from train_utils import compose_run_name

logger = logging.getLogger("__name__")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Filter out the tag registration warning
class IgnoreTagWarning(logging.Filter):
    def filter(self, record):
        return "is already registered as a group" not in record.getMessage()

logging.getLogger("lm-eval").addFilter(IgnoreTagWarning())

@hydra.main(config_path="configs", config_name="evaluate_ro.yaml")
def main(cfg: DictConfig):
    logger.info("Starting Romanian evaluation run")
    load_dotenv()

    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hf_token = os.getenv("HF_TOKEN", "")
    os.environ["HF_TOKEN"] = hf_token

    # Get global_step for proper x-axis alignment with training logs
    # Can be passed via config override or environment variable
    global_step = cfg.get("global_step", None)
    if global_step is None:
        global_step = int(os.getenv("EVAL_GLOBAL_STEP", 0))
    else:
        global_step = int(global_step)

    logger.info(f"Evaluation global_step: {global_step} (for WandB x-axis alignment)")

    # ---- Check CUDA availability and set device ----
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        
    elif cfg.use_mps:
        device = "mps"
        logger.info("Using MPS as specified in the config.")
    else:
        device = "cpu"
        logger.info("💻 CUDA not available. Using CPU")

    dtype = "bfloat16" if device=='cuda' else 'float16'
    max_length = cfg.get("max_length", 2048)  # Model's max context length
    eval_max_length = cfg.get("eval_max_length", max_length)  # Defaults to max_length if not specified

    # ---- Map PT models to their IT tokenizer equivalents ----
    PT_TO_IT_TOKENIZER_MAP = {
        "google/gemma-3-1b-pt": "google/gemma-3-1b-it",
        "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    }

    # Determine tokenizer path: use IT tokenizer for PT models when chat template is enabled AND use_sft is true
    use_sft = cfg.get("use_sft", True)  # Default to True for backwards compatibility
    tokenizer_path = cfg.model_path
    if use_sft and cfg.apply_chat_template and cfg.model_path in PT_TO_IT_TOKENIZER_MAP:
        tokenizer_path = PT_TO_IT_TOKENIZER_MAP[cfg.model_path]
        logger.info(f"Using IT tokenizer {tokenizer_path} for PT model {cfg.model_path}")

    model_args_str = (
        f"pretrained={cfg.model_path},tokenizer={tokenizer_path},"
        f"max_length={max_length},eval_max_length={eval_max_length},"
        f"truncation=True,dtype={dtype}"
    )
           

    # ---- Create results folder with timestamp ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = cfg.model_path.replace("/", "_")
    run_dir = Path(cfg.output_dir) / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Results will be saved in: {run_dir.resolve()}")

    # ---- Get run name from environment ----
    # WANDB_RUN_NAME must be set by the shell script (run_milestone_loop.sh)
    run_name = os.environ.get("WANDB_RUN_NAME")
    if not run_name:
        logger.warning("⚠️  WANDB_RUN_NAME not set! Run name should be provided by run_milestone_loop.sh")
        logger.warning("    Continuing without run name - WandB logging may be inconsistent")
    else:
        logger.info(f"Using run name: {run_name}")

    # ---- Initialize W&B with GPU monitoring ----
    if cfg.get("use_wandb", False):
        wandb.init(
            project=cfg.get("wandb_project", "romanian-llm-eval"),
            name=run_name,
            config={
                "model_path": cfg.model_path,
                "tasks": list(cfg.get("tasks_to_run", [])),
                "eval_batch_size": cfg.eval_batch_size,
                "max_length": max_length,
                "eval_max_length": eval_max_length,
                "use_sft": use_sft,
                "apply_chat_template": cfg.apply_chat_template,
                "device": device,
                "dtype": dtype,
            },
            settings=wandb.Settings(start_method="thread")
        )
        logger.info("📊 W&B logging enabled with GPU monitoring")

    selected_tasks = set(cfg.get("tasks_to_run", []))
    preplexity_tasks = set(cfg.get("preplexity_tasks", []))
    results = {}

    # ---- Safe JSON dump helper ----
    def safe_json_dump(obj):
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except TypeError:
            return json.dumps(json.loads(json.dumps(obj, default=str)), indent=2, ensure_ascii=False)

    # ---- Iterate through evaluation tasks ----
    for task_cfg in cfg.evaluation_tasks:
        task_name = task_cfg.name
        # Only apply chat template if: (1) not a perplexity task, (2) apply_chat_template is true, AND (3) use_sft is true
        apply_chat_template = (task_name not in preplexity_tasks) and cfg.apply_chat_template and use_sft

        if selected_tasks and task_name not in selected_tasks:
            logger.info(f"Skipping {task_name} (not in tasks_to_run)")
            continue

        try:
            fewshot = task_cfg.fewshot
            limit = task_cfg.limit
            batch_size = task_cfg.task_batch_size

            logger.info(f"🚀 Running evaluation for {task_name} | fewshot={fewshot} | limit={limit}")
            logger.info(f"Running {cfg.model_path} | use_sft={use_sft} | apply_chat_template={apply_chat_template}")

            # Prepare evaluation kwargs
            eval_kwargs = {
                "batch_size": batch_size,
                "model": "hf",
                "model_args": model_args_str,
                "apply_chat_template": apply_chat_template,
                "tasks": [task_name],
                "limit": limit,
                "verbosity": cfg.verbosity,
                "num_fewshot": fewshot,
                "log_samples": True,
                "write_out": False,
                "device": device,
                "fewshot_random_seed": 23,
                "random_seed": 23,
                "numpy_random_seed": 23,
                "torch_random_seed": 23,
            }

            # Add multiturn and system instruction only when using chat templates AND have fewshot examples
            if apply_chat_template and fewshot > 0:
                eval_kwargs["fewshot_as_multiturn"] = True
                #eval_kwargs["system_instruction"] = "Answer the following questions accurately."
                logger.info("📝 Using fewshot_as_multiturn=True")

            res = evaluator.simple_evaluate(**eval_kwargs)

            results[task_name] = res
            logger.info(f"✅ Finished {task_name}")

            # ---- Log to W&B ----
            if cfg.get("use_wandb", False):
                task_results = res.get("results", {}).get(task_name, {})
                metrics_to_log = {f"{task_name}/{k}": v for k, v in task_results.items() if isinstance(v, (int, float))}
                if metrics_to_log:
                    wandb.log(metrics_to_log)

            # ---- Save this task's results ----
            task_file = run_dir / f"{task_name}.json"
            with open(task_file, "w", encoding="utf-8") as f:
                f.write(safe_json_dump({task_name: res}))

            logger.info(f"💾 Saved partial results: {task_file.resolve()}")
        
        except Exception as e:
            logger.error(f"❌ Error running {task_name}: {str(e)}")
            results[task_name] = {"error": str(e)}
            continue

    # ---- Save combined results ----
    final_file = run_dir / "eval_results.json"
    with open(final_file, "w", encoding="utf-8") as f:
        f.write(safe_json_dump(results))

    logger.info(f"📁 Combined results saved to: {final_file.resolve()}")

    # ---- Optional: quick summary ----
    logger.info("📊 Summary:")
    for task_name, res in results.items():
        task_results = res.get("results", {}).get(task_name, {})
        if task_results:
            metrics = {k: v for k, v in task_results.items() if isinstance(v, (int, float))}
            logger.info(f"  - {task_name}: {metrics}")
        else:
            logger.info(f"  - {task_name}: No metrics found")

    # ---- Finish W&B run ----
    if cfg.get("use_wandb", False):
        wandb.finish()
        logger.info("📊 W&B run finished")

if __name__ == "__main__":
    main()

