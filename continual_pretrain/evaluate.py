import hydra
from omegaconf import DictConfig
import logging
from lm_eval import evaluator
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import torch

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

    hf_token = os.getenv("HF_TOKEN", "")
    os.environ["HF_TOKEN"] = hf_token

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
    
  
    # ---- Create results folder with timestamp ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = cfg.model_path.replace("/", "_")
    run_dir = Path(cfg.output_dir) / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📁 Results will be saved in: {run_dir.resolve()}")

    selected_tasks = set(cfg.get("tasks_to_run", []))
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
        if selected_tasks and task_name not in selected_tasks:
            logger.info(f"Skipping {task_name} (not in tasks_to_run)")
            continue

        try:
            fewshot = task_cfg.fewshot
            limit = task_cfg.limit

            logger.info(f"🚀 Running evaluation for {task_name} | fewshot={fewshot} | limit={limit}")
            logger.info(f"Running {cfg.model_path} with apply_chat_template {cfg.apply_chat_template}")
            res = evaluator.simple_evaluate(
                batch_size=cfg.eval_batch_size,
                model="hf",
                model_args=f"pretrained={cfg.model_path}",
                apply_chat_template=cfg.apply_chat_template,
                tasks=[task_name],
                limit=limit,
                verbosity=cfg.verbosity,
                num_fewshot=fewshot,
                log_samples=True,
                write_out=False,
                device=device,

                fewshot_random_seed = 23,
                random_seed = 23,
                numpy_random_seed = 23,
                torch_random_seed = 23
            )

            results[task_name] = res
            logger.info(f"✅ Finished {task_name}")

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

if __name__ == "__main__":
    main()