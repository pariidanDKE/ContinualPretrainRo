import hydra
from omegaconf import DictConfig
import logging
from lm_eval import evaluator
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("__name__")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@hydra.main(config_path="configs", config_name="evaluate_ro.yaml")
def main(cfg: DictConfig):
    logger.info("Starting Romanian evaluation run")
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN", "")
    os.environ["HF_TOKEN"] = hf_token

    # ---- Create results folder with timestamp ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg.output_dir) / timestamp
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

        fewshot = task_cfg.fewshot
        limit = task_cfg.limit

        logger.info(f"🚀 Running evaluation for {task_name} | fewshot={fewshot} | limit={limit}")

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
            device=cfg.device,
        )

        results[task_name] = res
        logger.info(f"✅ Finished {task_name}")

        # ---- Save this task’s results ----
        task_file = run_dir / f"{task_name}.json"
        with open(task_file, "w", encoding="utf-8") as f:
            f.write(safe_json_dump({task_name: res}))

        logger.info(f"💾 Saved partial results: {task_file.resolve()}")

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