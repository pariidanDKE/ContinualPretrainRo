import unsloth  # noqa: F401 — must be first import for Unsloth optimizations
import json
import logging
import os
import hydra
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed
from datetime import datetime
from dotenv import load_dotenv

def _cfg_from_checkpoint(checkpoint_path: str) -> tuple[dict, dict]:
    """Read model name and LoRA config entirely from the checkpoint's adapter_config.json.

    Returns (model_overrides, lora_cfg) where model_overrides contains only
    model_name and lora_cfg is built solely from the checkpoint — no yaml values involved.
    """
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)
    model_overrides = {"model_name": adapter_cfg["base_model_name_or_path"]}
    lora_cfg = {
        "r": adapter_cfg["r"],
        "lora_alpha": adapter_cfg["lora_alpha"],
        "lora_dropout": adapter_cfg.get("lora_dropout", 0.0),
        "bias": adapter_cfg.get("bias", "none"),
        "target_modules": adapter_cfg["target_modules"],
        "task_type": adapter_cfg.get("task_type", "CAUSAL_LM"),
        "modules_to_save": adapter_cfg.get("modules_to_save", None),
    }
    return model_overrides, lora_cfg

from milestone_trainer import create_milestone_trainer
from model import ModelBuilder, ModelBuilderConfig, LoraAdapterConfig
from train_utils import (
    to_dict,
    overwrite_pt_tokenizer,
    set_training_type,
    prepare_dataset,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
)


@hydra.main(config_path="configs", config_name="train_model.yaml", version_base=None)
def main(cfg: DictConfig):
    """
    Standalone evaluation entry point.

    Builds the model and tokenizer, optionally loads a LoRA checkpoint, creates a
    MilestoneTrainer, and calls trainer.evaluate(). With no eval dataset supplied,
    the loss loop is skipped and only the benchmark suite (evaluate_benchmarks) runs.

    Key overrides via CLI:
        benchmark_evaluation_cfg.tasks_to_run=[ro_mmlu,ro_grammar]
        evaluate.checkpoint_path=outputs/.../checkpoint-1000
    """
    load_dotenv()

    if cfg.get("seed"):
        set_seed(int(cfg.seed))

    logging.info("Full resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    training_kwargs = to_dict(cfg.training_args)
    model_cfg = to_dict(cfg.model.builder)
    lora_cfg = to_dict(cfg.model.lora)
    if "max_length" in training_kwargs:
        model_cfg["max_seq_length"] = training_kwargs["max_length"]

    benchmark_evaluation_cfg = to_dict(cfg.benchmark_evaluation_cfg)
    evaluate_cfg = to_dict(cfg.get("evaluate", {}))
    checkpoint_path = evaluate_cfg.get("checkpoint_path")

    # ---- If checkpoint given, ignore yaml lora_cfg entirely — read from adapter_config.json ----
    if checkpoint_path:
        model_overrides, lora_cfg = _cfg_from_checkpoint(checkpoint_path)
        model_cfg.update(model_overrides)
        logger.info(f"Loaded from checkpoint: model={model_cfg['model_name']}, r={lora_cfg['r']}")

    # ---- Build model ----
    model_cfg = ModelBuilderConfig(**model_cfg)
    lora_cfg = LoraAdapterConfig(**lora_cfg)
    builder = ModelBuilder(model_cfg, lora_cfg)
    model, tokenizer = builder.build()
    logger.info(f"Built model: {model_cfg.model_name}")

    use_sft = cfg.dataset.get("use_sft", False)
    if use_sft:
        tokenizer = overwrite_pt_tokenizer(model_name=model_cfg.model_name, tokenizer=tokenizer)

    milestone_cfg = to_dict(cfg.get("milestone", {}))
    use_unsloth = cfg.model.builder.get("use_unsloth", False)
    TrainerClass, TrainingArgsClass, training_kwargs = set_training_type(use_unsloth, training_kwargs)
    training_args = TrainingArgsClass(**training_kwargs)

    # Optionally load a validation dataset for eval loss (configured via dataset.* overrides)
    eval_dataset = None
    if cfg.dataset.get("name") and cfg.dataset.get("sample_size"):
        logger.info(f"Loading eval dataset: {cfg.dataset.name} (n={cfg.dataset.sample_size})")
        eval_dataset = prepare_dataset(cfg, tokenizer)
        
    dummy_dataset = Dataset.from_dict({"text": ["dummy"]})

    trainer = create_milestone_trainer(
        TrainerClass,
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dummy_dataset,
        eval_dataset=eval_dataset,
        benchmark_evaluation_cfg=benchmark_evaluation_cfg,
        num_milestones=milestone_cfg.get("num_milestones", 1),
        tokens_per_milestone=milestone_cfg.get("tokens_per_milestone", 1),
        run_benchmarks=True,
    )

    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        trainer._load_from_checkpoint(checkpoint_path)

    # modules_to_save (embed_tokens, lm_head) may land on CPU regardless of whether
    # a checkpoint was loaded — move all CPU params to match the main CUDA device
    cuda_device = next((p.device for p in model.parameters() if p.device.type == "cuda"), None)
    if cuda_device:
        for param in model.parameters():
            if param.device.type == "cpu":
                param.data = param.data.to(cuda_device)

    logger.info("Starting evaluation...")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation complete. Metrics: {metrics}")

    output_dir = benchmark_evaluation_cfg.get("output_dir", "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_path:
        parts = checkpoint_path.rstrip("/").split("/")
        run_name = f"{parts[-3]}_{parts[-1]}"  # e.g. "cpt-llama-ms3-300M_checkpoint-9765"
    else:
        run_name = "eval"
    output_path = os.path.join(output_dir, f"{timestamp}_{run_name}.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
