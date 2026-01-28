"""
Continuous milestone training in a single process.

This script runs all milestones in one training session with:
- Dataset packed once at the beginning
- Evaluation callback triggered at milestone boundaries
- No separate processes between milestones
"""

import os
import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
from transformers import set_seed, TrainerCallback
from dotenv import load_dotenv

from train_utils import (
    load_milestone_shard,
    to_dict,
    apply_wandb_config,
    calculate_max_steps,
    PT_TO_IT_TOKENIZER_MAP,
)
from trainer_callbacks import (
    CosineSchedulerCallback,
    EpochResetCallback,
)
from token_stats_callback import create_trainer_with_token_stats, TokenStatsCallback
from model import ModelBuilder, ModelBuilderConfig, LoraAdapterConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer
from unsloth import UnslothTrainer, UnslothTrainingArguments

logger = logging.getLogger(__name__)


class MilestoneEvalCallback(TrainerCallback):
    """
    Evaluate at milestone boundaries using lm-evaluation-harness.
    """
    def __init__(
        self,
        milestone_steps,
        eval_config,
        use_unsloth=False,
    ):
        """
        Args:
            milestone_steps: List of steps at which to evaluate (e.g., [76, 152, 228, ...])
            eval_config: DictConfig from evaluate_ro.yaml with task configurations
            use_unsloth: Whether using Unsloth (for inference mode switching)
        """
        self.milestone_steps = set(milestone_steps)
        self.eval_config = eval_config
        self.use_unsloth = use_unsloth
        self.completed_evals = set()

        # Parse config
        self.tasks_to_run = set(eval_config.get("tasks_to_run", []))
        self.perplexity_tasks = set(eval_config.get("preplexity_tasks", []))
        self.apply_chat_template_global = eval_config.get("apply_chat_template", True)
        self.max_length = eval_config.get("max_length", 2048)
        self.eval_max_length = eval_config.get("eval_max_length", self.max_length)
        self.eval_batch_size = eval_config.get("eval_batch_size", 8)

        # Build task config lookup
        self.task_configs = {
            task.name: task for task in eval_config.evaluation_tasks
        }

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Check if we've reached a milestone step and run evaluation."""
        current_step = state.global_step

        # Check if this is a milestone step and we haven't evaluated it yet
        if current_step in self.milestone_steps and current_step not in self.completed_evals:
            logger.info(f"\n{'='*80}")
            logger.info(f"MILESTONE EVALUATION AT STEP {current_step}")
            logger.info(f"{'='*80}\n")

            try:
                # Run evaluation using the logic from evaluate.py
                self._run_evaluation(current_step, model, tokenizer, state)

            except Exception as e:
                logger.error(f"Evaluation failed at step {current_step}: {e}")
                logger.exception("Full traceback:")

            # Mark this milestone as evaluated
            self.completed_evals.add(current_step)

            logger.info(f"\n{'='*80}")
            logger.info(f"RESUMING TRAINING")
            logger.info(f"{'='*80}\n")

    def _run_evaluation(self, current_step, model, tokenizer, state):
        """
        Run evaluation using lm-eval harness (adapted from evaluate.py).

        Args:
            current_step: Current training step
            model: The model being trained
            tokenizer: The tokenizer
            state: Trainer state
        """
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        import torch

        # 1. Switch model to inference mode (if using Unsloth)
        if self.use_unsloth:
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(model)
                logger.info("Switched to inference mode (Unsloth)")
            except Exception as e:
                logger.warning(f"Failed to switch to inference mode: {e}")

        # 2. Determine dtype based on device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "bfloat16" if device == "cuda" else "float16"

        # 3. Wrap the current model for lm-eval with custom eval_max_length
        logger.info(f"Wrapping model for lm-eval (eval_max_length={self.eval_max_length})")
        lm_obj = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            max_length=self.max_length,
            eval_max_length=self.eval_max_length,  # Custom parameter for your fork
            truncation=True,
            dtype=dtype,
        )

        # 4. Run evaluation for each selected task
        all_results = {}
        for task_name in self.tasks_to_run:
            if task_name not in self.task_configs:
                logger.warning(f"Task {task_name} not found in eval config, skipping")
                continue

            task_cfg = self.task_configs[task_name]

            # Determine whether to apply chat template for this task
            # Perplexity tasks never use chat template
            apply_chat_template = (
                task_name not in self.perplexity_tasks
                and self.apply_chat_template_global
            )

            logger.info(
                f"Evaluating {task_name} | "
                f"fewshot={task_cfg.fewshot} | "
                f"batch_size={task_cfg.task_batch_size} | "
                f"chat_template={apply_chat_template}"
            )

            # Prepare evaluation kwargs (from evaluate.py)
            eval_kwargs = {
                "batch_size": task_cfg.task_batch_size,
                "model": lm_obj,
                "apply_chat_template": apply_chat_template,
                "tasks": [task_name],
                "limit": task_cfg.get("limit"),
                "verbosity": self.eval_config.get("verbosity", "INFO"),
                "num_fewshot": task_cfg.fewshot,
                "log_samples": False,  # Don't save samples during training
                "write_out": False,
                "device": device,
                "fewshot_random_seed": 23,
                "random_seed": 23,
                "numpy_random_seed": 23,
                "torch_random_seed": 23,
            }

            # Add multiturn and system instruction when using chat templates with fewshot
            if apply_chat_template and task_cfg.fewshot > 0:
                eval_kwargs["fewshot_as_multiturn"] = True
                logger.info("Using fewshot_as_multiturn=True")

            # Run evaluation
            try:
                res = evaluator.simple_evaluate(**eval_kwargs)
                all_results[task_name] = res
                logger.info(f"✅ Finished {task_name}")
            except Exception as e:
                logger.error(f"❌ Error running {task_name}: {e}")
                all_results[task_name] = {"error": str(e)}
                continue

        # 5. Log results to console and wandb
        logger.info(f"\nEvaluation Results at Step {current_step}:")
        wandb_logs = {}

        for task_name, res in all_results.items():
            if "error" in res:
                logger.info(f"  {task_name}: ERROR - {res['error']}")
                continue

            task_results = res.get("results", {}).get(task_name, {})
            logger.info(f"  {task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {metric}: {value:.4f}")
                    wandb_logs[f"eval/{task_name}/{metric}"] = value

        # Log to wandb
        if state.is_world_process_zero and wandb_logs:
            wandb_logs['eval/step'] = current_step
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(wandb_logs, step=current_step)
                    logger.info("Logged evaluation results to wandb")
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        # 6. Switch model back to training mode
        if self.use_unsloth:
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_training(model)
                logger.info("Switched back to training mode (Unsloth)")
            except Exception as e:
                logger.warning(f"Failed to switch to training mode: {e}")

        # 7. Clear CUDA cache (optional - only helpful if close to OOM)
        # Note: This does NOT free optimizer states, gradients, or model params
        # It only releases cached allocator memory from temporary eval tensors
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache (released temporary eval tensors)")


@hydra.main(config_path="configs", config_name="train_model.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Train model continuously across all milestones.
    """
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    logger.info("Starting Continuous Milestone Training")

    # Set random seed
    if cfg.get("seed") is not None:
        set_seed(int(cfg.seed))

    # Extract configuration
    milestone_cfg = to_dict(cfg.get("milestone", {}))
    prebuilt_dataset_path = milestone_cfg.get("prebuilt_dataset_path")

    if not prebuilt_dataset_path or not os.path.exists(prebuilt_dataset_path):
        raise ValueError(
            f"prebuilt_dataset_path must be set and exist: {prebuilt_dataset_path}\n"
            "Run build_mixed_dataset.py first to create the milestone dataset."
        )

    # Load metadata to get milestone boundaries
    import json
    metadata_path = Path(prebuilt_dataset_path) / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    num_milestones = metadata['num_milestones']
    tokens_per_milestone = metadata['tokens_per_milestone']
    total_tokens = tokens_per_milestone * num_milestones

    logger.info(f"Milestone configuration:")
    logger.info(f"  Num milestones: {num_milestones}")
    logger.info(f"  Tokens per milestone: {tokens_per_milestone:,}")
    logger.info(f"  Total training tokens: {total_tokens:,}")

    # Build model and tokenizer
    builder_cfg = ModelBuilderConfig(**to_dict(cfg.model.builder))
    lora_cfg = LoraAdapterConfig(**to_dict(cfg.model.lora))
    builder = ModelBuilder(builder_cfg, lora_cfg)
    model, tokenizer = builder.build()

    # If training a PT model with SFT, load IT tokenizer for chat template
    model_name = builder_cfg.model_name
    use_sft = not cfg.dataset.get("tokenize", True)
    if use_sft and model_name in PT_TO_IT_TOKENIZER_MAP:
        it_tokenizer_name = PT_TO_IT_TOKENIZER_MAP[model_name]
        it_tokenizer = AutoTokenizer.from_pretrained(it_tokenizer_name, trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template

    # Load the FULL dataset (all milestones together)
    # SFTTrainer will pack it once at the beginning
    logger.info(f"Loading dataset from {prebuilt_dataset_path}")
    from datasets import load_from_disk
    full_dataset = load_from_disk(prebuilt_dataset_path)
    logger.info(f"Loaded {len(full_dataset)} total examples across all milestones")

    # Calculate training steps
    batch_size = cfg.training_args.per_device_train_batch_size
    grad_accum = cfg.training_args.gradient_accumulation_steps
    seq_length = cfg.training_args.get("max_length", 2048)
    world_size = cfg.training_args.get("world_size", 1)

    # Total steps for all milestones
    total_steps = calculate_max_steps(
        target_tokens=total_tokens,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_length=seq_length,
        world_size=world_size
    )

    # Steps per milestone (for evaluation callback)
    steps_per_milestone = calculate_max_steps(
        target_tokens=tokens_per_milestone,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_length=seq_length,
        world_size=world_size
    )

    # Calculate milestone evaluation steps
    milestone_eval_steps = [steps_per_milestone * (i + 1) for i in range(num_milestones)]

    logger.info(f"Training configuration:")
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info(f"  Steps per milestone: {steps_per_milestone}")
    logger.info(f"  Milestone eval steps: {milestone_eval_steps}")

    # Prepare training arguments
    training_kwargs = to_dict(cfg.training_args)
    wandb_cfg = to_dict(cfg.get("wandb"), resolve=False)
    training_kwargs = apply_wandb_config(cfg, training_kwargs, wandb_cfg)

    # Set max_steps to cover all milestones
    training_kwargs['max_steps'] = total_steps
    training_kwargs['save_strategy'] = 'steps'
    training_kwargs['save_steps'] = steps_per_milestone  # Save at each milestone
    training_kwargs['logging_steps'] = 2

    logger.info(f"Full training kwargs: {training_kwargs}")

    # Create trainer
    use_unsloth = cfg.model.builder.get("use_unsloth", False)
    text_field = cfg.dataset.get("text_field", "formatted_text")

    if use_unsloth:
        logger.info("Using UnslothTrainer with packing")

        training_args = UnslothTrainingArguments(
            **training_kwargs,
            packing=cfg.data_collator.get("packing", False),
            packing_strategy=cfg.data_collator.get("packing_strategy", "bfd"),  # NEW: specify strategy
            dataset_text_field=text_field,
        )

        trainer = create_trainer_with_token_stats(
            UnslothTrainer,
            model=model,
            args=training_args,
            train_dataset=full_dataset,
            tokenizer=tokenizer,
        )
    elif use_sft:
        logger.info("Using SFTTrainer with packing")

        training_args = SFTConfig(
            **training_kwargs,
            packing=cfg.data_collator.get("packing", False),
            packing_strategy=cfg.data_collator.get("packing_strategy", "bfd"),  # NEW: specify strategy
            dataset_text_field=text_field,
        )

        trainer = create_trainer_with_token_stats(
            SFTTrainer,
            model=model,
            args=training_args,
            train_dataset=full_dataset,
            processing_class=tokenizer,
        )
    else:
        raise ValueError("This script requires use_sft=True or use_unsloth=True for packing support")

    # Add callbacks
    num_train_epochs = training_kwargs.get('num_train_epochs', 1.0)

    # Scheduler callback for consistent LR across milestones
    scheduler_callback = CosineSchedulerCallback(total_steps, warmup_ratio=0.03)
    epoch_callback = EpochResetCallback(total_steps, num_train_epochs=num_train_epochs)

    # Token statistics callback
    token_stats_callback = TokenStatsCallback(tokenizer, log_every_n_steps=1)

    # Milestone evaluation callback
    # Load evaluation config from evaluate_ro.yaml
    from omegaconf import OmegaConf
    eval_config_path = Path(__file__).parent / "configs" / "evaluate_ro.yaml"
    eval_config = OmegaConf.load(eval_config_path)

    logger.info(f"Loaded evaluation config from {eval_config_path}")
    logger.info(f"Tasks to run: {eval_config.get('tasks_to_run', [])}")

    eval_callback = MilestoneEvalCallback(
        milestone_steps=milestone_eval_steps,
        eval_config=eval_config,
        use_unsloth=use_unsloth,
    )

    trainer.add_callback(token_stats_callback)
    trainer.add_callback(scheduler_callback)
    trainer.add_callback(epoch_callback)
    trainer.add_callback(eval_callback)

    # Start training (continuous across all milestones)
    logger.info("=" * 80)
    logger.info("Starting continuous training across all milestones")
    logger.info("=" * 80)

    trainer.train()

    logger.info("=" * 80)
    logger.info("Continuous milestone training complete!")
    logger.info(f"Total steps: {trainer.state.global_step}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
