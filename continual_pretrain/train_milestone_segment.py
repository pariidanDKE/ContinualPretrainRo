"""
Train for a single milestone segment (e.g., 100M tokens).

This script trains until a specific token milestone is reached, then exits.
It can resume from a previous checkpoint to continue training.

Usage:
    # Train from scratch to 100M tokens
    python train_milestone_segment.py milestone.target_tokens=100000000

    # Resume from checkpoint and train to 200M tokens
    python train_milestone_segment.py \
        milestone.target_tokens=200000000 \
        milestone.resume_from_checkpoint=outputs/checkpoint-6104
"""
import os
import unsloth
import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
from transformers import set_seed
from dotenv import load_dotenv

from train_utils import (
    prepare_dataset,
    load_milestone_shard,
    build_collator,
    to_dict,
    apply_wandb_config,
    calculate_max_steps,
    merge_lora_checkpoint,
    PT_TO_IT_TOKENIZER_MAP,
    SchedulerFixCallback
)
from model import ModelBuilder, ModelBuilderConfig, LoraAdapterConfig
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments, Trainer, AutoTokenizer
from data_module import DataPreprocessor

logger = logging.getLogger(__name__)




@hydra.main(config_path="configs", config_name="train_model.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Train model for a single milestone segment.
    """
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    logger.info("=" * 80)
    logger.info("Starting Milestone Segment Training")
    logger.info("=" * 80)

    # Set random seed
    if cfg.get("seed") is not None:
        set_seed(int(cfg.seed))

    # Extract milestone configuration
    milestone_cfg = to_dict(cfg.get("milestone", {}))
    target_tokens = milestone_cfg.get("target_tokens")
    total_training_tokens = milestone_cfg.get("total_training_tokens") or target_tokens
    resume_checkpoint = milestone_cfg.get("resume_from_checkpoint")
    prebuilt_dataset_path = milestone_cfg.get("prebuilt_dataset_path")
    current_milestone = milestone_cfg.get("current_milestone", 0)

    logger.info(f"Milestone configuration:")
    logger.info(f"  Target tokens: {target_tokens:,}")
    logger.info(f"  Total training tokens (for LR schedule): {total_training_tokens:,}")
    logger.info(f"  Resume from: {resume_checkpoint if resume_checkpoint else 'scratch'}")
    if prebuilt_dataset_path:
        logger.info(f"  Using prebuilt dataset: {prebuilt_dataset_path}")
        logger.info(f"  Current milestone: {current_milestone}")

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
        logger.info(f"Loading IT tokenizer {it_tokenizer_name} for chat template")
        it_tokenizer = AutoTokenizer.from_pretrained(it_tokenizer_name, trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template

    # Prepare dataset
    if prebuilt_dataset_path and os.path.exists(prebuilt_dataset_path):
        # Use prebuilt milestone-tagged dataset
        logger.info(f"Loading prebuilt milestone dataset from: {prebuilt_dataset_path}")
        dataset = load_milestone_shard(prebuilt_dataset_path, current_milestone)
    else:
        # Fallback: Build dataset on-the-fly
        if prebuilt_dataset_path:
            logger.warning(f"Prebuilt dataset path specified but not found: {prebuilt_dataset_path}")
            logger.warning("Falling back to on-the-fly dataset preparation")
        dataset = prepare_dataset(cfg, tokenizer)

    # Calculate max_steps for this milestone
    batch_size = cfg.training_args.per_device_train_batch_size
    grad_accum = cfg.training_args.gradient_accumulation_steps
    seq_length = cfg.training_args.get("max_length", 2048)
    world_size = cfg.training_args.get("world_size", 1)
    logger.info(f"  Tokens per step: batch_size{batch_size * grad_accum * seq_length * world_size:,}")

    max_steps = calculate_max_steps(
        target_tokens=target_tokens,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_length=seq_length,
        world_size=world_size
    )

    # Calculate total steps for scheduler (ensures consistent LR across milestones)
    total_steps = calculate_max_steps(
        target_tokens=total_training_tokens,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_length=seq_length,
        world_size=world_size
    )
    logger.info(
        f"Step Calculation Details:\n"
        f"  - Target Tokens: {total_training_tokens}\n"
        f"  - Batch Size: {batch_size}\n"
        f"  - Gradient Accumulation Steps: {grad_accum}\n"
        f"  - Sequence Length: {seq_length}\n"
        f"  - World Size (GPUs): {world_size}\n"
        f"  - Resulting Total Steps: {total_steps}"
    )

    logger.info(f"Token calculation:")
    logger.info(f"  Tokens per step: {batch_size * grad_accum * seq_length * world_size:,}")
    logger.info(f"  Max steps (current milestone): {max_steps:,}")
    logger.info(f"  Total steps (for scheduler): {total_steps:,}")

    # Prepare training arguments
    training_kwargs = to_dict(cfg.training_args)
    wandb_cfg = to_dict(cfg.get("wandb"), resolve=False)
    training_kwargs = apply_wandb_config(cfg, training_kwargs, wandb_cfg)

    # Override max_steps and save settings for milestone training
    training_kwargs['max_steps'] = max_steps
    training_kwargs['save_strategy'] = 'steps'
    training_kwargs['save_steps'] = max_steps  # Save exactly at milestone
    training_kwargs['logging_steps'] = 1  # Log frequently to monitor progress


    # Create trainer
    if use_sft:
        logger.info("Using SFTTrainer with chat template")
        text_field = cfg.dataset.get("text_field", "formatted_text")

        if not tokenizer.chat_template and model_name not in PT_TO_IT_TOKENIZER_MAP:
            preprocessor = DataPreprocessor(
                tokenizer=tokenizer,
                text_field=text_field,
                add_bos_eos=cfg.dataset.get("add_bos_eos", True),
                use_sft_config=True
            )
            chat_template = preprocessor.get_chat_template()
            tokenizer.chat_template = chat_template

        training_args = SFTConfig(
            **training_kwargs,
            packing=cfg.data_collator.get("packing", False),
            dataset_text_field=text_field,
            #max_length=seq_length,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        logger.info("Using regular Trainer with custom data collator")
        collator = build_collator(cfg, tokenizer)
        training_args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=collator,
        )


    scheduler_callback = SchedulerFixCallback(total_steps)
    trainer.add_callback(scheduler_callback)
    

    # Start training (resume if checkpoint provided)
    logger.info("=" * 80)
    if resume_checkpoint:
        logger.info(f"Resuming training from: {resume_checkpoint}")
        logger.info(f"Will train until step: {max_steps}")
    else:
        logger.info(f"Starting training from scratch")
        logger.info(f"Will train for: {max_steps} steps ({target_tokens:,} tokens)")
    logger.info("=" * 80)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info("=" * 80)
    logger.info("Milestone segment training complete!")
    logger.info(f"Checkpoint saved to: {training_args.output_dir}/checkpoint-{max_steps}")
    logger.info(f"Total steps: {trainer.state.global_step}")

    # Verify we reached the target
    if trainer.state.global_step < max_steps:
        logger.error(f"Training stopped early! Expected {max_steps} steps, got {trainer.state.global_step}")
        logger.error("Dataset exhausted before reaching milestone. Consider: smaller milestones, more data, or multiple epochs.")
        raise RuntimeError(f"Insufficient data: reached {trainer.state.global_step}/{max_steps} steps")

    # Merge LoRA weights for evaluation if using PEFT
    merged_output_dir = milestone_cfg.get("merged_output_dir")

    if merged_output_dir:
        logger.info("=" * 80)
        merged_path = Path(merged_output_dir) / f"checkpoint-{max_steps}"
        merge_lora_checkpoint(trainer, merged_path, tokenizer)

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
