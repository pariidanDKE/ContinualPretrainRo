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
from unsloth import unsloth_train
import os
import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
from transformers import set_seed
from dotenv import load_dotenv

from train_utils import (
    prepare_dataset,
    load_milestone_shard,
    to_dict,
    apply_wandb_config,
    calculate_max_steps,
    merge_lora_checkpoint,
    PT_TO_IT_TOKENIZER_MAP,
    setup_file_logging,
)
from trainer_callbacks import (
    EpochResetCallback,
    WSDSchedulerCallback,
    InitialLossCallback
)
from token_stats_callback import create_trainer_with_token_stats, TokenStatsCallback
from model import ModelBuilder, ModelBuilderConfig, LoraAdapterConfig
from trl import SFTConfig, SFTTrainer
from transformers import  AutoTokenizer, Trainer
from unsloth import UnslothTrainer, UnslothTrainingArguments
from train_utils import DebugUnslothTrainer




logger = logging.getLogger(__name__)



# Save original method
original_get_batch_samples = Trainer.get_batch_samples

def debug_get_batch_samples(self, epoch_iterator, num_batches, device):
    logging.info(f"\n{'='*80}")
    logging.info(f"[get_batch_samples] CALLED")
    logging.info(f"{'='*80}")
    logging.info(f"  num_batches requested: {num_batches}")
    logging.info(f"  model_accepts_loss_kwargs: {self.model_accepts_loss_kwargs}")
    logging.info(f"  compute_loss_func: {self.compute_loss_func}")
    
    # Collect batches
    batch_samples = []
    for i in range(num_batches):
        try:
            batch = next(epoch_iterator)
            batch_samples.append(batch)
            logging.info(f"  ✓ Collected batch {i+1}/{num_batches}")
        except StopIteration:
            logging.info(f"  ✗ StopIteration at batch {i+1}/{num_batches}")
            break
    
    logging.info(f"  → Total batches collected: {len(batch_samples)}")
    
    # Check if we should count items
    count_num_items_in_batch = (
        len(batch_samples) > 0
        and "labels" in batch_samples[0]
        and (self.model_accepts_loss_kwargs or self.compute_loss_func is not None)
    )
    
    logging.info(f"\n  Condition checks:")
    logging.info(f"    len(batch_samples) > 0: {len(batch_samples) > 0}")
    if len(batch_samples) > 0:
        logging.info(f"    'labels' in batch_samples[0]: {'labels' in batch_samples[0]}")
    logging.info(f"    model_accepts_loss_kwargs: {self.model_accepts_loss_kwargs}")
    logging.info(f"    compute_loss_func is not None: {self.compute_loss_func is not None}")
    logging.info(f"  → count_num_items_in_batch: {count_num_items_in_batch}")
    
    num_items_in_batch = None
    
    if count_num_items_in_batch:
        logging.info(f"\n  Counting items in batch...")
        try:
            num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            logging.info(f"    Sum of non-padding tokens: {num_items_in_batch}")
        except (TypeError, AttributeError) as e:
            logging.info(f"    Failed to count: {e}")
            pass
        
        if num_items_in_batch is not None:
            logging.info(f"    Before device operations: {num_items_in_batch}")
            
            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()
                logging.info(f"    After gather across devices: {num_items_in_batch}")
            
            import torch
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)
                logging.info(f"    After .to(device): {num_items_in_batch}")
            
            if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                num_items_in_batch = num_items_in_batch.unsqueeze(0)
                logging.info(f"    After unsqueeze (multi-GPU): {num_items_in_batch}")
            
            # Divide by number of devices with the same batch
            if pc := getattr(self.accelerator, "parallelism_config", None):
                num_items_in_batch = num_items_in_batch // pc.non_data_parallel_size
                logging.info(f"    After parallel division: {num_items_in_batch}")
    
    logging.info(f"\n  RETURN:")
    logging.info(f"    batch_samples length: {len(batch_samples)}")
    logging.info(f"    num_items_in_batch: {num_items_in_batch}")
    logging.info(f"{'='*80}\n")
    
    return batch_samples, num_items_in_batch

# Apply monkey patch
Trainer.get_batch_samples = debug_get_batch_samples

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

    logger.info("Starting Milestone Segment Training")

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
    logger.info(f"  Target tokens: {target_tokens:,} (minus remainder)")
    logger.info(f"  Total training tokens (for LR schedule): {total_training_tokens:,}")
    logger.info(f"  Resume from: {resume_checkpoint if resume_checkpoint else 'scratch'}")
    if prebuilt_dataset_path:
        logger.info(f"  Using prebuilt dataset: {prebuilt_dataset_path}")
        logger.info(f"  Current milestone: {current_milestone}")

    # Sync model.builder.max_seq_length with training_args.max_length
    # (UnslothTrainer uses max_seq_length from model config, not max_length from training args)
    builder_dict = to_dict(cfg.model.builder)
    if "max_length" in to_dict(cfg.training_args):
        max_length_override = cfg.training_args.max_length
        builder_dict["max_seq_length"] = max_length_override

    # Build model and tokenizer
    builder_cfg = ModelBuilderConfig(**builder_dict)
    lora_cfg = LoraAdapterConfig(**to_dict(cfg.model.lora))
    builder = ModelBuilder(builder_cfg, lora_cfg)
    model, tokenizer = builder.build()

    # If training a PT model with SFT, load IT tokenizer for chat template
    # Skip this for CPT mode (use_sft=False) where we want raw text training
    model_name = builder_cfg.model_name
    use_sft = cfg.dataset.get("use_sft", True)

    if use_sft and model_name in PT_TO_IT_TOKENIZER_MAP:
        it_tokenizer_name = PT_TO_IT_TOKENIZER_MAP[model_name]
        it_tokenizer = AutoTokenizer.from_pretrained(it_tokenizer_name, trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
        logger.info(f"Loaded chat template from {it_tokenizer_name} for SFT training")
    elif not use_sft:
        logger.info("CPT mode (use_sft=False) - skipping chat template loading")
        # Remove chat_template if it exists (some IT models have it by default)
        tokenizer.chat_template = None



    if prebuilt_dataset_path and os.path.exists(prebuilt_dataset_path):
        # Use prebuilt milestone-tagged dataset
        train_dataset, val_dataset = load_milestone_shard(prebuilt_dataset_path, current_milestone)
    else:
        # Fallback: Build dataset on-the-fly with 90/10 split
        if prebuilt_dataset_path:
            logger.warning(f"Prebuilt dataset path specified but not found: {prebuilt_dataset_path}")
            logger.warning("Falling back to on-the-fly dataset preparation")
        train_dataset, val_dataset = prepare_dataset(cfg, tokenizer, return_validation=True, validation_split=0.1)


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

    # tokens_per_step =  batch_size * grad_accum * world_size
    # samples_needed = max_steps * tokens_per_step 
    # samples_needed = samples_needed + (5 * tokens_per_step)
    # if len(train_dataset) > samples_needed:
    #     logger.info(f"Subsetting train_dataset from {len(train_dataset)} to {samples_needed} samples")
    #     train_dataset = train_dataset.select(range(samples_needed))
    # else:
    #     logger.warning(f"Train dataset has {len(train_dataset)} samples, but need {samples_needed}")
    #     logger.warning("Using full dataset - training may end early")


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
    logger.info(f"  Validation Dataset Length: {len(val_dataset):,}")


    # Prepare training arguments
    training_kwargs = to_dict(cfg.training_args)
    wandb_cfg = to_dict(cfg.get("wandb"), resolve=False)
    training_kwargs = apply_wandb_config(cfg, training_kwargs, wandb_cfg)

    # Set up file logging to output_dir/logs/
    output_dir = training_kwargs.get('output_dir', './outputs')
    run_name = training_kwargs.get('run_name', 'training')
    run_id = os.environ.get('WANDB_RUN_ID', '')
    setup_file_logging(output_dir, run_name=run_name, run_id=run_id)

    training_kwargs['max_steps'] = max_steps
    training_kwargs['save_strategy'] = 'steps'
    training_kwargs['save_steps'] = max_steps  # Save exactly at milestone
    training_kwargs['logging_steps'] = 1 # Log frequently to monitor progress


    # NOTE: Uncomment training args after diagnosing
    # Add evaluation configuration
    # training_kwargs['eval_strategy'] = 'steps'
    # training_kwargs['eval_steps'] = max(1, max_steps - 1)  # Evaluate just before end (avoids WandB finish race)
    # training_kwargs['per_device_eval_batch_size'] = 4 #training_kwargs.get('per_device_eval_batch_size', batch_size)
    # training_kwargs['prediction_loss_only'] = True  # Only compute loss, don't store logits/predictions (prevents OOM)

    # Patch TRL packing to prevent low-density bins that may cause sawtooth loss
    # if cfg.data_collator.get("packing", False):
    #     logger.info("Packing is enabled - applying TRL packing stability patch")
        #patch_trl_packing_for_stability(min_fill_ratio=0.7)

    # Create trainer
    use_unsloth = cfg.model.builder.get("use_unsloth", False)

    if use_unsloth:
        logger.info("Using UnslothTrainer with decoupled learning rates")
        # DataPreprocessor always outputs "formatted_text" when tokenize=False
        dataset_text_field = "formatted_text"

        training_args = UnslothTrainingArguments(
            **training_kwargs,
            packing=cfg.data_collator.get("packing", False),
            dataset_text_field=dataset_text_field,
        )

        trainer = create_trainer_with_token_stats(
            DebugUnslothTrainer,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            #eval_dataset=val_dataset,
            processing_class=tokenizer,
        )
    elif use_sft:
        logger.info("Using SFTTrainer")
        # DataPreprocessor always outputs "formatted_text" when tokenize=False
        dataset_text_field = "formatted_text"

        training_args = SFTConfig(
            **training_kwargs,
            packing=cfg.data_collator.get("packing", False),
            dataset_text_field=dataset_text_field,
        )

        trainer = create_trainer_with_token_stats(
            SFTTrainer,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            #eval_dataset=val_dataset,
            processing_class=tokenizer,
        )
  
    # Use actual max_length from training_args (handles Hydra overrides correctly)
    actual_seq_length = training_args.max_length
    trainer.step_size = batch_size * 1 * actual_seq_length * world_size # GradAccum is handled by callback aggregation logic(!)
    num_train_epochs = training_kwargs.get('num_train_epochs', 1.0)


    # Add callbacks for continuous epoch and LR tracking across milestones
    scheduler_callback = WSDSchedulerCallback(total_steps, warmup_ratio=training_args.warmup_ratio)
    epoch_callback = EpochResetCallback(total_steps, num_train_epochs=num_train_epochs)
    token_stats_callback = TokenStatsCallback(tokenizer, log_every_n_steps=1)
    #initial_loss_callback = InitialLossCallback(num_eval_batches=1)

    #trainer.add_callback(initial_loss_callback)
    trainer.add_callback(token_stats_callback)
    trainer.add_callback(scheduler_callback)
    trainer.add_callback(epoch_callback)

    # Start training (resume if checkpoint provided)
    logger.info("=" * 80)
    if resume_checkpoint:
        logger.info(f"Resuming training from: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    #unsloth_train(trainer,resume_from_checkpoint=resume_checkpoint)

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
