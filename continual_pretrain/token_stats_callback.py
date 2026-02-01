"""
Token Statistics Callback and Trainer Mixin for tracking real vs padding tokens.

This module provides tools to diagnose packing efficiency and token density issues
during training by logging the proportion of real tokens (non-padding, non-EOS) in each batch.

The statistics are computed for EVERY batch that goes through the model during training.
With gradient accumulation, we aggregate stats across all micro-batches within an optimizer step.
"""

import logging
import time
import torch
from typing import Dict, Any, Optional
from transformers import TrainerCallback
from transformers.integrations import WandbCallback
from collections import defaultdict
import wandb

logger = logging.getLogger(__name__)


class TokenStatsCallback(WandbCallback):
    """
    Callback to log token statistics (real vs padding vs EOS tokens) during training.

    This callback helps diagnose issues with packing efficiency by tracking:
    - Proportion of real tokens (non-padding, excluding EOS)
    - Proportion of padding tokens
    - Number of EOS tokens (indicates number of distinct sequences)
    - Effective token density (real tokens / total possible tokens)

    Statistics are computed for EVERY batch during training and aggregated
    across gradient accumulation steps before logging.

    Usage:
        callback = TokenStatsCallback(tokenizer, log_every_n_steps=10)
        trainer.add_callback(callback)
    """

    def __init__(self, tokenizer, log_every_n_steps: int = 10):
        """
        Args:
            tokenizer: The tokenizer containing pad_token_id and eos_token_id
            log_every_n_steps: How often to log aggregated statistics (default: 10)
        """
        super().__init__()  # Initialize parent WandbCallback
        self.tokenizer = tokenizer
        self.log_every_n_steps = log_every_n_steps
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.step_start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each optimizer step. Start timing."""
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each OPTIMIZER step (after gradient accumulation).
        Logs aggregated token statistics from all micro-batches in this step.
        """
        # Calculate throughput (tokens per second) for this step
        tokens_per_second = None
        elapsed_time = None

        if self.step_start_time is not None:
            elapsed_time = time.time() - self.step_start_time
            # Get total tokens from accumulated stats if available
            if hasattr(state, 'accumulated_batch_stats') and state.accumulated_batch_stats:
                total_tokens = state.accumulated_batch_stats.get('total_tokens', 0)
                if elapsed_time > 0 and total_tokens > 0:
                    tokens_per_second = total_tokens / elapsed_time

        # Check if we should log this step
        should_log = state.global_step % self.log_every_n_steps == 0

        # Only log at specified intervals
        if should_log:
            # Check if trainer has stored batch statistics for us
            if not hasattr(state, 'accumulated_batch_stats') or not state.accumulated_batch_stats:
                if state.global_step == self.log_every_n_steps:
                    logger.warning(
                        "TokenStatsCallback: No batch statistics found in trainer state. "
                        "Make sure you're using a trainer with TokenStatsTrainerMixin!"
                    )
            else:
                stats = state.accumulated_batch_stats

                # Prepare throughput info for logging
                throughput_info = ""
                if tokens_per_second is not None:
                    throughput_info = f"\n  Throughput: {tokens_per_second:,.0f} tok/s (elapsed: {elapsed_time:.2f}s)"

                # Log to console with detailed breakdown
                logger.info(
                    f"[Step {state.global_step}] Token Statistics across {stats['num_batches']} batches:\n"
                    f"  Padding tokens: {stats['padding_tokens']:,} ({stats['padding_ratio']:.2%})\n"
                    f"  EOS tokens:     {stats['eos_count']:,} (distinct sequences)\n"
                    f"  Token density:  {stats['token_density']:.2%}"
                    f"{throughput_info}"
                )

                # Log to WandB via WandbCallback integration
                if state.is_world_process_zero:
                    logs = {
                        "token_stats/padding_ratio": stats['padding_ratio'],
                        "token_stats/eos_count": stats['eos_count'],
                        "token_stats/total_tokens": stats['real_tokens'],
                        "token_stats/tokens_per_second" : tokens_per_second if tokens_per_second else 0,
                        "token_stats/elapsed_time" : elapsed_time if elapsed_time else 0
                    }

                    # Log directly to WandB using the WandbCallback's _wandb attribute
                    if hasattr(self, '_wandb') and self._wandb is not None:
                        self._wandb.log(logs) #, step=state.global_step)

        # CRITICAL: Clear accumulated stats after EVERY step, not just when logging
        # This prevents accumulation across multiple logging intervals
        state.accumulated_batch_stats = None


def compute_token_statistics(input_ids: torch.Tensor, step_size: int, pad_token_id: int, eos_token_id: int) -> Dict[str, int]:
    """
    Compute raw token counts from a batch of input_ids.

    This function is called for EVERY batch that goes through the model,
    so you get statistics for all incoming training data.

    Args:
        input_ids: Tensor of shape (batch_size, seq_length) - the actual training batch
        step_size: Total number of token positions in this batch
        pad_token_id: ID of the padding token
        eos_token_id: ID of the EOS token

    Returns:
        Dictionary containing raw counts:
        - total_tokens: Total number of token positions in this batch
        - real_tokens: Number of non-padding tokens (including EOS)
        - padding_tokens: Number of padding tokens
        - eos_count: Number of EOS tokens (= number of distinct sequences)
    """
    # Flatten input_ids for easier counting
    input_ids_flat = input_ids.view(-1)

    real_tokens = (input_ids_flat != pad_token_id).sum().item()
    padding_tokens = (input_ids_flat == pad_token_id).sum().item()
    eos_count = (input_ids_flat == eos_token_id).sum().item()

    # Real tokens = total - padding (EOS counts as a real token)
    padding_tokens = step_size - real_tokens

    return {
        'total_tokens': step_size,
        'real_tokens': real_tokens,
        'padding_tokens': padding_tokens,
        'eos_count': eos_count,
    }


def aggregate_token_statistics(accumulated_stats: Optional[Dict], new_stats: Dict[str, int]) -> Dict[str, Any]:
    """
    Aggregate token statistics across multiple batches (for gradient accumulation).

    Args:
        accumulated_stats: Previously accumulated stats (or None for first batch)
        new_stats: Statistics from the current batch

    Returns:
        Updated accumulated statistics with ratios computed
    """
    if accumulated_stats is None:
        accumulated_stats = {
            'total_tokens': 0,
            'real_tokens': 0,
            'padding_tokens': 0,
            'eos_count': 0,
            'num_batches': 0,
        }

    # Accumulate raw counts
    accumulated_stats['total_tokens'] += new_stats['total_tokens']
    accumulated_stats['real_tokens'] += new_stats['real_tokens']
    accumulated_stats['padding_tokens'] += new_stats['padding_tokens']
    accumulated_stats['eos_count'] += new_stats['eos_count']
    accumulated_stats['num_batches'] += 1

   

    # Compute ratios from accumulated totals
    total = accumulated_stats['total_tokens']
    accumulated_stats['real_token_ratio'] = accumulated_stats['real_tokens'] / total if total > 0 else 0.0
    accumulated_stats['padding_ratio'] = accumulated_stats['padding_tokens'] / total if total > 0 else 0.0
    accumulated_stats['token_density'] = accumulated_stats['real_tokens'] / total if total > 0 else 0.0

    return accumulated_stats


class TokenStatsTrainerMixin:
    """
    Mixin class to add token statistics tracking to any Trainer.

    This mixin overrides compute_loss to capture and analyze EVERY batch
    that comes from the dataloader during training. Statistics are aggregated
    across gradient accumulation steps.

    Usage:
        class CustomTrainer(TokenStatsTrainerMixin, Trainer):
            pass

        trainer = CustomTrainer(...)
        callback = TokenStatsCallback(tokenizer)
        trainer.add_callback(callback)
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Override compute_loss to capture batch statistics.

        This method is called for EVERY batch during training:
        - With gradient_accumulation_steps=1: called once per optimizer step
        - With gradient_accumulation_steps=N: called N times per optimizer step

        We accumulate statistics across all micro-batches within each optimizer step.
        """

        if model.training:
            output_parent = super().compute_loss(model, inputs, return_outputs = return_outputs, **kwargs)
            loss = output_parent[0] if return_outputs else output_parent

            # Log batch shape
            batch_size, seq_len = inputs['input_ids'].shape
            logger.info(f"Batch shape: ({batch_size}, {seq_len})")

            batch_stats = compute_token_statistics(
                inputs['input_ids'],
                step_size = self.step_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Aggregate with other micro-batches in this optimizer step
            if not hasattr(self.state, 'accumulated_batch_stats'):
                self.state.accumulated_batch_stats = None

            self.state.accumulated_batch_stats = aggregate_token_statistics(
                self.state.accumulated_batch_stats,
                batch_stats
            )
     
            return (loss, output_parent[1]) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


# Convenience function to create a trainer with token stats tracking
def create_trainer_with_token_stats(trainer_class, *args, **kwargs):
    """
    Create a trainer instance with token statistics tracking enabled.

    This wraps any trainer class (Trainer, SFTTrainer, UnslothTrainer, etc.)
    to add token statistics tracking for every training batch.

    Args:
        trainer_class: The base trainer class (Trainer, SFTTrainer, UnslothTrainer, etc.)
        *args, **kwargs: Arguments to pass to the trainer constructor

    Returns:
        Trainer instance with TokenStatsTrainerMixin applied

    Example:
        from transformers import Trainer
        from token_stats_callback import create_trainer_with_token_stats, TokenStatsCallback

        trainer = create_trainer_with_token_stats(
            Trainer,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            ...
        )

        callback = TokenStatsCallback(tokenizer, log_every_n_steps=10)
        trainer.add_callback(callback)
        trainer.train()
    """
    # Create a new class that inherits from both the mixin and the base trainer
    # The mixin must come first to properly override compute_loss
    class TrainerWithTokenStats(TokenStatsTrainerMixin, trainer_class):
        pass

    return TrainerWithTokenStats(*args, **kwargs)
