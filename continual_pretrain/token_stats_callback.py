"""
Token Statistics Callback and Trainer Mixin for tracking real vs padding tokens.

This module provides tools to diagnose packing efficiency and token density issues
during training by logging the proportion of real tokens (non-padding, non-EOS) in each batch.

The statistics are computed for EVERY batch that goes through the model during training.
With gradient accumulation, we aggregate stats across all micro-batches within an optimizer step.
"""

import logging
import torch
from typing import Dict, Any, Optional
from transformers import TrainerCallback
from collections import defaultdict

logger = logging.getLogger(__name__)


class TokenStatsCallback(TrainerCallback):
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
        self.tokenizer = tokenizer
        self.log_every_n_steps = log_every_n_steps
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each OPTIMIZER step (after gradient accumulation).
        Logs aggregated token statistics from all micro-batches in this step.
        """
        # Only log at specified intervals
        if state.global_step % self.log_every_n_steps != 0:
            return

        # Check if trainer has stored batch statistics for us
        if not hasattr(state, 'accumulated_batch_stats') or not state.accumulated_batch_stats:
            if state.global_step == self.log_every_n_steps:
                logger.warning(
                    "TokenStatsCallback: No batch statistics found in trainer state. "
                    "Make sure you're using a trainer with TokenStatsTrainerMixin!"
                )
            return

        stats = state.accumulated_batch_stats

        # Prepare dataset source proportion info
        source_info = ""
        if stats.get('source_counts'):
            total_examples = sum(stats['source_counts'].values())
            source_lines = []
            for source, count in sorted(stats['source_counts'].items(), key=lambda x: x[1], reverse=True):
                proportion = count / total_examples if total_examples > 0 else 0.0
                source_lines.append(f"    {source}: {count} examples ({proportion:.2%})")
            if source_lines:
                source_info = "\n  Dataset sources:\n" + "\n".join(source_lines)

        # Log to console with detailed breakdown
        logger.info(
            f"[Step {state.global_step}] Token Statistics across {stats['num_batches']} batches:\n"
            f"  Real tokens:    {stats['real_tokens']:,} / {stats['total_tokens']:,} ({stats['real_token_ratio']:.2%})\n"
            f"  Padding tokens: {stats['padding_tokens']:,} ({stats['padding_ratio']:.2%})\n"
            f"  EOS tokens:     {stats['eos_count']:,} (distinct sequences)\n"
            f"  Token density:  {stats['token_density']:.2%}"
            f"{source_info}"
        )

        # Log to WandB/TensorBoard via trainer's logging system
        if state.is_world_process_zero:
            logs = {
                "token_stats/real_token_ratio": stats['real_token_ratio'],
                "token_stats/padding_ratio": stats['padding_ratio'],
                "token_stats/eos_count": stats['eos_count'],
                "token_stats/token_density": stats['token_density'],
                "token_stats/total_tokens": stats['total_tokens'],
                "token_stats/real_tokens": stats['real_tokens'],
                "token_stats/num_batches": stats['num_batches'],
            }
            # Store for next on_log call
            if not hasattr(state, 'token_stats_logs'):
                state.token_stats_logs = {}
            state.token_stats_logs.update(logs)

        # Clear accumulated stats for next step
        state.accumulated_batch_stats = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when trainer logs metrics. Add our token stats to the logs.
        """
        if hasattr(state, 'token_stats_logs') and state.token_stats_logs:
            if logs is not None:
                logs.update(state.token_stats_logs)
            # Clear after logging
            state.token_stats_logs = {}


def compute_token_statistics(input_ids: torch.Tensor, step_size: int, pad_token_id: int, eos_token_id: int, ds_source=None) -> Dict[str, int]:
    """
    Compute raw token counts from a batch of input_ids.

    This function is called for EVERY batch that goes through the model,
    so you get statistics for all incoming training data.

    Args:
        input_ids: Tensor of shape (batch_size, seq_length) - the actual training batch
        step_size: Total number of token positions in this batch
        pad_token_id: ID of the padding token
        eos_token_id: ID of the EOS token
        ds_source: Optional list of dataset sources for each example in the batch

    Returns:
        Dictionary containing raw counts:
        - total_tokens: Total number of token positions in this batch
        - real_tokens: Number of non-padding tokens (including EOS)
        - padding_tokens: Number of padding tokens
        - eos_count: Number of EOS tokens (= number of distinct sequences)
        - source_counts: Dict mapping dataset source -> count of examples
    """
    # Flatten input_ids for easier counting
    input_ids_flat = input_ids.view(-1)

    real_tokens = (input_ids_flat != pad_token_id).sum().item()
    padding_tokens = (input_ids_flat == pad_token_id).sum().item()
    eos_count = (input_ids_flat == eos_token_id).sum().item()

    # Real tokens = total - padding (EOS counts as a real token)
    padding_tokens = step_size - real_tokens

    # Count dataset sources if provided
    source_counts = {}
    if ds_source is not None:
        for source in ds_source:
            source_counts[source] = source_counts.get(source, 0) + 1

    return {
        'total_tokens': step_size,
        'real_tokens': real_tokens,
        'padding_tokens': padding_tokens,
        'eos_count': eos_count,
        'source_counts': source_counts,
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
            'source_counts': defaultdict(int),
        }

    # Accumulate raw counts
    accumulated_stats['total_tokens'] += new_stats['total_tokens']
    accumulated_stats['real_tokens'] += new_stats['real_tokens']
    accumulated_stats['padding_tokens'] += new_stats['padding_tokens']
    accumulated_stats['eos_count'] += new_stats['eos_count']
    accumulated_stats['num_batches'] += 1

    # Accumulate source counts
    if 'source_counts' not in accumulated_stats:
        accumulated_stats['source_counts'] = defaultdict(int)
    for source, count in new_stats.get('source_counts', {}).items():
        accumulated_stats['source_counts'][source] += count

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
        # Only track statistics during TRAINING, not evaluation

      


        if model.training:
            output_parent = super().compute_loss(model, inputs, return_outputs = return_outputs, **kwargs)
            loss = output_parent[0] if return_outputs else output_parent

            # Log batch shape
            batch_size, seq_len = inputs['input_ids'].shape
            logger.info(f"Batch shape: ({batch_size}, {seq_len})")

            # Extract ds_source if available in the batch
            ds_source = inputs.get('ds_source', None)

            batch_stats = compute_token_statistics(
                inputs['input_ids'],
                step_size = self.step_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                ds_source=ds_source
            )

            # Aggregate with other micro-batches in this optimizer step
            if not hasattr(self.state, 'accumulated_batch_stats'):
                self.state.accumulated_batch_stats = None

            self.state.accumulated_batch_stats = aggregate_token_statistics(
                self.state.accumulated_batch_stats,
                batch_stats
            )

            real = batch_stats['real_tokens']
            total = batch_stats['total_tokens']
            
            # original_loss = loss
            # fill_ratio = real / total
            # scale_factor = torch.tensor(fill_ratio, device = loss.device, dtype = loss.dtype)
            
            
            # #loss = scale_factor * loss

            # # Try different approach of skipping steps which have different values
            # if fill_ratio < 0.9:
            #     loss = loss * 0.0 
            #     # Optional: Add a flag to your stats so you know you skipped it
            #     batch_stats['skipped_step'] = True
            #     print(f"\n[CRITICAL SKIP] Rogue batch detected ({real} tokens). Zeroing gradients for this step.")


            # logging.info(f"[TOKEN CALLBACK] Altered loss form {original_loss} to {loss}")

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
