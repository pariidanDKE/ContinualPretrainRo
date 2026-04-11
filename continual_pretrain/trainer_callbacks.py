"""
Trainer callbacks for milestone-based training.

This module contains specialized callbacks for managing continuous learning
rate scheduling, epoch tracking, and optimizer diagnostics across milestone
training boundaries.
"""
from unsloth import UnslothTrainer
import logging
from transformers import TrainerCallback
from transformers.integrations import WandbCallback
import time

logger = logging.getLogger(__name__)

# # =============================================================================
# # Evaluate Trigger per milestone
# # =============================================================================
class EvaluationTriggerCallback(TrainerCallback):
    """
    This callback will decide whether to run evals whenever a milestone is reached and whether to stop training when num_milestones are reached
    """
    def __init__(self, target_milestone_tokens : int, num_milestones: int, trainer: UnslothTrainer, do_evaluate: bool = True):
        self.target_milestone_tokens = target_milestone_tokens
        self.num_milestones = num_milestones
        self.do_evaluate = do_evaluate

        self.milestone_counter = 0
        self.total_token_counter = 0
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        milestone_token_counter = self.trainer.milestone_tokens
        self.total_token_counter += milestone_token_counter

        print(f"[step {state.global_step}] milestone_tokens: {self.trainer.milestone_tokens:,} | total_tokens_seen: {self.trainer.total_tokens_seen:,}")

        if self.trainer.milestone_tokens >= self.target_milestone_tokens:
            # Only trigger evaluation if do_evaluate is True
            if self.do_evaluate:
                control.should_evaluate = True # includes running benchmarks
            control.should_save = True # save every milestone

            self.trainer.milestone_tokens = 0
            self.milestone_counter+=1
            logging.info(f"Reached milestone {self.milestone_counter}! Total tokens: {self.total_token_counter}")

            if self.milestone_counter >= self.num_milestones:
                control.should_training_stop = True
                logging.info(f"Reached final milestone ({self.milestone_counter})! Total tokens: {self.total_token_counter}")

        return control


class EpochEndCheckpointCallback(TrainerCallback):
    """Saves a checkpoint at the end of each epoch."""

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control


class WSDDecayCheckpointCallback(TrainerCallback):
    """
    Saves a checkpoint exactly once when the WSD scheduler transitions into the
    decay phase (i.e. total_tokens_seen crosses total_tokens - decay_tokens).

    This gives a clean stable-phase snapshot you can resume from if you later
    want to extend training without re-running the warmup/stable phases.

    Args:
        total_tokens:  Total training tokens across all milestones
                       (num_milestones * tokens_per_milestone).
        decay_tokens:  Number of tokens in the decay phase
                       (same value passed to lr_scheduler_kwargs.num_decay_tokens).
        trainer:       The MilestoneTrainer instance (needs .total_tokens_seen).
    """

    def __init__(self, total_tokens: int, decay_tokens: int, trainer):
        self.decay_start_tokens = total_tokens - decay_tokens
        self.trainer = trainer
        self._saved = False

    def on_step_end(self, args, state, control, **kwargs):
        if self._saved:
            return control

        if self.trainer.total_tokens_seen >= self.decay_start_tokens:
            logger.info(
                f"WSD decay phase started at {self.trainer.total_tokens_seen:,} tokens "
                f"(threshold={self.decay_start_tokens:,}). Saving pre-decay checkpoint."
            )
            control.should_save = True
            self._saved = True

        return control


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

