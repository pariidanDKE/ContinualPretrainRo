"""
Trainer callbacks for milestone-based training.

This module contains specialized callbacks for managing continuous learning
rate scheduling, epoch tracking, and optimizer diagnostics across milestone
training boundaries.
"""
import os as _os
if _os.environ.get('USE_UNSLOTH', '1') != '0':
    from unsloth import UnslothTrainer
else:
    from trl import SFTTrainer as UnslothTrainer
import logging
from transformers import TrainerCallback
from transformers.integrations import WandbCallback
import time
import csv
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler, record_function

logger = logging.getLogger(__name__)


# =============================================================================
# PyTorch Profiler Callback
# =============================================================================
class ProfilerCallback(TrainerCallback):
    """
    Profiles a single window of training steps using PyTorch's profiler.

    Schedule: wait=10, warmup=1, active=10, repeat=1
    The profiler runs from step 10 through step 21, then exits cleanly.
    """

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._output_dir = args.output_dir
        self._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            profile_memory=True,  # memory events + stacks = OOM; CSV doesn't use them
            with_modules=True,
            with_stack=False,
        )
        self._profiler.__enter__()
        self._done = False
        self._hooks = []
        if model is not None:
            self._register_attention_hooks(model)

    def _register_attention_hooks(self, model):
        for _name, module in model.named_modules():
            if "attention" in type(module).__name__.lower():
                self._hooks.append(module.register_forward_pre_hook(self._attn_pre_hook))
                self._hooks.append(module.register_forward_hook(self._attn_post_hook))

    @staticmethod
    def _attn_pre_hook(module, input):
        module._rf_ctx = record_function("attention")
        module._rf_ctx.__enter__()

    @staticmethod
    def _attn_post_hook(module, input, output):
        if hasattr(module, "_rf_ctx"):
            module._rf_ctx.__exit__(None, None, None)
            del module._rf_ctx

    def _trace_handler(self, prof):
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        tensorboard_trace_handler(self._output_dir)(prof)
        self._export_kernel_csv(prof)

    def _export_kernel_csv(self, prof):
        try:
            rows = []
            for evt in prof.profiler.kineto_results.events():
                if evt.device_type().value != 1:  # 1 = CUDA kernel
                    continue
                rows.append({
                    "name": evt.name(),
                    "duration_us": evt.duration_ns() / 1000,
                    "blocks_per_sm": evt.blocks_per_sm(),
                    "occupancy": evt.estimated_occupancy(),
                })

            if not rows:
                return

            csv_path = _os.path.join(self._output_dir, "kernel_stats.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["name", "duration_us", "blocks_per_sm", "occupancy"])
                writer.writeheader()
                writer.writerows(rows)
        except Exception:
            pass

    def on_step_end(self, args, state, control, **kwargs):
        if self._done:
            return
        self._profiler.step()
        if state.global_step >= 5:
            self._profiler.__exit__(None, None, None)
            self._done = True
            control.should_training_stop = True
        

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self._done:
            self._profiler.__exit__(None, None, None)
            self._done = True
        for h in getattr(self, "_hooks", []):
            h.remove()
        self._hooks = []


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

        if self.trainer.milestone_tokens >= self.target_milestone_tokens:
            # Only trigger evaluation if do_evaluate is True
            if self.do_evaluate:
                control.should_evaluate = True # includes running benchmarks
            if args.save_strategy != "no":
                control.should_save = True # save every milestone

            self.trainer.milestone_tokens = 0
            self.milestone_counter+=1
            logging.info(f"Reached milestone {self.milestone_counter}! Total tokens: {self.total_token_counter}")

            if self.milestone_counter >= self.num_milestones:
                control.should_training_stop = True
                logging.info(f"Reached final milestone ({self.milestone_counter})! Total tokens: {self.total_token_counter}")

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

        tok/s = real_tokens / step_elapsed_time  (mirrors speed_metrics from transformers)
        """
        elapsed_time = time.time() - self.step_start_time if self.step_start_time is not None else None

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
                real_tokens = stats.get('real_tokens', 0)
                # Mirror speed_metrics: tokens_per_second = num_tokens / runtime
                tokens_per_second = (real_tokens / elapsed_time) if (elapsed_time and real_tokens) else 0

                # Log to WandB via WandbCallback integration
                if state.is_world_process_zero:
                    logs = {
                        "token_stats/padding_ratio": stats['padding_ratio'],
                        "token_stats/eos_count": stats['eos_count'],
                        "token_stats/total_tokens": real_tokens,
                        "token_stats/tokens_per_second": round(tokens_per_second, 3),
                        "token_stats/elapsed_time": round(elapsed_time, 4) if elapsed_time else 0,
                    }

                    # Log directly to WandB using the WandbCallback's _wandb attribute
                    if hasattr(self, '_wandb') and self._wandb is not None:
                        self._wandb.log(logs) #, step=state.global_step)

        # CRITICAL: Clear accumulated stats after EVERY step, not just when logging
        # This prevents accumulation across multiple logging intervals
        state.accumulated_batch_stats = None

