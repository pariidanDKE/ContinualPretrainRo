"""
Token Statistics Callback and Trainer Mixin for tracking real vs padding tokens.

This module provides tools to diagnose packing efficiency and token density issues
during training by logging the proportion of real tokens (non-padding, non-EOS) in each batch.

The statistics are computed for EVERY batch that goes through the model during training.
With gradient accumulation, we aggregate stats across all micro-batches within an optimizer step.
"""

import logging
import math
import torch
import time
from typing import Dict, Any, Optional, Union
from torch.utils.data import Dataset
from transformers.trainer_utils import speed_metrics
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import trainer
from torch.optim.lr_scheduler import LambdaLR


logger = logging.getLogger(__name__)

    # 'num_milestones': num_milestones,
    # 'tokens_per_milestone' : tokens_per_milestone,
class MilestoneTrainerMixin:
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

    def __init__(self, benchmark_evaluation_cfg : dict, num_milestones: int, tokens_per_milestone: int, run_benchmarks: bool, resume_tokens_seen: int = 0, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.milestone_tokens = 0
        self.total_tokens_seen = resume_tokens_seen  # cumulative across all milestones, never reset
        self.benchmark_evaluation_cfg = benchmark_evaluation_cfg
        self.num_milestones = num_milestones
        self.tokens_per_milestone = tokens_per_milestone
        self.run_benchmarks = run_benchmarks

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Override compute_loss to capture batch statistics.

        This method is called for EVERY batch during training:
        - With gradient_accumulation_steps=1: called once per optimizer step
        - With gradient_accumulation_steps=N: called N times per optimizer step

        We accumulate statistics across all micro-batches within each optimizer step.
        """

        if model.training:
            output_parent = super().compute_loss(model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch, **kwargs)
            loss = output_parent[0] if return_outputs else output_parent

            # Log batch shape

            batch_stats = compute_token_statistics(
                inputs['input_ids'],
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
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
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch = num_items_in_batch, **kwargs)
        

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        total_tokens = self.num_milestones * self.tokens_per_milestone

        # For non-WSD schedulers fall back to padded step estimate (original behaviour)
        if self.args.lr_scheduler_type != "warmup_stable_decay":
            tokens_per_step = (self.args.gradient_accumulation_steps * self.args.world_size *
                               self.args.max_length * self.args.train_batch_size)
            max_steps = math.ceil(total_tokens / tokens_per_step)
            return super().create_scheduler(num_training_steps=max_steps, optimizer=optimizer)

        # --- Token-aware WSD scheduler ---
        # warmup: prefer ratio over steps (steps would need the same padded conversion we're avoiding)
        if self.args.warmup_ratio > 0:
            warmup_tokens = int(self.args.warmup_ratio * total_tokens)
        else:
            tokens_per_step_padded = (self.args.gradient_accumulation_steps * self.args.world_size *
                                      self.args.max_length * self.args.train_batch_size)
            warmup_tokens = self.args.warmup_steps * tokens_per_step_padded

        lr_scheduler_kwargs = self.args.lr_scheduler_kwargs or {}
        min_lr_ratio = lr_scheduler_kwargs.get('min_lr_ratio', 0.1)
        decay_type = lr_scheduler_kwargs.get('decay_type', 'cosine')
        decay_tokens = lr_scheduler_kwargs.get('num_decay_tokens', int(0.05 * total_tokens))

        stable_tokens = total_tokens - warmup_tokens - decay_tokens
        logger.info(
            f"Token-aware WSD | warmup={warmup_tokens:,} | stable={stable_tokens:,} | "
            f"decay={decay_tokens:,} | total={total_tokens:,} tokens"
        )

        def wsd_token_lambda(_):  # step arg intentionally ignored; progress is token-based
            t = self.total_tokens_seen
            if t < warmup_tokens:
                return t / max(warmup_tokens, 1)
            elif t < total_tokens - decay_tokens:
                return 1.0
            else:
                decay_progress = (t - (total_tokens - decay_tokens)) / max(decay_tokens, 1)
                decay_progress = min(decay_progress, 1.0)
                if decay_type == 'cosine':
                    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                else:  # linear
                    return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - decay_progress)

        if self.lr_scheduler is None:
            opt = self.optimizer if optimizer is None else optimizer
            self.lr_scheduler = LambdaLR(opt, wsd_token_lambda)
            self._created_lr_scheduler = True
        return self.lr_scheduler
    
    def evaluate_benchmarks(self):
        """Evaluate model on benchmarks and extract metrics for logging."""
        perplexity_tasks = self.benchmark_evaluation_cfg['perplexity_tasks']
        tasks_to_run = self.benchmark_evaluation_cfg['tasks_to_run']

        # Store full results and extracted metrics separately
        full_results = {}
        metrics_dict = {}

        # Clear GPU cache before evaluation to free fragmented memory
        logger.info("🧹 Clearing GPU cache before evaluation...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        logger.info("🔧 Wrapping model for lm-eval...")
        lm_eval_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            max_length=self.benchmark_evaluation_cfg.get('max_length', 2048),
            eval_max_length=self.benchmark_evaluation_cfg.get('eval_max_length', 16384),  # For rolling window perplexity
            batch_size=self.benchmark_evaluation_cfg.get('eval_batch_size', 16),
            device=str(self.model.device),
            trust_remote_code=True,
        )
        logger.info(f"✅ Model wrapped with default max_length={self.benchmark_evaluation_cfg.get('max_length', 2048)}, "
                   f"eval_max_length={self.benchmark_evaluation_cfg.get('eval_max_length', 16384)}")

        # Define which metrics to extract for each task type
        METRICS_TO_EXTRACT = {
            'perplexity': ['word_perplexity,none','word_perplexity'],
            'reasoning': ['acc', 'acc_norm','acc,none','acc_stderr,none','acc_norm,none','acc_norm_stderr,none'],
        }

        # ---- Iterate through evaluation tasks ----
        for task_cfg in self.benchmark_evaluation_cfg['evaluation_tasks']:
            task_name = task_cfg['name']

            # Only apply chat template if: (1) not a perplexity task, (2) apply_chat_template is true, AND (3) use_sft is true
            apply_chat_template = (
                task_name not in perplexity_tasks
                and self.benchmark_evaluation_cfg['apply_chat_template']
                and self.benchmark_evaluation_cfg['use_sft']
            )

            if tasks_to_run is not None and task_name not in tasks_to_run:
                logger.info(f"⏭️  Skipping {task_name} (not in tasks_to_run)")
                continue

            try:
                fewshot = task_cfg['fewshot']
                limit = task_cfg.get('limit')
                batch_size = task_cfg['task_batch_size']

                # Get task-specific max_length, or use global default
                task_max_length = task_cfg.get('max_length', self.benchmark_evaluation_cfg.get('max_length', 2048))

                # Update the model wrapper's max_length for this task
                lm_eval_model._max_length = task_max_length

                logger.info(f"🚀 Running evaluation for {task_name} | fewshot={fewshot} | limit={limit} | max_length={task_max_length}")

                # Prepare evaluation kwargs
                eval_kwargs = {
                    "model": lm_eval_model,
                    "batch_size": batch_size,
                    "apply_chat_template": apply_chat_template,
                    "tasks": [task_name],
                    "limit": limit,
                    "verbosity": self.benchmark_evaluation_cfg['verbosity'],
                    "num_fewshot": fewshot,
                    "log_samples": True,
                    "write_out": False,
                    "device": str(self.model.device),
                    "fewshot_random_seed": 23,
                    "random_seed": 23,
                    "numpy_random_seed": 23,
                    "torch_random_seed": 23,
                }

                # Add multiturn for chat templates with fewshot examples
                if apply_chat_template and fewshot > 0:
                    eval_kwargs["fewshot_as_multiturn"] = True
                    logger.info("📝 Using fewshot_as_multiturn=True")

                # Run evaluation
                res = evaluator.simple_evaluate(**eval_kwargs)
                full_results[task_name] = res

                # Extract metrics based on task type
                if task_name in perplexity_tasks:
                    metrics_to_extract = METRICS_TO_EXTRACT['perplexity']
                else:
                    metrics_to_extract = METRICS_TO_EXTRACT['reasoning']

                # Extract metrics from results
                task_results = res.get('results', {}).get(task_name, {})
                for metric_name in metrics_to_extract:
                    if metric_name in task_results:
                        # Use format: task_name/metric_name
                        key = f"{task_name}/{metric_name}"
                        metrics_dict[key] = task_results[metric_name]
                        logger.info(f"  📊 {metric_name}: {task_results[metric_name]:.4f}")

                logger.info(f"✅ Finished {task_name}")

            except Exception as e:
                logger.error(f"❌ Error running {task_name}: {str(e)}")
                full_results[task_name] = {"error": str(e)}
                continue

        logger.info(f"📈 Extracted {len(metrics_dict)} metrics from {len(full_results)} tasks")

        # Clean up metric keys: remove ",none" suffix
        metrics_dict = {k.replace(',none', ''): v for k, v in metrics_dict.items()}

        return metrics_dict

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
        _recursive: bool = False,
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, dict[str, `Dataset`]], *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if eval_dataset is None:
            # No eval dataset — skip loss loop, run benchmarks only
            metrics = {}
            if not _recursive and self.run_benchmarks:
                benchmark_results = self.evaluate_benchmarks()
                metrics.update({
                    f"{metric_key_prefix}_{k.replace('/', '_')}": v for k, v in benchmark_results.items()
                })
            if not _recursive:
                self.log(metrics)
                self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                    _recursive=True,
                )
                metrics.update(dataset_metrics)
            if not _recursive and self.run_benchmarks:
                benchmark_results = self.evaluate_benchmarks()
                metrics.update({
                    f"{metric_key_prefix}_{k.replace('/', '_')}": v for k, v in benchmark_results.items()
                })
            if not _recursive:
                self.log(metrics)
                self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

        # get benchmark results
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        if self.run_benchmarks and not _recursive:
            benchmark_results = self.evaluate_benchmarks()
            #add custom benchmark metrics with proper eval prefix (using underscore like speed_metrics)
            prefixed_benchmarks = {
                f"{metric_key_prefix}_{k.replace('/', '_')}": v for k, v in benchmark_results.items()
            }
            output.metrics.update(prefixed_benchmarks)


        self.log(output.metrics)
       
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def get_batch_samples(self, epoch_iterator, num_batches, device):
            batch_samples, num_items_in_batch = super().get_batch_samples(epoch_iterator, num_batches, device)

            if isinstance(num_items_in_batch, torch.Tensor):
                tokens = num_items_in_batch.item()
                self.milestone_tokens += tokens
                self.total_tokens_seen += tokens
            
            print(f"[get_batch_samples] : Total tokens this milestone {self.milestone_tokens}")
            return batch_samples, num_items_in_batch
    


# Convenience function to create a trainer with token stats tracking
def create_milestone_trainer(trainer_class, *args, **kwargs):
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
            processing_class=tokenizer,
            ...
        )

        callback = TokenStatsCallback(tokenizer, log_every_n_steps=10)
        trainer.add_callback(callback)
        trainer.train()
    """
    # Create a new class that inherits from both the mixin and the base trainer
    # The mixin must come first to properly override compute_loss
    class MilestoneTrainer(MilestoneTrainerMixin, trainer_class):
        pass

    return MilestoneTrainer(*args, **kwargs)




def compute_token_statistics(input_ids: torch.Tensor, pad_token_id: int, eos_token_id: int) -> Dict[str, int]:
    """
    Compute raw token counts from a batch of input_ids.

    This function is called for EVERY batch that goes through the model,
    so you get statistics for all incoming training data.

    Args:
        input_ids: Tensor of shape (batch_size, seq_length) - the actual training batch
        pad_token_id: ID of the padding token
        eos_token_id: ID of the EOS token

    Returns:
        Dictionary containing raw counts:
        - real_tokens: Number of non-padding tokens (including EOS)
        - padding_tokens: Number of padding tokens
        - eos_count: Number of EOS tokens (= number of distinct sequences)
    """
    # Flatten input_ids for easier counting
    input_ids_flat = input_ids.view(-1)

    real_tokens = (input_ids_flat != pad_token_id).sum().item()
    padding_tokens = (input_ids_flat == pad_token_id).sum().item()
    eos_count = (input_ids_flat == eos_token_id).sum().item()

    return {
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
            'real_tokens': 0,
            'padding_tokens': 0,
            'eos_count': 0,
            'num_batches': 0,
        }

    # Accumulate raw counts
    accumulated_stats['real_tokens'] += new_stats['real_tokens']
    accumulated_stats['padding_tokens'] += new_stats['padding_tokens']
    accumulated_stats['eos_count'] += new_stats['eos_count']
    accumulated_stats['num_batches'] += 1

    # Calculate derived metrics
    total_tokens = accumulated_stats['real_tokens'] + accumulated_stats['padding_tokens']
    accumulated_stats['total_tokens'] = total_tokens
    accumulated_stats['padding_ratio'] = (
        accumulated_stats['padding_tokens'] / total_tokens if total_tokens > 0 else 0.0
    )

    return accumulated_stats
