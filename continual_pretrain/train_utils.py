"""
Shared utilities for training scripts.
"""

import os
import re
import math
import logging
from typing import Any, Dict
from pathlib import Path


from typing import Optional
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, load_from_disk, DatasetDict, interleave_datasets
from data_module import DataPreprocessor, SimplePaddingCollator, PackedSequenceDataCollator
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Utilities
# =============================================================================

def to_dict(cfg_section: Any, *, resolve: bool = True) -> Dict[str, Any]:
    """Convert an OmegaConf section or plain dict to a standard dict."""
    if cfg_section is None:
        return {}
    if isinstance(cfg_section, dict):
        return dict(cfg_section)
    if OmegaConf.is_config(cfg_section):
        return OmegaConf.to_container(cfg_section, resolve=resolve)  # type: ignore
    raise TypeError(f"Unsupported config section type: {type(cfg_section)!r}")


# =============================================================================
# Dataset Loading
# =============================================================================

def load_single_dataset(
    entry_cfg: Dict[str, Any],
    default_cfg: Dict[str, Any],
    tokenizer,
) -> Any:
    """Load and preprocess a single dataset entry."""
    name = entry_cfg["name"]
    split = entry_cfg.get("split", default_cfg["split"])
    sample_size = entry_cfg.get("sample_size", default_cfg.get("sample_size"))
    text_field = entry_cfg.get("text_field") or default_cfg["text_field"]
    add_bos_eos = (
        default_cfg["add_bos_eos"]
        if entry_cfg.get("add_bos_eos") is None
        else entry_cfg.get("add_bos_eos")
    )

    # Load dataset
    if os.path.isdir(name):
        raw = load_from_disk(name)
        if isinstance(raw, DatasetDict):
            ds = raw[split] if split in raw else raw[list(raw.keys())[0]]
        else:
            ds = raw
    else:
        ds = load_dataset(name, split=split)

   

    if sample_size:
        dataset_len = len(ds)

        if int(sample_size) > dataset_len:
            logger.info(f"Requested sample size larger than entry size, will load full dataset instead - {dataset_len} entries instead of {sample_size}")
            sample_size = dataset_len
        ds = ds.select(range(int(sample_size)))

    # Preprocess
    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        text_field=text_field,
        add_bos_eos=add_bos_eos,
        use_sft_config= not default_cfg.get("tokenize", True)
    )

    remove_cols = list(ds.column_names)
    ds = ds.map(
        preprocessor,
        remove_columns=remove_cols,
        num_proc=4,
    )
    return ds


def prepare_dataset(cfg, tokenizer, return_validation=False, validation_split=0.1):
    """
    Load, optionally mix, and tokenize datasets.

    Args:
        cfg: Configuration object
        tokenizer: Tokenizer to use for preprocessing
        return_validation: If True, split each dataset and return (train, validation)
        validation_split: Fraction of data to use for validation (default 0.1 = 10%)

    Returns:
        If return_validation is False: single dataset
        If return_validation is True: tuple of (train_dataset, val_dataset)
    """
    dataset_defaults = to_dict(cfg.dataset)
    data_builder = cfg.get("data_builder")
    seed = int(cfg.get("seed") or 0)


    if data_builder and data_builder.get("enabled"):
        builder_cfg = to_dict(data_builder, resolve=False)
        datasets_cfg = builder_cfg.get("datasets", [])

        # Collect proportions
        entries_with_proportions = []
        total_proportion = 0
        for entry in datasets_cfg:
            entry_dict = to_dict(entry)
            proportion = entry_dict.get("proportion", 1)
            if proportion <= 0:
                continue
            entries_with_proportions.append((entry_dict, proportion))
            total_proportion += proportion

        # Load datasets with adjusted sample sizes
        processed_train = []
        processed_val = []
        proportions = []
        total_sample_size = dataset_defaults.get("sample_size")

        for entry_dict, proportion in entries_with_proportions: # set same sample for all datasets (smoke tests)
            if total_sample_size and "sample_size" not in entry_dict:
                adjusted_sample_size = int(total_sample_size * proportion / total_proportion)
                entry_dict["sample_size"] = adjusted_sample_size

            ds = load_single_dataset(entry_dict, dataset_defaults, tokenizer)

            if return_validation:
                # Split this dataset 90/10 (or custom ratio)
                split_ds = ds.train_test_split(test_size=validation_split, seed=seed)
                processed_train.append(split_ds['train'])
                processed_val.append(split_ds['test'])
            else:
                processed_train.append(ds)

            proportions.append(proportion)

        probabilities = [p / total_proportion for p in proportions]

        train_dataset = interleave_datasets(processed_train, probabilities=probabilities, seed=seed)

        if return_validation:
            val_dataset = interleave_datasets(processed_val, probabilities=probabilities, seed=seed)
            return train_dataset, val_dataset
        else:
            return train_dataset

    # Single dataset mode
    ds = load_single_dataset(dataset_defaults, dataset_defaults, tokenizer)

    if return_validation:
        split_ds = ds.train_test_split(test_size=validation_split, seed=seed)
        return split_ds['train'], split_ds['test']
    else:
        return ds


# =============================================================================
# Data Collator
# =============================================================================

def build_collator(cfg, tokenizer):
    """Build appropriate data collator based on config."""
    collator_cfg = cfg.data_collator
    if not collator_cfg.use_custom_packed:
        return SimplePaddingCollator(tokenizer)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must have an `eos_token_id` for packing.")
    return PackedSequenceDataCollator(
        tokenizer=tokenizer,
        pack_length=collator_cfg.pack_length,
        eos_token_id=eos_id,
    )


# =============================================================================
# W&B Configuration
# =============================================================================

def slugify(text: Any) -> str:
    """Convert arbitrary text into a filesystem/W&B friendly token."""
    text = str(text).strip()
    text = re.sub(r"[^\w.-]+", "-", text)
    text = text.strip("-_")
    return text or "na"


def compose_run_name(cfg: DictConfig, training_kwargs: Dict[str, Any], wandb_cfg: Dict[str, Any]) -> str:
    """Compose a descriptive W&B run name."""
    parts = []
    user_prefix = wandb_cfg.get("run_name")
    if user_prefix:
        parts.append(slugify(user_prefix))

    model_name = cfg.model.builder.get("model_name")
    if model_name:
        parts.append(slugify(model_name.split("/")[-1]))

    dataset_name = cfg.dataset.get("name")
    if dataset_name:
        parts.append(f"ds-{slugify(dataset_name.split('/')[-1])}")

    sample_size = cfg.dataset.get("sample_size")
    parts.append(f"sample-{slugify(sample_size or 'full')}")

    for label, key in [
        ("bs", "per_device_train_batch_size"),
        ("ga", "gradient_accumulation_steps"),
        ("ep", "num_train_epochs"),
        ("lr", "learning_rate"),
    ]:
        value = training_kwargs.get(key)
        if value is not None:
            parts.append(f"{label}-{slugify(value)}")

    return "__".join(parts)


def apply_wandb_config(
    cfg: DictConfig,
    training_kwargs: Dict[str, Any],
    wandb_cfg: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Set up W&B environment variables and Trainer args."""
    wandb_cfg = wandb_cfg or {}
    project = wandb_cfg.get("project")
    group = wandb_cfg.get("group")
    run_name = wandb_cfg.get("run_name")

    if project:
        os.environ["WANDB_PROJECT"] = str(project)

    if group:
        os.environ["WANDB_RUN_GROUP"] = str(group)

    training_kwargs["report_to"] = "wandb"

    # Only set run_name if WANDB_RUN_ID not set (new run), otherwise reuse existing
    if "WANDB_RUN_ID" not in os.environ:
        if run_name:
            training_kwargs["run_name"] = run_name
        else:
            training_kwargs["run_name"] = compose_run_name(cfg, training_kwargs, wandb_cfg)

    return training_kwargs


# =============================================================================
# Milestone Training Utilities
# =============================================================================

# === OPTIMIZER STATE DIAGNOSTIC (remove after debugging) ===
class OptimizerStateDiagnosticCallback(TrainerCallback):
    """
    Diagnostic callback to check if optimizer state is preserved across checkpoint resume.
    Logs optimizer state info at train_begin to compare with previous run's end state.
    """
    def __init__(self, expected_step: int = None):
        """
        Args:
            expected_step: The step we expect to resume from (for comparison)
        """
        self.expected_step = expected_step

    def on_train_begin(self, args, state, control, **kwargs):
        """Check optimizer state at training start"""
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            logger.warning("[OptimizerDiag] No optimizer in kwargs")
            return

        logger.info(f"[OptimizerDiag] === OPTIMIZER STATE AT TRAIN BEGIN ===")
        logger.info(f"[OptimizerDiag] Global step: {state.global_step}")
        logger.info(f"[OptimizerDiag] Expected step: {self.expected_step}")

        # Check optimizer state dict
        opt_state = optimizer.state_dict()
        param_groups = opt_state.get('param_groups', [])
        state_dict = opt_state.get('state', {})

        logger.info(f"[OptimizerDiag] Num param groups: {len(param_groups)}")
        logger.info(f"[OptimizerDiag] Num params with state: {len(state_dict)}")

        # Check if state exists (Adam should have 'step', 'exp_avg', 'exp_avg_sq')
        if state_dict:
            first_key = list(state_dict.keys())[0]
            first_state = state_dict[first_key]
            logger.info(f"[OptimizerDiag] First param state keys: {list(first_state.keys())}")

            if 'step' in first_state:
                logger.info(f"[OptimizerDiag] First param 'step': {first_state['step']}")

            # 8-bit Adam (paged_adamw_8bit)
            if 'state1' in first_state:
                state1 = first_state['state1']
                logger.info(f"[OptimizerDiag] First param state1: mean={state1.float().mean():.6f}, std={state1.float().std():.6f}")
            if 'absmax1' in first_state:
                logger.info(f"[OptimizerDiag] First param absmax1: mean={first_state['absmax1'].mean():.6f}")
        else:
            logger.warning("[OptimizerDiag] ⚠️ OPTIMIZER STATE IS EMPTY - not restored from checkpoint!")

    def on_train_end(self, args, state, control, **kwargs):
        """Log optimizer state at training end for comparison with next milestone's start"""
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return

        logger.info(f"[OptimizerDiag] === OPTIMIZER STATE AT TRAIN END ===")
        logger.info(f"[OptimizerDiag] Global step: {state.global_step}")

        opt_state = optimizer.state_dict()
        state_dict = opt_state.get('state', {})
        logger.info(f"[OptimizerDiag] Num params with state: {len(state_dict)}")

        if state_dict:
            first_key = list(state_dict.keys())[0]
            first_state = state_dict[first_key]

            if 'step' in first_state:
                logger.info(f"[OptimizerDiag] First param 'step': {first_state['step']}")
            if 'state1' in first_state:
                state1 = first_state['state1']
                logger.info(f"[OptimizerDiag] First param state1: mean={state1.float().mean():.6f}, std={state1.float().std():.6f}")
            if 'absmax1' in first_state:
                logger.info(f"[OptimizerDiag] First param absmax1: mean={first_state['absmax1'].mean():.6f}")
# === END OPTIMIZER STATE DIAGNOSTIC ===


class EpochResetCallback(TrainerCallback):
    """
    Maintains continuous epoch tracking across milestone training.

    Similar to LR scheduler callbacks, this uses total_steps to calculate
    global epoch progress across all milestones rather than resetting at
    milestone boundaries.

    The epoch value represents progress through the entire training run,
    calculated as: epoch = (global_step / total_steps) * num_train_epochs
    """
    def __init__(self, total_steps: int, num_train_epochs: float = 1.0):
        """
        Args:
            total_steps: Total training steps across all milestones
            num_train_epochs: Number of epochs for the entire training run
        """
        self.total_steps = total_steps
        self.num_train_epochs = num_train_epochs

    def on_train_begin(self, args, state, control, **kwargs):
        """Log callback initialization"""
        logger.info(
            f"EpochResetCallback: Tracking continuous epochs across "
            f"{self.total_steps} total steps ({self.num_train_epochs} epochs)"
        )

    def _fix_epoch(self, state):
        """Calculate and set the correct global epoch value"""
        if self.total_steps > 0:
            # Calculate global epoch progress
            # epoch = (current_step / total_steps) * num_train_epochs
            global_epoch = (state.global_step / self.total_steps) * self.num_train_epochs
            state.epoch = global_epoch

    def on_step_end(self, args, state, control, **kwargs):
        """Fix epoch AFTER each step"""
        self._fix_epoch(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Override epoch in logs to show global training progress"""
        if logs is not None and self.total_steps > 0:
            # Calculate and log global epoch
            global_epoch = (state.global_step / self.total_steps) * self.num_train_epochs
            logs['epoch'] = global_epoch


class WSDSchedulerCallback(TrainerCallback):
    """
    Warmup-Stable-Decay (WSD) Learning Rate Scheduler Callback
    
    Implements three-phase learning rate schedule:
    1. Warmup: Linear warmup from warmup_starting_lr to base lr
    2. Stable: Constant learning rate
    3. Decay: Quick decay to lr_min in final portion of training
    """
    
    def __init__(
        self,
        total_steps: Optional[int] = None,
        num_warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.0,
        decay_phase_ratio: float = 0.1,  # Start decay when 90% done
        lr_min_ratio: float = 0.1,  # Decay to 10% of base lr
        warmup_starting_lr_ratio: float = 0.0,  # Start warmup from 0
        use_inverse_sqrt_decay: bool = True,  # Use inverse proportional decay
        lr_lambda_func: Optional[callable] = None,
    ):
        self.total_steps = total_steps
        self.num_warmup_steps = num_warmup_steps
        self.warmup_ratio = warmup_ratio
        self.decay_phase_ratio = decay_phase_ratio
        self.lr_min_ratio = lr_min_ratio
        self.warmup_starting_lr_ratio = warmup_starting_lr_ratio
        self.use_inverse_sqrt_decay = use_inverse_sqrt_decay
        self.lr_lambda_func = lr_lambda_func
    
    def _apply_fix(self, lr_scheduler, args):
        """Apply the WSD LR lambda fix to the scheduler"""

        # Calculate warmup steps if not provided
        if self.num_warmup_steps is None:
            # Prioritize self.warmup_ratio (from callback init), fallback to args
            warmup_ratio = self.warmup_ratio if self.warmup_ratio else getattr(args, 'warmup_ratio', 0.0)
            if warmup_ratio > 0:
                # Use max(1, ...) to ensure at least 1 warmup step when ratio > 0
                self.num_warmup_steps = max(1, round(warmup_ratio * self.total_steps))
            else:
                self.num_warmup_steps = getattr(args, 'warmup_steps', 0)
            logger.info(f"[UTILS] SETTING NUM WARMUP STEPS: {self.num_warmup_steps} (warmup_ratio={warmup_ratio}, total_steps={self.total_steps})")
        else:
            logger.info(f"[UTILS] NUM WARMUP STEPS ALREADY SET: {self.num_warmup_steps}")


        if self.lr_lambda_func is None:
            num_warmup_steps = self.num_warmup_steps
            total_steps = self.total_steps
            decay_phase_ratio = self.decay_phase_ratio
            lr_min_ratio = self.lr_min_ratio
            warmup_starting_lr_ratio = self.warmup_starting_lr_ratio
            use_inverse_sqrt_decay = self.use_inverse_sqrt_decay
            
            # Calculate when decay phase starts
            decay_start_step = int(total_steps * (1.0 - decay_phase_ratio))
            
            def lr_lambda(current_step: int):
                # Phase 1: Warmup
                if current_step < num_warmup_steps:
                    # Linear warmup from warmup_starting_lr_ratio to 1.0
                    warmup_progress = float(current_step) / float(max(1, num_warmup_steps))
                    return warmup_starting_lr_ratio + (1.0 - warmup_starting_lr_ratio) * warmup_progress
                
                # Phase 2: Stable (constant learning rate)
                if current_step < decay_start_step:
                    return 1.0
                
                # Phase 3: Decay
                # Calculate progress through decay phase (0.0 to 1.0)
                decay_steps = total_steps - decay_start_step
                decay_progress = float(current_step - decay_start_step) / float(max(1, decay_steps))
                
                if use_inverse_sqrt_decay:
                    # Inverse proportional decay: lr = 1 / (t * (1/lr_min - 1) + 1)
                    # This is steeper initially and approaches lr_min more gradually
                    lr_multiplier = 1.0 / (decay_progress * (1.0 / lr_min_ratio - 1.0) + 1.0)
                else:
                    # Sqrt decay: lr = lr_min + (1 - lr_min) * (1 - sqrt(t))
                    # More gradual and consistent decay
                    lr_multiplier = lr_min_ratio + (1.0 - lr_min_ratio) * (1.0 - math.sqrt(decay_progress))
                
                return max(lr_min_ratio, lr_multiplier)
            
            self.lr_lambda_func = lr_lambda
        
        # Fix ALL parameter groups
        num_param_groups = len(lr_scheduler.lr_lambdas)
        for i in range(num_param_groups):
            lr_scheduler.lr_lambdas[i] = self.lr_lambda_func
        
        return num_param_groups
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initial scheduler fix at training start"""
        if "lr_scheduler" in kwargs:
            lr_scheduler = kwargs["lr_scheduler"]
            num_param_groups = self._apply_fix(lr_scheduler, args)
            
            decay_start_step = int(self.total_steps * (1.0 - self.decay_phase_ratio))
            decay_type = "inverse proportional" if self.use_inverse_sqrt_decay else "sqrt"
            
            logger.info(
                f"Fixed scheduler for {num_param_groups} param group(s) to use WSD schedule:\n"
                f"  Total steps: {self.total_steps}\n"
                f"  Warmup steps: {self.num_warmup_steps} ({self.num_warmup_steps/self.total_steps*100:.1f}%)\n"
                f"  Stable phase: steps {self.num_warmup_steps}-{decay_start_step} ({(decay_start_step-self.num_warmup_steps)/self.total_steps*100:.1f}%)\n"
                f"  Decay phase: steps {decay_start_step}-{self.total_steps} ({self.decay_phase_ratio*100:.1f}%)\n"
                f"  Decay type: {decay_type}\n"
                f"  Min LR ratio: {self.lr_min_ratio}"
            )

class LinearSchedulerCallback(TrainerCallback):
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.num_warmup_steps = None
        self.lr_lambda_func = None

    def _apply_fix(self, lr_scheduler, args):
        """Apply the LR lambda fix to the scheduler"""
        if self.num_warmup_steps is None:
            self.num_warmup_steps = int(args.warmup_ratio * self.total_steps) if args.warmup_ratio else args.warmup_steps

        if self.lr_lambda_func is None:
            num_warmup_steps = self.num_warmup_steps
            total_steps = self.total_steps

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0,
                    float(total_steps - current_step) / float(max(1, total_steps - num_warmup_steps))
                )

            self.lr_lambda_func = lr_lambda

        # Fix ALL parameter groups
        num_param_groups = len(lr_scheduler.lr_lambdas)
        for i in range(num_param_groups):
            lr_scheduler.lr_lambdas[i] = self.lr_lambda_func

        return num_param_groups

    def on_train_begin(self, args, state, control, **kwargs):
        """Initial scheduler fix at training start"""
        if "lr_scheduler" in kwargs:
            lr_scheduler = kwargs["lr_scheduler"]
            num_param_groups = self._apply_fix(lr_scheduler, args)
            logger.info(f"Fixed scheduler for {num_param_groups} param group(s) to use total_steps={self.total_steps} (warmup={self.num_warmup_steps})")

    def on_step_end(self, args, state, control, **kwargs):
        """Reapply fix periodically to catch scheduler recreations"""
        # Check every 10 steps to minimize overhead
        if state.global_step % 10 == 0 and "lr_scheduler" in kwargs:
            lr_scheduler = kwargs["lr_scheduler"]
            # Verify our lambda is still there
            if lr_scheduler.lr_lambdas and lr_scheduler.lr_lambdas[0] != self.lr_lambda_func:
                logger.warning(f"⚠️  Scheduler lambda lost at step {state.global_step}! Reapplying fix...")
                self._apply_fix(lr_scheduler, args)


class CosineSchedulerCallback(TrainerCallback):
    """
    Cosine Annealing Learning Rate Scheduler Callback with Warmup

    Implements the same schedule as transformers.get_cosine_schedule_with_warmup,
    but correctly tracks global steps across milestone training.

    Schedule:
    1. Warmup: Linear warmup from 0 to base_lr
    2. Cosine Decay: lr = 0.5 * (1 + cos(pi * num_cycles * 2 * progress))

    Default num_cycles=0.5 produces a half-cosine from max lr to 0.
    """

    def __init__(
        self,
        total_steps: int,
        num_cycles: float = 0.5,  # HuggingFace default
        num_warmup_steps: Optional[int] = None,
        warmup_ratio: float = 0.0,
    ):
        """
        Args:
            total_steps: Total training steps across all milestones
            num_cycles: Number of cosine waves (0.5 = half-cosine decay, HF default)
            num_warmup_steps: Explicit warmup steps (overrides warmup_ratio)
            warmup_ratio: Warmup as fraction of total_steps
        """
        self.total_steps = total_steps
        self.num_cycles = num_cycles
        self.num_warmup_steps = num_warmup_steps
        self.warmup_ratio = warmup_ratio
        self.lr_lambda_func = None

    def _apply_fix(self, lr_scheduler, args):
        """Apply the cosine LR lambda fix to the scheduler"""

        # Calculate warmup steps if not provided
        if self.num_warmup_steps is None:
            # Prioritize self.warmup_ratio (from callback init), fallback to args
            warmup_ratio = self.warmup_ratio if self.warmup_ratio else getattr(args, 'warmup_ratio', 0.0)
            if warmup_ratio > 0:
                self.num_warmup_steps = max(1, round(warmup_ratio * self.total_steps))
            else:
                self.num_warmup_steps = getattr(args, 'warmup_steps', 0)
            logger.info(f"[CosineScheduler] Setting warmup steps: {self.num_warmup_steps} "
                       f"(warmup_ratio={warmup_ratio}, total_steps={self.total_steps})")
        else:
            logger.info(f"[CosineScheduler] Warmup steps already set: {self.num_warmup_steps}")

        if self.lr_lambda_func is None:
            num_warmup_steps = self.num_warmup_steps
            total_steps = self.total_steps
            num_cycles = self.num_cycles

            def lr_lambda(current_step: int):
                # Matches transformers._get_cosine_schedule_with_warmup_lr_lambda exactly
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

            self.lr_lambda_func = lr_lambda

        # Fix ALL parameter groups
        num_param_groups = len(lr_scheduler.lr_lambdas)
        for i in range(num_param_groups):
            lr_scheduler.lr_lambdas[i] = self.lr_lambda_func

        return num_param_groups

    def on_train_begin(self, args, state, control, **kwargs):
        """Initial scheduler fix at training start"""
        if "lr_scheduler" in kwargs:
            lr_scheduler = kwargs["lr_scheduler"]
            num_param_groups = self._apply_fix(lr_scheduler, args)

            logger.info(
                f"Fixed scheduler for {num_param_groups} param group(s) to use Cosine schedule:\n"
                f"  Total steps: {self.total_steps}\n"
                f"  Warmup steps: {self.num_warmup_steps} ({self.num_warmup_steps/self.total_steps*100:.1f}%)\n"
                f"  Cosine cycles: {self.num_cycles} (0.5 = half-cosine decay to 0, HF default)"
            )

    def on_step_end(self, args, state, control, **kwargs):
        """Reapply fix periodically to catch scheduler recreations"""
        # Check every 10 steps to minimize overhead
        if state.global_step % 10 == 0 and "lr_scheduler" in kwargs:
            lr_scheduler = kwargs["lr_scheduler"]
            # Verify our lambda is still there
            if lr_scheduler.lr_lambdas and lr_scheduler.lr_lambdas[0] != self.lr_lambda_func:
                logger.warning(f"⚠️  Scheduler lambda lost at step {state.global_step}! Reapplying fix...")
                self._apply_fix(lr_scheduler, args)




def calculate_max_steps(
    target_tokens: int,
    batch_size: int,
    grad_accum: int,
    seq_length: int,
    world_size: int = 1
) -> int:
    """
    Calculate max_steps needed to process target_tokens.

    tokens_per_step = batch_size × grad_accum × seq_length × world_size
    max_steps = target_tokens / tokens_per_step
    """
    tokens_per_step = batch_size * grad_accum * seq_length * world_size
    max_steps = target_tokens // tokens_per_step
    return max_steps


def merge_lora_checkpoint(trainer, output_path: Path, tokenizer):
    """
    Merge LoRA adapter weights into base model.

    Args:
        trainer: Trainer instance with PEFT model
        output_path: Where to save merged model
        tokenizer: Tokenizer to save alongside model
    """
    from peft import PeftModel

    if not isinstance(trainer.model, PeftModel):
        logger.warning("Not a PEFT model, skipping merge")
        return

    logger.info(f"Merging LoRA weights to: {output_path}")
    merged_model = trainer.model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("✅ Merged model saved")


# =============================================================================
# Milestone Dataset Loading
# =============================================================================

def load_milestone_shard(prebuilt_path: str, milestone_num: int):
    """
    Load a specific milestone shard from a prebuilt mixed dataset.

    Args:
        prebuilt_path: Path to prebuilt dataset
        milestone_num: Which milestone to load (0-indexed)

    Returns:
        tuple: (train_dataset, validation_dataset)
            - train_dataset: Examples for the requested milestone
            - validation_dataset: Validation examples (milestone_num==99, shared across all milestones)
    """
    from datasets import load_from_disk

    logger.info(f"Loading milestone {milestone_num} from: {prebuilt_path}")

    # Load full dataset
    dataset = load_from_disk(prebuilt_path)

    # Filter to requested milestone for training
    train_data = dataset.filter(
        lambda x: x['milestone_num'] == milestone_num,
        num_proc=4
    )

    # Filter to validation data (milestone_num == 99)
    val_data = dataset.filter(
        lambda x: x['milestone_num'] == 99,
        num_proc=4
    )

    # Log stats for training data
    if 'num_tokens' in train_data.column_names:
        train_tokens = sum(train_data['num_tokens'])
        logger.info(f"✓ Loaded training milestone {milestone_num}: "
                   f"{len(train_data):,} examples, {train_tokens:,} tokens")
        train_data = train_data.remove_columns(['num_tokens', 'milestone_num'])
    else:
        train_data = train_data.remove_columns(['milestone_num'])

    # Log stats for validation data
    if 'num_tokens' in val_data.column_names:
        val_tokens = sum(val_data['num_tokens'])
        logger.info(f"✓ Loaded validation data: "
                   f"{len(val_data):,} examples, {val_tokens:,} tokens")
        val_data = val_data.remove_columns(['num_tokens', 'milestone_num'])
    else:
        val_data = val_data.remove_columns(['milestone_num'])

    return train_data, val_data


# =============================================================================
# Model/Tokenizer Mappings
# =============================================================================

PT_TO_IT_TOKENIZER_MAP = {
    "google/gemma-3-1b-pt": "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
}
