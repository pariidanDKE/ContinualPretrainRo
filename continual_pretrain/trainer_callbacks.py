"""
Trainer callbacks for milestone-based training.

This module contains specialized callbacks for managing continuous learning
rate scheduling, epoch tracking, and optimizer diagnostics across milestone
training boundaries.
"""

import math
import logging
import torch
from typing import Optional
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


# =============================================================================
# Initial Loss Logging
# =============================================================================

class InitialLossCallback(TrainerCallback):
    """
    Logs the initial loss before any gradient updates (step 0).

    This callback computes and logs the model's loss on the first training batch
    before any optimizer steps are taken. Useful for tracking how much the model
    improves from its initial state.
    """

    def __init__(self, num_eval_batches: int = 1):
        """
        Args:
            num_eval_batches: Number of batches to average loss over (default: 1)
        """
        self.num_eval_batches = num_eval_batches
        self.logged = False

    def on_train_begin(self, args, state, control, **kwargs):
        """Compute and log initial loss before training starts"""
        if self.logged:
            return  # Already logged (e.g., on resume)

        model = kwargs.get("model")
        train_dataloader = kwargs.get("train_dataloader")

        if model is None or train_dataloader is None:
            logger.warning("[InitialLoss] Model or dataloader not available, skipping initial loss logging")
            return

        logger.info(f"[InitialLoss] Computing initial loss on {self.num_eval_batches} batch(es)...")

        # Set model to eval mode temporarily
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            dataloader_iter = iter(train_dataloader)
            for _ in range(self.num_eval_batches):
                try:
                    batch = next(dataloader_iter)

                    # Move batch to device
                    if isinstance(batch, dict):
                        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()}
                    else:
                        batch = tuple(t.to(model.device) if isinstance(t, torch.Tensor) else t
                                     for t in batch)

                    # Forward pass
                    if isinstance(batch, dict):
                        outputs = model(**batch)
                    else:
                        outputs = model(*batch)

                    # Extract loss
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    total_loss += loss.item()
                    num_batches += 1

                except StopIteration:
                    logger.warning(f"[InitialLoss] Dataloader exhausted after {num_batches} batches")
                    break

        # Restore model to train mode
        model.train()

        # Calculate average loss
        initial_loss = total_loss / max(num_batches, 1)

        logger.info(f"[InitialLoss] Initial loss (step 0): {initial_loss:.4f}")

        # Log to trainer's log history (will be picked up by WandB)
        log_dict = {
            "train/initial_loss": initial_loss,
            "train/loss": initial_loss,  # Also log as regular loss for step 0
            "step": 0,
            "epoch": 0.0
        }
        state.log_history.append(log_dict)

        # Also log to WandB directly if available
        if hasattr(state, "is_world_process_zero") and state.is_world_process_zero():
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(log_dict, step=0)
                    logger.info("[InitialLoss] Logged to WandB")
            except ImportError:
                pass

        self.logged = True


# =============================================================================
# Optimizer Diagnostics
# =============================================================================

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


# =============================================================================
# Epoch Tracking
# =============================================================================

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


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

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
            logger.info(f"[WSDScheduler] Setting warmup steps: {self.num_warmup_steps} (warmup_ratio={warmup_ratio}, total_steps={self.total_steps})")
        else:
            logger.info(f"[WSDScheduler] Warmup steps already set: {self.num_warmup_steps}")


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

        # CRITICAL: Force scheduler to recompute current LR with new lambda
        # Without this, step 1 uses LR computed with old lambda (before our fix)
        current_step = lr_scheduler.last_epoch
        lr_scheduler.step(current_step)

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
    """
    Linear Learning Rate Scheduler with Warmup

    Implements a linear decay schedule with optional warmup phase.
    Compatible with milestone-based training by tracking global steps.
    """

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
