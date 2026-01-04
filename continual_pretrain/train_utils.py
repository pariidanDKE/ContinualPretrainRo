"""
Shared utilities for training scripts.
"""

import os
import re
import logging
from typing import Any, Dict
from pathlib import Path

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

        if sample_size > dataset_len:
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


def prepare_dataset(cfg, tokenizer):
    """Load, optionally mix, and tokenize datasets."""
    dataset_defaults = to_dict(cfg.dataset)
    data_builder = cfg.get("data_builder")


    if data_builder and data_builder.get("enabled"):
        builder_cfg = to_dict(data_builder, resolve=False)
        datasets_cfg = builder_cfg.get("datasets", [])
        logger.info(f" DATASETS CFG : {datasets_cfg}")

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
        processed = []
        proportions = []
        total_sample_size = dataset_defaults.get("sample_size")

        for entry_dict, proportion in entries_with_proportions: # set same sample for all datasets (smoke tests)
            if total_sample_size and "sample_size" not in entry_dict:
                adjusted_sample_size = int(total_sample_size * proportion / total_proportion)
                entry_dict["sample_size"] = adjusted_sample_size

            ds = load_single_dataset(entry_dict, dataset_defaults, tokenizer)
            processed.append(ds)
            proportions.append(proportion)

        probabilities = [p / total_proportion for p in proportions]
        seed = int(cfg.get("seed") or 0)
        return interleave_datasets(processed, probabilities=probabilities, seed=seed)

    return load_single_dataset(dataset_defaults, dataset_defaults, tokenizer)


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

    if project:
        os.environ["WANDB_PROJECT"] = str(project)

    training_kwargs["report_to"] = "wandb"

    # Only set run_name if WANDB_RUN_ID not set (new run), otherwise reuse existing
    if "WANDB_RUN_ID" not in os.environ:
        training_kwargs["run_name"] = compose_run_name(cfg, training_kwargs, wandb_cfg)

    return training_kwargs


# =============================================================================
# Milestone Training Utilities
# =============================================================================

class SchedulerFixCallback(TrainerCallback):
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
    
    def on_train_begin(self, args, state, control, **kwargs):

        if "lr_scheduler" in kwargs:
            lr_scheduler  = kwargs["lr_scheduler"]

            num_warmup_steps = int(args.warmup_ratio * self.total_steps) if args.warmup_ratio else args.warmup_steps

            def lr_lambda(current_step : int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1,num_warmup_steps))

                return max(
                    0.0,
                    float(self.total_steps - current_step) / float(max(1, self.total_steps - num_warmup_steps))

                )     
            lr_scheduler.lr_lambdas[0] = lr_lambda
            logger.info(f"Fixed scheduler to use total_steps={self.total_steps} (warmup={num_warmup_steps})")                 

 


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
    """
    from datasets import load_from_disk

    logger.info(f"Loading milestone {milestone_num} from: {prebuilt_path}")

    # Load full dataset
    dataset = load_from_disk(prebuilt_path)

    # Filter to requested milestone
    milestone_data = dataset.filter(
        lambda x: x['milestone_num'] == milestone_num,
        num_proc=4
    )

    # Log stats and remove helper columns
    if 'num_tokens' in milestone_data.column_names:
        milestone_tokens = sum(milestone_data['num_tokens'])
        logger.info(f"✓ Loaded milestone {milestone_num}: "
                   f"{len(milestone_data):,} examples, {milestone_tokens:,} tokens")
        milestone_data = milestone_data.remove_columns(['num_tokens', 'milestone_num'])
    else:
        milestone_data = milestone_data.remove_columns(['milestone_num'])

    return milestone_data


# =============================================================================
# Model/Tokenizer Mappings
# =============================================================================

PT_TO_IT_TOKENIZER_MAP = {
    "google/gemma-3-1b-pt": "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
}
