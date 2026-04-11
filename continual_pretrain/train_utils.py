"""
Shared utilities for training scripts.
"""
import unsloth
from unsloth import UnslothTrainingArguments as _UnslothTrainingArguments, UnslothTrainer

# Fix unsloth bug: embedding_learning_rate is accepted but never stored as self attribute
class UnslothTrainingArguments(_UnslothTrainingArguments):
    def __init__(self, embedding_learning_rate=None, *args, **kwargs):
        self.embedding_learning_rate = embedding_learning_rate
        super().__init__(embedding_learning_rate, *args, **kwargs)
from trl import SFTConfig, SFTTrainer
import os
import re
import math
import logging
from typing import Any, Dict
from pathlib import Path
from transformers import AutoTokenizer

import torch
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from data_module import DataPreprocessor

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


def overwrite_pt_tokenizer(model_name, tokenizer):
    pt_to_it_tokenizer_map = {
    "google/gemma-3-1b-pt": "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    }

    if model_name in pt_to_it_tokenizer_map.values():
        logging.info('IT model : No need to overwrite')
        return tokenizer
    elif model_name  in pt_to_it_tokenizer_map:
        logging.info('PT model : overwrote chat_tempalte')

        it_tokenizer_name = pt_to_it_tokenizer_map[model_name]
        it_tokenizer = AutoTokenizer(it_tokenizer_name, trust_remote_code = True)
        tokenizer.chat_template = it_tokenizer.chat_template
        return tokenizer
    else:
        logging.info('Model not in mapper, chat_template set to None')
        tokenizer.chat_template = None
        return tokenizer



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
    add_bos_eos = default_cfg["add_bos_eos"]
      

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
        sample_offset = int(entry_cfg.get("sample_offset") or default_cfg.get("sample_offset") or 0)

        if sample_offset + int(sample_size) > dataset_len:
            logger.info(f"Requested sample size + offset larger than dataset size, clamping - {dataset_len} entries available")
            sample_size = dataset_len - sample_offset
        ds = ds.select(range(sample_offset, sample_offset + int(sample_size)))

    # Add source tracking column before preprocessing
    ds = ds.add_column("ds_source", [name] * len(ds))

    # Preprocess
    tokenize = default_cfg.get("tokenize", False)
    use_sft = default_cfg.get("use_sft", True)
    text_field = 'formatted_text' if use_sft else 'text'

    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        add_bos_eos=add_bos_eos,
        tokenize=tokenize,
        use_sft=use_sft,
        text_field = text_field
    )

    # Remove all columns except ds_source
    remove_cols = [col for col in ds.column_names if col != "ds_source"]
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

    # Calculate validation size based on target training tokens (not full dataset)
    val_size = None
    if return_validation:
        milestone_cfg = to_dict(cfg.get("milestone", {}))
        num_milestones = milestone_cfg.get('num_milestones', 1)
        tokens_per_milestone = milestone_cfg.get('tokens_per_milestone', 1_000_000)
        max_length = to_dict(cfg.training_args).get('max_length', 2048)

        # Calculate total target tokens for training
        total_target_tokens = num_milestones * tokens_per_milestone

        # Estimate validation samples: (target_tokens * validation_split) / avg_tokens_per_sample
        # Using 1000 as mean sample length (packing disabled, samples well below max_length)
        avg_tokens_per_sample = 1000
        val_size = int((total_target_tokens * validation_split) / avg_tokens_per_sample)
        logger.info(f"Validation size: {val_size} samples "
                   f"(based on {total_target_tokens:,} target tokens * {validation_split} split / {avg_tokens_per_sample} avg tokens)")


    if data_builder and data_builder.get("enabled"):
        builder_cfg = to_dict(data_builder, resolve=False)
        datasets_cfg = builder_cfg.get("datasets", [])

        # Collect proportions — top-level `proportions` list overrides per-dataset fields
        proportions_override = builder_cfg.get("proportions")
        entries_with_proportions = []
        total_proportion = 0
        for i, entry in enumerate(datasets_cfg):
            entry_dict = to_dict(entry)
            if proportions_override and i < len(proportions_override):
                proportion = proportions_override[i]
            else:
                proportion = entry_dict.get("proportion", 1)
            entries_with_proportions.append((entry_dict, proportion))
            total_proportion += proportion

        total_sample_size = data_builder.get("total_sample_size")

        # Load all datasets (each already has ds_source column from load_single_dataset)
        processed = []
        for i, (entry_dict, proportion) in enumerate(entries_with_proportions):
            if total_sample_size and ("sample_size" not in entry_dict or entry_dict['sample_size'] is None):
                adjusted_sample_size = int(total_sample_size * proportion / total_proportion)
                logger.info(f"Select {adjusted_sample_size} from dataset {entry_dict['name']}")
                entry_dict["sample_size"] = adjusted_sample_size
            ds = load_single_dataset(entry_dict, dataset_defaults, tokenizer)
            processed.append(ds)

        full_dataset = concatenate_datasets(processed).shuffle(seed=seed)
        logger.info(f"Concatenated {len(processed)} datasets with {len(full_dataset)} total examples")

        if return_validation:
            split_ds = full_dataset.train_test_split(test_size=val_size, seed=seed)
            train_dataset = split_ds['train']
            val_dataset = split_ds['test']
            # Group val set by ds_source — LLN ensures proportions are preserved naturally
            val_dict = {}
            for source_name in val_dataset.unique('ds_source'):
                source_key = source_name.rstrip('/').split('/')[-1]
                val_dict[source_key] = val_dataset.filter(lambda x, s=source_name: x['ds_source'] == s)
                logger.info(f"  val/{source_key}: {len(val_dict[source_key])} samples")
            logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val ({len(val_dict)} sources)")
            return train_dataset, val_dict

        return full_dataset

    # Single dataset mode
    ds = load_single_dataset(dataset_defaults, dataset_defaults, tokenizer)

    if return_validation:
        # Split using absolute validation size (based on target tokens)
        split_ds = ds.train_test_split(test_size=val_size, seed=seed)
        logger.info(f"Split dataset: {len(split_ds['train'])} train, {len(split_ds['test'])} validation")
        return split_ds['train'], split_ds['test']
    else:
        return ds

# =============================================================================
# Training Type utils
# =============================================================================


def set_training_type(use_unsloth : bool = False, training_kwargs: dict = None):
    TrainingArgsClass = UnslothTrainingArguments if use_unsloth else SFTConfig
    TrainerClass = UnslothTrainer if use_unsloth else SFTTrainer

    # unsloth specific arg
    if not use_unsloth:
        training_kwargs.pop('embedding_learning_rate', None)

    return TrainerClass, TrainingArgsClass, training_kwargs



# =============================================================================
# Logging Configuration
# =============================================================================

def setup_file_logging(output_dir: str, run_name: str = None, run_id: str = None, log_filename: str = None) -> None:
    """
    Set up file logging to save logs in the output directory.

    Creates a logs subdirectory in the output_dir and configures a FileHandler
    that writes all logs (INFO level and above) to a file.

    Args:
        output_dir: The training output directory (e.g., "./outputs/train_llama")
        run_name: WandB run name (optional, used to generate filename)
        run_id: WandB run ID (optional, used to generate filename)
        log_filename: Name of the log file (optional, overrides auto-generated name)
    """
    from pathlib import Path

    # Create logs directory
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename if not provided
    if log_filename is None:
        if run_name and run_id:
            # Use first 8 characters of run_id
            short_run_id = run_id[:8]
            log_filename = f"{run_name}-{short_run_id}.log"
        elif run_name:
            log_filename = f"{run_name}.log"
        else:
            log_filename = "training.log"

    log_file_path = logs_dir / log_filename

    # Create file handler with same format as console
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Use same format as console logging
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to root logger (captures all loggers in the project)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    logger.info(f"File logging enabled: {log_file_path}")
    return file_handler


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
    """
    Compose W&B run name with format:
    {run_type}-{model}-{custom_name}-ms{num_milestones}-{tokens}M

    Example: cpt-llama-myexp-ms10-900M or sft-qwen-ms5-100M
    """
    parts = []

    use_sft = cfg.dataset.get("use_sft", True)
    run_type = "sft" if use_sft else "cpt"
    parts.append(run_type)

    model_name = cfg.model.builder.get("model_name", "")
    if model_name:

        model_base = model_name.split("/")[-1].lower()
        for family in ["llama", "qwen", "gemma", "mistral", "phi"]:
            if family in model_base:
                parts.append(family)
                break
        else:
            parts.append(slugify(model_base.split("-")[0].split(".")[0]))

    custom_name = os.environ.get("CUSTOM_NAME", "").strip()
    if custom_name:
        parts.append(slugify(custom_name))


    num_milestones = os.environ.get("NUM_MILESTONES")
    if num_milestones:
        parts.append(f"ms{num_milestones}")


    milestone_tokens = os.environ.get("MILESTONE_TOKENS")
    if milestone_tokens:
        tokens = int(milestone_tokens)
        if tokens >= 1_000_000_000:
            formatted = f"{tokens // 1_000_000_000}B"
        elif tokens >= 1_000_000:
            formatted = f"{tokens // 1_000_000}M"
        else:
            formatted = f"{tokens // 1_000}K"
        parts.append(formatted)

    return "-".join(parts)


def apply_wandb_config(
    training_kwargs: Dict[str, Any],
    wandb_cfg: Dict[str, Any] | None,
    wandb_run_name: str= None
) -> Dict[str, Any]:
    """Set up W&B environment variables and Trainer args."""
    wandb_cfg = wandb_cfg or {}
    project = wandb_cfg.get("project")
    group = wandb_cfg.get("group")

    if project:
        os.environ["WANDB_PROJECT"] = str(project)

    if group:
        os.environ["WANDB_RUN_GROUP"] = str(group)

    training_kwargs["report_to"] = "wandb"

    # Get run_name from environment (set by run_milestone_loop.sh)
    # WANDB_RUN_NAME must be set by the shell script for consistent naming
    run_name = wandb_run_name
    if not run_name:
        logger.warning("⚠️  WANDB_RUN_NAME not set! Run name should be provided by run_milestone_loop.sh")
        logger.warning("    Continuing without run name - WandB logging may be inconsistent")
    else:
        logger.info(f"Using run name: {run_name}")
        training_kwargs["run_name"] = run_name

    return training_kwargs



def generate_run_name(
    model_name: str,
    use_sft: bool = False,
    custom_name: str = "",
    num_milestones: int = 1,
    milestone_tokens: int = 1_000_000
) -> str:
    """
    Generate a run name from model configuration.
    
    Args:
        model_name: Full model name/path (e.g., "meta-llama/Llama-3.2-1B")
        use_sft: Whether this is an SFT run (vs CPT)
        custom_name: Optional custom identifier to include
        num_milestones: Number of training milestones
        milestone_tokens: Tokens per milestone
        
    Returns:
        Generated run name (e.g., "cpt-llama-ms10-1B")
    """
    # Extract base model name (part after last /)
    model_base = model_name.split('/')[-1].lower()
    
    # Determine model family
    model_family = ""
    for family in ["llama", "qwen", "gemma", "mistral", "phi"]:
        if family in model_base:
            model_family = family
            break
    
    # If no family match, use first part of model name
    if not model_family:
        # Split by '-' or '.' and take first part
        model_family = model_base.split('-')[0].split('.')[0]
    
    # Build run name parts
    run_type = "sft" if use_sft else "cpt"
    parts = f"{run_type}-{model_family}"
    
    # Add custom name if set
    if custom_name:
        parts = f"{parts}-{custom_name}"
    
    # Add milestone count
    parts = f"{parts}-ms{num_milestones}"
    
    # Add token amount
    total_tokens = milestone_tokens * num_milestones
    if total_tokens >= 1_000_000_000:
        parts = f"{parts}-{total_tokens // 1_000_000_000}B"
    elif total_tokens >= 1_000_000:
        parts = f"{parts}-{total_tokens // 1_000_000}M"
    else:
        parts = f"{parts}-{total_tokens // 1_000}K"
    
    os.environ['WANDB_RUN_NAME'] = parts
    return parts
