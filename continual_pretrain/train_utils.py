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
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from data_module import DataPreprocessor, SimplePaddingCollator, PackedSequenceDataCollator

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

    # Add source tracking column before preprocessing
    ds = ds.add_column("ds_source", [name] * len(ds))

    # Preprocess
    tokenize = default_cfg.get("tokenize", False)
    use_sft = default_cfg.get("use_sft", True)

    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        text_field=text_field,
        add_bos_eos=add_bos_eos,
        tokenize=tokenize,
        use_sft=use_sft,
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

        # Concatenate all datasets and shuffle for random distribution
        logger.info(f"Concatenating {len(processed_train)} datasets with {len(processed_train[0]) if processed_train else 0} total examples")
        train_dataset = concatenate_datasets(processed_train).shuffle(seed=seed)

        if return_validation:
            val_dataset = concatenate_datasets(processed_val).shuffle(seed=seed)
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
# TRL Packing Patch for Stability
# =============================================================================

def patch_trl_packing_for_stability(min_fill_ratio: float = 0.7):
    """
    Patch TRL's _pack_bfd function to filter out low-density packed sequences.

    This addresses the sawtooth loss issue caused by randomly-sampled low-density
    packed sequences during training. BFD (Best-Fit Decreasing) packing optimizes
    for global padding efficiency but doesn't enforce minimum bin fill ratios,
    leading to pathological training samples with <10-20% real tokens.

    See: continual_pretrain/Notes/SawToothLossFix_OverridePacking.md

    Args:
        min_fill_ratio: Minimum fraction of seq_length that bins must contain (default 0.7)
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import numpy as np
        from collections import defaultdict, deque
        import trl.data_utils as data_utils
        from trl.data_utils import _SegmentTree
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not import TRL dependencies for patching: {e}. Skipping patch.")
        return

    def filtered_pack_bfd(examples: pa.Table, seq_length: int) -> pa.Table:
        """Pack sequences using BFD with low-density bin filtering."""

        # =====================================================================
        # PART 1: Original TRL fragmentation and binning logic
        # =====================================================================

        # Identify the list column and prepare all columns
        columns = []
        list_column_idx = None
        for idx, column in enumerate(examples.columns):
            if isinstance(column, pa.ChunkedArray):
                column = column.combine_chunks()
            if not (pa.types.is_list(column.type) or pa.types.is_large_list(column.type)):
                raise TypeError("pack_dataset(bfd) requires all columns to be list-like.")
            if list_column_idx is None:
                list_column_idx = idx
            columns.append(column)

        assert list_column_idx is not None
        list_column = columns[list_column_idx]
        offsets = np.asarray(list_column.offsets)
        values = list_column.values

        # Split every list row into fragments of length <= seq_length
        frag_lengths: list[int] = []
        frag_info: list[tuple[int, int, int]] = []  # (row_idx, split_start, frag_len)
        expanded_indices: list[int] = []

        for row_idx, (row_start, row_end) in enumerate(zip(offsets[:-1], offsets[1:], strict=False)):
            length = row_end - row_start
            for split_start in range(0, length, seq_length):
                frag_len = min(seq_length, length - split_start)
                frag_lengths.append(frag_len)
                frag_info.append((row_idx, split_start, frag_len))
                expanded_indices.append(row_idx)

        # Rebuild list columns with fragments
        offsets_type = list_column.offsets.type
        new_offsets = np.empty(len(frag_lengths) + 1, dtype=offsets_type.to_pandas_dtype())
        new_offsets[0] = 0
        new_offsets[1:] = np.cumsum(frag_lengths, dtype=offsets_type.to_pandas_dtype())
        new_offsets_array = pa.array(new_offsets, type=offsets_type)

        for idx, column in enumerate(columns):
            if idx == list_column_idx:
                slices = [
                    values.slice(offsets[row_idx] + split_start, frag_len)
                    for row_idx, split_start, frag_len in frag_info
                ]
                new_values = pa.concat_arrays(slices)
                columns[idx] = type(column).from_arrays(new_offsets_array, new_values)
                continue

            column_offsets = np.asarray(column.offsets)
            column_values = column.values
            slices = []
            for row_idx, split_start, frag_len in frag_info:
                row_len = column_offsets[row_idx + 1] - column_offsets[row_idx]
                if row_len < split_start + frag_len:
                    raise ValueError("List columns must have matching lengths when packing datasets.")
                start = column_offsets[row_idx] + split_start
                slices.append(column_values.slice(start, frag_len))
            column_offsets_array = pa.array(new_offsets, type=column.offsets.type)
            columns[idx] = type(column).from_arrays(column_offsets_array, pa.concat_arrays(slices))

        examples = pa.Table.from_arrays(columns, names=examples.column_names)
        ids = np.arange(len(examples))
        lengths = pc.list_value_length(examples[list_column_idx]).combine_chunks()
        examples = examples.append_column("seq_lengths", lengths)
        lengths = pc.make_struct(lengths, ids)
        lengths = lengths.sort("descending", by=0)

        # Greedy BFD binning
        segment_tree = _SegmentTree(seq_length)
        segment_tree.add(seq_length)
        space_to_bin = defaultdict(deque)
        bins: list[dict] = []

        for length, idx in zip(lengths.field(0).to_numpy(), lengths.field(1).to_numpy(), strict=True):
            space = segment_tree.search(length)
            if space < seq_length:
                bin = space_to_bin[space].popleft()
            else:
                bin = {"ids": [], "length": 0}
                bins.append(bin)
            bin["ids"].append(idx)
            bin["length"] += length
            if space < seq_length and not space_to_bin[space]:
                segment_tree.remove(space)
            space = space - length
            space_to_bin[space].append(bin)
            if space > 0:
                segment_tree.add(space)

        # =====================================================================
        # PART 2: Filter low-density bins (NEW FIX)
        # =====================================================================

        total_bins_before = len(bins)
        min_tokens = int(min_fill_ratio * seq_length)

        bins_filtered = [bin for bin in bins if bin["length"] >= min_tokens]
        bins_dropped = total_bins_before - len(bins_filtered)

        if bins_dropped > 0:
            dropped_ratio = bins_dropped / total_bins_before
            logger.info(
                f"Dropped {bins_dropped}/{total_bins_before} low-density packed bins "
                f"({dropped_ratio:.1%}) with fill ratio < {min_fill_ratio:.0%} "
                f"(min_tokens={min_tokens}, seq_length={seq_length})"
            )
            # Log details about dropped bins for debugging
            dropped_bins = [bin for bin in bins if bin["length"] < min_tokens]
            if dropped_bins:
                dropped_lengths = [bin["length"] for bin in dropped_bins]
                logger.debug(
                    f"   Dropped bin lengths: min={min(dropped_lengths)}, "
                    f"max={max(dropped_lengths)}, mean={np.mean(dropped_lengths):.0f}"
                )
        else:
            logger.info(
                f"✓ All {total_bins_before} packed bins passed min_fill_ratio={min_fill_ratio:.0%} filter"
            )

        bins = bins_filtered

        # =====================================================================
        # PART 3: Reconstruct PyArrow table from filtered bins (original logic)
        # =====================================================================

        examples = pc.take(examples, [id_ for bin in bins for id_ in bin["ids"]])
        offsets = np.cumsum([0] + [bin["length"] for bin in bins])

        assert all(column.num_chunks == 1 for column in examples.columns)
        lengths = examples["seq_lengths"].chunks[0]
        examples = examples.drop_columns("seq_lengths")
        lengths = pa.ListArray.from_arrays(
            np.cumsum([0] + [len(bin["ids"]) for bin in bins], dtype=np.int32),
            lengths
        )

        columns = []
        for column in examples.columns:
            column = column.chunks[0]
            if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
                dtype = column.offsets.type.to_pandas_dtype()
                column = type(column).from_arrays(offsets.astype(dtype), column.values)
            columns.append(column)

        return pa.Table.from_arrays(columns + [lengths], names=examples.column_names + ["seq_lengths"])

    # Patch the function in the data_utils module
    data_utils._pack_bfd = filtered_pack_bfd
    logger.info(
        f"✅ Patched TRL _pack_bfd with min_fill_ratio={min_fill_ratio:.0%} "
        f"to prevent low-density packed sequences"
    )


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
    cfg: DictConfig,
    training_kwargs: Dict[str, Any],
    wandb_cfg: Dict[str, Any] | None,
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
    run_name = os.environ.get("WANDB_RUN_NAME")
    if not run_name:
        logger.warning("⚠️  WANDB_RUN_NAME not set! Run name should be provided by run_milestone_loop.sh")
        logger.warning("    Continuing without run name - WandB logging may be inconsistent")
    else:
        logger.info(f"Using run name: {run_name}")
        training_kwargs["run_name"] = run_name

    return training_kwargs


# =============================================================================
# Milestone Training Utilities
# =============================================================================

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
