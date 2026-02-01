"""
Build mixed milestone dataset for continual pretraining.

This script:
1. Loads and mixes multiple datasets according to proportions
2. Adds token counts to each example
3. Calculates milestone boundaries based on cumulative token counts
4. Tags each example with its milestone number
5. Saves the dataset and metadata for use in milestone training

Usage:
    # Build with default config (5 milestones, 100M tokens each)
    python build_mixed_dataset.py

    # Build with custom milestones
    python build_mixed_dataset.py \
        milestone.num_milestones=10 \
        milestone.tokens_per_milestone=50000000

    # Quick test with sample data
    python build_mixed_dataset.py dataset.sample_size=10000
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from transformers import AutoTokenizer
from train_utils import prepare_dataset, to_dict

logger = logging.getLogger(__name__)


def add_token_counts(dataset, tokenizer, max_length):
    """Add token count column to dataset by tokenizing formatted_text."""
    logger.info("Adding token counts to dataset...")

    def count_tokens(example):
        """Count tokens by tokenizing the formatted_text field."""
        text = example['formatted_text']
        # Tokenize to count (don't save the tokens, just the count)
        token_ids = tokenizer(text, add_special_tokens=False, max_length = max_length, truncation = True, padding=False)['input_ids']
        return {'num_tokens': len(token_ids)}

    dataset = dataset.map(
        count_tokens,
        num_proc=4,
        desc="Counting tokens"
    )

    total_tokens = sum(dataset['num_tokens'])
    logger.info(f"✓ Total tokens in dataset: {total_tokens:,}")
    return dataset, total_tokens


def calculate_milestone_boundaries(token_counts, num_milestones, tokens_per_milestone, buffer_multiplier=1.5):
    """
    Calculate example indices where each milestone should end.

    Args:
        token_counts: Array of token counts per example
        num_milestones: Number of milestones to create
        tokens_per_milestone: Target training tokens per milestone
        buffer_multiplier: Multiplier for shard size (default 1.5x for 50% buffer)

    Returns:
        List of boundary indices, total_target_tokens, cumulative_tokens
    """
    logger.info("Calculating milestone boundaries...")
    logger.info(f"Using buffer multiplier: {buffer_multiplier}x")

    cumulative_tokens = np.cumsum(token_counts)
    total_tokens = cumulative_tokens[-1]
    total_target_tokens = num_milestones * tokens_per_milestone

    # Calculate boundaries for all milestones using buffer
    boundaries = []
    for i in range(1, num_milestones + 1):  # Include the last milestone
        # Apply buffer to create larger shards
        target_tokens_with_buffer = int(i * tokens_per_milestone * buffer_multiplier)

        if target_tokens_with_buffer >= total_tokens:
            logger.warning(
                f"Milestone {i} target with buffer ({target_tokens_with_buffer:,} tokens) exceeds "
                f"total dataset size ({total_tokens:,} tokens). "
                f"Will create {i} milestones instead of {num_milestones}."
            )
            break

        # Find first index where cumulative sum >= target
        boundary_idx = int(np.searchsorted(cumulative_tokens, target_tokens_with_buffer, side='left'))
        boundaries.append(boundary_idx)

    actual_num_milestones = len(boundaries)
    logger.info(f"✓ Calculated {actual_num_milestones} milestone boundaries")
    logger.info(f"✓ Target training tokens (per milestone): {tokens_per_milestone:,}")
    logger.info(f"✓ Actual shard size (with {buffer_multiplier}x buffer): {int(tokens_per_milestone * buffer_multiplier):,}")
    logger.info(f"✓ Total target training tokens: {total_target_tokens:,}")
    logger.info(f"✓ Total dataset tokens: {total_tokens:,}")

    if total_tokens > total_target_tokens:
        excess_tokens = total_tokens - total_target_tokens
        logger.info(f"✓ Excess tokens (will be unused): {excess_tokens:,}")

    return boundaries, total_target_tokens, cumulative_tokens


def assign_milestone_numbers(dataset, boundaries):
    """Assign milestone number to each example. Examples beyond boundaries are marked as -1 (unused)."""
    logger.info("Assigning milestone numbers to examples...")

    num_milestones = len(boundaries)

    def assign_milestone(example, idx):
        """Determine which milestone this example belongs to."""
        # Find which milestone this example falls into
        milestone_num = 0
        for i, boundary in enumerate(boundaries):
            if idx >= boundary:
                milestone_num = i + 1
            else:
                break

        # If idx is beyond all boundaries, mark as unused
        if num_milestones > 0 and idx >= boundaries[-1]:
            milestone_num = -1

        return {'milestone_num': milestone_num}

    dataset = dataset.map(
        assign_milestone,
        with_indices=True,
        num_proc=4,
        desc="Assigning milestones"
    )

    logger.info(f"✓ Assigned milestone numbers (0 to {num_milestones - 1}, -1 for unused)")
    return dataset


def print_statistics(dataset, num_milestones, tokens_per_milestone):
    """Print comprehensive statistics about the milestone dataset."""
    print("\n" + "=" * 70)
    print("MILESTONE DATASET STATISTICS")
    print("=" * 70)

    total_tokens_used = 0
    total_examples_used = 0

    for i in range(num_milestones):
        milestone_subset = dataset.filter(
            lambda x: x['milestone_num'] == i,
            num_proc=4
        )

        if len(milestone_subset) == 0:
            print(f"\nMilestone {i}: EMPTY (no data)")
            continue

        num_examples = len(milestone_subset)
        tokens = sum(milestone_subset['num_tokens'])
        avg_tokens = tokens / num_examples if num_examples > 0 else 0

        total_tokens_used += tokens
        total_examples_used += num_examples

        print(f"\nMilestone {i}:")
        print(f"  Examples:          {num_examples:>12,}")
        print(f"  Tokens:            {tokens:>12,}")
        print(f"  Avg tokens/example: {avg_tokens:>11.1f}")
        print(f"  Target tokens:     {tokens_per_milestone:>12,}")

        # Show percentage of target
        if tokens_per_milestone > 0:
            pct = (tokens / tokens_per_milestone) * 100
            print(f"  % of target:       {pct:>11.1f}%")

    # Check for validation data
    val_subset = dataset.filter(
        lambda x: x['milestone_num'] == 99,
        num_proc=4
    )

    if len(val_subset) > 0:
        val_tokens = sum(val_subset['num_tokens'])
        print(f"\nVALIDATION DATA (shared across all milestones):")
        print(f"  Examples:          {len(val_subset):>12,}")
        print(f"  Tokens:            {val_tokens:>12,}")
        avg_val_tokens = val_tokens / len(val_subset) if len(val_subset) > 0 else 0
        print(f"  Avg tokens/example: {avg_val_tokens:>11.1f}")

    # Check for unused data
    unused_subset = dataset.filter(
        lambda x: x['milestone_num'] == -1,
        num_proc=4
    )

    if len(unused_subset) > 0:
        unused_tokens = sum(unused_subset['num_tokens'])
        print(f"\nUNUSED DATA (beyond training target):")
        print(f"  Examples:          {len(unused_subset):>12,}")
        print(f"  Tokens:            {unused_tokens:>12,}")

    print("\n" + "-" * 70)
    print(f"TOTAL (used for training):")
    print(f"  Examples:          {total_examples_used:>12,}")
    print(f"  Tokens:            {total_tokens_used:>12,}")
    if total_examples_used > 0:
        print(f"  Avg tokens/example: {total_tokens_used/total_examples_used:>11.1f}")
    print("=" * 70 + "\n")


def save_metadata(output_dir, cfg, num_milestones, total_tokens, total_examples, boundaries):
    """Save metadata about the built dataset."""
    metadata = {
        "build_timestamp": datetime.now().isoformat(),
        "dataset_seed": int(cfg.get('dataset_seed', 42)),
        "num_milestones": num_milestones,
        "tokens_per_milestone": int(cfg.milestone.tokens_per_milestone),
        "total_tokens": int(total_tokens),
        "total_examples": int(total_examples),
        "milestone_boundaries": [int(b) for b in boundaries],
        "config_snapshot": {
            "data_builder": OmegaConf.to_container(cfg.data_builder, resolve=True),
            "model_name": cfg.model.builder.model_name,
        }
    }

    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved metadata to: {metadata_path}")
    return metadata


@hydra.main(config_path="configs", config_name="build_dataset.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build and save milestone-tagged dataset."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    logger.info("=" * 80)
    logger.info("BUILDING MIXED MILESTONE DATASET")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    dataset_seed = int(cfg.get('seed', 42))
    set_seed(dataset_seed)
    logger.info(f"Dataset seed: {dataset_seed}")

    # Extract configuration
    output_dir = cfg.output_dir
    num_milestones = int(cfg.milestone.num_milestones)
    max_length = int(cfg.data_builder.max_length)

    tokens_per_milestone = int(cfg.milestone.tokens_per_milestone)
    logger.info(f"Output directory: {output_dir}")


    tokenizer = AutoTokenizer.from_pretrained(cfg.model.builder.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and mix datasets with train/validation split
    logger.info("\n" + "-" * 80)
    train_dataset, val_dataset = prepare_dataset(cfg, tokenizer, return_validation=True, validation_split=0.005)
    # Shuffle with fixed seed for reproducibility
    train_dataset = train_dataset.shuffle(seed=dataset_seed)
    val_dataset = val_dataset.shuffle(seed=dataset_seed)

    # Add token counts to training data
    logger.info("\n" + "-" * 80)
    train_dataset, total_train_tokens = add_token_counts(train_dataset, tokenizer, max_length)

    # Calculate milestone boundaries (only for training data)
    logger.info("\n" + "-" * 80)
    token_counts = np.array(train_dataset['num_tokens'])
    boundaries, _, _ = calculate_milestone_boundaries(
        token_counts,
        num_milestones,
        tokens_per_milestone
    )

    # Actual number of milestones (may be less if dataset is too small)
    actual_num_milestones = len(boundaries)
    
    print(f"Set Max length of sequence to be {max_length} to reflect truncation during training!")

    # Assign milestone numbers to training data
    logger.info("\n" + "-" * 80)
    train_dataset = assign_milestone_numbers(train_dataset, boundaries)

    # Add token counts to validation data
    val_dataset, total_val_tokens = add_token_counts(val_dataset, tokenizer, max_length)

    # Assign milestone_num=99 to all validation examples
    val_dataset = val_dataset.map(
        lambda x: {'milestone_num': 99},
        num_proc=4,
        desc="Assigning validation milestone"
    )

    # Combine train and validation datasets
    from datasets import concatenate_datasets
    dataset = concatenate_datasets([train_dataset, val_dataset])
    total_tokens = total_train_tokens + total_val_tokens

    # Print statistics
    logger.info("\n" + "-" * 80)
    print_statistics(dataset, actual_num_milestones, tokens_per_milestone)

    # Save dataset
    logger.info("-" * 80)
    logger.info(f"Saving dataset to: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(output_path))

    # Save metadata
    logger.info("\n" + "-" * 80)
    save_metadata(
        output_dir,
        cfg,
        actual_num_milestones,
        total_tokens,
        len(dataset),
        boundaries
    )

if __name__ == "__main__":
    main()
