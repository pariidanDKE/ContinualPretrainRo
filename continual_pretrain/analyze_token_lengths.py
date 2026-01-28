#!/usr/bin/env python3
"""Analyze token length distribution in mixed milestone dataset."""

import os
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load the dataset
dataset_path = "data/mixed_milestone_dataset"
print(f"Loading dataset from {dataset_path}...")
dataset = load_from_disk(dataset_path)

print(f"Total examples: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

# Get unique milestones
milestones = sorted(set(dataset['milestone_num']))
print(f"\nMilestones found: {milestones}")

# Thresholds to check
thresholds = [1024, 2048, 4096, 8192]

# Analyze each milestone separately
print("\n" + "="*80)
for milestone in milestones:
    milestone_data = dataset.filter(lambda x: x['milestone_num'] == milestone)
    token_lengths = milestone_data['num_tokens']

    print(f"\nMILESTONE {milestone}")
    print(f"  Total examples: {len(token_lengths)}")
    print(f"  Mean token length: {np.mean(token_lengths):.2f}")
    print(f"  Median token length: {np.median(token_lengths):.2f}")
    print(f"  Min token length: {min(token_lengths)}")
    print(f"  Max token length: {max(token_lengths)}")
    print(f"\n  Examples exceeding thresholds:")

    for threshold in thresholds:
        count = sum(1 for length in token_lengths if length > threshold)
        percentage = (count / len(token_lengths)) * 100
        print(f"    > {threshold:5d} tokens: {count:6d} examples ({percentage:5.2f}%)")

# Create histograms
print("\n" + "="*80)
print("\nGenerating histograms...")

n_milestones = len(milestones)
fig, axes = plt.subplots(n_milestones, 1, figsize=(12, 4 * n_milestones))

# Handle single milestone case
if n_milestones == 1:
    axes = [axes]

for idx, milestone in enumerate(milestones):
    milestone_data = dataset.filter(lambda x: x['milestone_num'] == milestone)
    token_lengths = milestone_data['num_tokens']

    ax = axes[idx]

    # Create histogram
    bins = np.arange(0, max(token_lengths) + 500, 100)
    ax.hist(token_lengths, bins=bins, alpha=0.7, edgecolor='black')

    # Add vertical lines for thresholds
    colors = ['red', 'orange', 'green', 'blue']
    for threshold, color in zip(thresholds, colors):
        ax.axvline(x=threshold, color=color, linestyle='--', linewidth=2,
                   label=f'{threshold} tokens', alpha=0.7)

    ax.set_xlabel('Token Length')
    ax.set_ylabel('Number of Examples')
    ax.set_title(f'Milestone {milestone} - Token Length Distribution (n={len(token_lengths)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = "token_length_analysis.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved histogram to: {output_file}")

# Create a single combined histogram showing all milestones
print("\nGenerating combined histogram...")
fig2, ax2 = plt.subplots(figsize=(14, 6))

all_token_lengths = []
milestone_labels = []

for milestone in milestones:
    milestone_data = dataset.filter(lambda x: x['milestone_num'] == milestone)
    token_lengths = milestone_data['num_tokens']
    all_token_lengths.append(token_lengths)
    milestone_labels.append(f'Milestone {milestone}')

# Create overlaid histograms
bins = np.arange(0, max([max(tl) for tl in all_token_lengths]) + 500, 100)
ax2.hist(all_token_lengths, bins=bins, alpha=0.6, label=milestone_labels, edgecolor='black')

# Add threshold lines
threshold_colors = ['red', 'orange', 'green', 'blue']
for threshold, color in zip(thresholds, threshold_colors):
    ax2.axvline(x=threshold, color=color, linestyle='--', linewidth=2,
                label=f'{threshold} tokens', alpha=0.8)

ax2.set_xlabel('Token Length')
ax2.set_ylabel('Number of Examples')
ax2.set_title('Token Length Distribution - All Milestones Combined')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_file2 = "token_length_combined.png"
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"Saved combined histogram to: {output_file2}")

# Percentile analysis
print("\n" + "="*80)
print("\nPercentile Analysis:")
for milestone in milestones:
    milestone_data = dataset.filter(lambda x: x['milestone_num'] == milestone)
    token_lengths = milestone_data['num_tokens']

    print(f"\nMilestone {milestone}:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(token_lengths, p)
        print(f"  {p:2d}th percentile: {value:.0f} tokens")

print("\n" + "="*80)
print("Analysis complete!")
