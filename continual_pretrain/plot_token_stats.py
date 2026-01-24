"""
Plot token statistics from training logs to visualize packing efficiency.

This script parses train_milestone_segment.log and plots:
- Real token count per step
- Number of batches per step (to detect gradient accumulation anomalies)
- Milestone boundaries (vertical lines)
"""

import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_token_stats(log_file):
    """
    Parse token statistics from training log.

    Returns:
        steps: List of step numbers
        real_tokens: List of real token counts
        num_batches: List of batch counts per step
        eos_counts: List of EOS token counts
        milestones: List of (step, milestone_number) tuples for boundaries
        training_config: Dict with batch_size, grad_accum_steps, seq_length
    """
    steps = []
    real_tokens = []
    num_batches = []
    eos_counts = []
    milestones = []

    # Parse training configuration
    training_config = {
        'batch_size': 32,
        'grad_accum_steps': 2,  # default
        'seq_length': 2048
    }

    # Pattern: [Step 39] Token Statistics across 2 batches:
    #   Real tokens:    130,005 / 130,005 (100.00%)
    #   ...
    #   EOS tokens:     400 (distinct sequences)

    with open(log_file, 'r') as f:
        lines = f.readlines()

    i = 0
    current_milestone = None

    while i < len(lines):
        line = lines[i]

        # Parse training configuration from kwargs
        if "'gradient_accumulation_steps':" in line:
            match = re.search(r"'gradient_accumulation_steps':\s*(\d+)", line)
            if match:
                training_config['grad_accum_steps'] = int(match.group(1))
        if "'per_device_train_batch_size':" in line:
            match = re.search(r"'per_device_train_batch_size':\s*(\d+)", line)
            if match:
                training_config['batch_size'] = int(match.group(1))
        if "'max_length':" in line:
            match = re.search(r"'max_length':\s*(\d+)", line)
            if match:
                training_config['seq_length'] = int(match.group(1))

        # Detect milestone boundaries
        if "Current milestone:" in line:
            match = re.search(r'Current milestone:\s*(\d+)', line)
            if match:
                current_milestone = int(match.group(1))

        # Parse token statistics
        if "Token Statistics across" in line:
            # Extract step number
            step_match = re.search(r'\[Step (\d+)\]', line)
            # Extract batch count
            batch_match = re.search(r'across (\d+) batches', line)

            if step_match and batch_match:
                step = int(step_match.group(1))
                batch_count = int(batch_match.group(1))

                # Check if this is the first step of a new milestone
                if steps and step > steps[-1] + 1:
                    # Gap in steps indicates new milestone
                    milestones.append((step, current_milestone))
                elif not steps:
                    # Very first step
                    milestones.append((step, current_milestone))

                # Next line should have real tokens
                if i + 1 < len(lines):
                    real_token_match = re.search(r'Real tokens:\s*([\d,]+)\s*/\s*[\d,]+', lines[i + 1])
                    if real_token_match:
                        real_token_count = int(real_token_match.group(1).replace(',', ''))

                        # Look ahead for EOS tokens
                        eos_count = 0
                        for j in range(i + 1, min(i + 5, len(lines))):
                            eos_match = re.search(r'EOS tokens:\s*(\d+)', lines[j])
                            if eos_match:
                                eos_count = int(eos_match.group(1))
                                break

                        steps.append(step)
                        real_tokens.append(real_token_count)
                        num_batches.append(batch_count)
                        eos_counts.append(eos_count)

        i += 1

    return steps, real_tokens, num_batches, eos_counts, milestones, training_config


def plot_token_statistics(steps, real_tokens, num_batches, eos_counts, milestones, training_config):
    """
    Create visualization of token statistics.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Expected tokens per step from training config
    expected_tokens = (training_config['batch_size'] *
                      training_config['grad_accum_steps'] *
                      training_config['seq_length'])
    expected_batches = training_config['grad_accum_steps']

    # Plot 1: Real token count
    ax1 = axes[0]
    ax1.plot(steps, real_tokens, 'b-', linewidth=1, alpha=0.7, label='Real tokens')
    ax1.axhline(y=expected_tokens, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Expected ({expected_tokens:,})')
    ax1.set_ylabel('Real Tokens per Step', fontsize=11)
    ax1.set_title('Token Statistics Across Training', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(real_tokens) * 0.95, max(real_tokens) * 1.02])

    # Add milestone boundaries
    for step, milestone_num in milestones:
        ax1.axvline(x=step, color='green', linestyle=':', linewidth=1.5, alpha=0.6)
        ax1.text(step, ax1.get_ylim()[1] * 0.98, f'M{milestone_num}',
                fontsize=9, ha='left', va='top', color='green', fontweight='bold')

    # Plot 2: Packing efficiency (real tokens / expected tokens)
    ax2 = axes[1]
    efficiency = [(rt / expected_tokens) * 100 for rt in real_tokens]
    ax2.plot(steps, efficiency, 'g-', linewidth=1, alpha=0.7)
    ax2.axhline(y=100, color='r', linestyle='--', linewidth=1, alpha=0.5, label='100% efficiency')
    ax2.set_ylabel('Packing Efficiency (%)', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([min(efficiency) * 0.98, 101])

    # Add milestone boundaries
    for step, milestone_num in milestones:
        ax2.axvline(x=step, color='green', linestyle=':', linewidth=1.5, alpha=0.6)

    # Plot 3: Number of batches (should be constant = expected_batches)
    ax3 = axes[2]
    ax3.plot(steps, num_batches, 'orange', marker='o', markersize=3, linewidth=1, alpha=0.7)
    ax3.axhline(y=expected_batches, color='r', linestyle='--', linewidth=1, alpha=0.5,
                label=f'Expected (grad_accum={expected_batches})')
    ax3.set_ylabel('Batches per Step', fontsize=11)
    ax3.set_xlabel('Training Step', fontsize=11)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([min(num_batches) - 0.5, max(num_batches) + 0.5])

    # Add milestone boundaries
    for step, milestone_num in milestones:
        ax3.axvline(x=step, color='green', linestyle=':', linewidth=1.5, alpha=0.6)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'token_stats_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")

    # Show statistics
    print("\n" + "=" * 60)
    print("TOKEN STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Training Configuration:")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Gradient accumulation: {training_config['grad_accum_steps']}")
    print(f"  Sequence length: {training_config['seq_length']}")
    print(f"\nTotal steps analyzed: {len(steps)}")
    print(f"Step range: {min(steps)} - {max(steps)}")
    print(f"\nToken Count Statistics:")
    print(f"  Expected per step: {expected_tokens:,}")
    print(f"  Average: {sum(real_tokens) / len(real_tokens):,.0f}")
    print(f"  Min: {min(real_tokens):,} (step {steps[real_tokens.index(min(real_tokens))]})")
    print(f"  Max: {max(real_tokens):,} (step {steps[real_tokens.index(max(real_tokens))]})")
    print(f"\nPacking Efficiency:")
    print(f"  Average: {sum(efficiency) / len(efficiency):.2f}%")
    print(f"  Min: {min(efficiency):.2f}%")
    print(f"  Max: {max(efficiency):.2f}%")
    print(f"\nBatch Count per Step:")
    print(f"  Expected: {expected_batches} (gradient_accumulation_steps)")
    print(f"  Average: {sum(num_batches) / len(num_batches):.2f}")
    print(f"  Anomalies: {sum(1 for b in num_batches if b != expected_batches)} steps with != {expected_batches} batches")

    if any(b != expected_batches for b in num_batches):
        anomaly_steps = [steps[i] for i, b in enumerate(num_batches) if b != expected_batches]
        print(f"    Steps with anomalies: {anomaly_steps}")

    print(f"\nMilestone Boundaries:")
    for step, milestone_num in milestones:
        print(f"  Milestone {milestone_num} starts at step {step}")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    import sys

    # Allow log file to be passed as argument
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    else:
        log_file = Path(__file__).parent / "train_milestone_segment.log"

    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        exit(1)

    print(f"Parsing log file: {log_file}")
    steps, real_tokens, num_batches, eos_counts, milestones, training_config = parse_token_stats(log_file)

    if not steps:
        print("No token statistics found in log file!")
        exit(1)

    print(f"Found {len(steps)} steps with token statistics")
    plot_token_statistics(steps, real_tokens, num_batches, eos_counts, milestones, training_config)
