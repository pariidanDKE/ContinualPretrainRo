# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ContinualPretrainRo is a research framework for adapting English large language models (Llama, Gemma, Qwen) to Romanian through continual pretraining with LoRA (Low-Rank Adaptation). The project trains models incrementally on Romanian supervised fine-tuning datasets and evaluates them on Romanian benchmarks using a custom fork of lm-evaluation-harness.

## Key Architecture Concepts

### Milestone-Based Training
Training proceeds in token-based milestones (e.g., 1M, 2M, 3M tokens). At each milestone:
1. Train with LoRA adapters to target token count
2. Merge LoRA weights into base model
3. Evaluate merged model on Romanian benchmarks
4. Resume training from LoRA checkpoint for next milestone

All milestones in a session share a single WandB run ID for continuous tracking.

**LR Scheduling Across Milestones:**
Set `milestone.total_training_tokens` to enable consistent learning rate scheduling across all milestones. The `SchedulerFixCallback` (train_utils.py) ensures the LR scheduler uses the total token count, not just the current milestone, preventing the schedule from restarting at each checkpoint.

### Romanian Dialogue Tag System
The project uses special Romanian dialogue tags (`<utilizator>`, `<asistent>`, `<sistem>`) during data preparation. The `DataPreprocessor` (continual_pretrain/data_module.py) automatically converts these to model-specific chat templates (Llama, Gemma, Qwen) during training.

### Dual Model Support (PT/IT)
- **Pretrained (PT) models**: Base models without instruction tuning
- **Instruction-tuned (IT) models**: Models with chat capabilities
- During evaluation, PT models can use IT tokenizers' chat templates (see `PT_TO_IT_TOKENIZER_MAP` in train_utils.py)
- Perplexity tasks (ro_wiki, wikitext) never use chat templates; reasoning tasks do

### Dataset Registry Pattern
The `DatasetFormatterRegistry` (continual_pretrain/data_processing/dataset_registry.py) provides a plugin system for dataset-specific formatting. Register formatters with their corresponding column names, then apply them to datasets. Each dataset type (NoRobots, Dolly, Camel, etc.) in formatters.py has a dedicated formatter function that converts to the Romanian dialogue tag format (`<utilizator>`, `<asistent>`, `<sistem>`).

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (create .env file)
HF_TOKEN=your_huggingface_token
```

### Training

**Single training run:**
```bash
cd continual_pretrain
python train_milestone_segment.py \
    model.builder.model_name="meta-llama/Llama-3.2-1B" \
    model.lora.r=16 \
    milestone.target_tokens=1000000
```

**Milestone loop (recommended):**
```bash
cd continual_pretrain
./scripts/run_milestone_loop.sh
```
Edit the script to configure: model name, LoRA rank, batch size, tokens per milestone, number of milestones.

**Quick smoke test:**
```bash
python train_milestone_segment.py dataset.sample_size=1000
```

### Evaluation

**Evaluate a single model:**
```bash
cd continual_pretrain
python evaluate.py \
    model_path="google/gemma-3-1b-pt" \
    apply_chat_template=true \
    eval_batch_size=16
```

**Batch evaluation:**
```bash
./scripts/evaluate_model.sh
```
Edit the script to configure model paths and task lists.

**Available Romanian tasks:**
- `_ro_arc_challenge` - Science reasoning
- `_ro_winogrande` - Coreference resolution
- `_ro_belebele` - Reading comprehension
- `ro_wiki` - Perplexity on Romanian Wikipedia

**Available English baselines:**
- `arc_challenge`, `winogrande`, `wikitext`

### Data Processing

**Format datasets with Romanian dialogue tags:**
```bash
cd continual_pretrain
python data_processing/dataset_formatting.py
```

Configure datasets in `configs/dataset_formatting.yaml`. Output goes to `data/formatted_data/`.

## Hydra Configuration System

All scripts use Hydra for configuration. The base configs are in `continual_pretrain/configs/`.

### Override Syntax
```bash
# Single value
python train_milestone_segment.py model.lora.r=32

# Nested config
python train_milestone_segment.py model.builder.model_name="Qwen/Qwen2.5-1.5B"

# Boolean
python train_milestone_segment.py wandb.enabled=true

# Null value
python train_milestone_segment.py dataset.sample_size=null
```

### Key Config Sections

**train_model.yaml:**
- `dataset` - Single dataset config (legacy, use data_builder instead)
- `data_builder` - Multi-dataset mixing with proportions
- `data_collator` - Packing configuration (pack_length, packing enabled/disabled)
- `model.builder` - Model loading (model_name, use_unsloth, quantization)
- `model.lora` - LoRA parameters (r, lora_alpha, target_modules)
- `training_args` - Standard HuggingFace training args
- `milestone` - Milestone training config (target_tokens, resume_from_checkpoint, merged_output_dir)
- `wandb` - WandB tracking config

**evaluate_ro.yaml:**
- `model_path` - Model to evaluate
- `apply_chat_template` - Whether to use chat template
- `tasks_to_run` - List of tasks to evaluate
- `preplexity_tasks` - Tasks that should never use chat template

## Important File Locations

**Entry points:**
- `continual_pretrain/train_milestone_segment.py` - Milestone training
- `continual_pretrain/evaluate.py` - Evaluation runner
- `continual_pretrain/data_processing/dataset_formatting.py` - Dataset formatting

**Core modules:**
- `continual_pretrain/model.py` - Model loading with LoRA
- `continual_pretrain/data_module.py` - Data preprocessing and collators
- `continual_pretrain/train_utils.py` - Training utilities (dataset mixing, WandB, checkpointing)
- `continual_pretrain/data_processing/formatters.py` - Dataset-specific formatters

**Configs:**
- `continual_pretrain/configs/train_model.yaml`
- `continual_pretrain/configs/evaluate_ro.yaml`
- `continual_pretrain/configs/dataset_formatting.yaml`

**Outputs (gitignored):**
- `continual_pretrain/outputs/` - Training checkpoints
- `continual_pretrain/eval_results/` - Evaluation JSON results
- `continual_pretrain/wandb/` - Local WandB logs
- `continual_pretrain/data/` - Formatted datasets

**Evaluation harness:**
- `llm-eval-harness-ro/lm-evaluation-harness/` - Custom fork with Romanian tasks

## Working with LoRA Checkpoints

**Checkpoint structure:**
Checkpoints contain LoRA adapters only, not full models:
```
outputs/milestone_meta-llama__Llama-3.2-1B_r16/
├── checkpoint-488/          # LoRA adapters at step 488
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── merged/                  # Full merged models for evaluation
    └── checkpoint-488/
        ├── model.safetensors
        ├── config.json
        └── ...
```

**Merging LoRA weights:**
The `merge_lora_checkpoint()` function in train_utils.py merges LoRA adapters with the base model. This happens automatically at the end of each milestone when `milestone.merged_output_dir` is set.

**Resuming training:**
```bash
python train_milestone_segment.py \
    milestone.resume_from_checkpoint="outputs/milestone_model_r16/checkpoint-488" \
    milestone.target_tokens=2000000
```

## Unsloth Optimization

Set `model.builder.use_unsloth=true` for faster training with lower memory usage. Unsloth provides optimized model loading for Llama, Gemma, and Qwen models. It's particularly beneficial for longer sequences and larger batch sizes.

## WandB Integration

**Environment variables:**
- `WANDB_RUN_ID` - Reuse run ID across milestone loop iterations
- `WANDB_RESUME="allow"` - Allow resuming existing runs

**Configuration:**
```yaml
wandb:
  enabled: true
  project: "RoLLM"
  run_name: "descriptive_name"
```

**Milestone loop tracking:**
The `run_milestone_loop.sh` script generates a single WandB run ID for the entire training session, ensuring all milestones log to the same run.

## Dataset Mixing

Use `data_builder` in configs for proportional multi-dataset mixing:
```yaml
data_builder:
  enabled: true
  datasets:
    - name: "danp27/norobots_sft"
      split: "train"
      proportion: 0.5
    - name: "./data/formatted_data/dolly"
      split: "train"
      proportion: 0.5
```

The system automatically:
1. Loads and samples datasets according to proportions
2. Applies the DataPreprocessor to convert Romanian tags to chat templates
3. Tokenizes with proper loss masking (train only on assistant responses)

## Common Patterns

### Calculate training steps for token target
```python
from train_utils import calculate_max_steps

max_steps = calculate_max_steps(
    target_tokens=1_000_000,
    batch_size=32,
    grad_accum_steps=2,
    max_length=2048
)
```

### Load model with LoRA
```python
from model import ModelBuilder, ModelBuilderConfig, LoraAdapterConfig

model_cfg = ModelBuilderConfig(model_name="meta-llama/Llama-3.2-1B")
lora_cfg = LoraAdapterConfig(r=16, lora_alpha=32)
builder = ModelBuilder(model_cfg, lora_cfg)
model, tokenizer = builder.build()
```

### Format dataset with Romanian tags
```python
from data_processing.dataset_formatting import DatasetFormatterRegistry

registry = DatasetFormatterRegistry()
formatter = registry.get_formatter("norobots")
formatted = dataset.map(formatter)
```
