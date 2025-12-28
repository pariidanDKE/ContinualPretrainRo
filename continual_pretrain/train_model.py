import logging
import os
import re
from typing import Any, Dict
import unsloth
from unsloth import FastLanguageModel

import hydra
from datasets import Dataset, DatasetDict, interleave_datasets, load_dataset, load_from_disk
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments, set_seed
from trl import SFTConfig, SFTTrainer

from data_module import (
    DataPreprocessor,
    PackedSequenceDataCollator,
    SimplePaddingCollator,
)
from model import (
    LoraAdapterConfig,
    ModelBuilder,
    ModelBuilderConfig,
)

logger = logging.getLogger(__name__)


def _to_dict(cfg_section: Any, *, resolve: bool = True) -> Dict[str, Any]:
    """Convert an OmegaConf section or plain dict to a standard dict."""
    if cfg_section is None:
        return {}
    if isinstance(cfg_section, dict):
        return dict(cfg_section)
    if OmegaConf.is_config(cfg_section):
        return OmegaConf.to_container(cfg_section, resolve=resolve)  # type: ignore[arg-type]
    raise TypeError(f"Unsupported config section type: {type(cfg_section)!r}")


def _load_single_dataset(
    entry_cfg: Dict[str, Any],
    default_cfg: Dict[str, Any],
    tokenizer,
) -> Any:
    
    print(f"Per dataset config {entry_cfg}")
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
    num_proc = entry_cfg.get("num_proc", default_cfg["num_proc"])

    if os.path.isdir(name):
        raw = load_from_disk(name)
        if isinstance(raw, DatasetDict):
            ds = raw[split] if split in raw else raw[list(raw.keys())[0]]
        else:
            ds = raw
    else:
        ds = load_dataset(name, split=split)
    if sample_size:
        ds = ds.select(range(int(sample_size)))

    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        text_field=text_field,
        add_bos_eos=add_bos_eos,
        use_sft_config= ~default_cfg.get("tokenize") # if we use sft config we don't need to tokenize
    )

    remove_cols = list(ds.column_names)
    ds = ds.map(
        preprocessor,
        remove_columns=remove_cols,
        num_proc=num_proc,
    )
    return ds


def _prepare_dataset(cfg, tokenizer):
    """Load, optionally mix, and tokenize datasets."""
    dataset_defaults = _to_dict(cfg.dataset)
    data_builder = cfg.get("data_builder")

    if data_builder and data_builder.get("enabled"):
        builder_cfg = _to_dict(data_builder, resolve=False)
        datasets_cfg = builder_cfg.get("datasets", [])

        # First pass: collect proportions and calculate total
        entries_with_proportions = []
        total_proportion = 0
        for entry in datasets_cfg:
            entry_dict = _to_dict(entry)
            proportion = entry_dict.get("proportion", 1)
            if proportion <= 0:
                continue
            entries_with_proportions.append((entry_dict, proportion))
            total_proportion += proportion

        if not entries_with_proportions:
            raise ValueError("data_builder.enabled is true but no valid datasets were provided.")

        # Second pass: load datasets with adjusted sample sizes
        processed = []
        proportions = []
        total_sample_size = dataset_defaults.get("sample_size")

        for entry_dict, proportion in entries_with_proportions:
            # Adjust sample_size for this dataset based on its proportion
            if total_sample_size and "sample_size" not in entry_dict:
                adjusted_sample_size = int(total_sample_size * proportion / total_proportion)
                entry_dict["sample_size"] = adjusted_sample_size

            ds = _load_single_dataset(entry_dict, dataset_defaults, tokenizer)
            processed.append(ds)
            proportions.append(proportion)

        probabilities = [p / total_proportion for p in proportions]
        seed = int(cfg.get("seed") or 0)
        return interleave_datasets(processed, probabilities=probabilities, seed=seed)

    return _load_single_dataset(dataset_defaults, dataset_defaults, tokenizer)


def _build_collator(cfg, tokenizer):
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


def _slugify(text: Any) -> str:
    """Convert arbitrary text into a filesystem/W&B friendly token."""
    text = str(text).strip()
    text = re.sub(r"[^\w.-]+", "-", text)
    text = text.strip("-_")
    return text or "na"


def _compose_run_name(cfg: DictConfig, training_kwargs: Dict[str, Any], wandb_cfg: Dict[str, Any]) -> str:
    parts = []
    user_prefix = wandb_cfg.get("run_name")
    if user_prefix:
        parts.append(_slugify(user_prefix))

    model_name = cfg.model.builder.get("model_name")
    if model_name:
        parts.append(_slugify(model_name.split("/")[-1]))

    dataset_name = cfg.dataset.get("name")
    if dataset_name:
        parts.append(f"ds-{_slugify(dataset_name.split('/')[-1])}")

    sample_size = cfg.dataset.get("sample_size")
    parts.append(f"sample-{_slugify(sample_size or 'full')}")

    for label, key in [
        ("bs", "per_device_train_batch_size"),
        ("ga", "gradient_accumulation_steps"),
        ("ep", "num_train_epochs"),
        ("lr", "learning_rate"),
    ]:
        value = training_kwargs.get(key)
        if value is not None:
            parts.append(f"{label}-{_slugify(value)}")

    return "__".join(parts)


def _apply_wandb_config(
    cfg: DictConfig,
    training_kwargs: Dict[str, Any],
    wandb_cfg: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Set up W&B environment variables and Trainer args (always enabled)."""
    wandb_cfg = wandb_cfg or {}
    project = wandb_cfg.get("project")

    if project:
        os.environ["WANDB_PROJECT"] = str(project)

    training_kwargs["report_to"] = "wandb"
    training_kwargs["run_name"] = _compose_run_name(cfg, training_kwargs, wandb_cfg)
    return training_kwargs


@hydra.main(config_path="configs", config_name="train_model.yaml")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Starting training run")
    if cfg.get("seed") is not None:
        set_seed(int(cfg.seed))

    # ---- Map PT models to their IT tokenizer equivalents ----
    PT_TO_IT_TOKENIZER_MAP = {
        "google/gemma-3-1b-pt": "google/gemma-3-1b-it",
        "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    }

    builder_cfg = ModelBuilderConfig(**_to_dict(cfg.model.builder))
    lora_cfg = LoraAdapterConfig(**_to_dict(cfg.model.lora))
    builder = ModelBuilder(builder_cfg, lora_cfg)
    model, tokenizer = builder.build()

    # If training a PT model with SFT, load IT tokenizer for chat template
    model_name = builder_cfg.model_name
    use_sft = not cfg.dataset.get("tokenize", True)
    if use_sft and model_name in PT_TO_IT_TOKENIZER_MAP:
        from transformers import AutoTokenizer
        it_tokenizer_name = PT_TO_IT_TOKENIZER_MAP[model_name]
        logger.info(f"Loading IT tokenizer {it_tokenizer_name} for chat template (PT model: {model_name})")
        it_tokenizer = AutoTokenizer.from_pretrained(it_tokenizer_name, trust_remote_code=True)
        # Copy chat template to our tokenizer
        tokenizer.chat_template = it_tokenizer.chat_template
        logger.info(f"Applied IT chat template to PT model tokenizer")

    dataset = _prepare_dataset(cfg, tokenizer)

    # Check if we're using SFT config (i.e., not tokenizing in preprocessing)
    use_sft = not cfg.dataset.get("tokenize", True)

    training_kwargs = _to_dict(cfg.training_args)
    wandb_cfg = _to_dict(cfg.get("wandb"), resolve=False)
    training_kwargs = _apply_wandb_config(cfg, training_kwargs, wandb_cfg)

    if use_sft:
        # Use SFTTrainer with chat template
        logger.info("Using SFTTrainer with chat template")

        text_field = cfg.dataset.get("text_field", "formatted_text")

        # Only generate chat template if not already set (i.e., not a PT model using IT template)
        if not tokenizer.chat_template and model_name not in PT_TO_IT_TOKENIZER_MAP:
            # Create preprocessor to get the chat template for IT models
            preprocessor = DataPreprocessor(
                tokenizer=tokenizer,
                text_field=text_field,
                add_bos_eos=cfg.dataset.get("add_bos_eos", True),
                use_sft_config=True
            )
            chat_template = preprocessor.get_chat_template()
            logger.info(f"Generated chat template for {preprocessor.family} family")
            tokenizer.chat_template = chat_template
        else:
            logger.info(f"Using existing chat template (IT template already applied)")

        # Create SFTConfig from training kwargs
        training_args = SFTConfig(
            **training_kwargs,
            packing = cfg.data_collator.get("packing", False), # if false still enable auto-padding from unsloth 
            dataset_text_field=text_field,
            max_length=cfg.data_collator.get("pack_length", 512),
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
           
        )
    else:
        # Use regular Trainer with custom collator
        logger.info("Using regular Trainer with custom data collator")
        collator = _build_collator(cfg, tokenizer)
        training_args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=collator,
        )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training complete. Artifacts stored in %s", training_args.output_dir)


if __name__ == "__main__":
    main()
