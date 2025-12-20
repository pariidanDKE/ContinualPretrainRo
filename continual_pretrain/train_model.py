from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict

import hydra
from datasets import Dataset, DatasetDict, interleave_datasets, load_dataset, load_from_disk
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments, set_seed

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
        processed = []
        proportions = []
        for entry in datasets_cfg:
            entry_dict = _to_dict(entry)
            proportion = entry_dict.get("proportion", 1)
            if proportion <= 0:
                continue
            ds = _load_single_dataset(entry_dict, dataset_defaults, tokenizer)
            processed.append(ds)
            proportions.append(proportion)

        if not processed:
            raise ValueError("data_builder.enabled is true but no valid datasets were provided.")

        total = sum(proportions)
        probabilities = [p / total for p in proportions]
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

    builder_cfg = ModelBuilderConfig(**_to_dict(cfg.model.builder))
    lora_cfg = LoraAdapterConfig(**_to_dict(cfg.model.lora))
    builder = ModelBuilder(builder_cfg, lora_cfg)
    model, tokenizer = builder.build()

    dataset = _prepare_dataset(cfg, tokenizer)
    collator = _build_collator(cfg, tokenizer)

    training_kwargs = _to_dict(cfg.training_args)
    wandb_cfg = _to_dict(cfg.get("wandb"), resolve=False)
    training_kwargs = _apply_wandb_config(cfg, training_kwargs, wandb_cfg)
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
