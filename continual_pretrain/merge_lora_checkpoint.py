"""
Merge LoRA adapter weights into base model for standalone evaluation.

Usage:
    python merge_lora_checkpoint.py <checkpoint_path> <output_path>
"""

import sys
import json
import logging
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_lora_checkpoint(checkpoint_path: str, output_path: str):
    """Merge LoRA weights into base model and save."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    # Read adapter config to get base model
    adapter_config_path = checkpoint_path / "adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config["base_model_name_or_path"]

    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info(f"Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))

    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged model: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path))

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("✅ Merge complete")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_lora_checkpoint.py <checkpoint_path> <output_path>")
        sys.exit(1)

    merge_lora_checkpoint(sys.argv[1], sys.argv[2])
