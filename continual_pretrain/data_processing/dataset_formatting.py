import hydra
from omegaconf import DictConfig
import logging
from datasets import load_dataset
from pathlib import Path
import os
from dotenv import load_dotenv

from dataset_registry import DatasetFormatterRegistry
from formatters import (
    format_messages_standard,
    format_dolly_context_instruction,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Initialize global registry
formatter_registry = DatasetFormatterRegistry()

# Register all available formatters
formatter_registry.register(
    "OpenLLM-Ro/ro_sft_norobots",
    format_messages_standard,
    columns="messages"
)

formatter_registry.register(
    "OpenLLM-Ro/ro_sft_dolly",
    format_dolly_context_instruction,
    columns=["context", "instruction", "response"]
)



@hydra.main(config_path="../configs", config_name="dataset_formatting.yaml", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to format datasets based on Hydra configuration.
    
    Config structure:
        datasets:
          - name: "OpenLLM-Ro/ro_sft_norobots"
            split: "train"
            output_path: "./formatted_data/norobots"
            push_to_hub: false
          - name: "OpenLLM-Ro/ro_sft_dolly"
            split: "train"
            output_path: "./formatted_data/dolly"
            push_to_hub: false
        
        special_tokens:
          bos: "<s>"
          eos: "</s>"
          user: "<utilizator>"
          assistant: "<asistent>"
          system: "<sistem>"
        
        num_proc: 4
    """
    logger.info("Starting dataset formatting pipeline")
    load_dotenv()
    
    # Get HuggingFace token if needed
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        logger.info("✅ HuggingFace token loaded")
    
    # Get special tokens from config
    special_tokens = cfg.get("special_tokens", {})
    user_token = special_tokens.get("user", "<utilizator>")
    assistant_token = special_tokens.get("assistant", "<asistent>")
    system_token = special_tokens.get("system", "<sistem>")
    
    # Get number of processes
    num_proc = cfg.get("num_proc", 1)
    
    # List registered formatters
    logger.info("\n📋 Registered formatters:")
    for name, cols in formatter_registry.list_datasets():
        logger.info(f"  - {name} (columns: {cols})")
    
    # Process each dataset
    logger.info("\n" + "="*70)
    logger.info("PROCESSING DATASETS")
    logger.info("="*70)
    
    for dataset_cfg in cfg.datasets:
        dataset_name = dataset_cfg.name
        split = dataset_cfg.get("split", "train")
        output_path = dataset_cfg.get("output_path")
        push_to_hub = dataset_cfg.get("push_to_hub", False)
        hub_name = dataset_cfg.get("hub_name", None)
        
        logger.info(f"\n🔄 Processing: {dataset_name} (split: {split})")
        
        try:
            # Load dataset
            logger.info(f"  📥 Loading dataset...")
            dataset = load_dataset(dataset_name, split=split, token=hf_token if hf_token else None)
            logger.info(f"  ✅ Loaded {len(dataset)} examples")
            
            # Format dataset
            logger.info(f"  🔧 Formatting dataset...")
            formatted_dataset = formatter_registry.format_dataset(
                dataset,
                dataset_name=dataset_name,
                tokenizer=None,
                user_token=user_token,
                assistant_token=assistant_token,
                system_token=system_token,
                num_proc=num_proc
            )
            logger.info(f"  ✅ Formatted {len(formatted_dataset)} examples")
            
            # Show sample
            logger.info(f"\n  📄 Sample formatted output:")
            logger.info("  " + "-"*66)
            sample_text = formatted_dataset[0]['formatted_text']
            # Truncate if too long
            if len(sample_text) > 300:
                logger.info(f"  {sample_text[:300]}...")
            else:
                logger.info(f"  {sample_text}")
            logger.info("  " + "-"*66)
            
            # Save to disk if output path specified
            if output_path:
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"  💾 Saving to disk: {output_dir}")
                formatted_dataset.save_to_disk(str(output_dir))
                logger.info(f"  ✅ Saved to {output_dir}")
            
            # Push to hub if requested
            if push_to_hub:
                if not hub_name:
                    hub_name = f"{dataset_name}_formatted"
                
                logger.info(f"  📤 Pushing to HuggingFace Hub: {hub_name}")
                formatted_dataset.push_to_hub(
                    hub_name,
                    token=hf_token if hf_token else None
                )
                logger.info(f"  ✅ Pushed to {hub_name}")
            
            logger.info(f"  ✅ Successfully processed {dataset_name}")
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {dataset_name}: {str(e)}")
            continue
    
    logger.info("\n" + "="*70)
    logger.info("✅ Dataset formatting pipeline completed!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
