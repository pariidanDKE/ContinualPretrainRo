# # Basic usage with default config
# cd continual_pretrain/data_processing
# python dataset_formatting.py

# # Override specific settings
# python dataset_formatting.py num_proc=8

# # Process and push to HuggingFace Hub
# python dataset_formatting.py datasets.0.push_to_hub=true datasets.0.hub_repo_id="your-org/dataset-name"

python data_processing/dataset_formatting.py