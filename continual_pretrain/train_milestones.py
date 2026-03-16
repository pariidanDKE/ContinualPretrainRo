import unsloth
import wandb
import os
import logging
import hydra
from datetime import datetime
from omegaconf import DictConfig
from transformers import set_seed
from dotenv import load_dotenv

from milestone_trainer import create_milestone_trainer
from model import ModelBuilder, ModelBuilderConfig, LoraAdapterConfig
from trainer_callbacks import EvaluationTriggerCallback, TokenStatsCallback
from train_utils import (
    prepare_dataset,
    to_dict,
    apply_wandb_config,
    overwrite_pt_tokenizer,
    setup_file_logging,
    set_training_type,
    generate_run_name
)

logger = logging.getLogger(__name__)
logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )


@hydra.main(config_path="configs", config_name = "train_model.yaml", version_base= None)
def main(cfg : DictConfig):
    " Full training loop for model over n milestones (CPT or SFT)"
    load_dotenv()
    
    if cfg.get("seed"):
        set_seed(int(cfg.seed))
    
    # load milestone configuration
    training_kwargs = to_dict(cfg.training_args)
    
    milestone_cfg = to_dict(cfg.get("milestone", {}))

    tokens_per_milestone = milestone_cfg.get('tokens_per_milestone')
    num_milestones = milestone_cfg.get('num_milestones')
    resume_checkpoint = milestone_cfg.get("resume_from_checkpoint")


    from omegaconf import OmegaConf
    logging.info("Full resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logging.info("Starting Continuous Training")
    logging.info(f"Training on {tokens_per_milestone:,} tokens/milestone x {num_milestones} milestones = {tokens_per_milestone * num_milestones:,} total tokens")
    
    # override max_seq_length param for unlsoth to have one source of truth
    model_cfg = to_dict(cfg.model.builder)
    lora_cfg = to_dict(cfg.model.lora)
    if "max_length" in training_kwargs:
        model_cfg['max_seq_length'] = training_kwargs['max_length']

    # build model and tokenizer
    model_cfg = ModelBuilderConfig(**model_cfg)
    lora_cfg = LoraAdapterConfig(**lora_cfg)
    builder = ModelBuilder(model_cfg, lora_cfg)
    model, tokenizer = builder.build()

    import torch
    torch.cuda.synchronize()
    static_vram_gb = torch.cuda.memory_allocated() / 1e9
    logging.info(f"[VRAM] Static model memory (params only, no optimizer states): {static_vram_gb:.3f} GB  |  embedding_lora_mode={lora_cfg.embedding_lora_mode}")

    logging.info(f'Built Model and tokenizer : {model_cfg.model_name}')

    # if we do SFT then PT models should have chat_templates for fair comparison
    
    use_sft = cfg.dataset.get("use_sft", False)
    if use_sft:
        tokenizer = overwrite_pt_tokenizer(model_name=model_cfg.model_name, tokenizer = tokenizer)

    val_cfg = to_dict(cfg.validation_cfg)
    benchmark_evaluation_cfg = to_dict(cfg.benchmark_evaluation_cfg)
    result = prepare_dataset(cfg, tokenizer, return_validation=val_cfg.get('return_validation',False), validation_split= val_cfg.get('validation_split',0.1))
    train_dataset, val_dataset = result if isinstance(result, tuple) else (result, None)

    if isinstance(val_dataset, dict):
        val_info = f"{sum(len(v) for v in val_dataset.values())} validation examples across sources: {list(val_dataset.keys())}"
    else:
        val_info = f"{len(val_dataset) if val_dataset is not None else 0} validation examples"
    logging.info(f" Datasets loaded : {len(train_dataset)} training examples; {val_info}")


    # alter training_kwargs
    wandb_cfg = to_dict(cfg.get("wandb", {}), resolve=False)
    wandb_run_name = generate_run_name(model_name=model_cfg.model_name, use_sft=use_sft, num_milestones=num_milestones,milestone_tokens=tokens_per_milestone,custom_name=wandb_cfg.get('custom_run_name',''))
    wandb_run_id = wandb.util.generate_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    training_kwargs = apply_wandb_config(training_kwargs, wandb_cfg,wandb_run_name)
    training_kwargs['output_dir'] = f'outputs/{('sft' if use_sft else 'cpt')}/{wandb_cfg.get("group")}/{wandb_run_name}/{timestamp}-{wandb_run_id}'

    if not milestone_cfg.get('do_evaluate', False):
        training_kwargs.pop('per_device_eval_batch_size', None)
        training_kwargs.pop('prediction_loss_only', None)
        training_kwargs['eval_on_start'] = False

    # set logger to same file as output_dir/logs
    output_dir = training_kwargs.get('output_dir', './outputs')
    file_handler = setup_file_logging(output_dir, run_name=wandb_run_name, run_id=wandb_run_id)
    

    # setup trainer
    use_unsloth = cfg.model.builder.get("use_unsloth", False)
    TrainerClass, TrainingArgsClass, training_kwargs = set_training_type(use_unsloth,training_kwargs)

    milestone_trainer_params = {
    'model':model,
    'train_dataset':train_dataset,
    'processing_class':tokenizer,
    'benchmark_evaluation_cfg': benchmark_evaluation_cfg,
    'num_milestones': num_milestones,
    'tokens_per_milestone' : tokens_per_milestone,
    'run_benchmarks' : milestone_cfg.get('run_benchmarks',True)
    }

    if milestone_cfg.get('do_evaluate',False):
        milestone_trainer_params['eval_dataset'] = val_dataset


    training_args = TrainingArgsClass(
        **training_kwargs,
    )
    milestone_trainer_params['args'] = training_args

    trainer = create_milestone_trainer(
        TrainerClass,
        **milestone_trainer_params)

    # Re-attach file handler — Trainer init resets root logger handlers
    logging.getLogger().addHandler(file_handler)

    # setup callbacks
    token_stats_callback = TokenStatsCallback(tokenizer, log_every_n_steps = training_kwargs['logging_steps'])
    eval_trigger_callback = EvaluationTriggerCallback(
        target_milestone_tokens=tokens_per_milestone,
        num_milestones=num_milestones,
        trainer=trainer,
        do_evaluate=milestone_cfg.get('do_evaluate', False)
    )

    trainer.add_callback(token_stats_callback)
    trainer.add_callback(eval_trigger_callback)

        
    if wandb.run is not None:
        wandb.config.update({"hydra_cfg": OmegaConf.to_container(cfg, resolve=True)}, allow_val_change=True)

    if resume_checkpoint:
        logging.info(f' Resume training checkpoint with path {resume_checkpoint}..')
    else:
        logging.info('Milestone training start..')

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    logging.info('Milestone training complete!')


if __name__ == '__main__':
    main()