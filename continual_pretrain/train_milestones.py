import os
import sys
import torch
if os.environ.get('USE_UNSLOTH', '1') != '0':
    import unsloth
import wandb
import logging
import hydra
from datetime import datetime
from omegaconf import DictConfig
from transformers import set_seed
from dotenv import load_dotenv

from milestone_trainer import create_milestone_trainer
from model import ModelBuilder, ModelBuilderConfig, LoraAdapterConfig
from trainer_callbacks import EvaluationTriggerCallback, TokenStatsCallback, ProfilerCallback
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
    logging.info(f"Training on {tokens_per_milestone * num_milestones} tokens ({num_milestones} milestones)")
    
    # override max_seq_length param for unlsoth to have one source of truth
    model_cfg = to_dict(cfg.model.builder)
    lora_cfg = to_dict(cfg.model.lora)
    if "max_length" in training_kwargs:
        model_cfg['max_seq_length'] = training_kwargs['max_length']
    if "packing" in training_kwargs:
        model_cfg["packing"] = training_kwargs["packing"]
    if "gradient_checkpointing" in training_kwargs:
        model_cfg["gradient_checkpointing"] = training_kwargs["gradient_checkpointing"]
        model_cfg["unsloth_gradient_checkpointing"] = training_kwargs["gradient_checkpointing"]

    # build model and tokenizer
    model_cfg = ModelBuilderConfig(**model_cfg)
    lora_cfg = LoraAdapterConfig(**lora_cfg)
    builder = ModelBuilder(model_cfg, lora_cfg)
    model, tokenizer = builder.build()

    import torch
    torch.cuda.synchronize()
    static_vram_gb = torch.cuda.memory_allocated() / 1e9
    logging.info(f"[VRAM] Static model memory (params only, no optimizer states): {static_vram_gb:.3f} GB ")

    logging.info(f'Built Model and tokenizer : {model_cfg.model_name}')


    # if we do SFT then PT models should have chat_templates for fair comparison
    use_sft = cfg.dataset.get("use_sft", False)
    if use_sft:
        tokenizer = overwrite_pt_tokenizer(model_name=model_cfg.model_name, tokenizer = tokenizer)

    val_cfg = to_dict(cfg.validation_cfg)
    benchmark_evaluation_cfg = to_dict(cfg.benchmark_evaluation_cfg)
    result = prepare_dataset(cfg, tokenizer, return_validation=val_cfg.get('return_validation',False), validation_split= val_cfg.get('validation_split',0.1))
    train_dataset, val_dataset = result if isinstance(result, tuple) else (result, None)
    logging.info(f" Datasets loaded : {len(train_dataset)} training examples; {len(val_dataset) if val_dataset is not None else 0} validation examples")


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
    setup_file_logging(output_dir, run_name=wandb_run_name, run_id=wandb_run_id)
    
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

    nvtx_profile = os.environ.get('NVTX_PROFILE', '0') == '1'
    if nvtx_profile:
        # Inner range: just the attention score computation (QKt -> softmax -> V)
        # Patches the module-level function that LlamaAttention.forward dispatches to,
        # covering both eager and FA2 without copying any forward body.
        import transformers.models.llama.modeling_llama as _llama_mod
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        # Use range_push/pop (push/pop style NVTX) so NCU's --nvtx-include filter
        # can correlate GPU kernel launches to these ranges.
        # record_function + emit_nvtx emits start/end style ranges which NCU finds
        # but cannot use for kernel capture (causes "match only start/end ranges" warning).
        _orig_eager = _llama_mod.eager_attention_forward
        def _nvtx_eager(*args, **kwargs):
            torch.cuda.nvtx.range_push("attn_scores_eager")
            try:
                return _orig_eager(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        _llama_mod.eager_attention_forward = _nvtx_eager

        _orig_fa2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        def _nvtx_fa2(*args, **kwargs):
            torch.cuda.nvtx.range_push("attn_scores")
            try:
                return _orig_fa2(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _nvtx_fa2

        logging.info("[NVTX] Patched eager_attention_forward -> 'attn_scores_eager' (push/pop)")
        logging.info("[NVTX] Patched flash_attention_2 -> 'attn_scores' (push/pop)")

        # MLP forward range — layer 5 (index 4) only.
        # Instance-level patch: replaces forward on this specific LlamaMLP instance
        # so all other layers are unaffected. _orig_mlp_fwd is a bound method so
        # it still receives the correct self without re-binding.
        _mlp_l5 = model.model.model.layers[4].mlp
        _orig_mlp_fwd = _mlp_l5.forward
        def _nvtx_mlp_l5(*args, **kwargs):
            torch.cuda.nvtx.range_push("mlp_layer5")
            try:
                return _orig_mlp_fwd(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        _mlp_l5.forward = _nvtx_mlp_l5

        logging.info("[NVTX] Patched model.layers[4].mlp.forward -> 'mlp_layer5' (push/pop)")

    try:
        with torch.autograd.profiler.emit_nvtx(enabled=nvtx_profile):
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        logging.info('Milestone training complete!')
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        if wandb.run is not None:
            wandb.finish(exit_code=1 if sys.exc_info()[0] is not None else 0)


if __name__ == '__main__':
    main()