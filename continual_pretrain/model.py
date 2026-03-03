import os as _os
if _os.environ.get('USE_UNSLOTH', '1') != '0':
    from unsloth import FastLanguageModel
else:
    FastLanguageModel = None  # type: ignore[assignment]
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Union
import torch
from peft import (
    LoraConfig,
    get_peft_model as _get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


logger = logging.getLogger(__name__)


def _default_target_modules() -> Tuple[str, ...]:
    """Return modules commonly adapted in LLaMA-style models."""
    return (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    )

@dataclass
class LoraAdapterConfig:
    """High-level settings for constructing a LoRA adapter."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    embedding_lora_mode: str = "full_weights"
    target_modules: Sequence[str] = field(
        default_factory=lambda: list(_default_target_modules())
    )

    def to_peft_config(self) -> LoraConfig:
        """Create a `peft.LoraConfig` instance."""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
            target_modules=list(self.target_modules),
        )


@dataclass
class ModelBuilderConfig:
    """Configuration for constructing models."""

    model_name: str
    tokenizer_name: Optional[str] = None
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: Union[str, torch.dtype, None] = "bfloat16"
    device_map: Union[str, dict, None] = "auto"
    gradient_checkpointing: bool = True
    use_cache: bool = False
    use_lora: bool = True
    prepare_kbit_training: bool = True
    require_grads: bool = False
    max_seq_length: int = 4096
    use_flash_attention: bool = True
    packing: bool = True

    # Unsloth-related flags
    use_unsloth: bool = False
    unsloth_path: Optional[str] = None
    unsloth_dtype: Union[str, torch.dtype, None] = None 
    unsloth_gradient_checkpointing: Union[bool, str] = "unsloth"
    seed: int = 3407
    unsloth_use_rslora: bool = True
    unsloth_loftq_config: Optional[dict] = None
    unsloth_load_in_4bit: Optional[bool] = None


class ModelBuilder:
    """Create causal language models with optional 4-bit + LoRA or Unsloth support."""

    def __init__(
        self,
        config: ModelBuilderConfig,
        lora_config: Optional[LoraAdapterConfig] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.lora_config = lora_config or LoraAdapterConfig()
        self.logger = logger_ or logger

    def build(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Instantiate the configured model and tokenizer."""
        if self.config.use_unsloth:
            model, tokenizer = self._build_unsloth_model()
        else:
            model, tokenizer = self._build_standard_model()

        self._ensure_padding_token(tokenizer)
        return model, tokenizer

    def _log_weight_dtypes(self, model: PreTrainedModel) -> None:
        from collections import Counter, defaultdict
        dtype_counts: Counter = Counter()
        bf16_by_module: dict = defaultdict(int)
        for name, param in model.named_parameters():
            dtype_counts[str(param.dtype)] += param.numel()
            if param.dtype == torch.bfloat16:
                # group by top-level module path (e.g. "model.layers.0.self_attn.q_proj")
                bf16_by_module[name] += param.numel()
        total = sum(dtype_counts.values())
        self.logger.info("[Weight dtypes] Parameter counts by dtype:")
        for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
            self.logger.info(f"  {dtype}: {count:,} params ({100*count/total:.1f}%)")
        if bf16_by_module:
            self.logger.info("[Weight dtypes] bfloat16 params by module (descending):")
            for name, count in sorted(bf16_by_module.items(), key=lambda x: -x[1]):
                self.logger.info(f"  {name}: {count:,}")

    # --------------------------------------------------------------------- #
    # Hugging Face + BitsAndBytes + LoRA flow
    # --------------------------------------------------------------------- #
    def _build_standard_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        cfg = self.config

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        quant_config = None
        model_kwargs: dict[str, Any] = {"device_map": cfg.device_map}
        if cfg.load_in_4bit:
            compute_dtype = self._resolve_dtype(cfg.bnb_4bit_compute_dtype)
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["torch_dtype"] = compute_dtype
        else:
            dtype = self._resolve_dtype(cfg.bnb_4bit_compute_dtype)
            if dtype is not None:
                model_kwargs["dtype"] = dtype

        if not cfg.use_flash_attention:
            if cfg.packing:
                raise Exception("Mismatch : Cannot run trl with packing and no FA2")
            model_kwargs["attn_implementation"] = "eager"
        else:
            cfg.use_flash_attention = model_kwargs["attn_implementation"] = "flash_attention_2"
            
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        model.config.use_cache = cfg.use_cache

        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if cfg.use_lora:

            if cfg.require_grads and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            peft_config = self.lora_config.to_peft_config()
            model = _get_peft_model(model, peft_config)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()  # pragma: no cover - logging helper

        return model, tokenizer

    # --------------------------------------------------------------------- #
    # Unsloth flow
    # --------------------------------------------------------------------- #
    def _build_unsloth_model(
        self,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:

        cfg = self.config
        model_name = cfg.unsloth_path or cfg.model_name
        load_in_4bit = cfg.load_in_4bit

        fa_kwargs = {} if cfg.use_flash_attention else {"attn_implementation": "eager"}
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=self._resolve_dtype(cfg.bnb_4bit_compute_dtype),
            load_in_4bit=load_in_4bit,
            **fa_kwargs,
        )

        model = FastLanguageModel.get_peft_model(
            model=model,
            r=self.lora_config.r,
            target_modules=list(self.lora_config.target_modules),
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            use_gradient_checkpointing=cfg.unsloth_gradient_checkpointing,
            random_state=cfg.seed,
            use_rslora=cfg.unsloth_use_rslora,
            loftq_config=cfg.unsloth_loftq_config,
        )

        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()  # pragma: no cover - logging helper
        return model, tokenizer

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_padding_token(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Ensure pad_token exists to keep the Trainer happy."""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            self.logger.info(
                "Tokenizer lacked a pad token, defaulting pad_token to eos_token."
            )

    @staticmethod
    def _resolve_dtype(
        dtype: Union[str, torch.dtype, None],
    ) -> Optional[torch.dtype]:
        """Map user-provided dtype strings to `torch.dtype`."""
     
        normalized = dtype.strip().lower()
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }

        return mapping[normalized]

