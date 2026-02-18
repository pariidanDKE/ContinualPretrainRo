from unsloth import FastLanguageModel
from unsloth.models.llama import FastLlamaModel

from unsloth_zoo.hf_utils import (
    fix_lora_auto_mapping
)
from unsloth.models._utils import *
from unsloth.models._utils import (
    patch_unsloth_smart_gradient_checkpointing,
    #fix_lora_auto_mapping,
    __version__,
)
from unsloth.device_type import DEVICE_TYPE_TORCH, DEVICE_TYPE

from unsloth.models.vision import FastBaseModel

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Union
import torch
import gc
import os
import inspect
import functools
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model as _get_peft_model,
    prepare_model_for_kbit_training,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed as transformers_set_seed,
)



if DEVICE_TYPE == "xpu":
    clean_gpu_cache = torch.xpu.empty_cache
    get_current_device = torch.xpu.current_device
else:
    clean_gpu_cache = torch.cuda.empty_cache
    get_current_device = torch.cuda.current_device

class FastLanguageModelWithEmbeddings(FastLanguageModel):


    @staticmethod
    def get_peft_model(
        model,
        r = 16,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = 16,
        lora_dropout = 0.0,
        bias = "none",
        embedding_lora_mode = "full_weights",
        layers_to_transform = None,
        layers_pattern = None,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = 2048,  # not used anymore
        use_rslora = False,
        modules_to_save = None,
        init_lora_weights = True,
        loftq_config = {},
        temporary_location = "_unsloth_temporary_saved_buffers",
        qat_scheme = None,
        **kwargs,
    ):
        if os.environ.get("UNSLOTH_USE_NEW_MODEL", "0") == "1":
            # Check for other PEFT args in kwargs
            for peft_arg, flag in (
                ("finetune_vision_layers", False),
                ("finetune_language_layers", True),
                ("finetune_attention_modules", True),
                ("finetune_mlp_modules", True),
            ):
                if peft_arg not in kwargs:
                    kwargs[peft_arg] = flag
            return FastBaseModel.get_peft_model(
                model = model,
                r = r,
                target_modules = target_modules,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                bias = bias,
                layers_to_transform = layers_to_transform,
                layers_pattern = layers_pattern,
                use_gradient_checkpointing = use_gradient_checkpointing,
                random_state = random_state,
                max_seq_length = max_seq_length,
                use_rslora = use_rslora,
                modules_to_save = modules_to_save,
                init_lora_weights = init_lora_weights,
                loftq_config = loftq_config,
                temporary_location = temporary_location,
                **kwargs,
            )
        if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
            print(
                "Unsloth: Full finetuning is enabled, so .get_peft_model has no effect"
            )
            return model
        transformers_set_seed(random_state)

        if use_gradient_checkpointing == "unsloth":
            patch_unsloth_smart_gradient_checkpointing(
                dtype = model.get_input_embeddings().weight.dtype
            )

        if type(r) is not int:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be an integer.")
        if r <= 0:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be larger than 0.")

        if isinstance(model, PeftModelForCausalLM) or isinstance(
            model, PeftModelForSequenceClassification
        ):
            # Check if exactly the same and then pass through!
            assert hasattr(model, "peft_config")

            peft_config = model.peft_config["default"].to_dict()
            check_parameters = [
                "r",
                "lora_alpha",
                "lora_dropout",
                "bias",
                "layers_to_transform",
                "layers_pattern",
                "use_rslora",
                "init_lora_weights",
            ]
            check_all = True
            for param in check_parameters:
                check_all = check_all and (peft_config[param] == eval(param))

            # Check save_modules
            old_target_modules = list(peft_config["target_modules"])
            modules_to_save = peft_config["modules_to_save"]
            if modules_to_save is None:
                modules_to_save = {}
            modules_to_save = list(modules_to_save)
            old_target_modules += modules_to_save

            # Combine all
            new_target_modules = list(target_modules) + list(
                modules_to_save if modules_to_save is not None else []
            )

            # Now check!
            new_target_modules = set(new_target_modules)
            check_all = check_all and (
                len(set(old_target_modules) ^ new_target_modules) == 0
            )

            check_all = check_all and (
                (loftq_config == {} or loftq_config is None)
                and (
                    peft_config["loftq_config"] == {}
                    or peft_config["loftq_config"] is None
                )
            )

            if check_all:
                # Simply pass through!
                logger.warning(
                    "Unsloth: Already have LoRA adapters! We shall skip this step."
                )

                # Offload!
                # [TODO] First offload lm_head and embed_tokens to CPU (should be disk!!)
                if "embed_tokens" in new_target_modules:

                    if hasattr(model.get_input_embeddings(), "modules_to_save"):

                        print(
                            "Unsloth: Training embed_tokens in mixed precision to save VRAM"
                        )
                        new_dtype = model.get_input_embeddings().modules_to_save.default.weight.dtype
                        if new_dtype == torch.float16:
                            # See https://github.com/unslothai/unsloth/pull/1200
                            # Tesla T4 must use float32 and not float16
                            new_dtype = torch.float32

                        model.get_input_embeddings().modules_to_save.default.to(
                            device = DEVICE_TYPE_TORCH, dtype = new_dtype, non_blocking = True
                        )
                        model.get_input_embeddings().modules_to_save.default.requires_grad_(
                            True
                        )

                        # [TODO] Move old embed_tokens to CPU - should be disk!
                        model.get_input_embeddings().original_module.to(
                            device = "cpu", non_blocking = True
                        )
                        model.get_input_embeddings().original_module.requires_grad_(False)
                    
                    # PR code proposal (lora embeddings)
                    elif hasattr(model.get_input_embeddings(), "lora_A"):
                        print("Unsloth: Training embed_tokens wiht LoRA adapters")
                        embed_layer = model.get_input_embeddings()

                        for adapter_name in embed_layer.lora_A.keys():
                            embed_layer.lora_A[adapter_name].requires_grad(True)
                            embed_layer.lora_B[adapter_name].requires_grad(True)
                     
                if "lm_head" in new_target_modules:
                    print("Unsloth: Training lm_head in mixed precision to save VRAM")

                    if hasattr(model.get_output_embeddings(), "modules_to_save"):
                        new_dtype = model.get_output_embeddings().modules_to_save.default.weight.dtype
                        if new_dtype == torch.float16:
                            # See https://github.com/unslothai/unsloth/pull/1200
                            # Tesla T4 must use float32 and not float16
                            new_dtype = torch.float32

                        model.get_output_embeddings().modules_to_save.default.to(
                            device = DEVICE_TYPE_TORCH, dtype = new_dtype, non_blocking = True
                        )
                        model.get_output_embeddings().modules_to_save.default.requires_grad_(
                            True
                        )

                        # [TODO] Move old lm_head to CPU - should be disk!
                        model.get_output_embeddings().original_module.to(
                            device = "cpu", non_blocking = True
                        )
                        model.get_output_embeddings().original_module.requires_grad_(False)

                    # PR code proposal (lora embeddings)
                    elif hasattr(model.get_output_embeddings(), "lora_A"):
                        print("Unsloth: Training lm_head with LoRA adapters")

                        lm_head_layer = model.get_output_embeddings()
                        for adapter_name in lm_head_layer.lora_A.keys():
                            lm_head_layer.lora_A[adapter_name].requires_grad(True)
                            lm_head_layer.lora_B[adapter_name].requires_grad(True)


                return model
            else:
                raise TypeError(
                    "Unsloth: Your model already has LoRA adapters. Your new parameters are different."
                )

        if loftq_config is None:
            loftq_config = {}

        signature = str(inspect.signature(LoraConfig))
        SUPPORTS_LOFTQ = "loftq_config" in signature
        SUPPORTS_RSLORA = "use_rslora" in signature

        if lora_dropout != 0:
            logger.warning_once(
                f"Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = {lora_dropout}.\n"
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )

        if bias != "none":
            logger.warning_once(
                f"Unsloth: bias = `none` is supported for fast patching. You are using bias = {bias}.\n"
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )

        if not (
            type(init_lora_weights) is bool
            or init_lora_weights == "gaussian"
            or init_lora_weights == "loftq"
        ):
            raise ValueError(
                'Unsloth: `init_lora_weights` must be either [True, False, "gaussian", "loftq"].'
            )

        if init_lora_weights == "loftq":
            if not SUPPORTS_LOFTQ:
                import peft

                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support LoftQ init.\n"
                    "Please install PEFT 0.7.2 or higher.\n"
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )

            if loftq_config == {}:
                from peft import LoftQConfig

                logger.warning_once(
                    "Unsloth: init_lora_weights = `loftq` is set, but `loftq_config` is None.\n"
                    "We shall use `loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)`."
                )
                loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)

            if hasattr(model.config, "quantization_config"):
                raise ValueError(
                    "Unsloth: You are using `loftq` init, yet `load_in_4bit = True` was set.\n"
                    "Reload your model without any quantization by setting `load_in_4bit = False`."
                )

        assert type(use_rslora) is bool
        if use_rslora:
            if not SUPPORTS_RSLORA:
                # We manually check for PEFT
                import peft

                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support `use_rslora`.\n"
                    "Please install PEFT 0.7.2 or higher.\n"
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )

        accepted_modules = frozenset(
            (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        )
        model.config.update({"unsloth_version": __version__})

        if type(modules_to_save) is tuple:
            modules_to_save = list(modules_to_save)

        # PR code proposal - lora embeddings
        train_lm_head = False
        train_embed_tokens = False
        lora_lm_head = False
        lora_embed_tokens = False
        final_modules = []

        for module in target_modules:
            if module == "lm_head":
                if embedding_lora_mode == "full_weights":
                    # Current behavior: full weight training
                    train_lm_head = True
                    if modules_to_save is None:
                        modules_to_save = ["lm_head"]
                    else:
                        modules_to_save.append("lm_head")
                elif embedding_lora_mode == "lora":
                    # New behavior: LoRA on lm_head
                    lora_lm_head = True
                    final_modules.append(module)
                # else: frozen, don't add anywhere

            elif module == "embed_tokens":
                if embedding_lora_mode == "full_weights":
                    # Current behavior: full weight training
                    train_embed_tokens = True
                    if modules_to_save is None:
                        modules_to_save = ["embed_tokens"]
                    else:
                        modules_to_save.append("embed_tokens")
                elif embedding_lora_mode == "lora":
                    # New behavior: LoRA on embed_tokens
                    lora_embed_tokens = True
                    final_modules.append(module)
                # else: frozen, don't add anywhere

            else:
                try:
                    assert module in accepted_modules
                    final_modules.append(module)
                except AssertionError as e:
                    final_modules.append(module)
                    print(
                        "Unsloth: You added custom modules, but Unsloth hasn't optimized for this.\n"
                        "Beware - your finetuning might be noticeably slower!"
                    )
                pass
        #############################
        # for module in target_modules:
        #     if module == "lm_head":
        #         # logger.warning_once(
        #         #     "Unsloth: `lm_head` should be placed in `modules_to_save` and not `target_modules`. "\
        #         #     "Luckily, we shall do it for you!"
        #         # )
        #         train_lm_head = True
        #         if modules_to_save is None:
        #             modules_to_save = ["lm_head"]
        #         else:
        #             modules_to_save.append("lm_head")

        #     elif module == "embed_tokens":
        #         # logger.warning_once(
        #         #     "Unsloth: `embed_tokens` should be placed in `modules_to_save` and not `target_modules`. "\
        #         #     "Luckily, we shall do it for you!"
        #         # )
        #         train_embed_tokens = True
        #         if modules_to_save is None:
        #             modules_to_save = ["embed_tokens"]
        #         else:
        #             modules_to_save.append("embed_tokens")

        #     else:
        #         try:
        #             assert module in accepted_modules
        #             final_modules.append(module)
        #         except AssertionError as e:
        #             final_modules.append(module)
        #             print(
        #                 "Unsloth: You added custom modules, but Unsloth hasn't optimized for this.\n"
        #                 "Beware - your finetuning might be noticeably slower!"
        #             )
        #         pass

        # Check if we added new tokens!
        if hasattr(model, "_need_to_train_embeddings"):
            if not train_lm_head or not train_embed_tokens:
                print(
                    "Unsloth: You added new tokens but did not specify if you wanted to "
                    "train the lm_head and embed_tokens.\nWe must turn it on for you."
                )
                train_lm_head = True
                train_embed_tokens = True

                if modules_to_save is None:
                    modules_to_save = ["embed_tokens"]
                else:
                    modules_to_save.append("embed_tokens")

                if modules_to_save is None:
                    modules_to_save = ["lm_head"]
                else:
                    modules_to_save.append("lm_head")

        # Check for Llama-3
        # if hasattr(model._saved_temp_tokenizer, "_using_llama3_template"):
        #     if not train_embed_tokens and not train_lm_head:
        #         raise RuntimeError("")

        # First fix untrained tokens
        # Wrong - can cause reserved tokens to pop out!!
        # if train_embed_tokens or train_lm_head:
        #     fix_untrained_tokens(model, eps = 1e-16)
        # pass

        # Check modules_to_save
        if modules_to_save is not None:
            for module in modules_to_save:
                if module == "lm_head":
                    train_lm_head = True
                elif module == "embed_tokens":
                    train_embed_tokens = True
                else:
                    raise TypeError(
                        f"Unsloth: Module = {module} is not allowed. Only 'lm_head' and 'embed_tokens' is allowed."
                    )
        if isinstance(modules_to_save, (tuple, list)):
            modules_to_save = list(set(modules_to_save))

        vllm_engine = None
        if hasattr(model, "vllm_engine"):
            # Fast inference!
            vllm_engine = model.vllm_engine
            vllm_fast_generate = model.fast_generate
            vllm_fast_generate_batches = model.fast_generate_batches

            if modules_to_save is not None:
                raise NotImplementedError(
                    "Unsloth: Currently fast inference does not work with training embeddings or lm_head."
                )

            if bias != "none":
                raise NotImplementedError(
                    "Unsloth: Currently fast inference does not work with using biases for LoRA."
                )

        # Does not get lora yet, so get name from model, not base model
        is_classification = "Classification" in str(type(model))

        arguments = dict(
            r = r,
            lora_alpha = lora_alpha,
            target_modules = final_modules,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = TaskType.CAUSAL_LM if not is_classification else TaskType.SEQ_CLS,
            layers_to_transform = layers_to_transform,
            init_lora_weights = init_lora_weights,
            loftq_config = loftq_config,
            use_rslora = use_rslora,
            modules_to_save = modules_to_save,
            **kwargs,
        )
        if not SUPPORTS_LOFTQ:
            del arguments["loftq_config"]
        if not SUPPORTS_RSLORA:
            del arguments["use_rslora"]

        _saved_temp_tokenizer = model._saved_temp_tokenizer

        lora_config = LoraConfig(**arguments)
        # First offload lm_head and embed_tokens to disk
        input_embeddings_device = model.get_input_embeddings().weight.device
        if is_classification:
            output_embeddings_device = model.score.weight.device
        else:
            output_embeddings_device = model.get_output_embeddings().weight.device

        # PR code proposal - lora embeddings
        if use_gradient_checkpointing == "unsloth":
            if train_embed_tokens:
                print("Unsloth: Offloading input_embeddings to disk to save VRAM")
                offload_input_embeddings(model, temporary_location)
            
            if lora_embed_tokens:
                print("Unsloth: Using LoRA on embed_tokens - base weights stay on GPU")

            # Remove old items to save VRAM
            for _ in range(3):
                gc.collect()
                clean_gpu_cache()

            if train_lm_head:
                print("Unsloth: Offloading output_embeddings to disk to save VRAM")
                offload_output_embeddings(model, temporary_location)
            if lora_lm_head:
                print("Unsloth: Using LoRA on lm_head - base weights stay on GPU")


            # Remove old items to save VRAM
            for _ in range(3):
                gc.collect()
                clean_gpu_cache()

        model = _get_peft_model(model, lora_config)
        # Fix LoraConfig.auto_mapping is None
        fix_lora_auto_mapping(model)

        # Apply QAT + LoRA if specified
        if qat_scheme is not None:
            print("Unsloth: Applying QAT to mitigate quantization degradation")
            model = FastLlamaModel._prepare_for_qat(model, qat_scheme)

        model._saved_temp_tokenizer = _saved_temp_tokenizer

        model = FastLlamaModel.patch_peft_model(model, use_gradient_checkpointing)

        if train_embed_tokens:
            print("Unsloth: Training embed_tokens in mixed precision to save VRAM")
            assert hasattr(model.get_input_embeddings(), "modules_to_save")

            new_dtype = (
                model.get_input_embeddings().modules_to_save.default.weight.dtype
            )
            if new_dtype == torch.float16:
                # See https://github.com/unslothai/unsloth/pull/1200
                # Tesla T4 must use float32 and not float16
                new_dtype = torch.float32

            model.get_input_embeddings().modules_to_save.default.to(
                device = DEVICE_TYPE_TORCH, dtype = new_dtype, non_blocking = True
            )
            model.get_input_embeddings().modules_to_save.default.requires_grad_(True)
        elif lora_embed_tokens:
            # LoRA mode - different structure
            print("Unsloth: Training embed_tokens with LoRA adapters")

            # Check for LoRA structure instead of modules_to_save
            assert hasattr(model.get_input_embeddings(), "lora_A")

            # Base layer stays quantized (if load_in_4bit=True)
            # LoRA adapters are already in the right dtype from LoraConfig
            # Just ensure they're on GPU and trainable
            embed_layer = model.get_input_embeddings()

            # LoRA adapters should already be on GPU, but verify
            if hasattr(embed_layer, 'lora_A'):
                for adapter_name in embed_layer.lora_A.keys():
                    embed_layer.lora_A[adapter_name].requires_grad_(True)
                    embed_layer.lora_B[adapter_name].requires_grad_(True)
                        
            
        if train_lm_head:
            print("Unsloth: Training lm_head in mixed precision to save VRAM")
            assert hasattr(model.get_output_embeddings(), "modules_to_save")

            new_dtype = (
                model.get_output_embeddings().modules_to_save.default.weight.dtype
            )
            if new_dtype == torch.float16:
                # See https://github.com/unslothai/unsloth/pull/1200
                # Tesla T4 must use float32 and not float16
                new_dtype = torch.float32

            model.get_output_embeddings().modules_to_save.default.to(
                device = DEVICE_TYPE_TORCH, dtype = new_dtype, non_blocking = True
            )
            model.get_output_embeddings().modules_to_save.default.requires_grad_(True)
        elif lora_lm_head:
            # LoRA mode
            print("Unsloth: Training lm_head with LoRA adapters")

            assert hasattr(model.get_output_embeddings(), "lora_A")

            lm_head_layer = model.get_output_embeddings()
            if hasattr(lm_head_layer, 'lora_A'):
                for adapter_name in lm_head_layer.lora_A.keys():
                    lm_head_layer.lora_A[adapter_name].requires_grad_(True)
                    lm_head_layer.lora_B[adapter_name].requires_grad_(True)
        # Patch tokenizer to pad to the right
        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.padding_side = "right"
            # Also set is_loaded_in_8bit to disable incorrect DDP
            internal_model.is_loaded_in_8bit = True
            internal_model = internal_model.model
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.padding_side = "right"
        # Also set is_loaded_in_8bit to disable incorrect DDP
        internal_model.is_loaded_in_8bit = True

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            clean_gpu_cache()

        patch_peft_fast_inference(model)

        # Add for_inference and for_training
        model.for_training = functools.partial(FastLlamaModel.for_training, model)
        model.for_inference = functools.partial(FastLlamaModel.for_inference, model)
        m = model
        while hasattr(m, "model"):
            m.for_training = functools.partial(FastBaseModel.for_training, m)
            m.for_inference = functools.partial(FastBaseModel.for_inference, m)
            m = m.model
        return model



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
    prepare_kbit_training: bool = True
    require_grads: bool = True
    max_seq_length: int = 4096

    use_flash_attention: bool = True

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

    # --------------------------------------------------------------------- #
    # Hugging Face + BitsAndBytes + LoRA flow
    # --------------------------------------------------------------------- #
    def _build_standard_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        cfg = self.config

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        quant_config = None
        model_kwargs: dict[str, Any] = {"device_map": cfg.device_map}
        if cfg.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self._resolve_dtype(
                    cfg.bnb_4bit_compute_dtype
                ),
            )
            model_kwargs["quantization_config"] = quant_config
        else:
            dtype = self._resolve_dtype(cfg.bnb_4bit_compute_dtype)
            if dtype is not None:
                model_kwargs["dtype"] = dtype

        if not cfg.use_flash_attention:
            model_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        model.config.use_cache = cfg.use_cache

        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if cfg.prepare_kbit_training:
            model = prepare_model_for_kbit_training(model)

        if cfg.require_grads and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads() # since CPT traiuns embedding layers

        peft_config = self.lora_config.to_peft_config()
        model = _get_peft_model(model, peft_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()  # pragma: no cover - logging helper

        import os; os._exit(0)

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
        model, tokenizer = FastLanguageModelWithEmbeddings.from_pretrained(
            model_name=model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=self._resolve_dtype(cfg.bnb_4bit_compute_dtype),
            load_in_4bit=load_in_4bit,
            **fa_kwargs,
        )

        model = FastLanguageModelWithEmbeddings.get_peft_model(
            model=model,
            r=self.lora_config.r,
            target_modules=list(self.lora_config.target_modules),
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            embedding_lora_mode=self.lora_config.embedding_lora_mode,
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

