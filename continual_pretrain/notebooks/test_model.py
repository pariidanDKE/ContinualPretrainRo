from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import logging
from transformers import AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from PT models to IT tokenizers (for chat template)
PT_TO_IT_TOKENIZER_MAP = {
    "unsloth/llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
}

def model_size_gb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)

# ✅ Load with FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-1b",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

model = PeftModel.from_pretrained(model, "../outputs/cpt/Default/cpt-llama-TestContinuousLoop-ms8-200M/20260209_0122-8ttnhqbg/checkpoint-432")
model = model.merge_and_unload()

model = model.to(torch.float32)  # Dequantize to fp32 first
print(f"Before bf16: {model_size_gb(model):.3f} GB")
model = model.to(torch.bfloat16)
print(f"After bf16: {model_size_gb(model):.3f} GB")

# ====================================================================
# Load IT Tokenizer (for chat template support)
# ====================================================================
base_model_name = "unsloth/llama-3.2-1b"
if base_model_name in PT_TO_IT_TOKENIZER_MAP:
    it_tokenizer_name = PT_TO_IT_TOKENIZER_MAP[base_model_name]
    logger.info(f"Loading IT tokenizer from {it_tokenizer_name} for chat template support...")
    tokenizer = AutoTokenizer.from_pretrained(it_tokenizer_name)
    logger.info("✅ IT tokenizer loaded")
else:
    logger.info("Using original tokenizer (already has chat template)")

# ====================================================================
# Run Benchmarks on Full Precision Model
# ====================================================================

logger.info("🔧 Wrapping model for lm-eval...")
torch.cuda.empty_cache()

lm_eval_model = HFLM(
    pretrained=model,
    tokenizer=tokenizer,
    max_length=2048,
    eval_max_length=16384,
    batch_size=8,
    device=str(model.device),
    trust_remote_code=True,
)

logger.info("✅ Model wrapped for evaluation")

# Define benchmark configuration (matching evaluate_ro.yaml)
benchmark_tasks = [
    {"name": "ro_wiki", "fewshot": 0, "task_batch_size": 8, "limit": 1000},
    {"name": "_ro_mmlu", "fewshot": 5, "task_batch_size": 4, "limit": 1000},
    {"name": "_ro_winogrande", "fewshot": 5, "task_batch_size": 8, "limit": 1000},
    {"name": "_ro_belebele", "fewshot": 5, "task_batch_size": 4, "limit": 1000},
    {"name": "_ro_arc_challenge", "fewshot": 5, "task_batch_size": 8, "limit": 1000},
]

perplexity_tasks = {"ro_wiki", "wikitext"}
results = {}

# Run each benchmark
for task_cfg in benchmark_tasks:
    task_name = task_cfg["name"]
    fewshot = task_cfg["fewshot"]
    limit = task_cfg.get("limit")
    batch_size = task_cfg["task_batch_size"]

    # Only apply chat template for non-perplexity tasks
    apply_chat_template = task_name not in perplexity_tasks

    try:
        logger.info(f"🚀 Running {task_name} | fewshot={fewshot} | limit={limit}")

        eval_kwargs = {
            "model": lm_eval_model,
            "batch_size": batch_size,
            "apply_chat_template": apply_chat_template,
            "tasks": [task_name],
            "limit": limit,
            "verbosity": "INFO",
            "num_fewshot": fewshot,
            "log_samples": False,
            "write_out": False,
            "device": str(model.device),
            "fewshot_random_seed": 23,
            "random_seed": 23,
            "numpy_random_seed": 23,
            "torch_random_seed": 23,
        }

        if apply_chat_template and fewshot > 0:
            eval_kwargs["fewshot_as_multiturn"] = True

        res = evaluator.simple_evaluate(**eval_kwargs)
        results[task_name] = res

        # Print key metrics (use print to ensure visibility)
        task_results = res.get("results", {}).get(task_name, {})
        print(f"\n{'='*60}")
        print(f"✅ Finished {task_name}")
        print(f"{'='*60}")
        for metric_name, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"  📊 {metric_name}: {value:.4f}")
        print()

    except Exception as e:
        print(f"\n❌ Error running {task_name}: {str(e)}\n")
        logger.error(f"❌ Error running {task_name}: {str(e)}")
        results[task_name] = {"error": str(e)}

print("\n" + "=" * 80)
print("📈 FULL PRECISION BENCHMARK RESULTS SUMMARY")
print("=" * 80)
for task_name, res in results.items():
    task_results = res.get("results", {}).get(task_name, {})
    if task_results:
        metrics = {k: v for k, v in task_results.items() if isinstance(v, (int, float))}
        print(f"{task_name}: {metrics}")
    else:
        print(f"{task_name}: {res.get('error', 'No results')}")
print("=" * 80 + "\n")