#!/usr/bin/env python3
"""
Embedding similarity analysis: compare CPT vs base vs Romanian reference model.

Hypothesis: corr(CPT embeddings, RoLlama3.1) > corr(Base embeddings, RoLlama3.1)

Usage:
    python scripts/analyze_embedding_similarity.py
"""

import gc
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Config ───────────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.2-1B"

CPT_CHECKPOINT = (
    "/home/dan-parii/Documents/ContinualPretrainRo/continual_pretrain/outputs/cpt/"
    "embed_comparison_20260306_185038/cpt-llama-with_embeddings_r64-ms5-125M/"
    "20260306_1851-g8r0v0xz/checkpoint-1019"
)

RO_REFERENCE_MODEL = "OpenLLM-Ro/RoLlama3.1-8b-Instruct"

TOKENIZER_NAME = BASE_MODEL       # all three share the Llama 3 128k tokenizer
TOP_N_TOKENS   = 2_000            # most-frequent token IDs to start with
DATASET_NAME   = "OpenLLM-Ro/fineweb2-ro-human"
DATASET_SPLIT  = "train"
DATASET_ROWS   = 10_000           # rows to stream for frequency counting
SCRIPT_DIR     = __import__("pathlib").Path(__file__).parent
OUTPUT_FIGURE  = str(SCRIPT_DIR / "embedding_similarity_heatmaps.png")
OUTPUT_RESULTS = str(SCRIPT_DIR / "embedding_similarity_results.txt")

# ── Helpers ──────────────────────────────────────────────────────────────────

def build_token_frequency(tokenizer, n_rows: int = DATASET_ROWS) -> Counter:
    """Stream Romanian text and return a token-ID frequency Counter."""
    print(f"Streaming '{DATASET_NAME}' ({n_rows} rows) to build token frequencies …", flush=True)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    counter: Counter = Counter()
    for i, row in enumerate(ds):
        if i >= n_rows:
            break
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        counter.update(ids)
    print(f"  Unique token IDs observed: {len(counter):,}", flush=True)
    return counter


def filter_single_token_ids(tokenizer, token_ids: list[int]) -> list[int]:
    """
    Keep only token IDs whose decoded string re-encodes to exactly that one ID.
    This removes subword artifacts where decode→encode introduces extra tokens.
    """
    kept = []
    for tid in token_ids:
        surface = tokenizer.decode([tid])
        reenc   = tokenizer.encode(surface, add_special_tokens=False)
        if len(reenc) == 1 and reenc[0] == tid:
            kept.append(tid)
    return kept


def extract_embeddings(
    model_path: str,
    token_ids: list[int],
    is_peft: bool = False,
) -> np.ndarray:
    """
    Load a model (optionally merge LoRA adapters), extract embed_tokens rows
    for the given token_ids, then immediately delete the model to free RAM.

    Returns an L2-normalised float32 array of shape (n_tokens, hidden_dim).
    """
    print(f"\n  Loading: {model_path}", flush=True)
    if is_peft:
        base  = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="cpu"
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()      # folds LoRA deltas into weights
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cpu"
        )

    print("  Extracting embed_tokens …", flush=True)
    # Index in bfloat16 first to avoid materialising the full 128k-row matrix as float32
    ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    emb = model.model.embed_tokens.weight[ids_tensor].detach().float().cpu().numpy()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    emb = normalize(emb, norm="l2")           # unit vectors → cosine geometry
    print(f"  Embedding shape: {emb.shape}", flush=True)
    return emb


def upper_triangle(sim_matrix: np.ndarray) -> np.ndarray:
    """Return the upper triangle (k=1) of a square similarity matrix as 1-D vector."""
    n   = sim_matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return sim_matrix[idx]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ── 1. Build token frequency from Romanian text ──────────────────────────
    freq    = build_token_frequency(tokenizer)
    top_ids = [tid for tid, _ in freq.most_common(TOP_N_TOKENS)]

    # ── 2. Filter to single-token round-trips ───────────────────────────────
    single_ids = filter_single_token_ids(tokenizer, top_ids)
    print(f"\nSingle-token words retained: {len(single_ids)} / {len(top_ids)}", flush=True)

    # ── 3. Extract embeddings (one model at a time to keep peak RAM low) ─────
    print("\n[1/3] Base model", flush=True)
    emb_base = extract_embeddings(BASE_MODEL,          single_ids, is_peft=False)
    print("\n[2/3] CPT checkpoint (LoRA merge)", flush=True)
    emb_cpt  = extract_embeddings(CPT_CHECKPOINT,      single_ids, is_peft=True)
    print("\n[3/3] Romanian reference model", flush=True)
    emb_ro   = extract_embeddings(RO_REFERENCE_MODEL,  single_ids, is_peft=False)

    # ── 4. Full pairwise cosine similarity matrices ──────────────────────────
    S_base = emb_base @ emb_base.T   # (n, n)
    S_cpt  = emb_cpt  @ emb_cpt.T
    S_ro   = emb_ro   @ emb_ro.T

    # ── 5. Spearman correlations on upper triangles ──────────────────────────
    v_base = upper_triangle(S_base)
    v_cpt  = upper_triangle(S_cpt)
    v_ro   = upper_triangle(S_ro)

    corr_base, p_base = spearmanr(v_base, v_ro)
    corr_cpt,  p_cpt  = spearmanr(v_cpt,  v_ro)
    delta = corr_cpt - corr_base

    # ── 6. Report ────────────────────────────────────────────────────────────
    sep = "═" * 56
    print(f"\n{sep}")
    print("Spearman rho  —  upper-triangle pairwise cosine similarities")
    print(f"  Base  vs RoLlama3.1 :  rho = {corr_base:+.4f}   (p = {p_base:.2e})")
    print(f"  CPT   vs RoLlama3.1 :  rho = {corr_cpt:+.4f}   (p = {p_cpt:.2e})")
    print(f"  Delta (CPT - Base)  :  {delta:+.4f}")
    print(sep)
    if delta > 0:
        print("Hypothesis SUPPORTED: CPT moved embeddings closer to RoLlama3.1")
    else:
        print("Hypothesis NOT supported")
    print(sep)

    # ── 7. Heatmap visualisation ─────────────────────────────────────────────
    n = len(single_ids)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    panels = [
        (S_base, f"Base — Llama 3.2-1B\nhidden={emb_base.shape[1]}"),
        (S_cpt,  f"CPT — Romanian LoRA merged\nhidden={emb_cpt.shape[1]}"),
        (S_ro,   f"RoLlama3.1-8B (reference)\nhidden={emb_ro.shape[1]}"),
    ]
    for ax, (S, title) in zip(axes, panels):
        im = ax.imshow(S, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("token index")
        ax.set_ylabel("token index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Pairwise cosine similarity  —  top {n} single-token Romanian words\n"
        f"rho(Base, Ro) = {corr_base:+.4f}     "
        f"rho(CPT, Ro) = {corr_cpt:+.4f}     "
        f"Delta = {delta:+.4f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved -> {OUTPUT_FIGURE}", flush=True)

    # ── 8. Save numerical results ─────────────────────────────────────────────
    result_lines = [
        "Embedding Similarity Analysis Results",
        "=" * 56,
        f"Single-token words: {len(single_ids)}",
        f"Dataset rows sampled: {DATASET_ROWS}",
        f"CPT checkpoint: {CPT_CHECKPOINT}",
        f"Reference model: {RO_REFERENCE_MODEL}",
        "",
        "Spearman rho  —  upper-triangle pairwise cosine similarities",
        f"  Base  vs RoLlama3.1 :  rho = {corr_base:+.4f}   (p = {p_base:.2e})",
        f"  CPT   vs RoLlama3.1 :  rho = {corr_cpt:+.4f}   (p = {p_cpt:.2e})",
        f"  Delta (CPT - Base)  :  {delta:+.4f}",
        "",
        "Hypothesis SUPPORTED" if delta > 0 else "Hypothesis NOT supported",
    ]
    with open(OUTPUT_RESULTS, "w") as f:
        f.write("\n".join(result_lines) + "\n")
    print(f"Results saved  -> {OUTPUT_RESULTS}", flush=True)


if __name__ == "__main__":
    main()
