#!/usr/bin/env python3
"""
Per-token embedding drift analysis: Base vs CPT (Llama 3.2-1B).

Computes drift over the FULL 128k vocabulary — no frequency filtering needed
since both models share the same architecture and tokenizer.

Metrics per token:
  - L2 distance between raw base and CPT embedding vectors
  - Cosine distance (1 - cosine_similarity) between them
  - Relative drift (L2 / ||emb_base||)

Usage:
    python diagnostics_debugging/analyze_embedding_drift.py
"""

import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────────────

BASE_MODEL = "meta-llama/Llama-3.2-1B"
CPT_CHECKPOINT = (
    "/home/dan-parii/Documents/ContinualPretrainRo/continual_pretrain/outputs/cpt/"
    "embed_comparison_20260306_185038/cpt-llama-with_embeddings_r64-ms5-125M/"
    "20260306_1851-g8r0v0xz/checkpoint-1019"
)
TOKENIZER_NAME  = BASE_MODEL
TOP_DRIFT_SHOW  = 40

SCRIPT_DIR      = Path(__file__).parent
OUTPUT_FIGURE   = str(SCRIPT_DIR / "embedding_drift_top_tokens.png")
OUTPUT_RESULTS  = str(SCRIPT_DIR / "embedding_drift_results.txt")

# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_raw_embeddings(model_path: str, token_ids: list[int], is_peft: bool = False) -> np.ndarray:
    """Return raw (non-normalised) float32 embeddings of shape (n, hidden_dim)."""
    print(f"\n  Loading: {model_path}", flush=True)
    if is_peft:
        base  = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cpu")

    ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    emb = model.model.embed_tokens.weight[ids_tensor].detach().float().cpu().numpy()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Embedding shape: {emb.shape}", flush=True)
    return emb


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    tokenizer  = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    all_ids    = list(range(len(tokenizer)))
    print(f"Full vocabulary: {len(all_ids):,} tokens", flush=True)

    print("\n[1/2] Base model", flush=True)
    emb_base = extract_raw_embeddings(BASE_MODEL,      all_ids, is_peft=False)
    print("\n[2/2] CPT checkpoint", flush=True)
    emb_cpt  = extract_raw_embeddings(CPT_CHECKPOINT,  all_ids, is_peft=True)

    # ── Per-token drift metrics ──────────────────────────────────────────────
    diff        = emb_cpt - emb_base                           # (n, hidden_dim)
    l2_drift    = np.linalg.norm(diff, axis=1)                 # absolute vector movement

    # cosine distance = 1 − cosine_similarity
    # guard against zero-norm tokens (e.g. unused special tokens in base model)
    norm_base   = np.linalg.norm(emb_base, axis=1, keepdims=True)
    norm_cpt    = np.linalg.norm(emb_cpt,  axis=1, keepdims=True)
    zero_mask   = (norm_base.squeeze() == 0) | (norm_cpt.squeeze() == 0)
    safe_base   = np.where(norm_base == 0, 1.0, norm_base)
    safe_cpt    = np.where(norm_cpt  == 0, 1.0, norm_cpt)
    cos_sim     = np.sum((emb_base / safe_base) * (emb_cpt / safe_cpt), axis=1)
    cos_dist    = 1.0 - cos_sim
    cos_dist[zero_mask] = np.nan          # mask meaningless values

    # relative drift: L2 distance normalised by original vector norm
    rel_drift   = l2_drift / (norm_base.squeeze() + 1e-9)

    # Decode surface form for all tokens
    surfaces    = [tokenizer.decode([tid]) for tid in all_ids]

    # Sort by L2 drift (descending), exclude zero-norm tokens from ranking
    valid_mask  = ~zero_mask
    valid_order = np.argsort(np.where(valid_mask, l2_drift, -1))[::-1]
    order       = valid_order[valid_mask[valid_order]]

    # ── Print top drifters ───────────────────────────────────────────────────
    sep = "═" * 66
    header = f"{'Rank':>4}  {'Token':>20}  {'L2 drift':>10}  {'Cos dist':>10}  {'Rel drift':>10}"
    print(f"\n{sep}")
    print(f"Top {TOP_DRIFT_SHOW} tokens with largest embedding drift (Base → CPT)  [full 128k vocab]")
    print(sep)
    print(header)
    print("─" * 66)

    result_lines = [
        "Embedding Drift Analysis — Base vs CPT (Llama 3.2-1B)  [full 128k vocab]",
        sep,
        f"Vocabulary size: {len(all_ids):,}",
        "",
        f"Top {TOP_DRIFT_SHOW} most-drifted tokens",
        sep,
        header,
        "─" * 66,
    ]

    for rank, idx in enumerate(order[:TOP_DRIFT_SHOW], 1):
        tok  = repr(surfaces[idx])
        line = f"{rank:>4}  {tok:>20}  {l2_drift[idx]:>10.4f}  {cos_dist[idx]:>10.6f}  {rel_drift[idx]:>10.4f}"
        print(line)
        result_lines.append(line)

    print(sep)
    result_lines.append(sep)

    # Summary stats
    n_zero = int(zero_mask.sum())
    print(f"\nDrift summary across all {len(all_ids):,} tokens  ({n_zero} zero-norm tokens excluded from cos stats):")
    print(f"  L2 drift  — mean={l2_drift.mean():.4f}  median={np.median(l2_drift):.4f}  max={l2_drift.max():.4f}")
    print(f"  Cos dist  — mean={np.nanmean(cos_dist):.6f}  median={np.nanmedian(cos_dist):.6f}  max={np.nanmax(cos_dist):.6f}")
    print(f"  Rel drift — mean={rel_drift.mean():.4f}  median={np.median(rel_drift):.4f}  max={rel_drift.max():.4f}")

    result_lines += [
        "",
        f"Drift summary across all {len(all_ids):,} tokens  ({n_zero} zero-norm tokens excluded from cos stats):",
        f"  L2 drift  — mean={l2_drift.mean():.4f}  median={np.median(l2_drift):.4f}  max={l2_drift.max():.4f}",
        f"  Cos dist  — mean={np.nanmean(cos_dist):.6f}  median={np.nanmedian(cos_dist):.6f}  max={np.nanmax(cos_dist):.6f}",
        f"  Rel drift — mean={rel_drift.mean():.4f}  median={np.median(rel_drift):.4f}  max={rel_drift.max():.4f}",
    ]

    with open(OUTPUT_RESULTS, "w") as f:
        f.write("\n".join(result_lines) + "\n")
    print(f"\nResults saved -> {OUTPUT_RESULTS}", flush=True)

    # ── Plot ─────────────────────────────────────────────────────────────────
    top_idx     = order[:TOP_DRIFT_SHOW]
    top_labels  = [repr(surfaces[i]) for i in top_idx]
    top_l2      = l2_drift[top_idx]
    top_cos     = cos_dist[top_idx]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Bar chart: L2 drift
    ax = axes[0]
    bars = ax.barh(range(TOP_DRIFT_SHOW), top_l2[::-1], color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(TOP_DRIFT_SHOW))
    ax.set_yticklabels(top_labels[::-1], fontsize=8)
    ax.set_xlabel("L2 drift  (||emb_cpt − emb_base||)")
    ax.set_title(f"Top {TOP_DRIFT_SHOW} tokens by L2 embedding drift — Base → CPT")
    ax.invert_xaxis() if False else None

    # Bar chart: cosine distance
    ax = axes[1]
    ax.barh(range(TOP_DRIFT_SHOW), top_cos[::-1], color="darkorange", edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(TOP_DRIFT_SHOW))
    ax.set_yticklabels(top_labels[::-1], fontsize=8)
    ax.set_xlabel("Cosine distance  (1 − cos_sim)")
    ax.set_title(f"Top {TOP_DRIFT_SHOW} tokens by cosine distance — Base → CPT")

    plt.suptitle("Embedding drift: Base Llama 3.2-1B → CPT Romanian", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches="tight")
    print(f"Figure saved  -> {OUTPUT_FIGURE}", flush=True)


if __name__ == "__main__":
    main()
