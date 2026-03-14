# Embedding Analysis Findings
**Date:** 2026-03-08
**Branch:** refactor/training_loop
**CPT run:** `embed_comparison_20260306_185038` — `cpt-llama-with_embeddings_r64-ms5-125M` — checkpoint-1019
**Training budget:** ~125M Romanian tokens

---

## Setup

Three models compared:
| Model | Role | Hidden dim |
|---|---|---|
| `meta-llama/Llama-3.2-1B` | Baseline (no Romanian) | 2048 |
| CPT checkpoint-1019 | Llama 3.2-1B after Romanian CPT | 2048 |
| `OpenLLM-Ro/RoLlama3.1-8b-Instruct` | Romanian-exposed reference | 4096 |

All share the Llama 3 128k BPE tokenizer — token IDs are directly comparable across models.

**Token list:** Top 2000 most-frequent token IDs from 10k rows of `OpenLLM-Ro/fineweb2-ro-human`, filtered to single-token round-trips → **1994 tokens** retained.

**Embedding configuration:** `embed_tokens` and `lm_head` are **fully fine-tuned** (stored as complete weight matrices via `modules_to_save`, not LoRA-decomposed). Confirmed by inspecting `adapter_model.safetensors` — keys are `base_model.model.model.embed_tokens.weight` with shape `[128256, 2048]`.

---

## Experiment 1 — Structural Geometry Shift (Spearman ρ)
*Script: `analyze_embedding_similarity.py`*

**Method:** For each model, compute a pairwise cosine similarity matrix over the 1994 tokens (shape `1994×1994`). Flatten the upper triangle (~2M values) and compute Spearman ρ between models.

This is dimension-agnostic — comparing *relational structure* (which tokens are similar to which), not the vectors themselves.

**Results:**

| Comparison | Spearman ρ | p-value |
|---|---|---|
| Base vs RoLlama3.1 | +0.3096 | ~0 |
| CPT vs RoLlama3.1 | +0.3093 | ~0 |
| **Δ (CPT − Base)** | **−0.0003** | — |

**Finding:** Hypothesis NOT supported. CPT produced essentially zero measurable change in the cosine similarity structure of the embedding space relative to the Romanian reference model.

**Why this makes sense:**
- The full embedding matrix is 128,256 × 2048 ≈ 262M parameters. 125M training tokens means very sparse gradient coverage — most vocabulary items receive few meaningful updates.
- Cosine *structure* (clustering geometry) requires large, correlated movement across many tokens simultaneously to shift — much harder than just improving loss on frequent tokens.
- The +0.31 baseline correlation shows Llama 3.2's English embeddings already partially encode Romanian token relationships — plausible given Romanian's Latin roots overlapping with English vocabulary seen during pretraining.

---

## Experiment 2a — Per-Token Embedding Drift, Romanian-frequent subset (Base → CPT)
*Script: `analyze_embedding_drift.py` — 1994-token subset*

**Method:** For each of the 1994 Romanian-frequent tokens compute:
- **L2 drift:** `‖emb_cpt − emb_base‖` — absolute vector movement
- **Cosine distance:** `1 − cos_sim` — directional change (scale-invariant)
- **Relative drift:** `L2 drift / ‖emb_base‖` — movement relative to original magnitude

**Drift summary (1994 tokens):**

| Metric | Mean | Median | Max |
|---|---|---|---|
| L2 drift | 0.0903 | 0.0900 | 0.1898 |
| Cosine distance | 0.00397 | 0.00374 | 0.031 |
| Relative drift | 0.088 | 0.087 | 0.251 |

**Key observation:** Mean cosine distance is ~0.004 with a tight distribution (mean ≈ median) — almost every token moved in roughly the same direction. Confirms Exp 1: relational geometry barely changed.

**Top drifters (subset):**

| Rank | Token | L2 drift | Cos dist | Freq | Interpretation |
|---|---|---|---|---|---|
| 1 | `'.\n\n'` | 0.1898 | 0.031 | 85 | Paragraph break — highest cosine distance by 3×; Romanian text has distinct paragraph rhythm |
| 2–4 | `'atie'`, `'isme'`, `'iti'` | ~0.12 | ~0.006 | low | Romanian morphological suffixes — genuine CPT signal |
| 5 | `' Haskell'` | 0.1234 | 0.008 | 20 | Rare English proper noun in atypical Romanian context |
| 6–7 | `' ii'`, `'|'` | ~0.12 | ~0.009 | low | `' ii'` = Romanian genitive/dative suffix; `|` structural token |
| 22 | `'aram'` | 0.1141 | 0.005 | 31 | Romanian past tense suffix (`-aram`) |
| 37 | `' sau'` | 0.1108 | 0.007 | **254** | Romanian for "or" — only high-frequency word in top 40 |

**Patterns:** structural/whitespace tokens > Romanian morphological suffixes > low-freq tokens in unexpected contexts > high-freq Romanian function words.

---

## Experiment 2b — Per-Token Embedding Drift, Full 128k Vocabulary (Base → CPT)
*Script: `analyze_embedding_drift.py` — full vocab run*

**Motivation:** Exp 2a only covered tokens frequent in Romanian text. Since Base and CPT share the same architecture and tokenizer, the full 128k embedding matrices can be diffed directly with no subset required.

**Drift summary (128,256 tokens, 1 zero-norm token excluded from cosine stats):**

| Metric | Mean | Median | Max |
|---|---|---|---|
| L2 drift | 0.0508 | 0.0679 | 0.545 |
| Cosine distance | 0.00231 | 0.00217 | 0.031 |
| Relative drift | 0.050 | 0.066 | 1.000 |

The mean L2 drift is lower than Exp 2a (0.051 vs 0.090) — expected, because the Romanian frequency-weighted subset overrepresented tokens that CPT actually trained on. The full vocab includes tens of thousands of rare tokens that barely moved. Cosine distance mean also dropped (0.0023 vs 0.0040) for the same reason.

**Top drifters (full vocab):**

| Rank | Token | L2 drift | Cos dist | Note |
|---|---|---|---|---|
| 1 | `'<|finetune_right_pad_id|>'` | 0.545 | — (zero-norm) | Unused special token, zero base vector — excluded from ranking |
| 2 | `' duplicates'` | 0.205 | 0.019 | English word in Romanian tech/web context |
| 3 | `'Additional'` | 0.196 | 0.022 | English loanword frequent in Romanian UI/docs |
| 4 | `' corruption'` | 0.191 | 0.019 | English word with high Romanian news frequency |
| 5 | `'.\n\n'` | 0.190 | 0.031 | Paragraph break — still highest cosine dist across full vocab |
| 13 | `'<|end_of_text|>'` | 0.170 | 0.015 | EOS token — saw many document boundaries |
| 16 | `'categorie'` | 0.168 | 0.015 | Romanian for "category" — genuine Romanian signal |
| 37 | `'Gratis'` | 0.155 | 0.011 | Romanian/German loanword used in Romanian web text |

**Critical finding:** The top drifters from the full vocab are almost entirely **English tokens** — `' duplicates'`, `'Additional'`, `' corruption'`, `' plans'`, `' invasion'`, `' granite'`, `' snake'`, `'Login'`, `'chat'`, etc. Only `'categorie'` and `'Gratis'` in the top 40 are recognisably Romanian. This means CPT is spending significant gradient budget on English tokens appearing in Romanian web context rather than on core Romanian morphology.

`'.\n\n'` retains the highest **cosine** distance across both the subset and full-vocab runs — structural formatting is the most directionally shifted token class.

---

## Cross-Experiment Conclusions

1. **CPT did not shift the embedding geometry** measurably toward a Romanian reference model (Δρ = −0.0003). The relational structure of the embedding space is largely unchanged after 125M tokens.

2. **CPT did update individual embeddings** but the movement is nearly uniform in direction (cosine distance mean 0.002–0.004), resembling a near-global scaling rather than targeted reshaping. The geometry moves as a block, so pairwise structure is preserved.

3. **Gradient budget is misallocated:** The full-vocab drift analysis reveals the largest movers are English tokens appearing incidentally in Romanian web text — not core Romanian vocabulary. Romanian morphological tokens do appear in the top drifters of the frequency-filtered subset, but are absent from the full-vocab top 40.

4. **Practical implication:** If the goal is embedding-level Romanian adaptation, 125M tokens is insufficient to restructure the space. To improve: more training tokens, a higher embedding learning rate, a higher proportion of Romanian text in the mix, or a token-frequency-weighted gradient mask to concentrate updates on Romanian-relevant token IDs.

---

## Files
| File | Description |
|---|---|
| `analyze_embedding_similarity.py` | Spearman ρ analysis (Exp 1) |
| `analyze_embedding_drift.py` | Per-token drift analysis (Exp 2) |
| `embedding_similarity_results.txt` | Numerical results from Exp 1 |
| `embedding_similarity_heatmaps.png` | Pairwise similarity heatmaps (Exp 1) |
| `embedding_drift_results.txt` | Ranked drift table from Exp 2 |
| `embedding_drift_top_tokens.png` | Bar charts of top drifters (Exp 2) |
