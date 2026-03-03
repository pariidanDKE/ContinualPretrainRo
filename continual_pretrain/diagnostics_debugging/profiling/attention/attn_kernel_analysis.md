# Attention Kernel Analysis: Eager vs Flash Attention 2

## Profile

`fa_attention_comparison/bs1_ga32_20260227_1856/`
- `3_bf16_gc.sqlite` — BF16, gradient checkpointing, eager attention
- `4_bf16_gc_fa2.sqlite` — BF16, gradient checkpointing, FA2

**Config:** BS=1, GA=32, max_length=2048, no packing, Llama-3.2-1B, fullFT

---

## Method

**NVTX ranges:**
- `attn_scores/eager` — wraps `attention_interface(...)` call in eager config
- `attn_scores/fa2` — wraps `attention_interface(...)` call in FA2 config

Both added via `record_function` + `emit_nvtx`, patching `eager_attention_forward` and
`ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` at module level in `train_milestones.py`.

**Query approach — correlation ID join** (Query 3 in `profiling_querries/-- SQLite.sql`):
```
NVTX_EVENTS (CPU time)
    ↓ timestamp overlap
CUPTI_ACTIVITY_KIND_RUNTIME   ← cudaLaunchKernel calls, issued synchronously on CPU
    ↓ correlationId
CUPTI_ACTIVITY_KIND_KERNEL    ← actual GPU execution
```
Direct timestamp join on GPU kernels fails for ~200µs ranges because the GPU async lag
exceeds the range duration. The runtime table bridges the timelines via correlation ID.

Note: `record_function` appends `, op_id=N` — use `LIKE 'attn_scores/eager%'` not `=`.

---

## Eager `attn_scores` Kernels (forward pass, confirmed)

| Short name | Full demangled name | gpu_dur_us | Role |
|---|---|---|---|
| `elementwise_kernel` | `direct_copy_kernel_cuda ... BFloat16` | 8 µs | tensor reshape/copy |
| `elementwise_kernel` | `direct_copy_kernel_cuda ... BFloat16` | 8 µs | tensor reshape/copy |
| `Kernel2` | `cutlass::Kernel2<cutlass_75_tensorop_bf16_s1688gemm_bf16_64x128_tn_align1>` | 105 µs | **QKᵀ matmul** — `tn` = Q transposed × K |
| `vectorized_elementwise_kernel` | `AUnaryFunctor<..., MulFunctor<float>>` | 92 µs | **scale by 1/√d_head** |
| `elementwise_kernel` | `CUDAFunctor_add<BFloat16>` | 101 µs | **causal mask addition** (adds -inf) |
| `unrolled_elementwise_kernel` | `LoadWithCast<1> + StoreWithCast<1>` | 141 µs | dtype cast float32→bf16 |
| `softmax_warp_forward` | `softmax_warp_forward<float, float, float, 10, false, false>` | 186 µs | **softmax** (runs in float32 internally) |
| `vectorized_elementwise_kernel` | `bfloat16_copy_kernel_cuda` | 137 µs | bf16 copy of softmax output |
| `Kernel2` | `cutlass::Kernel2<cutlass_75_tensorop_bf16_s1688gemm_bf16_64x128_nn_align1>` | 92 µs | **softmax·V matmul** — `nn` = normal × normal |
| `elementwise_kernel` | `direct_copy_kernel_cuda ... BFloat16` | 10 µs | tensor reshape/copy |

Notes:
- `tn` vs `nn` layout is the definitive distinguisher between QKᵀ and softmax·V GEMMs
- Both use `cutlass_75` (Turing tiles) — the seq×seq matrix shape selects different tiles than the hidden_dim projections
- Softmax runs internally in float32 despite the model being bf16, explaining the surrounding dtype cast kernels

---

## FA2 `attn_scores` Kernels (forward pass, confirmed)

| Short name | Full demangled name | gpu_dur_us | Role |
|---|---|---|---|
| `elementwise_kernel_with_index` | `arange_cuda_out ... long` | 1 µs | generate position indices (`cu_seqlens` prep) |
| `reduce_kernel` | `ReduceOp<long, MinNanFunctor<long>>` | 4 µs | min over sequence lengths |
| `elementwise_kernel` | `CUDAFunctor_add<long>` | 2 µs | position index arithmetic |
| `vectorized_elementwise_kernel` | `CUDAFunctor_add<long>` | 2 µs | position index arithmetic |
| `vectorized_elementwise_kernel` | `AbsFunctor<long>` | 1 µs | absolute value on positions |
| `reduce_kernel` | `ReduceOp<long, sum_functor<long>>` | 3 µs | sum of sequence lengths |
| `unrolled_elementwise_kernel` | `direct_copy_kernel_cuda ... bool` | 2 µs | copy boolean mask |
| **`flash_fwd_kernel`** | `flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, bfloat16_t>, is_causal=true>` | **336 µs** | **QKᵀ + scale + mask + softmax + softmax·V — all fused** |

All 7 bookkeeping kernels operate on `long`/`bool` — they compute `cu_seqlens` metadata
required by FlashAttention as input. They are not part of the attention computation.

Key detail from `flash_fwd_kernel` template: `is_causal=true` is a compile-time parameter,
meaning causal masking is baked into the kernel — the separate `CUDAFunctor_add` mask
kernel is not fused but **eliminated entirely**.

---

## What FA2 Fuses vs Eliminates

| Eager kernel | Fate in FA2 |
|---|---|
| QKᵀ GEMM (`cutlass_75 tn`) | fused into `flash_fwd_kernel` SRAM tiling |
| scale `MulFunctor` | fused |
| causal mask `CUDAFunctor_add` | **eliminated** — `is_causal=true` compile-time flag |
| dtype cast `LoadWithCast/StoreWithCast` | fused |
| `softmax_warp_forward` | fused |
| bf16 copy post-softmax | fused |
| softmax·V GEMM (`cutlass_75 nn`) | fused into `flash_fwd_kernel` SRAM tiling |

The seq×seq attention matrix never materialises in HBM. HBM traffic reduced by
`2 × seq² × num_heads × 2 bytes` per forward pass (= 536 MB/layer at seq=2048, 32 heads,
bf16 — theoretical, not yet confirmed with NCU).

---

## NCU Comparison Results (confirmed, 2026-02-28)

**Profiles:** `ncu_attn_profiles/{eager,fa2}_bs1_seq2048_20260228_1700/`
**Analysis script:** `ncu_attn_comparison.py` → `results/ncu_comparison.{json,md}`

NCU captured invocations at an effective seq ≈ 168 tokens (warmup step, not
full 2048). **Absolute byte values are seq-length-dependent; ratios are valid.**

### Per-Layer Comparison (1 attention layer)

Eager layer = avg(QKᵀ_GEMM) + avg(softmax) + avg(PV_GEMM)
FA2 layer   = avg(flash_fwd_kernel)

| Metric | Eager | FA2 | FA2/Eager | Savings |
|---|---:|---:|---:|---:|
| HBM read (MB) | 7.40 | 1.10 | 0.149x | **+85%** |
| HBM write (MB) † | 4.95 | 2.29 | 0.462x | **+54%** |
| L2 total (MB) | 23.1 | 7.22 | 0.313x | **+69%** |
| GPU time (µs) | 38.6 | 25.0 | 0.647x | **+35%** |
| TensorCore util (%) | 9.4 | 22.6 | 2.41x | −141% |
| SM throughput (%) | 38.0 | 13.9 | 0.365x | +64% |
| Warp occupancy (%) | 44.7 | 8.19 | 0.183x | +82% |
| Regs / thread | 61 | 255 | 4.2x | — |

† Eager HBM write excludes PV_GEMM dirty-eviction artifact (~840 MB, see below).

### Key Findings

**HBM reads −85%** — confirms FA2 never materializes the seq×seq attention
matrix in HBM. The theoretical savings of `2 × seq² × H × 2B` per layer
are realized in the NCU counters.

**TensorCore util 2.4× higher in FA2** despite lower overall SM throughput.
These metrics use different denominators:
- `sm__pipe_tensor_op_hmma_cycles_active` → % of *active* cycles doing tensor ops
- `sm__throughput` → % of *elapsed* cycles (includes stall time)

FA2 stalls frequently on SRAM synchronization barriers between tile loops;
eager GEMM kernels stall less but waste elapsed cycles in separate HBM round-trips.

**255 regs/thread (FA2) vs 61 avg (eager)** — flash_fwd keeps the output
accumulator O and softmax statistics (m, l) in registers across inner loops,
enabling SRAM tiling. The register pressure crushes warp occupancy (8% vs 45%),
giving FA2 less latency hiding per SM — but the HBM traffic reduction dominates.

**SM throughput paradox**: FA2 has lower SM throughput (13.9% vs 38%) yet runs
35% faster per layer. The speedup is memory-bound, not compute-bound: FA2
eliminates 5+ HBM round-trips (QKᵀ write + scale + mask + cast + softmax read/write)
that take ~14 µs combined in eager, at the cost of lower SM efficiency per cycle.

### PV_GEMM dram__bytes_write Artifact

PV_GEMM reports ~840 MB HBM write but its grid is `(3, 1, 32)` = 96 blocks.
The actual output is O(few MB). The inflated value is dirty L2-line evictions:
the large QKᵀ output matrix cached in L2 is flushed to DRAM when PV_GEMM's
smaller working set evicts those lines. `lts__t_bytes.sum` (L2 total) is unaffected.

---

## What Is Still Missing

1. **Projections (q/k/v/o_proj)** — sit outside `attention_interface(...)` so not captured
   by `attn_scores` ranges. Need a separate NVTX range around the full `self_attn.forward`
   and correlation ID query to confirm their kernel names.

2. **Backward pass** — not isolated with correlation ID. `attn_scores` ranges only cover
   the forward `attention_interface` call; backward is not yet confirmed.

3. **Unsloth path not profiled** — configs here use standard HuggingFace `flash_attention_2`.

---

## Next Steps

- Profile at seq_len=4096 where seq×seq HBM savings are 4× larger (scales as seq²)
- Add correlation ID query on a range wrapping full `self_attn.forward` to confirm projections
- Profile backward pass: FA2 recomputes attention scores from SRAM during backward,
  so backward HBM savings should also be significant
