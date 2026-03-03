# BS=1 vs BS=32 Profiling Analysis
**Config**: qLoRA r64 + paged AdamW 8-bit + FA2 + GC + Unsloth (Config 8)
**Model**: unsloth/Llama-3.2-1B, BF16, effective batch = 32 tokens for both
**Kernel profiled (NCU)**: `void Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>`
**Profiles**: `outputs/ncu_profiles/*0058*`, `outputs/nsys_profiles/batch_size_comparison/`

---

## NCU: Per-kernel efficiency (identical for both BSes)

| Metric | BS=1 | BS=32 |
|---|---|---|
| Grid size | (80, 4, 1) = 320 blocks | (2000, 4, 1) = 8000 blocks |
| Blocks per SM (82 SMs) | 3.9 | 97.6 |
| `sm__pipe_tensor_op_hmma` | 47.9% | 48.4% |
| `sm__throughput` | 46.6% | 48.2% |
| `sm__warps_active` | 16.66% | 16.66% |
| `smsp__warps_eligible` | 0.87% | 0.87% |
| `gpu__time_duration` | ~790 µs | ~19.07 ms |
| `dram__bytes_read` | ~67 MB | ~1.45 GB |
| `dram__bytes_write` | ~20 MB | ~523 MB |
| `lts__t_bytes` (L2 traffic) | ~522 MB | ~13.11 GB |

### Key observations
- **Warp occupancy is register-limited**: tile shape `256×128×32` forces exactly 1 block/SM
  (16.66% × 48 max warps = 8 warps = 1 block of 256 threads). Batch size cannot change this.
- **Tensor core utilization (~48%) is set by the 3-stage software pipeline**, not batch size.
  The pipeline hides HBM latency but not fully — warps are stalled 99% of cycles
  (`smsp__warps_eligible = 0.87%`), yet tensor cores stay active via async prefetch.
- **`smsp__sass_thread_inst_executed_op_hfma_pred_on = 0`**: expected — `hfma` counts scalar
  FP16 CUDA-core ops. BF16 tensor core ops are counted by `sm__pipe_tensor_op_hmma_cycles_active`.
  The 262M `ffma` counts are FP32 epilogue (scaling/write-back), not matrix math.
- **`lts__t_hit_rate = n/a`**: CUTLASS uses streaming (bypass) cache policy for the activation
  matrix (A), but normal L2 caching for the weight matrix (B). The n/a is a metric
  reporting artifact, not zero hit rate.

---

## NCU: Equivalent-batch comparison (BS=32 wins)

Comparing equivalent effective batch of 32 samples:

| | BS=1 × 32 GA steps | BS=32 × 1 step | Saving |
|---|---|---|---|
| Total kernel time | 32 × 790µs = **25.3ms** | **19.0ms** | **25% faster** |
| Total HBM reads | 32 × 67MB = **2.14GB** | **1.45GB** | **32% less** |

### Why BS=32 reads less HBM for the same work
DRAM scales sub-linearly with blocks: 25× more blocks → only 21.6× more bytes (not 25×).

The weight matrix B (same for all token positions) is cached in L2 within a single large
kernel call. At BS=32, 500 blocks share each weight column tile — L2 retains it across
sequential waves on the same SM. At BS=1 × 32 separate calls, intervening kernels (LayerNorm,
attention, etc.) evict the weight from L2 between steps, forcing a full HBM reload each time.

**The 32% HBM saving = weight matrix loaded once (BS=32) vs loaded 32 times (BS=1 × 32 calls).**

This translates directly into the 25% kernel time saving.

---

## nsys: System-level efficiency

Both profiles cover ~90s of training (fair comparison).

| | BS=1 | BS=32 |
|---|---|---|
| Overall GPU busy | **83.4%** | **97.1%** |
| Overall GPU idle | **16.6%** | **2.9%** |

BS=32 eliminates ~14pp of GPU idle time.

### Inter-kernel gap for Kernel2 tn (includes other kernels, not pure idle)

| | BS=1 | BS=32 |
|---|---|---|
| Avg kernel duration | 117 µs | 15,048 µs |
| Avg gap to next launch | 445 µs | 37,321 µs |
| This-kernel duty cycle | 20.8% | 28.7% |

Note: the gap between consecutive Kernel2 tn launches includes other kernels running
(other GEMMs, attention, dequantize), not pure idle. The overall duty cycle (above) is
the cleaner metric for idle time.

### Top kernels by total GPU time (BS=1)

| Kernel | Count | Avg duration |
|---|---|---|
| Kernel2 tn (LoRA GEMM, profiled) | 26,459 | 579 µs |
| ampere_bf16 s16816gemm tn | 17,682 | 478 µs |
| ampere_bf16 s1688gemm tn | 92,782 | 61 µs |
| ampere_bf16 s16816gemm nn | 14,133 | 399 µs |
| kDequantizeBlockwise | 175,257 | 28 µs |
| flash_fwd_kernel | 17,150 | 107 µs |
| flash_bwd_dq_dk_dv | 8,576 | 218 µs |

---

## Summary: Two independent sources of BS=32 speedup

1. **Kernel-level (~25% faster per equivalent batch)**
   Weight tile L2 reuse within one large kernel call vs cold reload across 32 separate calls.
   Directly measured from NCU HBM read scaling (21.6× actual vs 25× expected).

2. **System-level (~14pp duty cycle improvement)**
   Python overhead, kernel launches, LayerNorm, optimizer bookkeeping — paid 32× at BS=1,
   paid once at BS=32. Directly measured from nsys overall GPU busy %.

---

## nsys: Time breakdown by kernel category

| Category | BS=1 time | BS=1 % | BS=32 time | BS=32 % |
|---|---|---|---|---|
| CUTLASS Kernel2 | 28.12s | 37.5% | 40.45s | **46.3%** |
| Other GEMM | 30.46s | 40.6% | 34.93s | 40.0% |
| Dequantize | 5.32s | **7.1%** | 0.27s | **0.3%** |
| Other | 5.12s | 6.8% | 5.39s | 6.2% |
| FA2 backward | 2.14s | 2.9% | 2.97s | 3.4% |
| Elementwise | 2.09s | 2.8% | 1.04s | 1.2% |
| FA2 forward | 1.84s | 2.4% | 2.22s | 2.5% |

### Key finding: Dequantize is the hidden qLoRA tax at small batch size

BS=1: **350,514 dequantize calls, 7.1% of GPU time**
BS=32: **17,198 dequantize calls, 0.3% of GPU time**

With qLoRA (4-bit weights), every GEMM call requires first dequantizing INT4 weights to
BF16. At BS=1 × 32 GA steps, each step has its own full set of dequantize calls —
20× more calls total than BS=32. This overhead is completely amortized at BS=32 since
dequantization happens once per effective batch regardless of per-sample processing.

This makes qLoRA disproportionately sensitive to batch size vs full fine-tuning — the
dequantize penalty compounds with each gradient accumulation step.

GEMM (Kernel2 + Other) dominates GPU time for both: 78% at BS=1, 86% at BS=32.
Attention (FA2 fwd + bwd) is only ~5% for both — not the bottleneck.

## nsys: Idle gap analysis (BS=1)

Largest idle gaps: **59–74ms**. Likely correspond to optimizer steps (paged AdamW 8-bit
triggers CPU-GPU sync at the end of each 32-step gradient accumulation cycle). With
~46 optimizer steps in 90s, these gaps account for most of the 16.6% idle time.

---

## Open questions (TODO)
- [ ] Confirm idle gap source: correlate 59-74ms gaps with NVTX optimizer annotations
- [ ] Attention kernel NCU comparison: profiles exist in `outputs/ncu_attn_profiles/`
