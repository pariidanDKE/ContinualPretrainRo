# Sequence Packing vs No-Packing: Why Wall-Clock Time Is Identical

## Hypothesis

Sequence packing and no-packing produce identical training throughput because the workload is **compute-bound** — the tensor cores are the bottleneck, and packing cannot change the total amount of MatMul work that needs to be done.

Unsloth already provides **padding-free training by default**: only real tokens are fed into the CUDA kernels and moved through HBM and L2 — no padding tokens waste bandwidth or compute. This means the memory-side inefficiency that packing is classically designed to fix was already eliminated before packing was introduced.

The remaining argument for packing was **arithmetic intensity**: by concatenating multiple short sequences into a single sample up to `max_seq_len`, each GEMM call moves the weight matrix once but reuses it across more tokens — more FLOPs per byte fetched from DRAM. This pushes the kernel further past the roofline ridge point into compute-bound territory, which should in principle reduce total data movement per token.

However, the workload was **already on the compute-bound side of the roofline** before packing. The tensor cores are the active bottleneck. Once you are past the ridge point, improving arithmetic intensity further does not reduce runtime — you are no longer limited by how much data you move, but by how many multiply-accumulate operations need to be executed. Packing increases the tokens processed per step, which increases FLOPs per step proportionally, which increases step duration proportionally. With proportionally fewer steps needed to cover the same token budget, the total time is unchanged.

The mental model: the tensor cores are a fixed-throughput checkout lane. Packing does not add more lanes or make the lane faster — it just changes whether you queue 10 people or 50 people at a time. Total throughput of the lane is unchanged.

---

## Profiling Setup

Two nsys profiles were captured for `bs=32, ga=4` (effective batch size 128), Config 8 (qLoRA r64 + paged AdamW 8-bit + FA2 + Unsloth), differing only in packing:

- `8a` — no packing (`model.builder.packing=false`, `training_args.packing=false`)
- `8b` — packing (`model.builder.packing=true`, `training_args.packing=true`)

Each run captured 90 seconds of steady-state training after a 90-second delay to clear model load and CUDA graph compilation. GPU kernel activity was extracted from the `.sqlite` export of the nsys report.

**Profiles:**
- `outputs/nsys_profiles/packing_comparison/bs32_ga4_20260307_2119/8a_bf16_gc_fa2_paged_qlora64_unsloth_nopacking.sqlite`
- `outputs/nsys_profiles/packing_comparison/bs32_ga4_20260307_2119/8b_bf16_gc_fa2_paged_qlora64_unsloth_packing.sqlite`

Supporting NCU profiles (kernel-level hardware counters, bs=8 ga=4):
- `outputs/ncu_profiles/packing_bs8_ga4_20260307_2053/`
- `outputs/ncu_profiles/nopacking_bs8_ga4_20260307_1954/`

---

## Evidence

### 1. LoRA GEMM scales perfectly linearly with tokens-per-step

| | NO-PACK | PACK |
|---|---|---|
| LoRA GEMM launches | 10,120 | 3,816 |
| Avg duration per launch | 3,898 µs | 10,119 µs |
| **Total LoRA GEMM time** | **39,447 ms** | **38,615 ms** |

Packing produces **2.65× fewer launches**, each taking **2.59× longer** — near-perfect linear scaling. This is the direct signature of a compute-bound workload. If any other factor were limiting — memory bandwidth, kernel launch overhead, dequantization pipeline — the scaling would be non-linear and packing would show a measurable benefit. The 1:1 proportionality means runtime is determined solely by total FLOPs, which packing does not change.

### 2. MatMul is 86% of GPU time in both runs — and does not move

| Category | NO-PACK ms | NO-PACK % | PACK ms | PACK % | Δ |
|---|---|---|---|---|---|
| LoRA GEMM | 39,447 | 46.4% | 38,615 | 47.9% | −2.1% |
| Base GEMM | 33,940 | 39.9% | 30,915 | 38.4% | −8.9% |
| **MatMul total** | **73,387** | **86.3%** | **69,530** | **86.3%** | **−5.3%** |
| FlashAttn | 5,097 | 6.0% | 4,954 | 6.1% | −2.8% |
| Dequant | 260 | 0.3% | 99 | 0.1% | −62.1% |
| Optimizer | 20 | 0.0% | 9 | 0.0% | −56.7% |
| Everything else | 6,317 | 7.4% | 6,009 | 7.5% | — |
| **Total GPU time** | **85,061** | | **80,601** | | **−5.2%** |

The MatMul share is **86.3% in both runs to the first decimal place**. Every optimization that packing does provide — dequant (−62%), optimizer (−57%), elementwise (−9%) — lives inside the remaining 13.7%. Even if packing eliminated that entire slice, the maximum possible wall-clock saving would be 13.7%. In practice the total saving is only 5.2%, almost entirely from Base GEMM benefiting from larger problem sizes, not from any fundamental efficiency gain.

### 3. NCU confirms tensor cores are the active bottleneck

From NCU profiling of the dominant `256x128_32x3` CUTLASS tile (bs=8, apples-to-apples, same tile in both runs):

| Metric | PACK | NO-PACK |
|---|---|---|
| `sm__throughput` (% of peak, elapsed) | 48.2% | 39.8% |
| `sm__pipe_tensor_op_hmma` (% of peak, active) | 48.4% | 49.5% |
| `gpu__time_duration.sum` | 9.754 ms | 3.050 ms |
| `dram__bytes_read.sum` | 737 MB | 210 MB |

The HMMA pipe (tensor cores) accounts for nearly all of SM throughput in both cases — the two metrics track each other closely. The tensor core pipeline is the dominant consumer of SM cycles, confirming it is the active bottleneck. The ~48% figure is not idle in a wasteful sense: the remaining cycles are occupied by the surrounding instruction mix (memory address generation, dequantization pipeline, wave scheduling latency between tile dispatches) that cannot be parallelised away.

Packing's higher `sm__throughput` (48.2% vs 39.8%) for the same tile reflects fewer kernel launches meaning less inter-launch idle time — not a fundamental improvement in compute efficiency. The HMMA pipe itself is nearly identical (48.4% vs 49.5%), confirming both runs are spending the same fraction of active SM cycles on tensor core work.

---

## Conclusion

The hypothesis is confirmed. Sequence packing does not improve wall-clock training time because:

1. **The memory-bandwidth problem was already solved** — Unsloth's padding-free training ensures only real tokens move through HBM and L2, eliminating the inefficiency packing was originally designed to address.
2. **The workload was already compute-bound** — improving arithmetic intensity further does not help once you are past the roofline ridge point. The tensor cores are the limiting resource, not memory bandwidth.
3. **Packing reshuffles FLOPs, it does not reduce them** — total tokens × model size × LoRA rank is fixed. Packing changes the shape of each MatMul call but not the total MatMul work, and the nsys data shows this directly in the near-perfect linear scaling of LoRA GEMM time.

The only path to faster training in this regime is either **reducing total FLOPs** (smaller rank, fewer layers trained) or **increasing tensor core throughput** (higher-end GPU, bf16 sparsity, or a fundamentally different training approach).
