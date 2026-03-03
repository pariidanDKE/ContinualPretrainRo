# GPU Profiling Insights

## Tools
- **nsys** — discovery: kernel timeline, counts, durations, launch overhead
- **ncu** — deep dive: SM utilization, occupancy, HBM bytes, compute vs memory bound
- Workflow: nsys to find bottlenecks → ncu to understand why

---

## FA2 vs Standard Attention

Standard attention launches a separate kernel per step, writing intermediates to HBM between each:
```
Standard: HBM → QK matmul → HBM → softmax → HBM → AV matmul → HBM
FA2:      HBM → [QK matmul → softmax → AV matmul] → HBM
                 └──── all in registers/SRAM ─────┘
```
**Same math, far less HBM traffic.**

Profiler evidence:
- 45 unique kernels → 5 with FA2
- Softmax kernels completely disappear (unambiguous attention-only ops)
- `cudaLaunchKernel` overhead: 15% → 2%

FA2 kernel stats (forward):
- 255 registers/thread — holds all intermediates on-chip
- 49KB dynamic shared memory — holds Q/K/V tiles
- 16.67% occupancy — intentional, register-limited (255 × 128 threads = 32,640 regs/block, max 2 blocks/SM)

Tradeoff: high register pressure → low occupancy, but eliminating HBM latency far outweighs it.

---

## Batch Size: BS1 vs BS32

**SM Utilization** — goes UP with batch size. Larger batch → larger matrices → more thread blocks → more SMs active. At BS1 some SMs sit idle; at BS32 all 82 SMs are saturated.

**Occupancy** — set by register/shared memory usage at compile time, does not change with batch size for register-limited kernels. Can increase for work-limited kernels at BS1.

**Compute vs Memory bound:**
- Compute bound: tensor cores always busy, arithmetic is the bottleneck (BS32 GEMMs)
- Memory bound: warps stall waiting on HBM loads (~700 cycle latency), bandwidth is the bottleneck (BS1 small GEMMs)

ncu Speed of Light: high `Compute (SM) %` + low `Memory %` = compute bound, reverse = memory bound.

---

## Memory Hierarchy
```
Registers (private/thread, ~0 latency)
→ Shared Memory/SRAM (per block, on-chip)
→ L2 Cache (on-chip, cross-SM)
→ HBM (off-chip, ~700 cycle latency, 24GB)
```

## SM Architecture
```
SM → Warp Schedulers → Warps (32 threads) → CUDA/Tensor Cores (inside SM)
```
Warp scheduler hides HBM latency by switching to another ready warp while one waits on a load. Requires sufficient occupancy to have other warps available.

---

## ncu Metrics: BS1 vs BS32
| Metric | BS1 | BS32 |
|--------|-----|------|
| `Compute (SM) [%]` | Low | High |
| `Memory [%]` | High | Lower |
| `Achieved Occupancy` | Low | Higher |
| `Waves Per SM` | < 1 | > 1 |
