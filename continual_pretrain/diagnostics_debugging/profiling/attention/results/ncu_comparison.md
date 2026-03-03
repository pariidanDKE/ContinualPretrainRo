# NCU Attention Comparison: Eager vs FA2
**Config:** BS=1, seq_len=2048 (Llama-3.2-1B, fullFT, BF16, gradient checkpointing)
**NCU invocations profiled:**
  - Eager: 4 QKᵀ_GEMM, 3 softmax, 3 PV_GEMM
  - FA2:   10 flash_fwd_kernel

> **Note:** NCU captured kernels at an effective seq ≈ 168 tokens (warmup step).
> Absolute byte values are not full-seq; **ratios are valid**.

---

## Per-Layer Comparison (1 attention layer)

Eager layer = avg(QKᵀ_GEMM) + avg(softmax) + avg(PV_GEMM)
FA2 layer   = avg(flash_fwd_kernel)

| Metric | Eager | FA2 | FA2/Eager | Savings |
|---|---:|---:|---:|---:|
| HBM read (MB) | 7.398 | 1.104 | 0.149x | +85.1% |
| HBM write (MB) (eager excludes PV artifact) | 4.95 | 2.287 | 0.462x | +53.8% |
| L2 total (MB) | 23.055 | 7.217 | 0.313x | +68.7% |
| GPU time (µs) | 38.614 | 24.992 | 0.647x | +35.3% |
| TensorCore util (%) | 9.402 | 22.643 | 2.408x | -140.8% |
| SM throughput (%) | 38.008 | 13.885 | 0.365x | +63.5% |
| Warp occupancy (%) | 44.737 | 8.193 | 0.183x | +81.7% |
| Regs / thread | 60.667 | 255.0 | 4.203x | -320.3% |

---

## Eager Kernel-Type Averages (per invocation)

| Metric | QKᵀ_GEMM | softmax | PV_GEMM |
|---|---:|---:|---:|
| HBM read (MB) | 1.385 | 3.543 | 2.470 |
| HBM write (MB) | 1.700 | 3.250 | 840.063 |
| L2 total (MB) | 10.065 | 8.010 | 4.980 |
| GPU time (µs) | 13.328 | 11.603 | 13.683 |
| TensorCore util (%) | 12.410 | 0.000 | 15.797 |
| SM throughput (%) | 55.292 | 44.003 | 14.727 |
| Warp occupancy (%) | 54.692 | 69.517 | 10.003 |
| L2 hit rate (%) | n/a | n/a | n/a |
| Shmem LD wavefronts | n/a | 0.000 | n/a |
| Shmem ST wavefronts | n/a | 0.000 | n/a |
| Regs / thread | 64.000 | 28.000 | 90.000 |

---

## Notes

### PV_GEMM dram__bytes_write.sum artifact
The PV_GEMM write counter reports ~840 MB per invocation but the actual output
is O(few MB) (grid=(3,1,32) = 96 blocks of tiny tiles). The inflated value
reflects dirty L2 evictions from QKᵀ and softmax — large QKᵀ output matrices
cached in L2 are evicted to DRAM when PV_GEMM accesses new addresses.
`lts__t_bytes.sum` (L2 total traffic) is unaffected and is the reliable metric.

### Shared memory (FA2 vs eager)
Flash attention uses **dynamic** shared memory for SRAM tiling.
`launch__shared_mem_per_block_static` is 0 for all kernels — the relevant
metric is `l1tex__data_pipe_lsu_wavefronts_mem_shared*` which captures
actual shared memory traffic at runtime.

### Invocation counts
NCU profiled 4 QKᵀ but only 3 softmax + 3 PV — the 4th QKᵀ has no
corresponding softmax/PV in the capture window. Per-layer averages use
all available invocations of each type independently.
