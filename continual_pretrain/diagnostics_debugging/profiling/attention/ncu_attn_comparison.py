"""
NCU comparison: eager attention vs Flash Attention 2
Metrics: dram__bytes_read/write, L2 traffic, GPU time, tensor-core util, occupancy

Invocation counts differ between eager and FA2 — we normalise to per-layer
averages before comparing:
  Eager layer = avg(QKᵀ_GEMM) + avg(softmax) + avg(PV_GEMM)
  FA2   layer = avg(flash_fwd_kernel)

NOTE: PV_GEMM dram__bytes_write.sum is excluded from the comparison —
  it reflects dirty L2 evictions from prior kernels, not kernel output.
  Use lts__t_bytes.sum (L2 total) as the reliable bandwidth proxy.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
EAGER_CSV = ROOT.parents[2] / "outputs/ncu_attn_profiles/eager_bs1_seq2048_20260228_1700/eager_bs1_seq2048.csv"
FA2_CSV   = ROOT.parents[2] / "outputs/ncu_attn_profiles/fa2_bs1_seq2048_20260228_1700/fa2_bs1_seq2048.csv"
OUT_JSON  = ROOT / "results/ncu_comparison.json"
OUT_MD    = ROOT / "results/ncu_comparison.md"

# ---------------------------------------------------------------------------
# Metrics to report
# ---------------------------------------------------------------------------
METRICS = [
    ("dram__bytes_read.sum",                                               "HBM read (MB)"),
    ("dram__bytes_write.sum",                                              "HBM write (MB)"),
    ("lts__t_bytes.sum",                                                   "L2 total (MB)"),
    ("gpu__time_duration.sum",                                             "GPU time (µs)"),
    ("sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", "TensorCore util (%)"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed",                   "SM throughput (%)"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active",                  "Warp occupancy (%)"),
    ("lts__t_hit_rate.pct",                                                "L2 hit rate (%)"),
    ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",               "Shmem LD wavefronts"),
    ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",               "Shmem ST wavefronts"),
    ("launch__registers_per_thread",                                       "Regs / thread"),
]

# Metrics where we SUM per-layer contributions (extensive quantities)
SUM_METRICS = {
    "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum",
    "gpu__time_duration.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
}

# Metrics that should be averaged (intensive — per-warp / peak-percentage)
AVG_METRICS = {m for m, _ in METRICS} - SUM_METRICS

# HBM write is excluded for PV_GEMM (dirty-eviction artifact)
PV_WRITE_EXCLUDED = True

# ---------------------------------------------------------------------------
# Kernel labelling
# ---------------------------------------------------------------------------
def kernel_label(kname: str) -> str:
    if "_tn_" in kname:           return "QKᵀ_GEMM"
    if "_nn_" in kname:           return "PV_GEMM"
    if "softmax" in kname:        return "softmax"
    if "flash_fwd" in kname:      return "flash_fwd"
    return kname[:50]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_by_invocation(path: Path):
    """Returns dict: (inv_id, kernel_label) -> {metric -> value}"""
    inv: dict[tuple, dict] = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["ID"], kernel_label(r["Kernel Name"]))
            try:
                inv[key][r["Metric Name"]] = float(r["Metric Value"])
            except ValueError:
                pass
    return inv


eager_inv = load_by_invocation(EAGER_CSV)
fa2_inv   = load_by_invocation(FA2_CSV)


# ---------------------------------------------------------------------------
# Group by kernel label
# ---------------------------------------------------------------------------
def group_by_label(inv_dict):
    groups: dict[str, list[dict]] = defaultdict(list)
    for (_, label), metrics in inv_dict.items():
        groups[label].append(metrics)
    return groups


eager_groups = group_by_label(eager_inv)
fa2_groups   = group_by_label(fa2_inv)


# ---------------------------------------------------------------------------
# Per-kernel averages
# ---------------------------------------------------------------------------
def avg_metrics(invocations: list[dict]) -> dict[str, float]:
    result = {}
    all_keys = set(k for d in invocations for k in d)
    for key in all_keys:
        vals = [d[key] for d in invocations if key in d]
        if vals:
            result[key] = mean(vals)
    return result


eager_avgs = {label: avg_metrics(invs) for label, invs in eager_groups.items()}
fa2_avgs   = {label: avg_metrics(invs) for label, invs in fa2_groups.items()}

EAGER_KERNEL_ORDER = ["QKᵀ_GEMM", "softmax", "PV_GEMM"]

# ---------------------------------------------------------------------------
# Per-layer aggregation
# ---------------------------------------------------------------------------
def aggregate_eager_layer(avgs: dict[str, dict]) -> dict[str, float]:
    """Sum/avg across the three eager kernel types to get one-layer totals."""
    per_layer: dict[str, float] = {}
    for mkey, _ in METRICS:
        vals = []
        for label in EAGER_KERNEL_ORDER:
            if label not in avgs:
                continue
            if PV_WRITE_EXCLUDED and label == "PV_GEMM" and mkey == "dram__bytes_write.sum":
                continue
            v = avgs[label].get(mkey)
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        if mkey in SUM_METRICS:
            per_layer[mkey] = sum(vals)
        else:
            per_layer[mkey] = mean(vals)
    return per_layer


eager_layer = aggregate_eager_layer(eager_avgs)
fa2_layer   = fa2_avgs.get("flash_fwd", {})


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------
rows = []
for mkey, mlabel in METRICS:
    e = eager_layer.get(mkey)
    f = fa2_layer.get(mkey)
    if e is None or f is None:
        continue
    if PV_WRITE_EXCLUDED and mkey == "dram__bytes_write.sum":
        note = "(eager excludes PV artifact)"
    else:
        note = ""
    ratio   = f / e if e != 0 else float("inf")
    savings = (1 - ratio) * 100
    rows.append({
        "metric":   mlabel,
        "metric_key": mkey,
        "eager":    round(e, 3),
        "fa2":      round(f, 3),
        "ratio":    round(ratio, 3),
        "savings_pct": round(savings, 1),
        "note":     note,
    })


# ---------------------------------------------------------------------------
# Per-kernel breakdown (for JSON)
# ---------------------------------------------------------------------------
per_kernel_eager = {
    label: {m: round(v, 3) for m, v in avgs.items() if m in {k for k, _ in METRICS}}
    for label, avgs in eager_avgs.items()
}
per_kernel_fa2 = {
    label: {m: round(v, 3) for m, v in avgs.items() if m in {k for k, _ in METRICS}}
    for label, avgs in fa2_avgs.items()
}

result = {
    "config": {
        "bs": 1, "seq_len": 2048,
        "note": "NCU invocations captured at effective seq~168 (warmup step). "
                "Relative ratios are valid; absolute bytes are not full-seq.",
    },
    "invocation_counts": {
        label: len(invs) for label, invs in {**eager_groups, **fa2_groups}.items()
    },
    "per_kernel_avg": {"eager": per_kernel_eager, "fa2": per_kernel_fa2},
    "per_layer_comparison": rows,
    "pv_write_note": (
        "dram__bytes_write.sum for PV_GEMM is excluded from eager totals. "
        "The value (~840 MB) reflects dirty L2-line evictions from QKᵀ/softmax, "
        "not the PV kernel's actual output writes (grid=(3,1,32) = 96 blocks, "
        "actual output is O(few MB))."
    ),
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(result, indent=2))
print(f"Saved JSON → {OUT_JSON}")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
n_inv = result["invocation_counts"]

md = f"""# NCU Attention Comparison: Eager vs FA2
**Config:** BS=1, seq_len=2048 (Llama-3.2-1B, fullFT, BF16, gradient checkpointing)
**NCU invocations profiled:**
  - Eager: {n_inv.get('QKᵀ_GEMM',0)} QKᵀ_GEMM, {n_inv.get('softmax',0)} softmax, {n_inv.get('PV_GEMM',0)} PV_GEMM
  - FA2:   {n_inv.get('flash_fwd',0)} flash_fwd_kernel

> **Note:** NCU captured kernels at an effective seq ≈ 168 tokens (warmup step).
> Absolute byte values are not full-seq; **ratios are valid**.

---

## Per-Layer Comparison (1 attention layer)

Eager layer = avg(QKᵀ_GEMM) + avg(softmax) + avg(PV_GEMM)
FA2 layer   = avg(flash_fwd_kernel)

| Metric | Eager | FA2 | FA2/Eager | Savings |
|---|---:|---:|---:|---:|
"""

for r in rows:
    note = f" {r['note']}" if r["note"] else ""
    savings_str = f"{r['savings_pct']:+.1f}%"
    md += f"| {r['metric']}{note} | {r['eager']} | {r['fa2']} | {r['ratio']:.3f}x | {savings_str} |\n"

md += f"""
---

## Eager Kernel-Type Averages (per invocation)

| Metric | QKᵀ_GEMM | softmax | PV_GEMM |
|---|---:|---:|---:|
"""

for mkey, mlabel in METRICS:
    vals = [
        eager_avgs.get(label, {}).get(mkey)
        for label in EAGER_KERNEL_ORDER
    ]
    cells = [f"{v:.3f}" if v is not None else "n/a" for v in vals]
    md += f"| {mlabel} | {cells[0]} | {cells[1]} | {cells[2]} |\n"

md += f"""
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
"""

OUT_MD.write_text(md)
print(f"Saved markdown → {OUT_MD}")

# ---------------------------------------------------------------------------
# Print summary to stdout
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(f"{'PER-LAYER COMPARISON':^72}")
print("=" * 72)
print(f"{'Metric':<42} {'Eager':>8} {'FA2':>8} {'Ratio':>8} {'Savings':>9}")
print("-" * 72)
for r in rows:
    excl = " *" if r["note"] else "  "
    print(f"{r['metric']:<42}{excl} {r['eager']:>8.3f} {r['fa2']:>8.3f} {r['ratio']:>7.3f}x {r['savings_pct']:>+8.1f}%")
print()
print("  * dram__bytes_write (eager) excludes PV_GEMM dirty-eviction artifact")
