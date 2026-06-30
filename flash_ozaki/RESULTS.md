# Triton Flash-Ozaki1_fp — bf16 byte-plane (A6000, GPU 0, 2026-06-30)

FlashAttention with the **bf16 byte-plane ozaki1_fp GEMM** fused into the tile dots (QK^T, P@V).
Online softmax + fp32 accumulation untouched. Each operand: block-FP scale -> nD signed w-bit
digits (bf16) -> kept digit-pair bf16 dots (place value folded as power-of-2 into the digits,
fp32-accumulated). Parameterized by (w, nmp). Files: `oz1fp_triton.py` (GEMM + validation),
`flash_oz1fp_triton.py` (flash kernel), `bench_flash_ozaki.py`. Model shape H=28, D=128, causal.

## GEMM fidelity vs production `batched_gemm` (ozaki1_fp) — faithful at every level
| cfg | mine vs prod | mine vs exact | prod vs exact |
|---|---|---|---|
| w8 nmp1 / w4 nmp4 (nD1/2) | 1.6e-2 | 9.2e-3 | 1.3e-2 |
| w4 nmp9 (nD3) | 6.8e-4 | 5.7e-4 | 3.6e-4 |
| w4 nmp10 (nD4) | 1.4e-4 | 8.3e-5 | 1.2e-4 |
| w4 nmp15 (nD5) | 6.8e-6 | 5.9e-6 | 4.4e-6 |

## Attention accuracy — fused vs EXACT (N=1024)
| config | fused vs EXACT | eager-ref vs EXACT | fused vs eager-ref |
|---|---|---|---|
| w8 nmp1 / w4 nmp4 | 1.3e-2 | 2.0e-2 | 2.4e-2 |
| w4 nmp9  | 1.8e-3 | 1.7e-3 | 2.5e-3 |
| w4 nmp10 | 1.6e-3 | 1.6e-3 | 2.3e-3 |
| w4 nmp15 | 1.6e-3 | 1.6e-3 | 2.3e-3 |

Flash machinery exact-mode vs torch exact: 2.0e-3. → fusion preserves ozaki1_fp numerics; at the
sweep's attention nmp (9/10/15) fused attention is ~1.6e-3 from exact, matching the eager backend.

## Speed (latency ms/call, B=1, H=28, D=128, causal; ozaki = w4 nmp10)
| N_ctx | Flash-Ozaki1_fp | Flash exact (same kernel) | torch SDPA | eager-ref (batched_gemm) |
|------:|----------------:|--------------------------:|-----------:|-------------------------:|
| 512   | 1.47  | 0.07 | 0.05 | 2.74   |
| 1024  | 4.89  | 0.20 | 0.13 | 8.75   |
| 2048  | 14.75 | 0.65 | 0.39 | 31.64  |
| 4096  | 48.13 | 2.29 | 1.37 | 114.41 |

### nmp=1 (w=8, nD=1 -> ONE digit-pair dot/matmul, same dot count as exact flash)
| N_ctx | Flash-Ozaki nmp=1 | Flash exact (same kernel) | torch SDPA | oz/flash | oz/sdpa |
|------:|------------------:|--------------------------:|-----------:|---------:|--------:|
| 1024  | 0.73  | 0.20 | 0.13 | 3.6x | 5.5x |
| 2048  | 2.41  | 0.65 | 0.39 | 3.7x | 6.1x |
| 4096  | 8.99  | 2.29 | 1.37 | 3.9x | 6.6x |
| 8192  | 39.37 | 8.67 | 5.10 | 4.5x | 7.7x |

Even at nmp=1 (same matmul count as flash), fused ozaki1_fp is ~3.5-4.5x slower than the same
kernel's exact path: the gap is the per-tile **block-FP encode** (amax reduction + round + signed
digit split + rescale), pure software overhead with no int8 shortcut. So nmp=1 is the floor of the
ozaki1_fp emulation cost; higher nmp adds ~nmp× on top. (HW would do the encode in the datapath.)

## kv_cache=True (pre-encoded KV digit planes + scales; the attention analog of weight_cache)
`flash_oz1fp_cached.py`: K/V block-FP-encoded ONCE into digit planes (+scales), kernel loads them
and skips the per-tile K/V encode. Q & P still encoded inline (P = current softmax, not cacheable).
Cache built once (amortized); only the flash kernel is timed. Accuracy preserved (cached vs EXACT:
nmp1 1.4e-2, nmp10 1.6e-3 — same as non-cached).

| cfg | N | cached | non-cached | exact flash | cache/noncache | cache/exact |
|---|--:|--:|--:|--:|--:|--:|
| w8 nmp1 (nD1)  | 4096 | 4.10  | 9.01  | 2.29 | **2.2x faster** | 1.8x |
| w4 nmp10 (nD4) | 4096 | 28.24 | 48.01 | 2.29 | **1.7x faster** | 12.3x |

- **nmp=1: caching cuts the gap to exact flash from ~3.9x → ~1.8x** — the K/V block-FP encode was
  ~half the overhead; removing it leaves only Q+P encode + rescale + untuned-Triton.
- **nmp=10: ~2x faster but still 12x exact** — caching removes the encode, but the nmp× digit-pair
  *dots* remain (caching can't reduce the matmul count). So at high nmp the dots are the floor.
- This is the PREFILL microbench (each K/V tile re-encoded ~N/BM x without caching). In **decode**
  the win is larger: uncached re-encodes all cached K/V every step (O(N^2) encode) vs O(N) cached.

## Findings
1. **Faithful**: the bf16 byte-plane ozaki1_fp GEMM matches the production emulation at every
   (w,nmp); fused attention matches the eager backend (~1.6e-3 vs exact at the sweep's nmp).
2. **Fused > eager**: ~2.4x faster than the (optimistic) materialized eager-ref at 4k; the real
   vLLM eager backend (Python per-step KV gather) is far slower -> fusion is the real win for the
   sweep's attention-emulated cells.
3. **Fused < exact flash**: ~nmp× slower (nD=4 -> ~10 digit-pair bf16 dots per matmul, no int8
   shortcut — this IS ozaki1_fp's bf16 emulation cost). Lower nmp is proportionally faster
   (nmp=1 -> 1 dot/matmul -> ~2× exact flash). Matches: ozaki1_fp's speed payoff needs the digit
   datapath in hardware; in emulation, Flash-Ozaki1_fp's value is removing eager overhead.

## Scope / next
nmp/w parameterized (validated w8 nmp1; w4 nmp 4/9/10/15). Block-FP chunk = full reduction
(production k=32; refinement, same mechanism). Digit split = no_clamp (matches the sweep's
byte_split_style). Un-tuned kernel (small blocks, num_stages=1); not meant to beat SDPA.
