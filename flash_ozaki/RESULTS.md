# Triton Flash-Ozaki1_fp — bf16 byte-plane (A6000, GPU 0, 2026-06-30)

FlashAttention with the **bf16 byte-plane ozaki1_fp GEMM** fused into the tile dots (QK^T, P@V).
Online softmax + fp32 accumulation untouched. Each operand: block-FP scale -> nD signed w-bit
digits (bf16) -> kept digit-pair bf16 dots (place value folded as power-of-2 into the digits,
fp32-accumulated). Parameterized by (w, nmp). Files: `flash_oz1fp_codegen.py` (fused flash + cached-KV
kernels, single source), `verification/standalone_oz_gemm.py` (isolated GEMM), `bench_flash_ozaki.py`,
`bench_cached.py`. Model shape H=28, D=128, causal.

> **Update:** the kernels below converged onto the production-faithful codegen path (optimal pack plan
> + single-peel place-folded planes + frexp `int_bits` block-FP scale). The earlier hand-rolled
> `_po2/maxmag` kernels (`flash_oz1fp_triton.py`, `flash_oz1fp_cached.py`, `oz1fp_triton.py`'s GEMM)
> were removed. Faithful-path numbers (bit-identical operands vs production) are in
> `verification/README.md`; the tables here are the pre-convergence hand-rolled measurements.

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

## kv_cache=True (pre-encoded KV; the attention analog of weight_cache)
`flash_oz1fp_cg_cached` + `encode_kv` (in `flash_oz1fp_codegen.py`): K/V block-FP-encoded ONCE into
nD place-folded digit planes; the kernel loads them and skips the per-tile K/V encode. Q & P encoded
inline (P = current softmax, not cacheable). EXACT vs the non-cached codegen kernel up to fp
accumulation order. (The earlier experimental super-digit / plan-cache formats were dropped -- the
`bench_cached.py` "super-plane packing" it claimed to measure was never actually exercised.)

**EXACT** vs non-cached ozaki1_fp. Format chosen per config so the truncation is exact: FULL
(nmp=nD^2) packs to nS byte super-planes (no boundary → exact + smaller); TRIANGULAR (nmp=T_nD)
stores nD digit planes + exact w-bit truncation (boundary sub-pairs CANNOT be reconstructed from
packed super-digits → individual digits required → no super-digit memory saving). Memory + accuracy
+ speed (N=2048, H=28, D=128; ms = pack / unpack / non-cached):

| cfg | storage | cache MB pk/unpk | mem | cached vs EXACT | cached vs non-cache | ms |
|---|---|---|--:|---|---|---|
| w4 nmp9 (full)  | super(nS) | 59 / 88  | **0.67x** | 1.67e-3 | 1.1e-3 | 5.4/6.8/14.0 |
| w4 nmp16 (full) | super(nS) | 59 / 118 | **0.50x** | 1.61e-3 | 2.6e-4 | 8.7/10.3/18.9 |
| w4 nmp10 (tri)  | digit(nD) | 118 / 118 | 1.00x   | 1.61e-3 | 3.8e-4 | 7.7/7.7/21.2 |
| w4 nmp15 (tri)  | digit(nD) | 147 / 147 | 1.00x   | 1.61e-3 | 1.0e-4 | 11.0/11.0/28.2 |

- **FULL configs pack -> 0.5-0.67x cache, exact.** TRIANGULAR stay at nD planes (1.0x) but EXACT.
  All ~2-3x faster than non-cached (from skipping the K/V encode).
- (Fundamental: exact triangular needs the individual digits; memory-saving packing only for full.)

### Triangular = exact rect-cover (whole super-pair MINUS dropped boundary terms)
The cached triangular path computes the EXACT w-bit truncation via the signed rect-cover (same as
production): per super-pair, `super_q x super_k` minus the dropped boundary sub-pairs (e.g. nmp10
SP(0,1) = super0_q x super1_k - q0*k2, excluding (0,2); SP(1,0) - q2*k0 excluding (2,0)) -> 5 GEMMs
vs unpacked 10. Confirmed exact (cached-vs-non-cached residual = exact-truncation level, only the V
block-FP chunk + fp order; (0,2),(2,0) properly excluded). Storage stays nD (the dropped-term
subtraction needs the individual boundary digits k0/k2 -> no sub-nD storage; same as production,
verified: production caches 4 super-planes = nD for nmp10). The 10->5 GEMM drop is ~free but does
NOT speed it up at attention tile sizes -- digit extraction/memory dominates, not dots; the ~2.7x
over non-cached is all from caching the encode.
- Earlier (nmp=1, nD=1) cached numbers: cache cuts the gap to exact flash ~3.9x -> ~1.8x (the K/V
  encode was ~half the nmp=1 overhead). At high nmp the nmp× digit-pair *dots* are the floor
  (caching/packing reduce encode + plane count, not the matmul count).
- PREFILL microbench (each K/V tile re-encoded ~N/BM x without caching). In **decode** the win is
  larger: uncached re-encodes all cached K/V every step (O(N^2) encode) vs O(N) cached.

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

## Eager-option support (added 2026-06-30) — all validated vs production batched_gemm (w=4)
- **nmp, gemm_bits (w)**: full. (w=8 nmp>1 uses production's native byte-split path, not the generic
  w-bit path here — out of scope; w=8 nmp=1 coincides. The sweep is w=4.)
- **k (chunk_size)**: per-chunk block-FP along the reduction. QK^T chunks head_dim into KQ=min(k,D);
  P@V chunk = BLOCK_N (set BLOCK_N=k). w4 nmp10 vs prod: k=32 1.2e-4, k=64 1.4e-4, k=128 1.4e-4.
- **byte_split_style**: `all_signed_no_clamp` (top digit absorbs carry) + `all_signed_clamp_pos`
  (all digits clamped, conservative maxmag). w4 nmp10/9/15 clamp_pos vs prod: 1.4e-4 / 6.3e-4 / 6.4e-6.
- **g=8//w packing**: group g adjacent digits into a byte; interior super-pairs -> 1 GEMM (boundary
  unpacked, **bit-identical**: pack=True == pack=False output). Speed is config-dependent because at
  these tile sizes **digit-extraction dominates, not the dots**: nmp4 (nD2, full) **5.7x faster**
  (4 dots->1); nmp9/16 (nD3/4, full) ~1.0-1.06x; nmp10/15 (triangular) ~0.85x (boundary unpack +
  super-digit recompute > dot saving). So packing helps only when the dot count dominates (nD=2);
  the extraction bottleneck is what kv_cache=True addresses.

## Decode (q_len=1) + GQA head-folding (added 2026-06-30)
Both kernels now take `q:[B,Hq,T,D]` with `k,v:[B,Hkv,N,D]` (prefill T==N is the old path; decode
T=1, chunked T<N). A query token at row-block position is at absolute kv pos `N-T+i`, so q_len=1
attends to all kv. **GQA**: when `Hq=Hkv*G`, the G query heads that share a kv head are folded into
the query-row (M) dim (`row = t*G+g`, token = `row//G`), so **one loaded/encoded K/V tile feeds all
G heads** — decode QK^T becomes a `[G,D]` GEMM, not a `[1,D]` GEMV. MHA is the `G=1` special case
(byte-identical to the old path). Auto-detected from shapes; BLOCK_M auto-shrinks to `max(16, pow2(T*G))`.

Accuracy (fused, N=2048, Qwen-7B shape Hq28/Hkv4 → G=7) — **decode == prefill ozaki1_fp, and the GQA
fold is bit-transparent (MHA-replicate == GQA-fold)**:
| cfg | MHA decode (G=1) | GQA decode (fold) | cached GQA decode |
|---|---|---|---|
| w8 nmp1  | 1.37e-2 | 1.37e-2 | 1.37e-2 |
| w4 nmp9  | 1.72e-3 | 1.72e-3 | 1.73e-3 |
| w4 nmp10 | 1.63e-3 | 1.63e-3 | 1.69e-3 |
| w4 nmp15 | 1.69e-3 | 1.69e-3 | 1.71e-3 |

Batched-decode speedup of GQA-fold vs replicating K/V to all Hq heads (w4 nmp10, N=2048, A6000):
| B | programs (B·Hkv) | GQA-fold ms | replicate ms | speedup |
|--:|--:|--:|--:|--:|
| 1  | 4   | 1.46 | 1.49 | 1.02x |
| 8  | 32  | 1.50 | 4.33 | 2.89x |
| 32 | 128 | 2.34 | 15.2 | 6.48x |
| 64 | 256 | 4.40 | 29.1 | 6.63x |

- The fold's win is the **~G× reduction in K/V encode+load** (each tile reused by G heads). It only
  materializes once `B·Hkv` is large enough to fill the GPU: at **B=1 it's occupancy-bound** (only
  `B·Hkv=4` programs → most SMs idle → ~1.0x), but it climbs to **~6.6x (→ the G=7 ceiling) at B=64**.
- Single-request decode (B=1) stays SM-starved; the standard fix is split-KV / FlashDecoding (split N
  across programs + combine partial softmax) — not done here. The fold is the prerequisite layout for it.

## Scope / next
Un-tuned kernel (small blocks, num_stages=1; not meant to beat SDPA). QK k-chunk reloads Q per
(kv-tile x d-chunk). Optimal rect-cover packing (w4 nmp10 -> 5 GEMMs vs super-pair 7) not done
(needs the rectangle list; Triton-list-hard). cached variant is nmp-only (no pack/k/clamp yet).
Decode supports q_len=1 + GQA head-folding (above); **split-KV / FlashDecoding** for B=1 occupancy
is the next decode step (the fold is its prerequisite layout). Page-table / non-contiguous KV-cache
gather (paged-attention style) not done — current path assumes a contiguous `[B,Hkv,N,D]` cache.
