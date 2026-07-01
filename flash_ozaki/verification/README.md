# flash_ozaki verification — vs production `ozaki1_batched_gemm_fp` and vs SDPA

> 최신 재현 수치(2026-07-01, A6000 gpu:0, 코드젠 수렴 이후)는 **[RESULTS.md](RESULTS.md)** (한국어) 참고.
> 이 README는 방법론/파일 설명 + 초기 측정치.

Validation that the **flash-attention ozaki1_fp kernels** (codegen path: optimal pack plan +
single-peel place-folded planes) faithfully reproduce the production emulation
`emulation/llm/ozaki_matmul.py::ozaki1_batched_gemm_fp`, plus a full-attention speed/accuracy
comparison against torch SDPA. All numbers: **RTX A6000, gpu:2, torch 2.5.1 / triton 3.1.0,
byte_split_style=`all_signed_no_clamp`, K-chunk=32, bf16 in / fp32 accum.**

> **ozaki1 is an int8-GEMM hardware scheme; this code is the fast bf16 *emulation* of that int8
> datapath.** The reference is therefore the production emulation (which matches the fp64 reference
> to the fp32 floor), NOT the un-quantized product. "More accurate than production" would mean
> *un-faithful* (modeling a more precise machine than the int8 HW). Faithful = match production.

## Files
| file | what |
|---|---|
| `standalone_oz_gemm.py` | the QK/PV ozaki1_fp matmul (A@B) extracted from the attention kernel (codegen plan+peel, production-faithful block-FP scale). `oz1fp_gemm_cg(...)`, `encode_B(...)` (weight cache). |
| `bench_gemm_vs_production.py` | **GEMM part**: standalone QK (Q@K^T) & PV (P@V) vs production, cached/non-cached × nmp × w — relerr, bit-exact, latency. |
| `verify_faithfulness.py` | encode faithfulness: Â bit-identical, no_clamp top-digit range, digit-split identical, kernel == own fp64 product. |
| `bench_attention_vs_sdpa.py` | **full attention**: flash-ozaki1_fp vs torch SDPA vs the same kernel's exact bf16 flash path — accuracy + latency, prefill + decode(GQA). |
| `analyze_triton_vs_cublas.py` | why the Triton GEMM beats production at attention shapes but loses to cuBLAS on large compute-bound GEMMs (tiling sweep). |

Run any script from the repo root, e.g. `CUDA_VISIBLE_DEVICES=2 python flash_ozaki/verification/bench_gemm_vs_production.py`.

---

## Part A — GEMM part: standalone QK/PV vs production `ozaki1_batched_gemm_fp`

### A.1 Faithfulness (`verify_faithfulness.py`) — exact emulation of the int8 datapath
After matching production's block-FP scale convention (`scale = 2^(frexp_exp(amax) − int_bits)`,
`int_bits = w·nD−1`, clamp to `[−2^int_bits, 2^int_bits−1]`):

- **Dequantized operand Â bit-identical to production** (`torch.equal`, relerr `0.00e+00`) for w4 nmp9/16, w8 nmp1, w2 nmp16.
- **no_clamp top digit ∈ `[−2^(w-1), 2^(w-1)]`** (the int8 range + 1-bit MSB overflow flag): w4 → `[−8, 8]`, w8 → `[−128, 127]`, w2 → `[−2, 2]`. ✔ (Earlier the flash kernel used `maxmag = 2^(w·nD)−1`, a ~1-bit-too-fine scale → top digit reached `[−2^w, 2^w]`, i.e. a *wider* range than the int8 HW can produce — fixed.)
- **Signed digit-split identical** to `_oz1_wbit_digit_split` for all (w, nD, clamp).
- **Kernel result == its own quantized product (fp64)** to `5.7e-9 / 5.7e-8 / 0.0` (w4 nmp9 / nmp16 / w8 nmp1) → the only deviation from the ideal of *its operands* is fp32 accumulation order.

→ Remaining flash-vs-production difference is **purely fp32 accumulation order** (Triton tile-dot vs cuBLAS full-K + combine); bit-exact where the order coincides.

### A.2 Accuracy (`bench_gemm_vs_production.py`, QK & PV, M=N=1024, K=128/1024)
`mine/prod` = standalone vs production; `mine|prod/exact` = vs fp32 `A@B`.

| op | w/nmp | mine/prod | bit-eq | mine/exact | prod/exact |
|---|---|---|---|---|---|
| QK | w8 nmp1 | **0.0** | yes | 1.28e-2 | 1.28e-2 |
| QK | w4 nmp9 | 5.7e-9 | no | 3.61e-4 | 3.61e-4 |
| QK | w4 nmp10 | **0.0** | yes | 1.21e-4 | 1.21e-4 |
| QK | w4 nmp15 | 7.7e-9 | no | 4.38e-6 | 4.38e-6 |
| PV | w4 nmp9 | 9.8e-8 | no | 2.63e-4 | 2.63e-4 |
| PV | w4 nmp16 | 1.1e-7 | no | 4.20e-6 | 4.20e-6 |

**`mine/exact == prod/exact` exactly** (faithful), and `mine/prod` ≈ fp32-accumulation floor (`1e-9–1e-7`, `0.0` where order coincides). Same for cached (`encode_B`) — cached == non-cached on our side.

### A.3 Speed (latency ms; `prod/mine` = speedup of standalone over production)
| shape | op | w/nmp | mine ms | prod ms | prod/mine |
|---|---|---|---|---|---|
| PREFILL M=N=1024 | QK | w8 nmp1 | 0.26 | 0.23 | **0.87×** |
| | QK | w4 nmp9 (cache) | 0.48 | 2.04 | **4.3×** |
| | PV | w4 nmp16 (cache) | 0.52 | 4.68 | **9.0×** |
| DECODE Mq=16 N=2048 | QK | w4 nmp9 (cache) | 0.078 | 0.19 | 2.4× |
| | PV | w4 nmp10 (cache) | 0.21 | 0.21 | 1.0× |

The standalone Triton matmul is **3–9× faster than production for nmp≥9**, ~equal/slightly slower for
nmp1. **Why** (`analyze_triton_vs_cublas.py`): the attention QK/PV are **low-arithmetic-intensity**
GEMMs (K=128, or N=128). Production issues `#GEMM` *separate* cuBLAS calls + a combine + Python
orchestration → re-reads the operands `#GEMM×` (memory-bound) + launch overhead. The fused single
Triton kernel reads operands **once** and accumulates all `#GEMM` digit-pair dots in-kernel.
**This is NOT beating cuBLAS** — on a LARGE compute-bound GEMM (Z=1, M=2048, K=N=4096) production ≈
`#GEMM × bf16` and the Triton kernel is *slower* (`prod/mine ≈ 0.74×` at w4): a plain bf16 Triton GEMM
at BM=BN=64 is already 1.6× cuBLAS, and only matches it at 128×128 — but the ozaki digit-plane SRAM
pressure forces the smaller tiles. So: **fused-Triton wins at attention's low-intensity shapes; cuBLAS
(production) wins at large linear-layer GEMMs.**

---

## Part B — Full attention: flash-ozaki1_fp vs SDPA vs exact flash (`bench_attention_vs_sdpa.py`)

Ozaki config w4 nmp10. flash-exact = the same kernel's `ozaki=False` bf16 path.

### B.1 Accuracy (vs fp32 exact attention)
| case | flash-ozaki | flash-exact (bf16) | torch SDPA |
|---|---|---|---|
| PREFILL MHA N=1024 | **1.61e-3** | 1.99e-3 | 1.99e-3 |
| DECODE GQA B=32 N=2048 | **1.69e-3** | 2.25e-3 | 2.25e-3 |

flash-ozaki (nmp10) is **as accurate as — slightly better than — plain bf16 attention** (SDPA / flash-exact),
because the block-FP + multi-digit polynomial has less error than a single bf16 product.

### B.2 Latency (ms/call)
| case | flash-ozaki | flash-exact | torch SDPA | oz/sdpa |
|---|---|---|---|---|
| PREFILL MHA N=1024 | 1.79 | 0.16 | 0.11 | 15.9× |
| PREFILL MHA N=2048 | 6.33 | 0.49 | 0.33 | 19.4× |
| PREFILL MHA N=4096 | 23.3 | 1.90 | 1.26 | 18.5× |
| DECODE GQA B=8 N=2048 | 0.74 | 0.10 | 0.85 | **0.9×** |
| DECODE GQA B=32 N=2048 | 1.14 | 0.19 | 3.06 | **0.4×** |

- **Prefill**: flash-ozaki is ~16–19× SDPA — the emulation cost (nmp10 ≈ several digit-pair dots + the
  per-tile block-FP encode, no int8 shortcut). It is *not* meant to beat SDPA; it reproduces the int8
  ozaki numerics.
- **Decode**: flash-ozaki is **faster than torch SDPA** (~2.7× at B=32) because the GQA head-fold packs
  the 7 query heads sharing a KV head into one `[G,D]` tile (one KV load feeds all G), whereas SDPA with
  repeated KV handles q_len=1 GQA poorly. It is still ~6× the exact bf16 flash path (the ozaki cost).

---

## Key findings
1. **Faithful**: with the production block-FP scale convention, the flash codegen kernel's dequantized
   operands are **bit-identical** to production and outputs match to the **fp32-accumulation floor**;
   accuracy `== production` exactly. The int8-HW digit ranges (`top ∈ [−2^(w-1), 2^(w-1)]`) are respected.
2. **The earlier `maxmag = 2^(w·nD)−1`** made the flash encode ~1 bit finer than the int8 datapath
   (top digit up to `±2^w`) → looked "more accurate" but was **un-faithful**; corrected to
   `int_bits = w·nD−1` (frexp scale + clamp).
3. **Fusing all digit-pair dots in one kernel** (what flash attention does) beats production's separate
   cuBLAS-per-slot at the **low-intensity attention shapes** (3–9×), but loses to cuBLAS on **large
   compute-bound GEMMs** — confirming the fused approach is right for attention, slot-cuBLAS for linears.
