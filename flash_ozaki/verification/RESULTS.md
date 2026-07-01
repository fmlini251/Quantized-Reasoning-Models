# flash_ozaki 검증 결과 — production `ozaki1_batched_gemm_fp` 및 torch SDPA 대비

flash-attention용 **ozaki1_fp 코드젠 커널**(`flash_oz1fp_codegen.py`: fused flash + cached-KV,
`verification/standalone_oz_gemm.py`: 독립 GEMM)이 production 에뮬레이션
`emulation/llm/ozaki_matmul.py::ozaki1_batched_gemm_fp`을 **충실히(faithful)** 재현하는지 검증하고,
전체 어텐션의 속도/정확도를 torch SDPA와 비교한다.

- **환경:** NVIDIA RTX A6000 (gpu:0), conda `quantized-reasoning-models`, torch 2.5.1+cu124 /
  triton 3.1.0, `byte_split_style=all_signed_no_clamp`, bf16 입력 / fp32 누산.
- **reduction chunk = 32 (전 실험 통일)** — production의 block-FP chunk_size와 동일. 모든 GEMM 리덕션
  (QK head_dim, PV kv, 독립 GEMM의 K)에 chunk=32 적용. 각 실험 표에 chunk를 명시한다.
- **측정 일자:** 2026-07-01. 모든 수치는 이 문서 작성 시점에 **직접 재실행**해 얻은 값이다.
- **대상 코드:** production-faithful 코드젠 경로 (optimal pack plan + single-peel place-folded planes +
  frexp `int_bits` block-FP 스케일). 예전 hand-rolled `_po2/maxmag` 커널은 제거됨.
- **production 기준 = ozaki1_fp** (`emulation/llm/ozaki_matmul.py::ozaki1_batched_gemm_fp`, `rslt_type=
  "ozaki1_fp"`). 본 문서의 **모든 production 수치는 int8 실경로가 아니라 우리와 동일한 bf16 ozaki1_fp 경로**
  이므로 apples-to-apples 비교다(속도 차이는 커널 구현 차이지 datapath 차이가 아님).

> **ozaki1은 int8-GEMM HW 방식이고 이 코드(와 production ozaki1_fp)는 둘 다 그 int8 데이터패스의 bf16
> *에뮬레이션*이다.** 기준은 production 에뮬레이션이지 un-quantized 곱이 아니다. "production보다 더 정확"하면
> int8 HW보다 정밀한 기계를 모델링한 것이므로 오히려 *un-faithful*이다. **Faithful = production(ozaki1_fp)과 일치.**

## 파일
| 파일 | 내용 |
|---|---|
| `standalone_oz_gemm.py` | 어텐션에서 떼어낸 QK/PV ozaki1_fp 행렬곱(A@B), 코드젠 plan+peel. `oz1fp_gemm_cg`, `encode_B`(weight cache). |
| `bench_gemm_vs_production.py` | **GEMM**: 독립 QK·PV vs production (cache/non-cache × nmp × w) — relerr, 비트동일, 지연. |
| `verify_faithfulness.py` | 인코딩 충실도: Â 비트동일, no_clamp 최상위 자릿수 범위, 자릿수 분해 동일, 커널==자기 fp64 곱. |
| `bench_attention_vs_sdpa.py` | **전체 어텐션**: flash-ozaki vs SDPA vs exact bf16 flash — 정확도 + 지연, prefill/decode(GQA). |
| `analyze_triton_vs_cublas.py` | fused Triton이 어텐션 형상에선 이기고 큰 GEMM에선 cuBLAS에 지는 이유(타일링 스윕). |

실행: 리포 루트에서 `CUDA_VISIBLE_DEVICES=0 python flash_ozaki/verification/<script>.py`

---

## Part A — GEMM: 독립 QK/PV vs production **ozaki1_fp** (`ozaki1_batched_gemm_fp`, bf16, chunk=32)

> **flash 커널과의 관계:** 독립 GEMM과 flash 커널은 codegen의 `pack_plan`/`_emit_peel`/`_emit_dots`/
> `_bfp_scale`를 **그대로 공유**하고, 이제 둘 다 **동일한 chunk=32** block-FP 리덕션을 쓴다.

### A.1 충실도 (`verify_faithfulness.py`, chunk=32)
- **역양자화 피연산자 Â가 production과 비트동일** (`torch.equal`, relerr `0.00e+00`): w4 nmp9/16, w8 nmp1, w2 nmp16.
- **no_clamp 최상위 자릿수 ∈ `[−2^(w-1), 2^(w-1)]`** (int8 범위 + 1비트 MSB 플래그): w4 `[−8,8]`, w8 `[−128,127]`, w2 `[−2,2]`. ✔
- **부호 자릿수 분해가 `_oz1_wbit_digit_split`과 동일** — 모든 (w,nD,clamp)에서 `True`.
- **커널 == 자기 자신의 fp64 양자화 곱**: w4 nmp9 / nmp16 / w8 nmp1 → `5.7e-9 / 5.7e-8 / 0.0`.

### A.2 정확도 (`bench_gemm_vs_production.py`, Z=28, prefill, **chunk=32**)
형상(M×K×N): **QK = 1024×128×1024**(reduction K=128=head_dim), **PV = 1024×1024×128**(reduction K=1024=kv,
N=128=head_dim). `mine/prod` = 독립 vs production, `*/exact` = fp32 `A@B` 대비. cache==fresh(정확도 동일).

| op | w/nmp | mine/prod | 비트동일 | mine/exact | prod/exact |
|---|---|---|---|---|---|
| QK | w8 nmp1 | **0.00e+00** | yes | 1.28e-2 | 1.28e-2 |
| QK | w4 nmp9 | 5.67e-9 | no | 3.61e-4 | 3.61e-4 |
| QK | w4 nmp10 | **0.00e+00** | yes | 1.21e-4 | 1.21e-4 |
| QK | w4 nmp15 | 7.69e-9 | no | 4.38e-6 | 4.38e-6 |
| QK | w4 nmp16 | 5.73e-8 | no | 5.67e-6 | 5.67e-6 |
| PV | w4 nmp9 | 9.75e-8 | no | 2.63e-4 | 2.63e-4 |
| PV | w4 nmp10 | **0.00e+00** | yes | 1.97e-4 | 1.97e-4 |
| PV | w4 nmp16 | 1.14e-7 | no | 4.20e-6 | 4.20e-6 |

**`mine/exact == prod/exact` 정확히 일치** → faithful. `mine/prod`는 fp32 누산 바닥. **cache(`encode_B`) == fresh**.

### A.3 속도 전체 스윕 (QK prefill, Z=28 M=N=1024 K=128, chunk=32; 지연 ms)
w∈{2,4,8} × nmp∈{1,3,4,6,9,10,15,16}, cache/non-cache 분리. `#G`=자릿수 dot 수(w<8은 pack-plan, w=8은 nmp).
`p/m`=production/mine 배속(>1 = mine 빠름). cuBLAS bf16 bmm = 0.177 ms.

| w | nmp | #G | mine_fresh | mine_cache | prod_fresh | prod_cache | p/m fresh | p/m cache |
|---|---|---|---|---|---|---|---|---|
| 2 | 1  | 1 | 0.491 | 0.315 | 0.385 | 0.359 | 0.78× | 1.14× |
| 2 | 3  | 2 | 0.658 | 0.661 | 1.354 | 1.106 | 2.06× | 1.67× |
| 2 | 4  | 1 | 0.591 | 0.404 | 0.831 | 0.583 | 1.41× | 1.44× |
| 2 | 6  | 3 | 0.778 | 0.750 | 2.124 | 1.512 | 2.73× | 2.02× |
| 2 | 9  | 1 | 0.650 | 0.445 | 1.141 | 0.740 | 1.76× | 1.66× |
| 2 | 10 | 4 | 0.913 | 0.935 | 3.057 | 1.931 | 3.35× | 2.07× |
| 2 | 15 | 5 | 1.048 | 1.100 | 3.701 | 2.564 | 3.53× | 2.33× |
| 2 | 16 | 1 | 0.749 | 0.509 | 1.451 | 0.896 | 1.94× | 1.76× |
| 4 | 1  | 1 | 0.491 | 0.314 | 0.386 | 0.361 | 0.79× | 1.15× |
| 4 | 3  | 2 | 0.658 | 0.662 | 1.357 | 1.108 | 2.06× | 1.67× |
| 4 | 4  | 1 | 0.590 | 0.409 | 0.854 | 0.586 | 1.45× | 1.43× |
| 4 | 6  | 3 | 0.741 | 0.739 | 1.987 | 1.727 | 2.68× | 2.34× |
| 4 | 9  | 4 | 0.849 | 0.628 | 2.391 | 2.132 | 2.81× | **3.40×** |
| 4 | 10 | 5 | 1.022 | 1.153 | 3.036 | 2.547 | 2.97× | 2.21× |
| 4 | 15 | 6 | 1.151 | 1.195 | 3.690 | 3.182 | 3.21× | 2.66× |
| 4 | 16 | 4 | 0.898 | 0.701 | 2.838 | 2.356 | 3.16× | **3.36×** |
| 8 | 1  | 1 | 0.494 | 0.318 | 0.284 | 0.248 | **0.58×** | **0.78×** |
| 8 | 3  | 3 | 0.678 | 0.465 | 1.411 | 1.375 | 2.08× | 2.96× |
| 8 | 4  | 4 | 0.708 | 0.488 | 1.814 | 1.780 | 2.56× | 3.65× |
| 8 | 6  | 6 | 1.010 | 0.632 | 2.635 | 2.583 | 2.61× | 4.09× |
| 8 | 9  | 9 | 1.093 | 0.795 | 9.123 | 9.075 | 8.35× | **11.42×** |
| 8 | 10 | 10 | 1.223 | 1.024 | 10.192 | 10.136 | 8.33× | 9.90× |
| 8 | 15 | 15 | 3.244 | 1.565 | 15.508 | 15.439 | 4.78× | 9.86× |
| 8 | 16 | 16 | 1.473 | 1.178 | 16.575 | 16.504 | **11.25×** | **14.01×** |

**핵심:**
1. **nmp1(단일 dot)만 mine이 느리다**(w8 0.58×/0.78×, w2·4 ~0.79×/1.15×) — 손익분기(A.4). production의 단일
   cuBLAS + 가벼운 combine이 fused 단일-dot의 중복 인코딩 오버헤드보다 유리.
2. **`#G`↑ → mine이 크게 이긴다**, 특히 **w=8(패킹 없음, #G=nmp)**: nmp16 = **cache 14.0× / fresh 11.3×**.
   production은 `#G`개 개별 cuBLAS(+피연산자 재-read)를 내지만 mine은 단일 fused 커널에서 한 번 읽고 누산.
3. **w=2/4는 패킹이 full-nmp를 #G=1로 접음**(nmp 4/9/16→#G1) → 거기선 mine 이점 작음(1.1~1.9×). triangular nmp
   (#G>1: 3/6/10/15)는 2~3.5×.
4. **cache vs non-cache**: mine은 항상 cache가 빠름(B 인코딩 생략). p/m 배속은 대개 **cache에서 더 큼**(mine이
   캐시 이득을 더 봄). production의 cache-fresh 격차는 **w8에서 작음**(#G개 cuBLAS가 지배, 인코딩은 소수) —
   nmp16 16.58→16.50; w2/4에선 큼(인코딩 비중↑).
5. **w8 nmp15 mine_fresh=3.24 이상치**: nD5 fresh는 B 인라인 인코딩으로 SRAM↑ → OOM 재시도(BLOCK_M 축소) → 느림.
   cache(1.57)는 B가 캐시라 회피.

### A.4 linear-layer GEMM 모양 스윕 (Z=1 M=2048 K=N=4096, chunk=32) — A.3과 정반대로 여기선 production 유리
어텐션(K=128, 저강도)과 달리 linear는 큰 compute-bound GEMM. cuBLAS bf16 bmm = 0.714 ms. `p/m`>1 = mine 빠름.

| w | nmp | #G | mine_fresh | mine_cache | prod_fresh | prod_cache | p/m fresh | p/m cache | mine_c/cuBLAS |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1  | 1 | 4.099 | 2.371 | 1.090 | 0.994 | 0.27× | 0.42× | 3.32× |
| 2 | 4  | 1 | 5.015 | 3.140 | 2.504 | 1.457 | 0.50× | 0.46× | 4.40× |
| 2 | 9  | 1 | 5.481 | 3.587 | 3.542 | 1.820 | 0.65× | 0.51× | 5.02× |
| 2 | 10 | 4 | 8.096 | 8.292 | 8.224 | 3.413 | 1.02× | 0.41× | 11.61× |
| 2 | 16 | 1 | 6.471 | 4.161 | 4.527 | 2.152 | 0.70× | 0.52× | 5.83× |
| 4 | 1  | 1 | 4.077 | 2.343 | 1.069 | 0.973 | 0.26× | 0.42× | 3.28× |
| 4 | 4  | 1 | 5.016 | 3.166 | 2.532 | 1.475 | 0.50× | 0.47× | 4.43× |
| 4 | 9  | 4 | 7.419 | 5.587 | 4.942 | 3.844 | 0.67× | 0.69× | 7.83× |
| 4 | 10 | 5 | 9.191 | 9.794 | 6.684 | 4.632 | 0.73× | 0.47× | 13.72× |
| 4 | 16 | 4 | 7.985 | 5.902 | 6.389 | 4.339 | 0.80× | 0.74× | 8.27× |
| 8 | 1  | 1 | 4.078 | 2.342 | 0.808 | 0.713 | 0.20× | 0.30× | 3.28× |
| 8 | 4  | 4 | 6.177 | 4.060 | 3.207 | 3.065 | 0.52× | 0.75× | 5.69× |
| 8 | 9  | 9 | 9.549 | 6.811 | 8.568 | 8.383 | 0.90× | **1.23×** | 9.54× |
| 8 | 10 | 10 | 10.409 | 9.090 | 9.566 | 9.324 | 0.92× | 1.03× | 12.73× |
| 8 | 16 | 16 | 13.756 | 10.863 | 15.241 | 15.005 | 1.11× | **1.38×** | 15.21× |

**핵심 (A.3 어텐션과 정반대):**
1. **linear에선 mine이 대부분 진다**(p/m<1, 0.2~0.9×), mine_cache는 cuBLAS 대비 **3~15× 느림**. Part D대로 — 큰
   compute-bound GEMM은 cuBLAS가 128×128 타일로 near-peak인데 ozaki는 자릿수-평면 SRAM 때문에 큰 타일을 못 써
   느리다. production은 slot당 cuBLAS라 그 효율을 그대로 쓴다.
2. **mine이 이기는 건 #G가 아주 클 때만**: w8 nmp16(#G16) cache **1.38×**/fresh 1.11×, nmp9 cache 1.23×. `#G`개
   cuBLAS 호출 오버헤드가 커져야 fused가 앞선다.
3. **여기선 caching이 mine에 더 중요**: 큰 weight(4096²)를 fresh는 m-타일마다 재인코딩 → mine_fresh가 크게 느림
   (w4 nmp1 fresh 4.08 vs cache 2.34). = production의 "모델-레벨 weight_cache가 큰 이득"(ozaki1_fp_speed.md)과 같은
   원리(weight가 클수록 1회 인코딩 amortize↑). low-#G에서 mine이 지는 근본은 **activation/weight의 m·n-타일 중복
   인코딩**(A.3 nmp1과 동일 메커니즘; QK 격리 마이크로벤치는 커밋 7acbc21 — encode 0.130 vs 1회 0.022, ~5.8×).
4. w2/4 nmp10은 mine_cache가 fresh보다 느린 이상치(nD4 평면 로드 / OOM 재시도).

**결론 (A.3 + A.4)**: **fused-Triton(mine)은 어텐션 저강도 형상에서 production을 최대 14× 앞서고(A.3), linear의 큰
compute-bound GEMM에선 cuBLAS(slot-cuBLAS)에 3~15× 진다(A.4).** → 어텐션엔 fused-Triton, linear엔 production
slot-cuBLAS. flash_ozaki가 어텐션 전용인 이유.

---

## Part B — 전체 어텐션: flash-ozaki vs SDPA vs exact flash (`bench_attention_vs_sdpa.py`, chunk=32)

ozaki w4 nmp10, **chunk=32**. flash-exact = 같은 커널 `ozaki=False`(순수 bf16) 경로.

### B.1 정확도 (fp32 exact 대비, chunk=32)
| case | flash-ozaki (bf16 out) | flash-exact (bf16) | torch SDPA (bf16) | prod-EAGER (fp32) | prod-FLASH (fp32) |
|---|---|---|---|---|---|
| PREFILL MHA N=1024 | **1.61e-3** | 1.99e-3 | 1.99e-3 | 1.66e-4 | 1.96e-4 |
| DECODE GQA B=32 N=2048 | **1.68e-3** | 2.25e-3 | 2.25e-3 | — | — |

- flash-ozaki(nmp10)는 **순수 bf16 어텐션(SDPA/flash-exact)만큼 — 오히려 약간 더 — 정확**하다. (P를 fp32로
  유지한 채 block-FP 인코딩하므로, P를 bf16으로 truncate하는 SDPA/flash-exact보다 낫다.)
- prod-EAGER/FLASH = **production `ozaki1_batched_gemm_fp`** 로 QK·PV를 돌린 torch 어텐션. EAGER는 전체 S를
  materialize→전체행 fp32 softmax→PV, FLASH는 online-softmax(BN=32 타일별 QK/PV, P를 fp32 유지). 둘 다
  **fp32를 반환**한다.

> **핵심(정정): flash-ozaki의 "1.6e-3"은 >99%가 bf16 출력 반올림이지 ozaki·flash 알고리즘 오차가 아니다.**
> - score(QK)·softmax 통계(m/l)·accumulator는 flash-ozaki도 **전부 fp32**다(SDPA·Triton flash와 동일). bf16은
>   **최종 출력 저장**(`acc.to(Out.dtype)`) 한 곳뿐이며, bf16 모델의 정상 동작이다.
> - 증거: **SDPA도 bf16 입력→bf16 출력(정상 사용법)이면 fp32-exact 대비 1.99e-3**, fp32로 돌리면 **3.4e-7**.
>   flash-ozaki 출력을 fp32로 두면(=codegen twin 에뮬) 실제 ozaki 오차는 **~2e-4**로, prod-FLASH(1.96e-4)와
>   일치한다. `emul(codegen twin)→bf16` 은 flash-ozaki와 **relerr까지 bit-일치**(1.534e-3=1.534e-3).
> - GEMM 단독 검증: codegen QK/PV는 production `ozaki1_batched_gemm_fp`와 **relerr=0.0(bit-identical)**.

**flash-ozaki는 production-based EAGER보다 production-based FLASH 에뮬레이션에 훨씬 가깝다** (출력 정밀도를 bf16로 맞춰 측정):

| flash-ozaki 까지의 거리 (bf16 출력 일치) | relerr |
|---|---|
| ↔ **prod-FLASH**→bf16 (online softmax, 동일 구조) | **3.2e-5** |
| ↔ **prod-EAGER**→bf16 (materialized full-row softmax) | 8.5e-4 |

online-softmax flash 알고리즘을 production ozaki GEMM으로 **충실히 구현**했음을 확인(FLASH에 ~26× 더 가까움).
eager와의 8.5e-4 차이는 알고리즘 차이(전체행 정규화 P vs per-tile 비정규화 P). fp32로 보면 두 거리가
1.60e-3≈1.62e-3로 구분 안 되는데, 이는 flash-ozaki의 bf16 출력 반올림(~1.6e-3)이 3e-5 신호를 덮기 때문.

### B.2 지연 (ms/call, chunk=32) — **non-cached flash 속도**
| case | flash-ozaki | flash-exact | torch SDPA | oz/sdpa |
|---|---|---|---|---|
| PREFILL MHA N=1024 | 1.732 | 0.205 | 0.132 | 13.1× |
| PREFILL MHA N=2048 | 6.247 | 0.652 | 0.393 | 15.9× |
| PREFILL MHA N=4096 | 23.611 | 2.318 | 1.361 | 17.4× |
| DECODE GQA B=8 N=2048 | 0.755 | 0.103 | 0.856 | **0.9×** |
| DECODE GQA B=32 N=2048 | 1.002 | 0.188 | 3.235 | **0.3×** |

- **Prefill**: flash-ozaki는 SDPA의 ~13~17×. int8 ozaki 수치를 재현하는 비용(nmp10 ≈ 여러 자릿수쌍 dot +
  타일별 block-FP 인코딩). (참고: chunk=None보다 chunk=32가 **더 빠른데**, 이는 "chunk가 작아서"가 아니라
  chunk=None의 hoisted 경로가 `nD`개 Q 자릿수-평면을 **head_dim 전체 폭**으로 상주시켜 nD≥4에서 공유메모리
  한도를 넘겨 OOM 재시도가 BLOCK_M을 접기 때문 — 상세 원인은 Part C.)
- **Decode**: flash-ozaki가 **torch SDPA보다 빠르다**(B=32에서 ~3.2×). GQA 헤드 폴딩으로 7개 쿼리 헤드를
  하나의 `[G,D]` 타일로 묶어 q_len=1 GQA를 SDPA보다 잘 처리. exact bf16 대비는 ~5×(ozaki 비용).

### B.3 prefill이 flash-exact보다 ~8× (SDPA 대비 ~13×) 느린 원인 분해 (N=2048, 측정)
| config | ms | /exact |
|---|---|---|
| flash-exact (ozaki off, bf16 1 dot) | 0.651 | 1× |
| nmp1 w8 (자릿수 dot **1개** + 인코딩) | 2.391 | **3.7×** |
| nmp4 w4 (pack=1 dot, nD2) | 3.116 | 4.8× |
| nmp9 w4 (pack=4 dots, nD3) | 3.997 | 6.1× |
| nmp10 chunk=None (QK 1청크) | 63.49 | **97.6×** (OOM 재시도) |
| nmp10 chunk=32 (pack=5 dots) | 5.197 | **8.0×** |

1. **블록-FP 인코딩이 지배(~2.7×)**: nmp1은 자릿수 dot이 exact와 같은 1개인데도 3.7× → 느린 건 dot이 아니라
   Q/K/P/V 4개 피연산자의 타일별 인코딩(amax 리덕션 + `a/scale` fp32 나눗셈 + round/clamp + int→bf16 캐스트).
   flash-exact는 raw bf16이라 인코딩 0.
2. **다중 자릿수 dot(+~2×)**: pack-plan dot 수 1→4→5 (nmp1/9/10)에 비례해 3.7×→6.1×→8.0×.
3. **chunk=32가 Q/K를 head_dim 청크별로 재인코딩**해 인코딩 증폭. 단 chunk=None은 nD≥4에서 OOM 재시도로
   **97.6×**(Part C) — chunk=32가 정답.

본질적으로 int8 ozaki 수치를 SW로 재현하는 비용(HW int8 데이터패스 대체).

**시도했다 기각한 op-레버 3종 (모두 bit-exact, 모두 속도 무변):**
| 시도 | 결과 |
|---|---|
| `a/scale` → 역수 곱 (scale=2^k) | GEMM 0.314→0.309, prefill nmp10 8.0×→7.9× (노이즈) |
| `_bfp_scale` exp2 → 정수지수 비트구성 | nmp1 3.7×, nmp9/10 무변 |
| P amax → softmax의 타일 max 재사용(리덕션 제거) | nmp9 6.1→6.0×, 나머지 무변 |

셋 다 faithfulness 유지(Â bit-identical)이나 이득 없음 → **인코딩 비용은 개별 산술/리덕션이 아니라 구조적**
(Q/K/P/V 4개 피연산자의 메모리 트래픽 + int↔bf16 캐스트 + occupancy)이다. Triton/ptxas가 나눗셈/exp2를 이미
최적화하고, 리덕션도 병목이 아니라 op 하나 빼도 안 변함.

**occupancy 오토튜닝도 실측 — 무의미:** num_warps×num_stages×BLOCK_M 스윕(prefill nmp10: 3×3×4=36 config,
decode nmp10: 32 config). prefill 최적 = default 대비 **1.02×**(nw4/ns1/bm64가 이미 rank-3), decode = **1.00×**
(default가 최적). 즉 커널은 **이미 near-optimal**하게 튜닝돼 있고 ~8×(prefill)/~5×(decode) 오버헤드는 구현
비효율이 아니라 **ozaki 에뮬레이션의 본질적 비용**(다중 자릿수 dot + 타일별 4-피연산자 인코딩)이다.

**결론 — 실효 레버는 사실상 하나:** ① **K/V 캐싱**(구현됨, 4개 중 2개 인코딩을 루프에서 제거, ~1.3×) — 검증된
유일한 방법. ② op-튜닝/오토튜닝은 exhausted(무효). ③ 큰 이득은 근본적으로 **HW int8 데이터패스**(에뮬 대상)나
**더 거친/적은 인코딩**(faithfulness 희생)뿐.

---

## Part C — chunk=32를 표준으로 쓴 근거

`chunk_size`는 **모든 GEMM 리덕션**의 block-FP chunk를 정한다(QK head_dim → `KQ=next_pow2(min(cs,D))`,
PV kv → 타일 `BLOCK_N=next_pow2(cs)`). production이 chunk=32이므로 전 실험을 32로 통일했다.

**non-cached flash vs production ozaki1_fp eager (materialized `batched_gemm`, bf16, N=1024)** — chunk=32가 항상 chunk=None 이상으로 근접:

| w/nmp | fused vs eager (chunk=None) | fused vs eager (chunk=32) |
|---|---|---|
| w4 nmp9  | 2.55e-3 | **2.47e-3** |
| w4 nmp10 | 2.28e-3 | **2.27e-3** |
| w4 nmp15 | 2.26e-3 | **2.26e-3** |

정확도 gap 개선은 작지만(남은 `fused vs eager` ~2.3e-3은 대부분 fp32 누산 순서), **chunk=32는 production과
정합하면서 더 빠르다**(Part B.2). 그래서 32를 표준으로 채택.

### 왜 chunk=32가 chunk=None보다 빠른가 (원인 분석)
"chunk가 작아서"가 **아니다** — 속도는 chunk에 대해 단조가 아니다. prefill N=2048, `BLOCK_M` 요청=64,
괄호는 OOM 재시도 후 실제 사용 BLOCK_M:

| nmp (nD) | chunk=None BN=64 | BN=32 | BN=16 | chunk=32 (BN=32) |
|---|---|---|---|---|
| 9 (nD3) | 5.86 (BM64) | **3.74** (BM64) | 5.37 (BM64) | 3.99 (BM64) |
| 10 (nD4) | 9.18 (BM32) | 12.94 (BM32) | 8.43 (BM64) | **5.22** (BM64) |
| 15 (nD5) | 177 (BM16) | 36.6 (BM32) | 10.4 (BM64) | **5.91** (BM64) |

두 가지 효과(둘 다 공유메모리 문제):
1. **주효과(nD≥4): chunk=32가 상주 Q-평면 크기를 줄여 OOM 재시도를 피한다.** chunk=None의 hoisted 경로는
   `nD`개 Q 자릿수-평면을 **head_dim 전체**(`nD×[BM,128]` bf16)로 kv 루프 내내 상주 → nD=4/5에서 100KB/SM
   초과 → OOM 재시도가 **BLOCK_M을 접어**(64→32→16) occupancy 붕괴(9~177ms). chunk=32는 현재 `KQ=32` 조각만
   상주(`nD×[BM,32]`, 4× 작음) → BM=64 유지 → 5~6ms. nmp10/15 격차의 대부분.
2. **부효과: BLOCK_N=32 occupancy 스윗스팟.** OOM 없는 nmp9에서 BN64→5.86, **BN32→3.74**, BN16→5.37 —
   BN16이 BN32보다 **느리다** → 단조 감소가 아니라 32 근처가 최적. chunk=32가 BLOCK_N=32를 설정해 여기에 안착.

부수 발견: 현재 OOM 재시도가 **BLOCK_M을 줄이는 것은 나쁜 선택**이다(nmp15 chunk=None: BN64/BM16=177ms vs
BN16/BM64=10.4ms). BLOCK_N을 줄이거나 hoisted Q-평면 폭을 제한하면 이 경로를 ~17× 회복 가능 — 후속 최적화.

---

## Part D — Triton vs cuBLAS (`analyze_triton_vs_cublas.py`, chunk-무관)

ozaki 없는 순수 bf16 Triton GEMM을 타일링별로 cuBLAS `torch.bmm`과 비교(block-FP 없음 → chunk 무관).

**LARGE compute-bound (Z=1, M=2048, K=N=4096)** — cuBLAS 0.705 ms

| 타일 | Triton ms | vs cuBLAS |
|---|---|---|
| BM64 BN64 | 1.165 | 1.65× |
| BM128 BN128 | **0.674** | **0.96×** |

**QK-small low-intensity (Z=28, M=1024, K=128, N=1024)** — cuBLAS 0.177 ms

| 타일 | Triton ms | vs cuBLAS |
|---|---|---|
| BM64 BN64 | 0.138 | 0.78× |
| BM128 BN128 | **0.119** | **0.67×** |

큰 compute-bound GEMM은 ozaki 자릿수-평면 SRAM 압력으로 큰 타일을 못 써 cuBLAS(production) 유리, 저강도
어텐션 형상에선 fused-Triton 유리.

---

## Part E — 사전 인코딩 KV 캐시 (`flash_oz1fp_cg_cached`, `flash_ozaki/bench_cached.py`, chunk=32)

weight_cache의 어텐션 판. K/V를 block-FP로 **한 번** 인코딩해 place-folded 자릿수 평면으로 저장하고, 커널은
이를 로드해 타일별 K/V 인코딩(amax+반올림+분해+캐스트)을 건너뛴다. Q·P는 인라인. `encode_kv(chunk_size=32)`가
K head_dim을 chunk별로 스케일(k_scale `[Z,nchd,N]`), V kv를 chunk로 나눠 non-cached와 같은 세분성을 갖는다.

### E.1 정확도 + 속도 (N=2048, H=28, D=128, causal, **chunk=32**)
| cfg | K/V 평면 | cache MB | cached vs EXACT | cached vs non-cached | ms cached / non-cached |
|---|---|---|---|---|---|
| w4 nmp9 | 3/3 | 89.9 | 1.86e-3 | 4.00e-6 | **3.01** / 3.98 |
| w4 nmp16 | 4/4 | 119.3 | 1.61e-3 | 9.97e-6 | **3.54** / 4.54 |
| w4 nmp10 | 4/4 | 119.3 | 1.62e-3 | 6.47e-6 | **4.10** / 5.24 |
| w4 nmp15 | 5/5 | 148.6 | 1.61e-3 | 3.25e-6 | **4.73** / 5.91 |

Decode(chunk=32): MHA 1.92e-3, GQA(4 kv헤드, G=7 fold) 1.87e-3 (vs EXACT). **캐시가 모든 config에서
non-cached보다 빠르다**(타일별 K/V 인코딩 제거; chunk=32는 BLOCK_N=32라 공유메모리 여유가 있어 OOM 재시도도 없음).

### E.2 캐시 이득은 prefill·decode 모두 ~1.3× (production의 "decode에서 큰 이득"과 다른 이유)
`nc/c` = non-cached(매 호출 전체 K/V 재인코딩) / cached(사전 인코딩) 지연 비.

| regime | w4 nmp9 nc/c | w4 nmp10 nc/c |
|---|---|---|
| PREFILL MHA N=2048 | 1.30× | 1.27× |
| DECODE GQA B=32 N=2048 | 1.29× | 1.28× |

production(`ozaki1_fp_speed.md`)은 **decode 캐시 이득이 큼**(w4 nmp10 decode 18.1×→5.1×, ~3.5×) / prefill은
작음(6.5×→5.6×). 우리는 두 regime 모두 ~1.3×로 **평탄**한데, 이유는 **캐시 대상이 다르기 때문**:
- production `weight_cache`는 **정적 weight**(K×N, 매 step 동일)를 **전체 추론에서 1회** 인코딩→모든 토큰 재사용.
  non-cached decode는 매 M=1 GEMV(50µs)마다 거대한 weight(4096²)를 재인코딩→인코딩≫GEMV→decode 이득 큼.
- 우리 attention KV 캐시는 **동적**(decode마다 새 토큰 K/V가 append)이고, 인코딩이 커널에 **융합돼 저렴**(호출당
  ~22–28%). 그래서 전체 재인코딩을 캐시로 없애도 ~1.3×.
- production식 decode 대박 이득을 보려면 **증분 인코딩**(새 토큰 K/V만 인코딩해 캐시에 append, 과거 재사용)이
  필요 — 현 bench는 매 호출 전체 N개를 재인코딩하므로 그 이득을 측정하지 않는다(실서빙 KV-cache는 증분).

**증분 인코딩 실측 (`encode_kv_append`, 새 32-토큰 블록만 인코딩해 append):** 정확성은 완벽(append 캐시 ==
전체 encode 캐시, **bit-identical**, attn relerr 0.0). 그러나 **지금 구현으론 오히려 손해**다:

| N | naive per-step(전체 재인코딩+attn) | cached(사전인코딩) | naive/cached | append 1블록(torch enc + O(N) concat) |
|---|---|---|---|---|
| 1024 | 0.380 | 0.296 | 1.28× | ~2.7ms (enc 1.95 + concat 0.71) |
| 2048 | 0.753 | 0.586 | 1.29× | ~2.9ms (enc 1.46 + concat 1.41) |
| 4096 | 1.499 | 1.161 | 1.29× | ~4.3ms (enc 1.46 + concat 2.87) |

- per-step 이득은 컨텍스트 길이 무관하게 **flat ~1.3×** (production의 "N이 클수록 커지는 decode 이득"과 다름).
- **append 자체(1.5~4.3ms)가 naive step(0.38~1.5ms)보다 크다** → 증분이 net 손해. 분해하면 torch `encode_kv`
  (32토큰)=~1.5ms(토큰 수 무관, frexp/파이썬 루프/pack-커널 launch 오버헤드) + O(N) concat(캐시 전체 복사).
- **병목은 "증분 vs 전체"가 아니라 인코더가 torch 참조 구현**이라는 점 — naive는 전체 K/V 인코딩을 **fused
  triton 커널 안에서**(수십 µs) 하므로 오히려 빠르다.

**fused-triton 증분 인코더 (`encode_kv_append_fused`, in-place, 옵션 구현):** torch append(≈2187µs)를 2개
fused triton 커널(`_enc_k_kernel`/`_enc_v_kernel`)이 사전할당 캐시에 **in-place 기록**(concat 제거)하는 것으로
대체. 결과(N=2048, B=32 GQA decode, w4 nmp9): **append 2187µs → 91µs (24× 빠름)**, bit-identical. 이제 증분
per-step(append 91 + cached-attn 590 = 682µs)이 naive(757µs)를 **1.11× 이긴다**(torch일 땐 졌음).

- 그래도 이득은 작다(상한 ~1.3×): cached-attention(590µs)이 지배, naive 재인코딩도 fused라 싸고(전체 N ≈167µs),
  fused append도 2-커널 launch 오버헤드(~91µs, 32토큰엔 과함). 짧은 컨텍스트(<~1100토큰)에선 append(91µs) >
  naive 재인코딩이라 오히려 손해, 긴 컨텍스트에서만 이득.
- **production식 3.5×가 안 나오는 근본 이유**: production은 **거대 정적 weight(4096²)**를 매 M=1 GEMV마다
  재인코딩→인코딩 ≫ GEMV. 우리 attention의 K/V 인코딩은 attention 대비 작고 naive조차 fused라 절약분이 적다.
  → 더 밀어붙이려면 **새 토큰 인코딩을 decode attention 커널에 융합**(별도 launch 제거)해야 함(production이 그렇게 함).
  현 결론: attention 쪽 캐시 이득 상한 ~1.3×; fused 증분 인코더로 그 상한을 실제로 달성(net 1.11×).

### E.3 cached vs non-cached = ~1e-5는 **순수 fp 누산 순서**(값 차이 아님) 확인
- 인코딩은 비트동일 — GEMM에서 `cached==fresh`가 w4 nmp10 chunk=32에서 `0.00e+00`; non-cached 커널에 raw K를
  넣든 캐시를 역양자화한 K를 넣든 출력이 `0.00e+00`(재인코딩 idempotent).
- 두 커널 모두 결정적(각자 2회 = `0.00e+00`), `num_warps` 4↔8도 `0.00e+00`.
- 그럼에도 **비트동일 피연산자**로 cached vs non-cached = `6.47e-6` → 원인은 두 커널이 **서로 다른 컴파일
  프로그램**(cached는 평면을 `tl.load`, non-cached는 인라인 peel)이라 자릿수쌍 dot + chunk별 `qk +=`를 약간
  다른 fp 순서로 합하고, 64단계 online-softmax rescale이 ~ulp를 ~6e-6로 증폭하기 때문.
- ozaki 오차(1.6e-3)의 약 1/250 → 무시 가능. (chunk=None에선 두 경로 구조가 정렬돼 cached vs non-cached가
  정확히 `0.00e+00`; decode도 `0.00e+00`.)

---

## 핵심 결론
1. **Faithful (chunk=32)**: 역양자화 피연산자는 production과 **비트동일**, 출력은 **fp32 누산 바닥**까지 일치
   (정확도 `== production`). int8-HW 자릿수 범위도 지켜진다.
2. **어텐션 정확도**: flash-ozaki(nmp10, chunk=32)는 bf16 SDPA만큼(약간 더) 정확. **Decode(GQA)에선 SDPA보다
   빠르고**(~3.2×), prefill은 int8 재현 비용으로 SDPA의 ~13~17×.
3. **chunk=32 통일**: 모든 리덕션(QK head_dim·PV kv·독립 GEMM K)을 production과 같은 32로 청크 — 더 faithful
   하고 chunk=None보다 빠르다. cached 경로도 동일 지원(cached vs non-cached: chunk=32 prefill은 fp-순서 ~6e-6,
   chunk=None/decode는 비트동일).
4. **fused-Triton은 저강도 어텐션에서 production을 3~9× 앞서고**, 큰 compute-bound GEMM은 cuBLAS가 유리.
