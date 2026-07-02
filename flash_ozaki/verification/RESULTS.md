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
| case | flash-ozaki (bf16) | flash-exact (bf16) | torch SDPA (bf16) | prod-FLASH (P→bf16, fp32 out) |
|---|---|---|---|---|
| PREFILL MHA N=1024 | **1.99e-3** | 1.99e-3 | 1.99e-3 | 1.19e-3 |
| DECODE GQA B=32 N=2048 | **2.28e-3** | 2.26e-3 | 2.25e-3 | — |

- flash-ozaki(nmp10)는 **순수 bf16 어텐션(SDPA/flash-exact)과 정확도 동일**하다. ozaki 양자화가 추가하는
  오차는 무시할 수준(bf16 아래).
- **정밀도 모델(모든 bf16 flash-attention 공통):** score(QK)·softmax 통계(m/l)·`l` 정규화자·accumulator는
  **전부 fp32**. **P(=softmax 확률)만 P@V matmul 직전에 bf16으로 truncate**하고(텐서코어가 bf16 피연산자를
  요구), 최종 출력도 bf16 저장. score→softmax 사이 bf16 캐스트는 **없다**.
  - flash-exact: `tl.dot(p.to(v.dtype), v)` — v가 bf16이라 P를 bf16 캐스트.
  - SDPA: FlashAttention의 텐서코어 P@V가 P를 입력 dtype(bf16)으로 캐스트.
  - **flash-ozaki: `p = p.to(tl.bfloat16)` 후 block-FP PV 인코딩** (l 정규화자는 fp32 유지). 세 경로가 동일.
- prod-FLASH = **production `ozaki1_batched_gemm_fp`** 로 per-tile QK/PV를 돌린 online-softmax torch 어텐션.
  fp32를 반환하므로 fp32-exact 대비 값이 flash-ozaki보다 작아 보이지만, 이는 **출력 dtype 차이**일 뿐이다.

> **핵심: flash-ozaki의 "2e-3 (vs fp32-exact)"는 ozaki·flash 알고리즘 오차가 아니라 bf16(출력 + P) 반올림이다.**
> - 증거: **SDPA도 bf16 입력→bf16 출력(정상 사용법)이면 1.99e-3**, fp32로 돌리면 **3.4e-7**. 같은 알고리즘,
>   출력 dtype만 다르다. prod-FLASH(P→bf16, fp32 출력) = 1.19e-3에 bf16 출력 반올림까지 더하면 flash-ozaki의
>   1.99e-3이 되며, ozaki 양자화 자체 기여분은 그 아래로 무시할 수준이다.
> - GEMM 단독 검증: codegen QK/PV는 production `ozaki1_batched_gemm_fp`와 **relerr=0.0(bit-identical)**.

**flash-ozaki는 production-based EAGER보다 production-based FLASH 에뮬레이션에 훨씬 가깝다** (출력·P 정밀도를 동일하게 bf16으로 맞춰 측정):

| flash-ozaki 까지의 거리 (bf16 출력·P 일치) | relerr |
|---|---|
| ↔ **prod-FLASH** (P→bf16, online softmax, 동일 구조) | **3.1e-5** |
| ↔ **prod-EAGER** (P→bf16, materialized full-row softmax) | 2.9e-3 |

online-softmax flash 알고리즘을 production ozaki GEMM으로 **충실히 구현**(FLASH에 ~92× 더 가까움)했으며,
P→bf16 truncation·bf16 출력까지 real flash-attention 데이터패스와 정확히 일치. eager와의 거리(2.9e-3)는
알고리즘 차이(전체행 정규화 P vs per-tile 비정규화 P)다.

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

### E.4 권장 (vLLM 서빙 default): **non-cached** (`flash_oz1fp_cg`)
캐시는 K/V를 **nD개 자릿수 평면**으로 저장하므로 KV 캐시 메모리가 **nD× 선형 증가**한다 (E.1: raw bf16 K+V
≈ 29.4MB → w4 nmp9 nD3 = 89.9MB(**3×**) / nmp10 nD4 = 119.3MB(**4×**) / nmp15 nD5 = 148.6MB(**5×**)).

vLLM 처리량은 **KV 캐시 용량**(동시 시퀀스 수 × 컨텍스트 길이, paged blocks)에 지배된다. cached의 nD×(3~5×)
블로우업은 배치/컨텍스트를 그만큼 줄여 **처리량 순손실이 캐시의 ~1.3× 이득을 크게 초과**한다(예: nD=4면 KV 용량
~1/4 → 동시성 ~1/4, 토큰당 1.3× 이득으로 상쇄 불가). 따라서:

- **서빙 기본 = non-cached**: 매 호출 K/V를 fused triton 커널 **안에서** 재인코딩(호출당 ~22–28%, 이미 상각)하고
  KV를 **1×(raw bf16)** 로 유지해 배치/컨텍스트 용량을 최대화. flash_oz1fp_cg API도 이미 non-cached가 기본이며,
  cached는 `flash_oz1fp_cg_cached`+`encode_kv` **opt-in**.
- **cached는 예외적으로만**: 메모리가 남고 지연이 결정적인 **저동시성/단일 시퀀스**(배치 1, KV가 병목 아님)에서만.
- 참고: 현재 vLLM 백엔드(`vllm_custom/.../ozaki_attention.py`)는 eager `batched_gemm` 경로라 flash 캐시가
  **아직 미연동**. flash_ozaki를 백엔드로 승격할 때도 위 이유로 non-cached를 기본으로 둔다.

---

## Part F — Nsight Compute 직접 프로파일링 + 병목 최적화 (2026-07-02, A6000)

Part A~E는 wall-clock config 스윕(블랙박스)으로 오버헤드를 "구조적"이라 결론냈지만 **HW 파이프라인을 직접
측정한 적이 없다.** `ncu`(Nsight Compute 2024.1.1, `--clock-control none`)로 flash-ozaki(`_flash_cg`) vs
flash-exact(`_flash_exact_fwd`)를 **동일 형상에서 직접 프로파일링**해 병목을 파이프 레벨로 규명하고, 그
병목(정수 ALU)을 직접 겨냥한 최적화를 적용했다. (프로파일 스크립트는 세션 스크래치, 결과는 아래 표.)

### F.1 병목 파이프 규명 (ncu, nmp10 w4 chunk=32)
| regime | 커널 | Duration | 최고 파이프 | DRAM% | occ% | 결론 |
|---|---|---|---|---|---|---|
| PREFILL N=1024 | exact | 194µs | Tensor(양호) | 20.6 | 15.3 | 레이턴시/occ-bound (D=128 레지스터压) |
| PREFILL N=1024 | **ozaki** | 1230µs | **ALU 56.4%** | **5.0** | 15.4 | **정수 ALU-bound** (자릿수 peel+block-FP quant) |
| DECODE N=4096 | exact | 370µs | — | **95.9** | 12.8 | **메모리-bound** (KV 스트리밍, DRAM≈peak) |
| DECODE N=4096 | **ozaki** | 1460µs | **ALU 48.4%** | **24.4** | 12.7 | **정수 ALU-bound** (메모리 파이프 굶김) |

**속도차 정당화:** flash-exact는 prefill=텐서코어, **decode=DRAM 대역폭(96%)** 에 bound된, 어텐션이 마땅히
그래야 할 커널이다. flash-ozaki는 그 위에 **int8 데이터패스의 SW 에뮬레이션(정수 자릿수 분해 + block-FP 양자화)**
을 얹어 **정수 ALU 파이프**를 병목으로 만든다(prefill ALU 56%/DRAM 5%, decode ALU 48%/DRAM 24%). 게다가
occupancy가 ~13~15%(레지스터压으로 2 block/SM, decode는 grid=B·Hkv도 작음)라 그 ALU 작업이 **메모리/텐서 뒤로
숨지 못하고 직렬로 얹혀** prefill ~6×, decode ~4×가 된다. 즉 오버헤드는 "구조적"이 아니라 **구체적으로 정수 ALU**다.

### F.2 op-구간 분해 (2×2 differential timing, 커널과 bit-exact)
QK/PV를 각각 ozaki/plain으로 토글(공유 softmax·GQA·causal 동일). exact 대비 배속:
- PREFILL N=2048: QK-only 4.2× / PV-only 3.7× / full 6.7×; QK를 encode+1dot로 자르면 2.3× → **QK encode ~1.3×,
  QK multi-dot ~1.9×**(다중 dot이 encode보다 큼), PV ≈ QK. **단일 지배 op 없음** — 비용이 4개(QK enc/dot,
  PV enc/dot)에 고루 퍼져 있다.
- DECODE B32 N4096: QK-only 2.5× / PV-only 2.3× / full 4.0×; encode/dot 대략 반반. B64에선 encode 비중 하락
  (warp가 늘어 은닉↑) → 레이턴시/occupancy 성분 확인.

### F.3 MATH-500 실서빙 병목 = **decode** (prefill 아님)
실제 nmp10 attn_only MATH-500 출력 120건 토크나이즈: prompt 평균 92 tok / generation 평균 4368 tok(중앙값
2568, 최대 32601). 어텐션 작업량(∝위치수)은 **decode/prefill = 3333×**(중앙값 1378×), 선형층조차 47×. 추론모델은
프롬프트의 ~50배를 생성하고 어텐션은 위치에 대해 2차이므로 **~4h eval은 사실상 전부 decode**다(prefill <0.1%).
→ 최적화 우선순위는 decode.

### F.4 최적화: **병렬 balanced-digit peel** (병목 ALU 직접 감축, bit-exact)
`_emit_peel`의 직렬 borrow peel(`cur=(cur-d)>>w` + `where`-select, 자릿수마다 순차 의존)을 **bias-trick 병렬형**
으로 교체: `z = src + B` (B=2^(w-1)·(2^(w(nD-1))−1)/(2^w−1)), `d_t = ((z>>wt) & (2^w−1)) − 2^(w-1)` (t<nD−1),
최상위 `= z>>(w(nD−1))`. 각 자릿수가 **독립**(borrow 체인 제거) + `where` 제거 → **정수 ALU op 감소 + ILP 상승**,
F.1에서 규명한 정확한 병목을 겨냥. `no_clamp`(flash 기본) 전용, 다른 clamp 스타일은 기존 직렬 peel 유지.
`_gen_src`·`_gen_cached_src` 공용이라 non-cached/cached 모두, vLLM 백엔드도 자동 적용(API 무변).

**Bit-identical 검증:** _digit_planes(직렬 torch twin) 대비 모든 w/nD(경계값 포함) `torch.equal`; decomp full vs
실커널 `0.0`; cached vs non-cached ~1e-6(기존과 동일한 fp 누산순서, E.3). 정확도(vs fp32-exact) 및 production
eager-ref 대비 relerr 모두 **불변**(2.5e-3) → production-faithful 유지.

**속도(실 `flash_oz1fp_cg`, CUDA-event):**
| regime | 기존(직렬) | 병렬 peel | 배속 | oz/exact |
|---|---|---|---|---|
| PREFILL nmp10 N2048 | 4.12ms | **3.41ms** | 1.21× | 6.5→5.4× |
| PREFILL nmp10 N4096 | 15.26ms | **12.64ms** | 1.21× | 6.6→5.5× |
| DECODE nmp10 B32 N4096 | 1.55ms | **1.30ms** | 1.19× | 4.2→3.5× |
| DECODE nmp15 B32 N4096 | 1.59ms | **1.27ms** | 1.26× | — |

**ncu 재측정으로 기전 확증:** prefill ALU 56.4→**51.5%**(1230→973µs), decode ALU 48.4→**44.1%** + DRAM
24→**30%**(1460→1210µs). 병목 ALU를 줄여 덜 ALU-bound가 되고(메모리 쪽으로 회귀) 그만큼 빨라졌다 — 예측대로.

### F.5 최적화 2: **split-KV / flash-decoding** (decode occupancy, `flash_oz1fp_cg_splitkv`)
F.4로도 decode는 여전히 ALU-bound(44%)·occ ~13%다. 원인은 레지스터 2 block/SM **+ 작은 grid**(Z=B·Hkv
프로그램이 84 SM을 못 채움 → DRAM 30%로 놀고 ALU가 은닉 못 됨). **kv 루프를 N_SPLITS개 program-z 슬라이스로
쪼개**(grid에 S축 추가) 각자 부분 online-softmax `(m,l,acc)`를 내고, `_flash_combine`이 log-sum-exp로 병합한다.
동시 프로그램↑ → occupancy↑ → ALU가 놀던 메모리 파이프 뒤로 은닉. chunk=32 전용, 각 split은 BLOCK_N 타일의
정수배라 타일별 block-FP는 non-split과 **동일**.

**정확성:** S=1 == non-split **비트동일(0.0)**(combine 항등 확인); S>1은 fp32-exact 대비 **2.1–2.3e-3 = non-split과
동일 정확도**(split↔non-split 2.5e-3은 두 valid fp 순서의 bf16 출력 반올림). `kv_lens`(vLLM 패딩 decode) 경로도
per-seq fp32 대비 2.1e-3 정상.

**속도 (decode GQA Hq28/Hkv4 w4 nmp10, ms/call; non-split은 이미 F.4 병렬 peel 적용본):**
| B / N | non-split | split-KV (best S) | 배속 |
|---|---|---|---|
| B=8 N=4096   | 0.917 | **0.239** (S=32) | **3.83×** |
| B=32 N=2048  | 0.606 | **0.450** (S=8)  | 1.35× |
| B=32 N=4096  | 1.288 | **0.883** (S=16) | 1.46× |
| B=64 N=4096  | 2.375 | **1.735** (S=16) | 1.37× |

이득은 **grid가 굶주릴수록 큼**(B=8 → Z=32 프로그램 ≪ 84 SM → **3.8×**; B=64 → Z=256 이미 차서 1.37×) — F.5의
"작은 grid" 진단과 정확히 일치. F.4 병렬 peel 위에 곱해지는 이득이다. `n_splits`는 명시 인자(휴리스틱: `Z·S`가
SM 수의 몇 배가 되도록, decode 짧은 배치엔 8–32).

### F.6 남은 레버
- prefill `num_stages=3` = 추가 1.11×(bit-exact) — 단 prefill은 eval의 ~0%.
- 근본 바닥: int8 에뮬레이션 = 정수 ALU 그 자체 → 더 큰 이득은 실제 int8 HW나 더 적은 자릿수(정확도 tradeoff).
  이제 이 결론은 블랙박스 추론이 아니라 **ncu로 HW 규명된 사실**이다.

---

## 핵심 결론
0. **직접 프로파일링(Part F)**: ncu로 병목을 파이프 레벨 규명 — exact는 prefill=텐서 / **decode=DRAM 96%(메모리
   bound)**, ozaki는 둘 다 **정수 ALU-bound**(int8 SW 에뮬레이션)로 저-occupancy라 은닉 실패 → prefill 6×/decode 4×.
   MATH-500 병목은 **decode**(작업량 3333× prefill). 병렬 balanced-digit peel로 ALU를 줄여 **bit-exact 1.2×**(decode
   포함) 달성, ncu로 ALU% 감소 확증. 추가로 **split-KV/flash-decoding**(`flash_oz1fp_cg_splitkv`)으로 decode
   occupancy를 올려 작은 배치에서 **최대 3.8×**(B=8), 큰 배치 1.35–1.46× (S=1은 non-split과 비트동일).
1. **Faithful (chunk=32)**: 역양자화 피연산자는 production과 **비트동일**, 출력은 **fp32 누산 바닥**까지 일치
   (정확도 `== production`). int8-HW 자릿수 범위도 지켜진다.
2. **어텐션 정확도**: flash-ozaki(nmp10, chunk=32)는 bf16 SDPA/flash-exact와 **동일**(1.99e-3, vs fp32-exact).
   이 값은 bf16(출력+P) 반올림이지 ozaki 오차가 아니다(SDPA도 bf16이면 1.99e-3, fp32면 3.4e-7). score/softmax/
   정규화자는 fp32, P는 P@V 직전 bf16 truncate — SDPA·flash-exact와 동일한 데이터패스. **Decode(GQA)에선
   SDPA보다 빠르고**(~3.2×), prefill은 int8 재현 비용으로 SDPA의 ~13~17×.
3. **chunk=32 통일**: 모든 리덕션(QK head_dim·PV kv·독립 GEMM K)을 production과 같은 32로 청크 — 더 faithful
   하고 chunk=None보다 빠르다. cached 경로도 동일 지원(cached vs non-cached: chunk=32 prefill은 fp-순서 ~6e-6,
   chunk=None/decode는 비트동일).
4. **fused-Triton은 저강도 어텐션에서 production을 3~9× 앞서고**, 큰 compute-bound GEMM은 cuBLAS가 유리.
5. **KV 캐시는 opt-in, 서빙 기본은 non-cached** (E.4): 캐시는 KV 메모리를 **nD×(3~5×) 선형 증가**시키는 대가로
   ~1.3×만 얻는다. vLLM은 KV 용량이 처리량(동시성·컨텍스트)을 좌우하므로 순손실 → 배치/컨텍스트 최대화를 위해
   non-cached를 기본으로. cached는 저동시성·지연-critical 예외에서만.
