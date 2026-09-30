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

> **NOTE (2026-07-02, 병렬 peel 재측정):** 이제 **양쪽 다 병렬 balanced-digit peel 적용본**이다 — mine(독립
> GEMM)은 공유 `_emit_peel`, production(`ozaki1_batched_gemm_fp`, `~/ozaki_npu`)은 CUDA `wbit_super_encode`
> /Triton fold/torch split. **정확도(A.1/A.2)·mine==prod 비트동일성은 불변**(bit-exact 최적화).
>
> **속도 영향 — 통제 측정(serial vs parallel `.so`, idle gpu, min-over-windows) 결과:**
> - **production은 ~1.0× (변화 없음)**: 실제 DeepSeek-7B linear(qkv/o/gate_up/down)·attn QK/PV 전부 0.99–1.03×.
>   production encode는 **메모리-bound**(K×N weight+nD plane 스트리밍)이라 자릿수 split(소수 ALU op)을 병렬화해도
>   메모리 뒤에 숨어 이득 없음. (초기 ad-hoc run의 2.5–3.1×는 GPU 경합/콜드스타트 아티팩트 — 통제 측정으로 정정.)
> - **mine(fused flash)은 ~1.2×**: flash 커널은 ALU-bound(ncu Part F)라 같은 peel이 실제 이득. → mine의 기존
>   어텐션 우위(A.3, p/m ~3×)가 peel로 소폭 더 벌어짐(mine만 ~1.2× 당겨짐, prod 불변).
>
> 즉 **같은 코드 변경, 반대 roofline → 반대 결과**. A.3/A.4의 절대 ms 표는 pre-peel 측정이며 지연-bound라
> idle-GPU min-over-windows로만 신뢰(재실행 시 mine은 ~1.2× 하향, prod 거의 불변).

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

### D.1 mine vs prod op-by-op + 커널 분해 (torch profiler, idle gpu, 2026-07-02)
같은 ozaki1_fp A@B(둘 다 병렬 peel, bit-exact)를 **연산 단위로** 추적:
- **mine** (`oz1fp_gemm_cg`): **커널 1개**. program당 K-chunk 루프에서 A/B를 **한 번 읽어** 인라인 인코드(peel)
  → #G개 plan-dot를 **레지스터 `cacc`에 누적** → `acc += cacc·sA·sB` → C 1회 store. 중간결과 HBM 왕복 없음.
- **prod** (`ozaki1_batched_gemm_fp`): **커널 #G+3개**. `wbit_super_encode_A`+`_B`(CUDA 인코드) → **rectangle당
  cuBLAS bf16 GEMM #G개**(각각 super-plane 피연산자를 HBM에서 재-read, [b,m,n] fp32 partial을 HBM에 write) →
  `combine_cast`(#G개 partial을 재-read해 가중합).

**커널별 시간 (us/call, w4 nmp10):**
| | ATTN QK (Z28 M1024 K128 N1024) | LINEAR gate_up (M2048 K3584 N37888) |
|---|---|---|
| **mine** | **706** (1 커널) | 51369 (1 커널) |
| **prod** | **2523** (8 커널) | 28402 (8 커널) |
| prod 분해 | cuBLAS×5 1137 (45%) · **combine 955 (38%)** · encA 318 · encB 113 | cuBLAS×5 21428 **(75%)** · encB 3773 · combine 2519 · encA 683 |

**속도차 원인 (roofline 의존, 정반대):**
- **어텐션(저강도) → mine 3.6× 승**: prod의 `combine_cast`(38%, 955us) 하나가 mine 커널 전체(706us)보다 크다 —
  prod는 5개 partial [28,1024,1024] fp32(~600MB)를 HBM에 쓰고 combine에서 되읽는 **순수 HBM 왕복**을 지불하는데,
  mine은 5개 dot를 레지스터 `cacc`에 누적해 그 왕복을 **완전히 제거**. + prod는 작은 GEMM에 cuBLAS 5회 launch.
- **linear(compute-bound) → prod 1.8× 승**: prod의 cuBLAS×5(75%)가 **256×128 타일 + 32×3 파이프**로 near-peak.
  mine은 nD개 자릿수 평면이 상주해 BM=BN=64/CHUNK=32 작은 타일밖에 못 써 compute 효율↓ → 51.4ms vs 28.4ms.
- **decode M=1(메모리/launch-bound) → mine 다시 승**(linear에서도 2.2–3.1×): M=1이면 cuBLAS GEMM도 GEMV(작음)라
  prod의 #G launch+partial+combine 오버헤드 > mine의 단일 fused 커널.

즉 **fused(mine)는 HBM 왕복·launch를 없애 저강도/decode에서 이기고, slot-cuBLAS(prod)는 big-tile near-peak로
compute-bound linear prefill에서 이긴다.** (같은 이유로 flash_ozaki 어텐션 백엔드는 fused, linear는 production을 쓴다.)

### D.2 mine이 큰 타일을 못 쓰는 원인 — 레지스터 벽 (측정: `tile_wall.py`, w4 nmp10 nD4 #G5)
mine의 program당 **상주 레지스터 상태**가 plain GEMM보다 근본적으로 무겁다:
- **[BM,BN] fp32 accumulator 2개**: `acc`(청크 누적 총합) + `cacc`(청크 내 #G dot을 block-FP 스케일 전에 모으는
  버퍼). plain GEMM은 1개.
- **operand당 nD개 자릿수 평면 상주**: `ap[0..nD-1]`([BM,CHUNK]), `bp[0..nD-1]`([CHUNK,BN]). plain은 1쌍.
- **#G개 dot의 super-digit 중간값**(각 dot이 `super(ap[i0:i1])`=평면 합을 만들어 tl.dot).

**측정된 레지스터/스레드 (64×64 / nw4, 0 spill):** #G/nD에 비례해 증가 →
`nD1 #G1: 128r · nD2 #G1: 128r · nD3 #G4: 207r · nD4 #G5: 226r · nD5 #G6: 223r`.
즉 정확도 sweet-spot(nmp9/10/15, #G4~6)은 이미 **255 상한 근처(207~226r)**.

**128×128로 키우면** accumulator만 acc+cacc `2×[128,128]fp32` = nw8에서도 128 r/thread → 평면·중간값 얹으면:
| tile/nw | regs | spill | shmem | time | 결과 |
|---|---|---|---|---|---|
| 64×64 / nw4  | 226 | 0   | 32KB | **52.4ms** | 유일한 무-spill big-ish, 최적 |
| 128×64 / nw4 | 255 | 14  | 52KB | 60.9ms | 상한 도달→spill |
| 128×128 / nw4| 255 | **250** | 64KB | **178ms** | 대량 spill (3.4× 악화) |
| 128×128 / nw8| 255 | 26  | 84KB | 53.3ms | 여전히 spill, 이득 없음 |
| 256×128 / nw8| — | — | 131KB | **OOM** | 공유메모리 초과(>101KB) |

**대조:** #G=1 config(nmp1 w8 / nmp4·16 w4, 평면 1쌍·dot 1개)는 128×128이 **220r·0 spill로 들어간다** → 큰 타일
가능. plain bf16 Triton도 128×128 = cuBLAS의 0.96×(Part D). 즉 **큰 타일을 막는 건 GEMM 구조가 아니라 ozaki
자릿수-평면 상태(2 accumulator + nD 평면 + #G dot)** — nmp10/15는 그 상태 때문에 64×64에 고정되고, 64×64 MMA는
compute-bound GEMM에서 cuBLAS의 256×128(32×3 파이프) 대비 효율이 낮아 12.8× cuBLAS / 1.8× prod가 된다.

**함의:** ① 저강도 어텐션·decode는 GEMM이 병목이 아니라(메모리/launch) 64×64로도 충분 → mine의 fusion이 이김.
② compute-bound linear만 big-tile 효율이 지배 → 레지스터 벽 때문에 mine이 짐. (cacc를 없애고 per-dot 스케일로
acc에 직접 누적하면 accumulator 1개로 ~64r 절약해 128×64 정도는 가능하나, nD 평면·#G dot 압력은 남고 prod의
cuBLAS가 이미 효율 상한이라 compute-bound에서 mine이 prod를 이기진 못함 → 벽은 fused 자릿수-평면 방식의 구조적 성질.)

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

## Part G — flash-exact vs **vllm-flash-attn** 단계별 오차 분해 + decode P-cast 제거 (`analyze_vs_vllm_flash_attn.py`, 2026-08-06, A6000)

동기: MATH-500에서 우리 triton 경로(0.914~0.938)가 native `vllm-flash-attn`(0.948)에 7/7 런 모두 뒤졌다.
커널 결함인지 노이즈인지 가리려고 **fp64 기준으로 단계별 분해**했다. (Part B.1은 SDPA와만 비교했었다.)

### G.1 단계 분해 (decode B=8 N=2048, relerr vs fp64) — 오차는 전부 bf16 캐스트 2개
| 단계 | relerr |
|---|---|
| fp64 tiled online softmax (tile=32) | 1.16e-15 |
| + fp32 accumulator (m/l/acc) | 2.45e-07 |
| **+ P → bf16 (P@V 직전)** | **1.52e-03** |
| **+ bf16 출력 저장** | **2.24e-03** |
| 같은 경로 tile=128 (FA식) | 2.24e-03 |
| tile 32 vs 128 (캐스트 없이) | 2.68e-07 |

→ `chunk=32`로 인한 작은 BLOCK_N, online-softmax 리스케일 횟수, GQA head-fold는 **1e-7대 = 오차원이 아니다.**

### G.2 갈리는 지점은 **decode 전용**, N에 비례 (20 draws/shape)
| case | ours(구) | vllm-FA | ratio | ours 열세 |
|---|---|---|---|---|
| PREFILL T=N=92 | 1.861e-3 | 1.868e-3 | **0.997** | 0/20 |
| PREFILL T=N=512 | 1.959e-3 | 1.967e-3 | **0.996** | 0/20 |
| DECODE T=1 N=512 | 2.189e-3 | 2.158e-3 | 1.014 | 20/20 |
| DECODE T=1 N=4368 | 2.268e-3 | 2.168e-3 | **1.047** | 20/20 |

prefill은 오히려 우리가 근소 우위. decode에서만 뒤지고 컨텍스트가 길수록 벌어진다. 부호 편향은 양쪽 다
≤1e-5로 **무편향** — 즉 결함이 아니라 반올림 분산 차이다.

### G.2b FA는 P를 몇 비트로 나르는가 — 가수 비트 스윕 (decode N=2048)
P를 k개 유효비트로 반올림한 시뮬레이터와 대조:

| P 유효비트 | relerr | |
|---|---|---|
| 7 | 3.469e-3 | |
| **8 (= bf16)** | **2.236e-3** | **← vllm-FA(2.163e-3)와 같은 급** |
| 9 | 1.807e-3 | |
| 10 | 1.682e-3 | |
| 13~16 | **1.640e-3** | 포화 — 여기가 **bf16 출력 저장** 바닥 |

**→ FA도 P를 bf16으로 떨군다.** (초기에 "FA는 P를 bf16까지 안 떨군다"고 추정했으나 이 스윕이 반증했다.)
구-ours 2.235e-3과 FA 2.163e-3은 **같은 8비트-P 급 안에서의 ~3% 차이**이고, 그 원인은 dot 내부 MMA 누산
순서·타일 경계 등 Triton이 추상화해 버리는 스케줄 차이다. 정밀도 클래스 차이가 아니다.

### G.2c 진짜 원인 — FA decode는 **split-KV(flash-decoding)**를 쓴다
`flash_attn_with_kvcache`가 `num_splits`를 노출하므로 직접 고정해 봤다 (decode B=8 N=2048):

| FA `num_splits` | 1 | 2 | 4 | 8 | 16 | auto |
|---|---|---|---|---|---|---|
| relerr | **2.2339e-3** | 2.2226e-3 | 2.2007e-3 | 2.1625e-3 | 2.1410e-3 | **2.1625e-3** (=8) |

- **`num_splits=1`이면 우리 단일패스 경로(2.2353e-3)와 사실상 동일.** 즉 dot도 softmax도 아니고, **split 수가
  격차의 전부**였다. auto는 8을 골랐고 그게 정확히 앞서 측정한 2.163e-3이다.
- 시뮬레이터로도 재현된다(P→bf16 고정, n_splits=1/2/4/8/16 → 2.236/2.217/2.180/2.137/2.074e-3).
- **exp2 vs 자연지수는 무관**(2.2366 vs 2.2360e-3) — FA2가 `exp2`를 쓰는 건 속도용이지 정확도 요인이 아니다.
- **BLOCK_N도 무관**(32/64/128/256 → 2.236/2.241/2.238/2.247e-3).
### G.2d 왜 split이 많을수록 정확한가 — **bf16 반올림의 값 의존성** (누산 사슬 길이가 아니다)
"사슬이 짧아져서"가 직관이지만 **틀렸다.** 항을 하나씩 꺼서 측정 (B=8 N=2048, S=split 수):

| datapath | S=1 | S=2 | S=4 | S=8 | S=16 | S=32 |
|---|---|---|---|---|---|---|
| P→bf16 + bf16 출력 (실제 커널) | 2.236e-3 | 2.217e-3 | 2.180e-3 | 2.137e-3 | 2.074e-3 | 2.042e-3 |
| **P 무손실**, bf16 출력 | 1.640e-3 | 1.640e-3 | 1.640e-3 | 1.640e-3 | 1.640e-3 | **1.640e-3** |
| P→bf16, fp32 출력 | 1.517e-3 | 1.478e-3 | 1.440e-3 | 1.362e-3 | 1.290e-3 | 1.230e-3 |
| P 무손실, fp32 출력 (fp32 누산만) | 2.45e-7 | 2.02e-7 | 1.86e-7 | 1.80e-7 | 1.84e-7 | 2.05e-7 |

- **P를 무손실로 두면 split 의존성이 완전히 사라진다**(1.640e-3 고정). fp32 누산 사슬은 2e-7로 애초에 무관.
  즉 이득은 **오직 P→bf16 항**에서 온다. `l`을 반올림된 p로 계산해 분자/분모 공통모드를 만들어도 변화 없음.
- 그런데 상대 오차는 이후 `alpha` 재스케일을 그대로 통과하므로 **이론상 split과 무관해야 한다.** 실제로
  bf16 캐스트를 **값과 무관한 ±2^-9 상대오차 모델**로 바꾸면 split 의존성이 **완전히 사라진다**:

  | 반올림 모델 | S=1 | S=8 | S=32 |
  |---|---|---|---|
  | 진짜 bf16 캐스트 | 2.236e-3 | 2.137e-3 | 2.042e-3 |
  | 균일 ±2^-9 상대오차(모델) | 1.760e-3 | 1.762e-3 | **1.763e-3** (평탄) |

- **따라서 원인은 bf16 반올림이 값에 의존한다는 것 하나다.** 구체적으로: split마다 자기 local max로 정규화하니
  **각 split의 최대 항이 정확히 `p = exp(0) = 1.0`** 이 되고, 1.0은 bf16으로 **정확히 표현**되어 ε=0이다.
  그리고 그 항은 그 split에서 **가중치가 가장 큰 항**이다. 측정된 "정확히 표현되는 p의 비율":

  | | S=1 | S=8 | S=16 | S=32 |
  |---|---|---|---|---|
  | ε=0인 p 비율 | 0.0023 | 0.0105 | 0.0164 | **0.0235** |
  | 그중 `p==1.0` 비율 | 0.00049 (=1/2048) | — | — | 0.01562 (=32/2048) |

  가수 위치 분포 자체는 거의 안 변한다(mean 1/mantissa 1.4425 → 1.4489, 오히려 미세하게 나빠짐).
  즉 **"오차 0으로 고정되는 앵커 항이 split 수만큼 늘어나는" 효과**이지, 알고리즘이 근본적으로 나아지는 게 아니다.
- 규모도 작다: S=1→32에서 전체 오차 −8.7%. bf16 출력 저장(1.640e-3)이 남는 한 바닥은 못 넘는다.

### G.3 수정: exact 경로에 **split-KV 도입** + split 수를 **ozaki 경로와 공유**
`_flash_exact_split_fwd` 추가 — 생성형 `_flash_split`의 순수 bf16 쌍둥이. split마다 fp32 `(m, l, acc)` 부분합을
쓰고 기존 `_flash_combine`의 fp32 LSE 병합을 재사용한다.

**split 수는 `decode_num_splits()` 하나를 ozaki 경로와 대조군이 공유한다** (= 기존 ozaki 휴리스틱
`min(ceil(16·SM/zc), n_tiles, 32)` 그대로). G.2d에서 보듯 split 수는 **수치적으로 중립이 아니므로**, 대조군이
피험군과 다르게 쪼개면 앵커 아티팩트를 서로 다르게 받는다 — 실제로 그 전까지는 ozaki가 32-way, 대조군이
단일패스라 **대조군이 더 부정확한 팔**이었다(아래 표). 공유하면 ozaki−대조군 차이가 digit-plane GEMM 하나로
귀속된다. FA 수준에 맞추는 `fa_num_splits()`(= min(ceil(n_tiles/4), ceil(1.5·SM/zc)))는 **FA와 비교할 때만**
쓰는 별도 함수로 남겨 둔다.

| shape | splits | ozaki(w4 nmp10) | **대조군(공유)** | 대조군(FA 수) | vllm-FA | ozaki/대조군 |
|---|---|---|---|---|---|---|
| B=8 N=512 | 16 | 2.018e-3 | **2.012e-3** | 2.062e-3 | 2.109e-3 | 1.003 |
| B=8 N=2048 | 32 | 2.093e-3 | **2.042e-3** | 2.137e-3 | 2.163e-3 | 1.025 |
| B=32 N=2048 | 21 | 2.118e-3 | **2.082e-3** | 2.207e-3 | 2.223e-3 | 1.017 |
| B=8 N=4368 | 32 | 2.192e-3 | **2.119e-3** | 2.195e-3 | 2.200e-3 | 1.034 |
| B=32 N=4368 | 21 | 2.188e-3 | **2.129e-3** | 2.246e-3 | 2.214e-3 | 1.028 |

- **ozaki/대조군 > 1** = ozaki 팔이 공통 바닥 위에 자기 양자화 오차를 얹고 있다는 뜻 — 통제 실험의 올바른 방향.
  그 잔차 **+0.3~3.4%가 곧 digit-plane GEMM의 기여분**이다.
- 수정 전에는 방향이 거꾸로였다: B=8 N=2048에서 ozaki(32) 2.093e-3 vs 대조군(단일패스) 2.235e-3 → **비 0.936**,
  즉 피험군이 대조군보다 정확했다. 0.926을 낸 bf16 런(9cf34c33)이 바로 그 단일패스 대조군이다.
- **ozaki 경로 수치는 불변**: 공유 함수가 기존 인라인 식과 8개 shape 전부에서 동일한 split 수를 반환함을 확인.

#### FA와의 비교 (원인 규명 시점의 데이터, `fa_num_splits` 사용)

| shape | splits | 단일패스 | **split-KV** | vllm-FA | 신/FA | ours 열세 |
|---|---|---|---|---|---|---|
| B=2 N=512 | 4 | 2.193e-3 | **2.112e-3** | 2.163e-3 | 0.976 | 0/20 |
| B=8 N=512 | 4 | 2.189e-3 | **2.106e-3** | 2.158e-3 | 0.976 | 0/20 |
| B=2 N=2048 | 16 | 2.246e-3 | **2.086e-3** | 2.148e-3 | 0.972 | 0/20 |
| B=8 N=2048 | 8 | 2.252e-3 | **2.133e-3** | 2.169e-3 | 0.983 | 0/20 |
| B=32 N=2048 | 2 | 2.252e-3 | **2.220e-3** | 2.230e-3 | 0.996 | 1/20 |
| B=4 N=4368 | 16 | 2.268e-3 | **2.139e-3** | 2.168e-3 | 0.987 | 1/20 |
| B=16 N=4368 | 4 | 2.274e-3 | **2.220e-3** | 2.222e-3 | 0.999 | 7/20 |
| B=2 N=16384 | 32 | 2.312e-3 | **2.186e-3** | 2.195e-3 | 0.996 | 7/20 |

- split 수를 FA에 맞추면 **구 −1.4~−4.7% 열세 → +0.1~+2.8%**로 같은 수준에 들어온다. 다만 이건 "FA와
  비교"용이고, **서빙 기본값은 위의 공유 휴리스틱**이다(그쪽이 통제 실험에 맞다).
- 남은 ~2%는 동일 split 수에서도 남으므로(B=2 N=512: 양쪽 4 splits, 0.976) FA 내부 잔여 요소다. 컴파일된
  `.so`뿐이라 그 이상은 외부에서 특정 불가 — G.4 참조.
- **CUDA 커널은 손대지 않았고 손댈 수도 없다**: `vllm_flash_attn`은 `_vllm_fa2_C.abi3.so` 바이너리로만 존재한다.
  이번 변경은 전부 우리 Triton 커널이고, 그중에서도 **exact 대조군만** 바뀌었다(ozaki 경로는 수치 불변).
- **속도는 덤으로 큰 이득**: B=8 N=2048 0.194→0.042ms(**−79%**), B=8 N=4368 0.478→0.098ms(−80%). exact decode의
  grid가 `B·Hkv`뿐이라 SM을 못 채우던 걸 split이 메운다. 큰 배치(zc≥0.8·SM)는 splits=1로 떨어져 영향 없음.
- `n_splits=1`은 단일패스와 **비트동일**(max diff 0.00e+00). `kv_lens` 패딩 마스킹도 누수 없음(per-seq 대비
  최대 ~2 bf16 ulp — split 수가 달라 fp 순서만 다름).

### G.3b `PV_SPLIT`(P를 hi/lo bf16 쌍으로)은 **기본 OFF**
`p_hi = bf16(p)`, `p_lo = bf16(p - p_hi)` → p를 ~2^-17로 재현. decode 오차를 **1.640e-3**(= G.2b 스윕의 포화
바닥 = bf16 **출력 저장**만 남은 상태)까지 낮추지만, **FA도 P를 bf16으로 떨구므로(G.2b) 이걸 켜면 대조군이
FA보다 구조적으로 더 정확해진다**(0.77×). 대조군으로서 부적격이라 기본 OFF, P 항을 따로 연구할 때만 opt-in.
ozaki 경로는 어느 쪽이든 플래그 무시 — production `ozaki1_batched_gemm_fp` 비트동일성 유지.

### G.4 Triton으로 CUDA FA와 **bit-exact**를 만들 수 있는가 — 아니오, 그리고 원하는 목표도 아니다
현재 출력 bf16 원소 일치율(decode N=2048): 구 **53.9%**, 신 **60.9%**(1 ulp 이내 84.5% → 88.2%).
정확해질수록 FA와의 비트 일치가 **올라간다** — 둘 다 참값이라는 같은 끌개로 수렴하기 때문.
(원소별 ulp의 max는 부호비트 인코딩 때문에 0 근처 부호 반전에서 무의미하게 커지므로 해석하지 말 것.)

100% 비트일치는 실무적으로 불가능하다. 맞춰야 할 것이:
1. **`tl.dot` 내부 MMA 누산 순서** — Triton 백엔드가 MMA 레이아웃과 K-스텝 순서를 정한다. BLOCK_M/N을 같게
   해도 CUTLASS 파이프라인과 fp32 누산 트리가 달라 마지막 비트가 갈린다. Triton은 이걸 **의도적으로 감춘다**.
2. **softmax 형태** — FA2는 scale을 Q에 접어 넣고(`q*scale*log2e`) `exp2`를 쓴다. Triton `tl.exp`는 내부
   `*log2(e)` 후 `ex2.approx.f32`(2 ULP 근사)로 내려가, 곱셈 위치가 달라 결과 비트가 다르다.
3. **decode split-KV** — 구조(per-split fp32 m/l/acc + fp32 LSE 결합)와 split 수는 G.3에서 맞췄지만, FA의
   휴리스틱 **상수**는 복원 불가다: 여러 `num_splits` 요청이 같은 분할로 접혀(등가류) 비트매칭으로도 auto의
   정확한 값이 아니라 구간만 나온다. 남은 ~2%가 여기서 온다.
4. 헤드차원·아키텍처별 `kBlockM/kBlockN` 선택도 FA 내부 결정이다.

즉 bit-exact = **명령어 스케줄 수준에서 FA를 Triton으로 재구현**하는 일이고, Triton의 추상화가 막는 지점이다.

더 중요한 건 **목표로도 틀렸다**는 것:
- bit-exact는 FA의 **오차까지 그대로 재현**한다는 뜻이다. 방금 없앤 P→bf16 캐스트를 도로 넣어야 한다.
  정확도와 정반대 방향이다.
- ozaki 효과를 격리하는 **대조군은 FA가 아니라 우리 자신의 flash-exact**여야 한다(`--ozaki_flash_exact`).
  같은 커널·같은 타일링·같은 softmax에서 QK/PV dot만 바뀌므로 이미 apples-to-apples다. FA와의 비트일치는
  이 실험에 아무것도 보태지 않는다.
- 도달 가능한 유일한 의미 있는 목표는 **참값(fp64)과의 거리**이고, 그건 이미 바닥(1.640e-3 = bf16 출력)이다.
  더 내리려면 어텐션 출력 dtype을 fp32로 바꿔야 하는데 그건 모델 계약을 바꾸는 일이다.

### G.5 정확도 갭에 대한 판정
**이 수정으로 MATH-500 0.926 vs 0.948이 설명되지는 않는다.** 두 런은 이미 2.2e-3 무편향 섭동으로 완전히 다른
토큰열을 만들며(발산 포화: native↔triton 불일치 문항 22.6개 ≈ triton끼리 25.7개), 섭동을 4.7% 줄인다고
정확도가 2.2pp 움직일 근거가 없다. triton 계열 내부 21쌍 중 McNemar p<0.05는 1개(α=0.05 기대값과 동일)로
계열 내부는 완전 대칭. 깨끗한 대조군(bf16 EXACT)의 p=0.035는 7회 다중비교를 못 넘는다. **결정적 한계는
native 표본이 1개**라는 것 → native seed 리플리킷 2~3개가 유일한 판정 실험.

---

## Part H — `gemm_bits`(w)를 2~8 임의 값으로 쓸 수 있는가 (2026-08-06, A6000)

동기: 스윕은 w=4로 돌고 있는데 **w=5**도 같은 커널로 되는지, 정확도가 이론과 달라지는 데가 없는지.

### H.1 결론: **된다.** w=5는 production과 비트일치하고 int_bits 법칙을 그대로 따른다
codegen GEMM vs production `ozaki1_batched_gemm_fp(gemm_bits=w)`, chunk=32, [1,256,256]×[1,256,256]:

| w | nmp | nD | int_bits | mine vs **prod** | mine vs fp64 | prod vs fp64 |
|---|---|---|---|---|---|---|
| 5 | 1 | 1 | 4 | **0.00e+00** | 9.37e-2 | 9.37e-2 |
| 5 | 3 | 2 | 9 | **0.00e+00** | 5.15e-3 | 5.15e-3 |
| 5 | 4 | 2 | 9 | **0.00e+00** | 2.72e-3 | 2.72e-3 |
| 5 | 6 | 3 | 14 | **0.00e+00** | 1.97e-4 | 1.97e-4 |
| 5 | 9 | 3 | 14 | 6.8e-08 | 1.60e-5 | 1.60e-5 |
| 5 | 10 | 4 | 19 | 4.4e-08 | 8.81e-6 | 8.81e-6 |
| 5 | 16 | 4 | 19 | 7.3e-08 | 1.04e-7 | 8.24e-8 |

(~1e-8은 fp32 누산 순서 차이. w=4/w=8과 동일한 품질.)

**정확도는 int_bits = w·nD − 1 하나로 결정되고 w=5도 예외가 아니다.** 어텐션 relerr(decode B=8 N=2048):
`int_bits 3(w4nmp1) 3.20e-1 → 4(w5nmp1) 1.65e-1 → 7(w4nmp4) 2.42e-2 → 9(w5nmp4) 5.62e-3 →
11(w4nmp9) 2.43e-3 → 14(w5nmp9) 2.24e-3 → 15+ 2.24e-3(bf16 바닥)`. **int_bits가 같으면 값도 같다**
(w4 nmp4 = w8 nmp1 = 2.417e-2, w4 nmp16 = w8 nmp4 = 5.71e-6 — 정확히 일치).

불변식도 전부 성립(w=2~8, nmp 1~16): top digit ∈ [−2^(w−1), 2^(w−1)], 하위 digit 동일, 평면 bf16 **정확**,
Σ평면 == xI **무손실**, max|d_i·d_j|·K = 8192(w5) < 2^24 → fp32 dot 누산 **정확**.

### H.2 유일한 실질 비용: **w가 8을 나누지 않으면 팩킹이 사라진다** (`g = 8 // w`)
bf16 유효비트가 8이라 super-digit span은 `g = 8//w`로 제한된다 → **w∈{2,4,8}만 정수 분할**:

| nmp | w=4 (g=2) | w=5 (g=1) | w=8 (g=1) |
|---|---|---|---|
| 4 | **1 dot** | 4 dots | 4 dots |
| 9 | **4 dots** | 9 dots | 9 dots |
| 16 | **4 dots** | 16 dots | 16 dots |

즉 w=5는 **bf16 에뮬레이션 비용이 w=8과 같고 w=4의 최대 4배**다. 단 이건 **에뮬레이션 비용**이지
모델링 대상 HW의 비용이 아니다 — 실제 5-bit MAC 하드웨어에서 w5 nmp4(5-bit MAC 4회)가
w4 nmp6(4-bit MAC 6회)보다 유리할 수 있다. 두 축을 섞지 말 것. (참고 g: w2→4, w3→2, w5/6/7→1.)

### H.3 이 스윕에서 드러난 **기존 버그**: w=8 nmp≥10은 조용히 깨진다
커널은 블록-FP 정수를 **int32**로 peel한다. `int_bits = w·nD−1`이 31에 닿으면 `2^int_bits + peel_bias`가
int32를 넘는다:

| w | nmp | int_bits | mine vs prod | prod vs fp64 |
|---|---|---|---|---|
| 8 | 10 | 31 | **5.88e-02** | 5.35e-08 |
| 8 | 16 | 31 | **5.88e-02** | 5.35e-08 |
| 8 | 15 | 39 | **1.00e+00** | 5.35e-08 |

**production은 멀쩡하고 우리 Triton 커널만 틀린다.** 예외도 NaN도 없이 틀린 값을 반환하므로 그동안
드러나지 않았다(Part A는 w4 nmp9/16, w8 nmp1, w2 nmp16만 검증 — nD≥4의 w8 구석은 미검증).
**`assert_int32_peel_fits(nmp, w)`** 를 추가해 `_gen_src` / `_gen_split_src` / standalone `_gen_gemm_src`
에서 막았다. 차단되는 조합은 **w=8 nmp∈{10,15,16}과 w=7 nmp=15뿐**이고, w≤6은 전 nmp 안전
(w=4 최대 int_bits 19, w=5 최대 24).

### H.4 실행하려면
`inference_vllm.py`의 `--gemm_bits`가 `choices=[2,4,8]`이라 5를 거부했다 → **2~8로 확장**했다.
production은 원래 `assert 2 <= w <= 8`이고 nmp 집합(`{1,3,4,6,9,10,15,16}`)은 w와 무관하므로 그 외
변경은 필요 없다. 스모크(nmp=4, decode): w=2/3/4/5/6/7/8 → relerr 3.16e-1 / 8.84e-2 / 2.42e-2 /
**5.60e-3** / 2.39e-3 / 2.20e-3 / 2.21e-3 — 전부 정상, int_bits 순으로 단조.

### H.5 fp16 packing 경로 (`pack_dtype="fp16"`) — w=5의 팩킹 복구 (2026-08-06)
bf16 유효비트 8 → `g = 8//w`가 w=5에서 1이 되는 게 H.2의 유일한 손실이었다. **fp16은 유효비트 11**이라
`g = 11//w = 2`. 다만 fp16은 지수가 5비트뿐(max 65504)이라 **절대-place fold를 못 쓴다**(plane = 2^int_bits →
int_bits≥16에서 inf). 그래서 fp16 경로는 **production 규약**을 쓴다: rectangle 내부 **상대 place** +
외부 `2^(w(i0+j0))`를 fp32 누산기에 적용. 상대 super-digit 최대는 maxS(≤682)뿐이라 지수 문제가 사라진다.

구현: `pack_dtype` 인자를 `pack_plan / _emit_peel / _super / _emit_dots / _gen_src / _get_kernel /
flash_oz1fp_cg`(+ standalone GEMM)에 관통. 상대 fold는 **int32에서 합친 뒤 한 번만 캐스트**한다 —
fp16 텐서에 파이썬 float를 곱하면 Triton이 fp32로 승격시켜 fp16 텐서코어를 놓친다.

```
cacc += tl.dot(((qd0 + qd1 * 32).to(tl.float16)), tl.trans(((kd0 + kd1 * 32).to(tl.float16))),
               out_dtype=tl.float32) * 1.0        # w=5 nmp=4: dot 4개 -> 1개
```

**dot 수** (w=5): nmp 3→3/2, 4→**4/1**, 6→6/3, 9→9/4, 10→10/5, 15→15/6, 16→**16/4**.

### H.6 bf16 경로와 bit-exact 한가 — **정확한 구간에서는 완전히 일치, 아니면 fp32 바닥에서 갈린다**
GEMM 레벨(fp32 출력이라 bf16 출력 반올림에 가려지지 않음). `exact` = 같은 자릿수 다항식을 int64/fp64로 평가:

| w | nmp | int_bits | dots bf/fp | **bf16 vs fp16** | bf16 vs exact | fp16 vs exact | prod vs exact |
|---|---|---|---|---|---|---|---|
| 5 | 1 | 4 | 1/1 | **BIT-EQ** | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 5 | 3 | 9 | 3/2 | **BIT-EQ** | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 5 | 4 | 9 | 4/1 | **BIT-EQ** | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 5 | 6 | 14 | 6/3 | **BIT-EQ** | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 5 | 9 | 14 | 9/4 | 5.25e-08 | 5.08e-08 | **4.43e-08** | 2.59e-08 |
| 5 | 10 | 19 | 10/5 | 5.67e-09 | 2.41e-08 | 2.42e-08 | 2.07e-08 |
| 5 | 15 | 24 | 15/6 | 6.01e-08 | 5.43e-08 | **4.63e-08** | 2.59e-08 |
| 5 | 16 | 19 | 16/4 | 6.38e-08 | 5.64e-08 | **4.70e-08** | 2.60e-08 |

- **패턴이 명확하다: `bf16 vs exact = 0.00e+00`인 셀에서 두 경로는 BIT-EQ**이고, 각 경로가 **개별적으로**
  fp32 정확성을 잃는 지점(≈1e-8)부터 서로 갈린다. 즉 **팩킹 그룹핑이나 dtype 때문이 아니라**, 둘 다 부정확해진
  뒤 누산 순서가 달라서 갈리는 것이다 — 차이는 **fp32 엡실론 수준(~5e-8)**.
- 갈리는 셀에서는 **fp16/상대 경로가 항상 exact에 더 가깝다**(4.43 vs 5.08, 4.63 vs 5.43, 4.70 vs 5.64e-8)
  — 상대 place가 중간값을 작게 유지하고 dot 수도 적어 누산 단계가 짧기 때문. production(2.6e-8)에도 더 가깝다.
- 어텐션 레벨(bf16 출력)에서는 w=5 nmp **1/3/4/6이 `torch.equal` 완전 일치**, 9/15/16은 불일치, 10은
  대부분 일치(차이가 bf16 출력 ULP 아래).
- **w=4는 영향 없음**(g=2 동일): 기본값 == bf16 경로 비트동일, nmp 4/9는 fp16과도 비트동일.

> ⚠️ 구현 중 발견한 함정: `p = p.to(bf16)`(P→bf16 truncation)를 pack_dtype에 묶으면 안 된다. 그건 **실제
> flash-attention 데이터패스 모델링**이지 팩킹 세부사항이 아니다. 처음에 같이 fp16으로 바꿨더니 P가 3비트
> 더 정확해져 relerr이 2.24e-3 → 1.65e-3(= P-무손실 바닥)으로 떨어지며 대조가 깨졌다. 항상 bf16 고정.

### H.7 속도 (decode, ms/call, w=5)
| shape | nmp | bf16 | fp16 | |
|---|---|---|---|---|
| B=8 N=2048 | 4 | 0.641 | 0.550 | **1.17×** |
| B=8 N=2048 | 9 | 0.719 | 0.725 | 0.99× |
| B=8 N=2048 | 16 | 0.995 | 0.730 | **1.36×** |
| B=32 N=2048 | 4 | 0.708 | 0.516 | **1.37×** |
| B=8 N=4368 | 16 | 1.992 | 1.512 | **1.32×** |

dot 수가 4배 줄어도 1.2~1.4×에 그치고 **nmp=9는 이득 없음**: 상대 fold가 rectangle마다 int32 super를
재계산하는데 이 커널은 **정수 ALU-bound**(Part F)라 dot 감소분을 ALU 증가가 상쇄한다. 겹치는 super를
호이스팅하면 더 나올 여지가 있다(미구현).

---

## 핵심 결론
0. **직접 프로파일링(Part F)**: ncu로 병목을 파이프 레벨 규명 — exact는 prefill=텐서 / **decode=DRAM 96%(메모리
   bound)**, ozaki는 둘 다 **정수 ALU-bound**(int8 SW 에뮬레이션)로 저-occupancy라 은닉 실패 → prefill 6×/decode 4×.
   MATH-500 병목은 **decode**(작업량 3333× prefill). 병렬 balanced-digit peel로 ALU를 줄여 **bit-exact 1.2×**(decode
   포함) 달성, ncu로 ALU% 감소 확증. 추가로 **split-KV/flash-decoding**(`flash_oz1fp_cg_splitkv`)으로 decode
   occupancy를 올려 작은 배치에서 **최대 3.8×**(B=8), 큰 배치 1.35–1.46× (S=1은 non-split과 비트동일).
1. **Faithful (chunk=32)**: 역양자화 피연산자는 production과 **비트동일**, 출력은 **fp32 누산 바닥**까지 일치
   (정확도 `== production`). int8-HW 자릿수 범위도 지켜진다.
2'. **flash-exact를 vllm-flash-attn과 같은 구조로 정렬(Part G)**: fp64 기준 단계 분해 결과 오차는 전부
   `P→bf16` + `bf16 출력` 두 캐스트이고(타일링·softmax·GQA fold는 1e-7대). FA 대비 decode 열세(−1.4~−4.7%)의
   원인은 **FA decode가 split-KV(flash-decoding)를 쓰기 때문**이었다 — FA를 `num_splits=1`로 고정하면
   2.2339e-3으로 우리 단일패스(2.2353e-3)와 동일. exact 경로에 split-KV(`_flash_exact_split_fwd`)를 도입하고
   split 수를 FA의 실제 선택값에 맞추자(`fa_num_splits`) **+0.1~+2.8% 이내로 수렴**(부호도 혼재), 덤으로 decode
   **−79% 지연**. P를 hi/lo로 쪼개는 `pv_split`은 FA보다 더 정확해져서(0.77×) 대조군 실격 → 기본 OFF.
   prefill·ozaki 경로는 비트동일 유지. **단, MATH-500 0.926 vs 0.948은 이걸로 설명 안 된다**
   (G.5: native 표본 1개가 병목 — 리플리킷 필요).
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
