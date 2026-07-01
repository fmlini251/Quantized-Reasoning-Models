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

### A.3 속도 (Z=28, chunk=32; 지연 ms, `prod/mine` = 독립 커널의 production 대비 배속)
형상은 op별로 다르다(M×K×N, K=reduction):

| op | M×K×N | w/nmp | mine ms | prod ms | prod/mine |
|---|---|---|---|---|---|
| PREFILL QK | 1024×128×1024 | w8 nmp1 (cache) | 0.317 | 0.246 | 0.78× |
| PREFILL QK | 1024×128×1024 | w4 nmp9 (cache) | 0.628 | 2.135 | **3.40×** |
| PREFILL PV | 1024×1024×128 | w4 nmp16 (cache) | 0.677 | 4.978 | **7.35×** |
| DECODE QK | 16×128×2048 | w4 nmp16 (fresh) | 0.123 | 1.109 | **8.99×** |
| DECODE PV | 16×2048×128 | w4 nmp10 (fresh) | 0.270 | 1.161 | **4.30×** |

nmp≥9에서 **production 대비 3~9× 빠르다**. 어텐션 QK/PV는 연산강도가 낮은 GEMM이라 production은 `#GEMM`개
cuBLAS 호출+combine으로 피연산자를 여러 번 재-read하지만, fused 단일 Triton 커널은 한 번 읽고 자릿수쌍 dot을
커널 안에서 누산한다.

### A.4 cached vs non-cached — production(ozaki1_fp)도 gap이 작다 (QK 1024×128×1024, chunk=32)
`nc`=non-cached(매 호출 weight 인코딩), `c`=cached(production weight_cache / 우리 `encode_B`). 지연 ms.
cuBLAS bf16 GEMM(ozaki 없음) = 0.177 ms.

| w/nmp | prod nc | prod c | prod nc/c | mine nc | mine c | mine nc/c | mine_c / prod_c |
|---|---|---|---|---|---|---|---|
| w8 nmp1 | 0.273 | 0.247 | 1.1× | 0.491 | 0.314 | 1.56× | 1.27× (mine 느림) |
| w4 nmp9 | 2.393 | 2.130 | 1.1× | 0.849 | 0.627 | 1.35× | **0.29×** (mine 3.4× 빠름) |
| w4 nmp10 | 3.027 | 2.546 | 1.2× | 1.022 | 1.149 | 0.89× | **0.45×** |

- **production ozaki1_fp의 cache gap도 1.1~1.2×로 작다.** 단일 GEMM 호출 안에서 weight 인코딩은 `#GEMM`개
  cuBLAS 호출 대비 작은 비중이라 캐시 이득이 작다. (production의 "큰 cache 이득"은 **모델-레벨 weight_cache** —
  weight를 전체 추론에서 **한 번만** 인코딩해 모든 토큰/레이어에 재사용 — 로, 여기 단일-GEMM 벤치와는 amortize
  범위가 다르다.)
- **w8 nmp1은 손익분기 (mine 1.27× 느림) — 원인은 activation A의 N-타일 중복 인코딩** (둘 다 bf16 ozaki1_fp,
  int8 무관). 마이크로벤치로 격리(타일 고정 BM=BN=64, grid의 n-타일 수만 변경):
  cuBLAS 0.175 / mine 0.314(오버헤드 **+0.139**). A block-FP 인코딩을 **GEMM처럼 16 n-타일**로 돌리면 **0.130ms**
  (오버헤드 거의 전부), **1회만**이면 0.022ms → fused는 A[m-타일]을 **n-타일마다(=`#GEMM`이 아니라 N/BN=16번)
  재인코딩**해 ~5.8× 부풀린다(448 프로그램에선 A6000 미포화라 16×가 아닌 5.8× wall). dot 자체는 문제 없음
  (순수 Triton GEMM 0.138 ≤ cuBLAS 0.175). production은 A를 **1회**만(별도 패스) 인코딩 후 cuBLAS → `#GEMM`=1
  에선 이 중복 인코딩이 GEMM 전체 비용에 맞먹어 mine이 짐. (앞서 BN 스윕으로 "중복 아님"이라 했던 것은 BN이
  타일크기와 n-타일수를 동시에 바꿔 SRAM 페널티가 가린 것 — 타일 고정 격리로 중복이 지배적임이 확정됨.)
- **nmp≥9에서 역전**: 같은 0.13ms 인코딩이 타일당 9~16개 자릿수 dot에 분산되고, production은 `#GEMM`개 개별
  cuBLAS(2~3ms)를 내야 하므로 mine c가 prod c의 0.29~0.45×(2~3.4× 빠름). non-cached는 더 극적 — mine nc(0.85)가
  prod nc(2.39)의 ~1/3.

---

## Part B — 전체 어텐션: flash-ozaki vs SDPA vs exact flash (`bench_attention_vs_sdpa.py`, chunk=32)

ozaki w4 nmp10, **chunk=32**. flash-exact = 같은 커널 `ozaki=False`(순수 bf16) 경로.

### B.1 정확도 (fp32 exact 대비, chunk=32)
| case | flash-ozaki | flash-exact (bf16) | torch SDPA |
|---|---|---|---|
| PREFILL MHA N=1024 | **1.61e-3** | 1.99e-3 | 1.99e-3 |
| DECODE GQA B=32 N=2048 | **1.68e-3** | 2.25e-3 | 2.25e-3 |

flash-ozaki(nmp10)는 **순수 bf16 어텐션(SDPA/flash-exact)만큼 — 오히려 약간 더 — 정확**하다.

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
- production식 decode 대박 이득을 보려면 **증분 인코딩**(새 토큰 K/V만 O(D) 인코딩해 캐시에 append, 과거 재사용)이
  필요 — 현 bench는 매 호출 전체 N개를 재인코딩하므로 그 이득을 측정하지 않는다(실서빙 KV-cache는 증분).

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
