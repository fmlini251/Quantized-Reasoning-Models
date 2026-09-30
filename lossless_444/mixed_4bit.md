# Track C 실험 스크립트 — 4bit 포맷 혼용 (mixed 4-bit × ozaki1)

> 실험 실행 전용 문서. 이론(digit 비용 유도, lattice 설계 공간, codesign)은
> [notes/mixed_4bit_theory.md](notes/mixed_4bit_theory.md), 포맷·논문 서베이는
> [notes/mixed_4bit_survey.md](notes/mixed_4bit_survey.md), 트랙 전체 맥락은
> [PROPOSAL.md](PROPOSAL.md) Track C/H0.
> **목표**: W/A/KV 유효 ≤4.5bit + ozaki1 연산으로 MATH-500에서 project target ≥93.0
> 달성 (판정 기준은 §0.1). 이길 대상: FlatQuant W4A4KV4 = 82.8[레포 논문] / 84.1±1.3[PROPOSAL].
>
> **개정 (2026-07-09)**: outlier는 fp8/int8 채널 승격이 아니라 **FlatQuant식 변환(Track B)** 이
> 평탄화한다 (PROPOSAL 개정). 여기 포맷 혼용(C0/C4/C5)은 "outlier 대응"이 아니라 **변환 후
> 잔여 분포에 4bit 격자를 정합**하는 층 — C5는 그 잔여 이득을 측정하도록 재정의됨.

---

## 0. 공통 프로토콜

| 항목 | 값 |
|---|---|
| 모델 | `modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B` (revision/sha 고정·기록) |
| 디코딩 | seed 42, temp 0.6, top_p 0.95, max 32k (기존 스윕과 동일) |
| 스크리닝 | GSM8K — **broken-run detector로만** (§C1, FlatQuant조차 −0.6pt라 성능 변별 불가) |
| 본 측정 | MATH-500. 최종 후보만 AIME-120 + GPQA-Diamond |
| 판정 | §0.1. 경계 셀은 McNemar + paired bootstrap |
| 병기 지표 | §0.2 cost accounting (W/A/KV bits-per-value + plane 3종) |
| calibration | `datasets/gen_data/DeepSeek-R1-Distill-Qwen-7B/NuminaMath-1.5.jsonl` (G0가 검증) |
| 결과 기록 | 모든 run은 `results/<run_id>/`에 manifest·metrics·per-problem JSONL·quant_config·cost_summary (§0.3). 스윕은 `sweep_ozaki_nmp_grid.py`의 resumable/캐노니컬-해시 패턴 복제 |

실험군: **A (C0–C6)** = 기존 표준 포맷만. **B (OP0–OP3)** = op-aware custom 포맷
(A의 승자가 B의 대조군).

### 0.1 목표와 판정 기준

기준점은 **동일 harness로 재측정한 `BF16_MATH500`** (G1). 예상값 94.8.

- **Project target**: MATH-500 ≥ 93.0 (PROPOSAL 첫 마일스톤).
- **Lossless**: BF16_MATH500 − 1.0pt 이내 (예상 ≥93.8).
- **Fair**: BF16_MATH500 − 3.0pt 이내 (예상 91.8–93.8).
- **Risky**: −3.0pt 초과 (예상 <91.8).

⚠ project target(≥93.0)은 예상 기준 **fair 구간**이지 strict lossless가 아니다 — 결과
기록 시 둘을 분리 표기한다. ±2pt는 노이즈, 경계는 McNemar.

### 0.2 Cost accounting

모든 셀은 아래 cost를 기록한다. "≤4.5bit 달성" 주장은 이 공식으로만 판정.

**Storage (role r ∈ {W,A,KV}, 단일 포맷 f):**
```
b_eff(r,f) = 4 + (scale_bits + format_bits + zero_point_bits + extra_meta_bits) / block_size
```
**혼용:** `b_eff(r) = Σ_f ρ_f · b_eff(r,f)` (ρ_f = 해당 role 내 **value 기준** 비율).

검산 예: int4_g128 = 4+16/128 = 4.125; mxint4 = 4+8/32 = 4.25; nvfp4 = 4+8/16 = 4.5;
mixfp4 ρ 승격 = 4.5 (format bit는 scale sign bit에 은닉, +0).

**Plane (연산 비용, 3종 구분 기록):**
- `stored_plane`: 저장값을 exact digit으로 표현하는 이론상 plane 수.
- `executed_plane`: 실제 실행한 GEMM plane 수.
- `nonzero_density`: 2nd+ plane의 non-zero digit 비율 (희소성 실측).

예: int4/mxint4 → stored=executed=1. mxfp4/e2m1 → stored=executed=2, density 별기.
mixfp4 ρ 승격 → avg_stored = 1+ρ, executed는 dense/sparse 구현별 기록.

**최종 표 필수 컬럼:**
`run_id | W_bpv | A_bpv | KV_bpv | W_plane | A_plane | KV_plane | P_plane | Q_plane | GSM8K | MATH500`

### 0.3 결과 기록 & 통계 스키마

**`results/<run_id>/manifest.json`** (재현성):
`run_id, parent_run_id, git_commit, model_sha, tokenizer_sha, dataset_sha,
calibration_file_sha256, eval_harness_commit, flash_ozaki_commit, vllm/torch/cuda_version,
seed, decoding_config, quant_config, format_config, cost_summary, created_at`.

**`results/<run_id>/metrics.json`**:
`GSM8K_acc, MATH500_acc, MATH500_bootstrap_ci, truncation_rate, extraction_failure_rate,
avg_output_tokens, W/A/KV_bpv, W/A/KV_plane, tokens_per_sec(있으면)`.

**`results/<run_id>/problems.jsonl`** (문제 단위, McNemar 필수):
`run_id, task, problem_id, prompt_hash, gen_seed, extracted_answer, gold_answer,
is_correct, num_output_tokens, stopped_by, format_config_hash, parent_baseline_run_id`.

**Multiple-comparison guard**: C1/후보 목록은 MATH-500 실행 **전에** `c1_candidates.json`
으로 freeze. 비교는 동일 `problem_id` paired. 경계 셀은 McNemar + paired bootstrap CI.
최종 후보는 MATH-500 단독 확정 금지 — AIME-120 + GPQA-Diamond sanity check 필수.

### 0.4 실행 전 gate (순서대로 통과)

- **G0 calibration reader**: `generated_text` str/list 양쪽에서 reasoning 누락 없음. 샘플
  100개의 token count·empty rate·parse error 기록.
- **G1 baseline** (§C1 A1과 동일): BF16 재측정이 기존과 ±1.0pt 초과 다르면 harness 점검
  후 진행. FlatQuant W4A4KV4가 보고값과 ±2.0pt 초과 다르면 C1 중단, mismatch 분석.
- **G2 fmt_lib**: 전 포맷 quant/dequant round-trip test, scale finite, NaN/Inf 없음,
  block remainder 처리 확인.
- **G3 C0 완료**: format map 생성 + C1 후보 8–12개 freeze.
- **G4 C2 exactness** (§C2): 통과 전 C3 결과는 "R3 저장 오차만"으로 **해석 금지**.

---

## 1. 포맷 스펙 (구현 참조표)

`fmt_lib.py`가 구현할 전부. "plane"은 w=4 기준 ozaki digit plane 수. **구현자는 codebook을
직접 해석하지 말고 `fmt_lib.py`의 canonical code→digit-plane 테이블만 사용**한다
(특히 e2m1/mxfp4/nvfp4).

### 1.1 표준 포맷 (실험군 A)

| id | 격자 (블록 정렬 후 정수) | block / scale | bits/val | plane | 비고 |
|---|---|---|---|---|---|
| `int4_pc` | {−8..7} | 채널 / fp16 | 4.004 | 1 | FlatQuant W 구성 |
| `int4_g128` | {−8..7} | 128 / fp16 | 4.125 | 1 | GPTQ/AWQ 관행 |
| `int4_pt` | {−8..7} | 행(per-token) / fp16 | 4.004 | 1 | FlatQuant A 구성 (대조군) |
| `nvint4` | {−8..7} | 16 / E4M3 | 4.5 | 1 | |
| `mxint4` | {−8..7} | 32 / E8M0 | 4.25 | 1 | **≡ ozaki 1-digit encode** (A는 추가 양자화 불필요) |
| `mxfp4` | E2M1: ±{0,1,2,3,4,6,8,12}(×½) | 32 / E8M0 | 4.25 | **2** (2nd ternary·희소: ±4/±6 코드만) | |
| `nvfp4` | E2M1 동일 | 16 / E4M3 + fp32/tensor | 4.5 | **2** | chunk=16(기존재) + E4M3 = in-reduction 곱셈기(§1.3) — 두 겹 비쌈 |
| `e1m2` | {0..7}(×½) ≡ INT4 | 16 / E4M3 | 4.5 | 1 | MixFP4의 INT-격자 |
| `nf4` | 무리수 분위수 | 64 / fp32 absmax | 4.5 | **연산 불가** | R1 대조군 **전용** — C3/C4/C5 exact 후보 제외, 정확해도 winner 아님 |
| `mixfp4` | 블록별 e2m1 ∨ e1m2 | 16 / E4M3 (sign bit = 포맷 비트) | 4.5 | 1–2 | 선택: per-block MSE argmin, κ\*=2.2243 |

### 1.2 custom 포맷 (실험군 B)

| id | 격자 | plane | 비고 |
|---|---|---|---|
| `uint4` | unsigned {0..15} | 1 (int_bits w·nD−1→w·nD) | 비음수 전용, 해상도 2×. ≡ unsigned E1M3 |
| `e2m2u` | unsigned {0..7, 8,10,12,14, 16,20,24,28} | 2 (상위 ∈{0,1,2}) | |
| `e3m1u` | unsigned {0..192} | 2 | 큰 dynamic range |
| `lut16` | 임의 정수 16개 (Lloyd-Max, lattice 스냅) | 1–2 (최대 코드 비트) | tail 코드 소수로 제한 → 상위 plane 희소 |
| `azp` | int4/uint4 + 정적 per-channel zero-point | 1 | bias fold 조건은 §1.3-3 |

### 1.3 스케일/오프셋 fold 규칙 (C2가 검증) — **하드웨어 비용 위치가 다름**

1. 2의 거듭제곱 스케일(E8M0) → prealign 지수 (shift, 공짜).
2. fp 스케일: **reduction 밖**(per-tensor/out-channel/token)이면 epilogue 후처리 = dtype
   무관 공짜(모든 int-GEMM의 dequant 경로 재사용). **per-block mantissa**(NVFP4의 E4M3)면
   `standalone_oz_gemm.py:60`의 `cacc·sA·sB`가 reduction 안 곱셈 = **곱셈기 추가**
   (pow2 E8M0는 shift로 족함). 상세: [theory §2](notes/mixed_4bit_theory.md).
3. **Zero-point fold**: 비대칭은 `x ≈ s·(q−z)`, linear `y=Wx`면
   `y ≈ s·(Wq) − s·(Wz)`. 보정항이 **완전 오프라인 bias**가 되는 조건:
   (a) z 정적, (b) s가 fold 단위에서 정적(또는 s·z 정적), (c) W 고정, (d) colsum(W) 사전
   계산 가능. → 정적 z + **정적** s(per-tensor/channel)면 runtime 0.
   **s가 per-token 동적이면 정적 z라도 보정항이 rank-1 런타임**(per-token s × precomputed
   per-out const)이 된다. 1차 실험은 rank-1 런타임 보정이 필요한 asymmetric activation을
   **제외**. (OP2 `azp` 해석에 직결.)

### 1.4 Quantizer 공통 규칙

- rounding: deterministic round-to-nearest-even (SR은 G-트랙 별도).
- clipping: 표현 가능 code range로 clamp.
- scale 선택: int4/mxint4 = block absmax 기본; mxfp4/nvfp4 = block MSE-opt scale (C0에서는
  absmax·MSE-opt 둘 다 기록); lut16 = calibration Lloyd-Max 후 lattice snap.
- block axis: role별 명시 — W = out-channel 방향 / A = per-token 행 / K,V = head_dim (§1.5)
  / P,Q = §3.3 지도 참조. reduction 축(k)은 chunk(32/16) 단위.
- block remainder: 나누어떨어지지 않는 마지막 block은 실제 길이로 scale 계산, packed GEMM
  에서는 padding.
- scale dtype: E4M3/E8M0 rounding·saturation 규칙 명시 (round-to-nearest, max 포화).
- NaN/Inf: quantize 전 오류 → run fail. zero: codebook의 zero는 단일 canonical zero.

### 1.5 KV cache quantization policy

KV는 inference 중 누적되는 cache라 별도 정의:
- block axis = **head_dim**, granularity = per-token × per-head × block.
- 저장: quantized K/V + scale 함께. decode step은 새 token K/V만 quantize·append,
  prefill은 전 sequence 동일 policy.
- head_dim remainder block은 padding 후 packed path.
- **b_eff(KV)에 scale overhead 반드시 포함** (§0.2).

---

## 2. 선행 구현 작업

| # | 작업 | 난이도 | 막는 실험 |
|---|---|---|---|
| W1 | `scripts/fmt_lib.py` (§1 전 포맷 + 아래 API 계약) | S | C0, C1, OP0 |
| W2 | per-채널/그룹 fp 스케일 combine fold (`combine_cast` 스칼라 weight → per-(chunk,채널) 벡터; 참조 `standalone_oz_gemm.py:60`) | M | C2, C3 |
| W3 | 비대칭 nD (nD_A≠nD_B) pack plan/GEMM — H0 공유 | M | C3 |
| W4 | GPTQ int4 체크포인트 → packed digit-plane 컨버터 (+아래 invariant 검증) | M | C3 |
| W5 | chunk=16 경로 검증 (nvfp4) | S | C2 |
| W6 | per-block 혼용 2nd-plane 희소 GEMM (K축 블록 gather → skinny GEMM) | L | C4 (1차는 per-tensor 회피) |
| W7 | `scripts/sweep_c1_formats.py` (resumable 복제) | S | C1 |
| W8 | unsigned digit split (int_bits=w·nD, byte_split_style 신설) | M | OP1, OP2 |
| W9 | lut16 인코드 (16-entry 코드→digit plane 테이블; GEMM 경로 불변) | S | OP2, OP3 |
| W10 | zero-point bias-fold pass (§1.3-3 조건, 오프라인 z·colsum(W)) | S | OP2 |
| W11 | plane-비용 제약 Lloyd-Max 격자 설계기 | S | OP0, OP3 |
| W12 | eval harness 확장: per-problem `problems.jsonl` + manifest/cost_summary 방출 (§0.3) | S | 전 실험(McNemar) |
| W13 | C0 streaming-stat dumper (§C0 dump policy) | S | C0, OP0 |

**W1 fmt_lib API 계약** (구현 흔들림 방지):
```python
@dataclass
class QuantConfig:      # fmt_id, block_size, scale_dtype, axis, symmetric, allow_mse_scale
@dataclass
class QuantizedTensor:  # codes, scales, zero_points(opt), metadata
def quantize(x, cfg) -> QuantizedTensor
def dequantize(q, cfg) -> Tensor
def to_digit_planes(q, cfg) -> list[Tensor]   # canonical code→plane 테이블
def estimate_cost(q, cfg) -> dict             # b_eff, plane 3종 (§0.2)
def block_stats(x, cfg) -> dict               # κ, kurtosis, per-format MSE
```
모든 포맷은 round-trip unit test 보유 (G2).

**W4 GPTQ converter invariants**: group_size·scale·zero-point shape 일치; act-order/
permutation 존재 시 inverse perm 적용; sym/asym 자동 감지; dequant(GPTQ) vs dequant
(converter) max_abs_err 기록; layer별 mismatch summary; tolerance 초과 layer는 **C3 진입
금지**.

fake-quant 실험(C1/OP1/OP2의 R1 셀)은 `fake_quant_utils.py` 확장으로 vLLM 경로에,
미시(C0/OP0)는 transformers hook 덤프로 수행.

---

## 3. 실험군 A — 표준 포맷 (C0–C6)

### C0 — 미시: 블록 통계·포맷 선호 지도 (P0, GPU-경량, 최우선)

**dump policy** (raw 전체 저장 금지 — streaming 통계 중심, W13):
- 저장: per-layer/role histogram, per-block κ=max/RMS, per-block MSE(각 포맷),
  outlier channel index set, E2M1 promotion gain. shard별 JSONL.
- raw tensor는 debug용만: sample ≤ 8, layer = first/middle/last 각 2, bf16,
  `results/c0_debug_tensors/`.

**절차:**
1. calibration forward에서 W / A / K / V / Q / P 통계 수집 (block 32, 16).
2. κ 히스토그램·kurtosis·outlier 정상성(아래) → 텐서 역할×layer 승자 지도, κ\* 예측 대조.
3. FlatQuant fake-quant A4(per-token) MSE vs 같은 텐서 `mxint4` MSE (가설: block-32 승).
4. E2M1 승격 커버리지 곡선: MSE-이득 순 ρ(%) vs 잔여 MSE → C4의 ρ 후보.

**Outlier channel 정의:**
- channel score = mean over tokens of (max_abs 또는 RMS-normalized max).
- outlier = 상위 p% channel, p ∈ {0.5, 1, 2}.
- IoU는 calibration shard ≥2개로 계산. IoU ≥0.5 → channel-stationary; <0.3 → block-local
  /token-dependent. **용도(개정)**: fp8 채널 승격 근거가 아니라, (a) 변환(B)이 없앨 outlier의
  방향 진단(채널-고정이면 회전으로 잘 흡수), (b) 변환 후 잔여에 대한 포맷 혼용(C) 필요성 판단.

**C0 → C1 후보 selection rule** (G3에서 freeze, 사후 편향 방지):
- 필수 포함: ① FlatQuant W4A4KV4, ② int4 최선(W int4_g128 / A int4_pt / KV int4_g128),
  ③ A=mxint4 가설, ④ A=mxfp4 상한, ⑤ W=mxfp4, ⑥ KV=mxint4, ⑦ role별 MSE 최저 조합,
  ⑧ role별 plane≤1 최저 조합.
- 추가: role별 MSE가 최저 대비 5% 이내인 포맷만 retain, 단 W/A/KV 각 b_eff≤4.5.
- 12개 초과 시 C0 predicted total MSE 상위 12개만. → `results/c1_candidates.json` freeze.
- 산출물: `results/c0_format_maps.md`.

### C1 — R1 레짐: 역할별 uniform 포맷 그리드 (P0)

**A1 baseline gate** (G1, C1 전 필수, 동일 harness):
- 재측정: ① BF16, ② FlatQuant W4A4KV4, ③ FlatQuant transform-only(no quant),
  ④ FlatQuant transform + reference quant. 전부 raw output·extracted·correct flag 저장.
- Gate: BF16 ±1.0pt 초과 → harness 점검. FlatQuant ±2.0pt 초과 → C1 중단, mismatch 분석.

**셀**: 연산 bf16(저장 오차만 분리). `c1_candidates.json`의 8–12셀. 비교선에 **FlatQuant
변환(`flat_matrices.pth` 재사용) + 우리 포맷** 포함(변환/격자 기여 분리).

**GSM8K filter** (broken-run detector, 성능 비교 아님). Fail 조건 (하나라도):
BF16 대비 GSM8K drop >5pt / extraction failure >1% / avg 생성 길이 ≥2× / truncation >2% /
NaN·Inf·empty 발생. **Fail 셀은 MATH-500 미실행.**

- 통과 기준: `A=mxint4` 셀이 `A=int4_pt`를 paired로 >2pt 이기면 block-32 가설 확정.
- 산출물: `results/c1_format_grid.md`.

### C2 — exactness 단위 검증 (P0, =H0-a, **G4**)

`flash_ozaki/verification/verify_faithfulness.py` 확장. **아래 전부 통과해야 C3 해석 유효:**
1. **Integer accumulation**: digit-plane GEMM = int64 reference와 bit-exact. shape sweep
   (M,N,K small/med/large, **K가 chunk 배수 아닌 경우 포함**).
2. **Scale fold equivalence**: `dequant(A)@dequant(B)` vs `ozaki_digit_gemm(planes, folded
   scales)`, fp32 기준 max_abs_err·rel_err 기록. **허용 오차는 scale dtype별 별기.**
3. **e2m1 decomposition**: 전 codebook value의 digit-plane 분해 테이블 test, 2nd-plane
   sparsity가 이론값 일치.
4. **nvfp4 chunk=16**: chunk boundary, leftover chunk, E4M3 saturation case.
5. **unsigned digit**: uint4/e2m2u/e3m1u의 sign-extension 오류 없음.
- **C2 실패 시 C3가 accuracy를 내도 "R3 저장 오차만 검증"으로 해석 금지.**

### C3 — R3 레짐: 4bit 저장 + ozaki 연산 e2e (P0-2)

W2–W4 선행. C1 최선 셀을 ozaki 경로로 재실행 (W 1-plane, A 1-plane(+승격), KV 1-plane,
Q 2–3 digit, P는 OP1 결과). **단일 질문**: C1(같은 포맷, bf16 연산) 대비 차이 = 잔여 연산
오차 → H0 예측대로 ~0인가.

**Failure triage** (C3가 동일 포맷 C1 대비 MATH-500 ≥1pt 하락 시, 순서대로):
1. layer-wise numeric replay: C1 bf16 path vs C3 ozaki path의 layer output relerr.
2. op-wise replay: QK / softmax P / PV / MLP up·gate·down.
3. scale-fold ablation: fold on/off, fp32 scale reference, chunk boundary.
4. digit-plane ablation: 1-plane only / dense 2-plane / sparse 2-plane.
- 산출물: `results/c3_r3_combined.md`, `results/c3_error_trace.jsonl`.

### C4 — per-block 적응 혼용 (P1, MixFP4 방식)

W/KV(오프라인): per-block MSE argmin으로 INT-격자 vs e2m1, Type-in-Scale 저장.
A(online): κ-임계 동적 vs calibration 고정. e2e는 per-tensor 선택 시작(W6 회피 → 이후 확장).

**Promotion budget ρ** (role별 **block 비율** 기본):
- 예: W ρ=10% = 전 W block 중 승격 score 상위 10%를 e2m1로. A/KV 동형(관측 block 분포 기준).
- score(block) = MSE_INT4 − MSE_E2M1. 동률/근소 차는 **낮은 plane 포맷 우선.**
- ρ ∈ {5, 10, 25, 100}% 스윕 (100 = MixFP4 원안).
- 별기: block_ratio, value_ratio, 결과 bits/val, avg plane, 2nd-plane nonzero density.
- 산출물: `results/c4_adaptive.md`.

### C5 — 변환 후 잔여에 대한 포맷 혼용 이득 (변환 B × 블록-e2m1 C) (P1)

**개정**: 기존 "채널-fp8(B) vs 블록-e2m1(C)"의 fp8 축은 폐기(B가 변환 평탄화로 전환). 이제
질문은 **변환(B, 저장 +0bit)으로 평탄화한 뒤에도 블록-e2m1(C) 혼용의 잔여 이득이 있는가**:
A·K 각각 (i) 변환 + 균일 int4, (ii) 변환 + ρ% 블록 e2m1, (iii) 변환 없이 e2m1만. 사전 예측:
변환이 κ를 낮춰 e2m1 여지가 줄지만(survey §2의 RHT-후에도 혼용이 남던 관찰과 대조), diffuse
/고κ 잔여 블록에선 (ii)가 남을 수 있음. 산출물: `results/c5_transform_x_fmt.md`.

### C6 — rotation 합성 (P1–P2, Track F 합동)

FlatQuant 변환(또는 QuaRot Hadamard) 후 C0 재측정(κ 이동, 필요 ρ 감소) → 최종 후보
"변환 + mxint4 + 소량 승격 + exact 연산" e2e. 산출물: `results/c6_rotation.md`.

### 판정·비용

- 판정 §0.1, cost §0.2, 통계 §0.3. GSM8K는 filter만.
- 비용: C0/C2/OP0 = GPU-경량(수 시간). C1/OP1/OP2 R1 셀 = 셀당 기존 inference. C3만
  attention 에뮬 — flash_ozaki + digit collapse로 기존 그리드 셀보다 쌀 것.

---

## 4. 실험군 B — operation-aware custom 포맷 (OP0–OP3)

표준 포맷은 텐서를 익명 숫자로 본다. 우리는 각 텐서가 **어느 op의 출력인지** 알고, op가
값 domain을 결정한다 (softmax [0,1] 비음수, SiLU [−0.278,∞) 비대칭). lattice 설계 근거는
[theory §7](notes/mixed_4bit_theory.md).

### OP0 — op별 출력 분포 통계 (P0 말미, C0 확장)

C0 덤프에 추가: SiLU(gate) 출력, SwiGLU 곱(down_proj 입력), softmax P(head·위치별) —
skewness, 음수 질량, 비음수 여부, spike/row outlier 강도. W11로 op별 최적 `lut16` 격자 +
plane 비용 산출. **게이트**: op-aware 격자 MSE 이득이 표준 포맷 대비 <10%면 해당 op는
e2e로 올리지 않고 실험군 A로 회귀.

### OP1 — softmax P의 unsigned 포맷 (P1)

W8 선행. 후보 {`uint4`(1 plane), `e2m2u`, `e3m1u`, 대조 `int4` sym(=sign bit 낭비 정량화)}.

**Row-type별 분리 지표** (P는 row entropy에 따라 PV 오차가 크게 다름):
- row bins: top-heavy max(P)≥0.5 / medium 0.1–0.5 / diffuse <0.1.
- 기록: P MSE, row-sum error |ΣP̂−1|, KL(P‖P̂) zero-safe, PV relerr, attn output cosine,
  downstream MATH-500.
- 성공 기준: `uint4`가 P 2-digit 대비 **diffuse row PV relerr 악화 없이** 동급 → H0-b의
  P 2-digit 가정을 1 unsigned digit로, PV = GEMM 1개 exact.
- 산출물: `results/op1_softmax_fmt.md`.

### OP2 — SiLU/SwiGLU 비대칭 포맷 (P1)

W8–W10 선행. **양자화 지점을 별도 role로 기록**: gate_pre(SiLU 전), gate_act(SiLU 후),
up_act(up_proj 출력), down_in(gate_act·up_act = down_proj 입력).

- 후보: (i) `azp` 정적 zero-point(bias fold — §1.3-3 조건 충족 확인), (ii) `lut16`(분위수
  스냅), (iii) 대조 sym {int4, mxint4, mxfp4}.
- 지점 비교: **separated**(gate_act·up_act 각각 양자화 후 곱) vs **product**(down_in을
  bf16으로 계산 후 양자화) vs **hybrid**(OP0의 dominant-error branch만).
- OP0에서 spike 지배로 판별되면 포맷 교체 대신 "비대칭 격자 + 변환 평탄화(B)".
- 기록: down_proj output relerr + 최종 MATH-500. 산출물: `results/op2_swiglu_fmt.md`.

### OP3 — lut16 자동 설계 + 최종 대결 (P2)

W11로 전 op per-op 포맷 테이블 자동 생성 (1-digit 우선, MSE 이득 임계 초과 시만 2-digit
tail 허용). **최종 셀: 실험군 A 승자 조합 vs op-aware 조합** — 같은 bits/val·plane 예산,
MATH-500 paired 비교. 산출물: `results/op3_final_duel.md`.

---

## 5. 실행 순서

```
G0,G1 (baseline/calib gate)
  │
W1,W13 ─► C0 ─► [C1 후보 freeze] ─► C1 (W7, A1 gate, GSM8K filter)
W2,W5  ─► C2 (G4) ──────────────────┐
                                     ├─► C3 (W3,W4) ─► C4 (ρ; per-block은 W6) ─► C5 ─► C6
W11 ─► OP0 ─► OP1 (W8) ─► OP2 (W8–W10) ──────────────────────────────► OP3
                    │
                    └─(OP1 결과 = C3의 P 인코딩에 반영)
W12 (per-problem 로깅) = 전 e2e 실험의 전제
```

- 모든 셀 기록: 포맷 id(§1)·b_eff·plane 3종·GSM8K(filter)·MATH500 + manifest/metrics/
  problems.jsonl (§0.3). C2(G4) 전 C3 결론 사용 금지.
