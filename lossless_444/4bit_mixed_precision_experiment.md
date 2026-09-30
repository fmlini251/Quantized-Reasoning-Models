# 4bit Mixed-Precision 실험 계획서 — ozaki1 4bit-포맷 혼용 ALU × FlatQuant

> **독립 실험 계획서.** 공통 인프라(포맷 스펙 §1, gate G0–G4, cost accounting §0.2,
> 결과 스키마 §0.3, 선행작업 W1–W13)는 [mixed_4bit.md](mixed_4bit.md)를 참조하고
> 여기서는 재정의하지 않는다. 이론 근거(digit 비용, fold 규칙, lattice 설계 공간)는
> [notes/mixed_4bit_theory.md](notes/mixed_4bit_theory.md), 포맷·논문 서베이는
> [notes/mixed_4bit_survey.md](notes/mixed_4bit_survey.md), 트랙 전체는
> [PROPOSAL.md](PROPOSAL.md).
>
> 이 문서의 새 축: **FlatQuant(GPTQ/RTN) 변환 학습과 혼용 포맷의 결합**을 1급 실험
> 대상으로 승격 — mixed_4bit.md의 C1은 "FlatQuant 변환 재사용 + 포맷 교체"를 비교선
> 하나로만 두지만, 여기서는 **전 셀 format-aware 재학습**이 기본이다 (§6 결정 D2).
>
> **용어 (2026-07-09 개정)**: 이 문서의 "mixed-dtype / dtype 혼용"은 전부 **≤4bit 포맷 혼용**
> (int4/mxint4/mxfp4/nvfp4/e1m2/mixfp4)을 뜻한다 — outlier를 fp8/int8 **8bit로 승격하는 방식은
> 폐기**됐고(PROPOSAL 개정), outlier는 FlatQuant식 변환(Track B)이 상류에서 평탄화한다. 즉 여기
> 포맷 혼용은 "outlier 대응"이 아니라 "평탄화된 분포에 4bit 격자를 정합"하는 층이다.

---

## 1. 배경 — 왜 ozaki1 ALU가 4bit 포맷 혼용의 하드웨어적 근거인가

ozaki1 연산기는 어떤 입력 dtype이든 prealign(블록 공통 지수 정렬) 후 **공통 int8
digit-GEMM**으로 계산한다. 따라서:

1. **dtype 추가 비용 = 연산기 앞단의 dtype→int cast unit 하나.** datapath(정수
   곱셈기·누산기)는 불변이고, 포맷 차이는 encode 시점의 code→digit-plane 변환
   테이블로만 존재한다. 비교: MixFP4는 두 포맷을 지원하려고 MAC 루프 **안에**
   E2M2 디코더를 넣어 tensor core 대비 +3.1% 면적/+1.5% 전력을 냈다
   ([survey §2](notes/mixed_4bit_survey.md)). ozaki1에서는 같은 기능이 encode-side
   테이블(양자화/쓰기 시점)이라 GEMM datapath 비용이 ~0이다
   ([theory §8](notes/mixed_4bit_theory.md)).
2. **두 operand의 dtype이 달라도 동일 방식으로 연산.** 양쪽 모두 digit plane으로
   내려온 뒤에는 격자 출신이 무엇이든 int GEMM 하나다. `A=mxint4 × W=mxfp4`,
   `Q=int4 × K=uint4`처럼 **역할별·블록별로 독립적인 포맷 선택**이 조합 폭발 없이
   가능 — 전용 유닛 방식이라면 포맷 쌍마다 datapath가 필요했을 자유도다.
3. **정직한 비용 단위는 plane.** "HW overhead 최소"는 공짜라는 뜻이 아니다 — 포맷의
   실제 비용은 digit plane 수 × 상위 plane 희소성으로 나타난다 (INT-격자 = 1 plane,
   E2M1 = 2 plane[2nd는 ternary 희소], [theory §1](notes/mixed_4bit_theory.md)).
   따라서 모든 셀은 §0.2 cost accounting(bpv + plane 3종)을 병기해야 하고, "같은
   4.x bit"끼리도 plane이 다르면 다른 비용의 설계다.

이 특성 위에서 두 실험을 세운다:

- **실험 1 (§3)**: 표준 4bit 포맷(INT4-계열/MXINT4/MXFP4/NVFP4/E1M2/MixFP4)을
  W·A·KV의 데이터 분포에 따라 matrix/chunk 단위로 혼용 — 단일 dtype(uniform INT4)만
  쓰는 FlatQuant 대비 정확도 이득을 GPTQ/RTN 두 스타일에서 정량화.
- **실험 2 (§4)**: 표준 포맷 밖의 custom/biased 포맷 — softmax 출력(비음수)에
  unsigned 격자, SiLU/SwiGLU 출력(biased)에 정적 zero-point — 을 op-aware로 적용해
  실험 1 승자 위에서 추가 이득을 측정.

## 2. 공통 프로토콜

mixed_4bit.md §0 전체를 그대로 따른다 — 모델(R1-Distill-Qwen-7B), 디코딩(seed 42,
temp 0.6, top_p 0.95, 32k), 판정 기준(§0.1: project target ≥93.0 / lossless ≥
BF16−1.0), cost accounting(§0.2), 결과·통계 스키마(§0.3: manifest + metrics +
per-problem `problems.jsonl` + McNemar), gate(§0.4: G0 calib reader → G1 baseline
재측정 → G2 fmt_lib round-trip → G3 후보 freeze → G4 exactness).

이 문서에서 추가하는 것 한 가지:

**WikiText-2 PPL 스크리닝 단계.** FlatQuant 재학습이 끝난 셀마다 MATH-500 전에
WikiText-2 PPL을 잰다 (FlatQuant repo `eval_utils.py` 경로 재사용, 분 단위 비용).
용도 둘: (i) FlatQuant·MixFP4 논문 수치와의 직접 비교 가능 지표 확보(두 논문 모두
PPL 보고), (ii) GSM8K filter보다 먼저 도는 초경량 broken-run detector. **주의**:
PPL은 reasoning 붕괴를 못 본다 (FlatQuant W4A4KV4가 GSM8K −0.6인데 MATH-500
−11.8) — PPL은 스크리닝·비교용 병기 지표일 뿐, 판정은 MATH-500.

**평가 사다리 (셀당, 순서대로·앞이 fail이면 중단):**
```
PPL(WikiText-2, 분 단위) → GSM8K filter(§C1 fail 조건) → MATH-500(본 판정)
→ [최종 후보만] AIME-120 + GPQA-Diamond
```

## 3. 실험 1 — 표준 4bit 포맷 혼용 × FlatQuant 재학습

### 3.1 질문과 가설

**질문**: W/A/KV 각각의 데이터 분포에 맞춰 표준 4bit 포맷을 골라 쓰면(matrix 단위 +
chunk/block 단위), 단일 uniform INT4에 최적화된 FlatQuant 대비 MATH-500을 몇 pt
회복하는가. RTN과 GPTQ 중 어느 스타일에서 이득이 큰가.

- **H1 (granularity)**: FlatQuant A4는 per-token(행 K≈3584에 스케일 1개) INT4,
  MXINT4는 32-elem마다 지수 — 스케일 입자 ~112× 고움. A=mxint4만으로 A4 오차가
  유의미하게 준다 ([theory §6](notes/mixed_4bit_theory.md)).
- **H2 (격자-분포 정합)**: 고κ(crest factor) 블록은 E2M1이, 저κ 블록은 INT-격자가
  이긴다 (MixFP4 κ\*=2.2243). 회전(FlatQuant 변환)으로 Gaussian화된 뒤에도 혼용
  이득이 남는다 — MixFP4가 RHT 하에서 격차가 오히려 벌어진 것의 재검 (survey §2).
- **H3 (스타일 상호작용)**: GPTQ는 quantize-보상 루프가 grid에 적응하므로 포맷
  교체 이득이 RTN보다 작을 것 (RTN이 격자 민감도가 크다) — 반대로 나오면 그것대로
  격자·보상 상호작용의 발견.
- **H4 (transform-격자 co-adaptation)**: format-aware 재학습(§3.2)이 변환 재사용
  대비 추가 이득을 낸다 — 변환이 uniform INT4 격자에 맞춰 평탄화하는 대신 혼용
  격자의 잔여 오차 방향으로 평탄화를 재배치.

### 3.2 FlatQuant 결합 방식 — 전 셀 format-aware 재학습 (결정 D2)

모든 셀은 **혼용 quantizer를 학습 loop 안에 넣고 FlatQuant 변환(Θ = P, c, α)을
재학습**한다. 기존 `flat_matrices.pth` 재사용 셀은 attribution ablation(§3.4
E1-4)에만 등장한다.

**Hook 지점** (`methods/flatquant/`):

| 대상 | 현재 코드 | 교체 |
|---|---|---|
| A/KV fake-quant | `flatquant/quant_utils.py` `ActivationQuantizer.fake_quant` (round_ste + sym/asym uniform) | `fmt_lib.quantize→dequantize` STE 래퍼 (W14) |
| W RTN | 동일 파일 `WeightQuantizer` (find_params/quantize) | fmt_lib grid로 find_params 확장 (W14) |
| W GPTQ | `gptq_utils.py` (column-loop 보상) | quantize 콜만 fmt_lib grid로 치환 (W15) |
| 학습 진입 | `main.py` (`--gptq` 플래그, 15 epoch, lr 5e-3) | `--fmt_config <json>` 추가 |

**W14 설계 규칙 (STE·scale):**
- 비균일 격자(E2M1 등)의 backward는 round_ste와 동일한 straight-through — codebook
  스냅은 forward만.
- scale 선택: INT-격자 = absmax, E2M1-계열 = block MSE-opt (mixed_4bit.md §1.4).
  FlatQuant의 학습형 클리핑 α(sigmoid-bounded)는 유지하되 **α가 fmt_lib scale
  탐색과 이중 클리핑이 되지 않게** α는 pre-quant 텐서에만 적용, fmt_lib은 α 적용
  후 텐서를 받는다.
- **per-block 포맷 map은 학습 전에 freeze** (calibration 통계 → κ/MSE argmin →
  고정 map). 학습 중 재선택(dynamic re-argmin)은 승자 셀 1개에서만 ablation —
  map이 흔들리면 학습이 비정상 (loss 불연속).

**GPTQ × per-block 포맷의 순서 문제**: GPTQ 보상이 뒤 column의 분포를 바꾸므로
"보상 후 분포로 포맷 선택"은 닭-달걀. 1차 규칙 = **RTN 통계로 포맷 map을 freeze한
뒤 GPTQ 실행** (map은 입력이지 출력이 아님). 승자 셀에서 map 재선택 1회 반복(2-pass)
ablation.

**calibration**: 레포 관행대로 R1-Distill에 seqlen 4096 (survey §3). **주의**
NuminaMath calib 파일은 G0 gate(gen_calib str/list 포맷 함정) 통과 필수.

**비용**: FlatQuant 학습 ~1h/config (L3-8B, seqlen 2048 기준) → seqlen 4096
R1-Distill-Qwen-7B에서 셀당 ~1.5–2 GPU·h 가정. **따라서 셀 수 통제가 예산의
전부** — E1-0에서 후보를 RTN/GPTQ 각 8–12셀로 freeze (G3와 동일한 사전등록 규칙).

### 3.3 Granularity 규칙 (결정 D3)

- **R1 (fake-quant, 연산 bf16)**: matrix(per-tensor/role) 혼용 + **per-block
  혼용까지 전부** — 시뮬레이션이라 granularity가 공짜. per-block은 MixFP4 방식
  (블록별 MSE argmin, Type-in-Scale로 메타 0bit) + κ-임계 policy 비교.
- **R3 (ozaki e2e)**: **per-tensor 혼용만.** per-block 혼용의 희소 2nd-plane
  GEMM(W6)은 1차 범위 밖 — R1에서 per-block 이득의 상한을 먼저 확정하고 W6 투자
  여부를 결정한다.

### 3.4 실험 셀

포맷 pool = mixed_4bit.md §1.1 (int4_pc/int4_g128/int4_pt, nvint4, mxint4, mxfp4,
nvfp4, e1m2, mixfp4; nf4는 R1 대조군 전용). 기존 C-실험과의 대응을 병기한다.

| id | 내용 | 대응 | 선행 |
|---|---|---|---|
| **E1-0** | 블록 통계·포맷 선호 지도 + 후보 freeze. C0 절차 그대로 + **FlatQuant 변환 후 텐서에 대해서도 동일 통계** (변환이 κ 분포를 어디로 옮기는지 = H2 입력) | C0 (+C6 통계부) | W1, W13 |
| **E1-1** | **RTN × 포맷 grid**: freeze된 후보 8–12셀, 각각 format-aware 재학습 → PPL → GSM8K filter → MATH-500. 비교선: FlatQuant 원 구성(W=int4_pc, A=int4_pt, KV=int4_g128 asym) 동일 harness 재학습·재측정 | C1의 재학습판 | W14 |
| **E1-2** | **GPTQ × 포맷 grid**: E1-1과 동일 후보 (`--gptq`). RTN grid와 paired 비교 → H3 | 신규 | W14, W15 |
| **E1-3** | **per-block 적응 혼용 (R1)**: E1-1/E1-2 승자 조합에서 W·A·KV 각각 per-block MixFP4-방식 ON. policy 비교 {κ\*=2.2243 임계, per-block MSE argmin, calibration-고정 vs A만 online} × 승격예산 ρ ∈ {5,10,25,100}% | C4 | W14 (W6 불필요 — R1) |
| **E1-4** | **attribution 2×2** (승자 조합 1개만): {uniform INT4, 혼용} × {변환 재사용, format-aware 재학습} → 이득을 격자 vs co-adaptation으로 분해 (H4). 재사용 arm이 mixed_4bit.md C1의 원안과 접점 | C1 비교선 | — |
| **E1-5** | **R3 ozaki e2e**: E1-1/E1-2/E1-3 승자(단, per-tensor로 축약)를 ozaki 경로로 재실행. W 1-plane(E2M1 승자면 2), A 1-plane, KV 1-plane, Q 2–3 digit. 단일 질문 = R1 동일 셀 대비 차이 ≈ 0인가 (H0 digit collapse). failure 시 C3 triage 절차 | C3 | W2–W4, G4(C2) |

**산출물**: `results/e1_0_format_maps.md`, `results/e1_rtn_gptq_grid.md`,
`results/e1_3_adaptive.md`, `results/e1_4_attribution.md`, `results/e1_5_r3.md`.
전 셀 §0.3 스키마 + §0.2 cost 병기.

**예산 추정**: 재학습 (8–12셀)×2스타일×~2h ≈ 30–50 GPU·h + 셀당 MATH-500 inference
(기존 스윕과 동일 규모, GSM8K filter로 fail 셀 조기 중단). E1-3은 재학습 없이
승자 체크포인트 위 quantizer 교체만 (map freeze 재계산) → inference 비용만.

## 4. 실험 2 — op-aware custom/biased 포맷

표준 포맷은 텐서를 익명 숫자로 보지만, 우리는 **텐서가 어느 op의 출력인지** 안다.
softmax P는 [0,1] 비음수, SiLU는 [−0.278,∞) biased — sign bit와 대칭 격자가 구조적
낭비인 지점이다. 1-digit unsigned(int_bits w·nD−1→w·nD)는 같은 1 plane에서 해상도
2×, 정적 zero-point는 z·colsum(W) bias fold로 런타임 0
([theory §7](notes/mixed_4bit_theory.md)).

**1차 범위 (결정 D4)**: softmax P unsigned + SiLU/SwiGLU 정적 zero-point.
**이월**: `lut16`(lattice-LUT, W9·W11 선행 필요)과 **동적 per-token asymmetric**
(보정항이 rank-1 런타임 — mixed_4bit.md §1.3-3) 은 2차. 이월 사유: 1차는 "런타임
추가 비용 0인 custom 포맷"만으로 이득 존재를 먼저 증명.

### 4.1 실험 셀

| id | 내용 | 대응 | 선행 |
|---|---|---|---|
| **E2-0** | op별 출력 분포 통계: softmax P (row-type: top-heavy max(P)≥0.5 / medium / diffuse), SiLU(gate)·up_act·down_in의 skewness·음수질량·spike. E1-0 덤프에 편승 | OP0 | W13 |
| **E2-1** | **softmax P unsigned**: {uint4(1 plane), e2m2u, e3m1u} vs 대조 int4 sym(sign bit 낭비 정량화). 지표는 OP1 그대로 — row-type별 P MSE, \|ΣP̂−1\|, KL, PV relerr, attn output cosine → MATH-500. 성공 = uint4가 P 2-digit과 diffuse-row PV relerr 동급 → PV가 1-GEMM exact | OP1 | W8 |
| **E2-2** | **SwiGLU 정적 zero-point (azp)**: 양자화 지점 {gate_act, up_act, down_in} 별도 role 기록, separated vs product vs hybrid 비교. 후보 (i) azp (ii) 대조 sym {int4, mxint4, mxfp4} | OP2 subset | W8, W10 |
| **E2-3** | **최종 대결**: 실험 1 승자 조합 vs 실험 1 승자 + E2-1/E2-2 승자 포맷 — 같은 bpv·plane 예산, MATH-500 paired (McNemar) | OP3 축소판 | 위 전부 |

### 4.2 공정 비교를 위한 주의 두 가지

1. **P 양자화는 FlatQuant 대비 실험이 아니다.** FlatQuant W4A4KV4는 P를 bf16으로
   둔다 — E2-1은 attention을 4bit로 **연산**하는 ozaki 경로(또는 fake-quant
   attention 셀)에서만 의미가 있고, 결과 표에서 "FlatQuant보다 양자화 표면이 넓은
   설정"임을 명시한다. E2-1의 대조군은 FlatQuant가 아니라 "P를 signed int4/2-digit
   으로 연산한 같은 ozaki 셀"이다.
2. **azp의 bias-fold 조건**: 보정항이 완전 오프라인이 되려면 z 정적 **그리고** s가
   fold 단위에서 정적이어야 한다 (mixed_4bit.md §1.3-3). FlatQuant A는 per-token
   동적 스케일이므로, **E2-2의 azp 셀은 해당 지점만 정적(per-channel, calibration)
   스케일로 전환**하고, "동적 per-token sym"을 별도 대조로 둔다 — azp의 이득이
   비대칭 격자에서 오는지, 정적 스케일 손해에 먹히는지 분해된다.

## 5. 선행 구현 (mixed_4bit.md §2에 추가되는 항목)

W1–W13은 [mixed_4bit.md §2](mixed_4bit.md) 그대로 유효. 이 문서가 추가하는 것:

| # | 작업 | 난이도 | 막는 실험 | 상태 (2026-07-09) |
|---|---|---|---|---|
| W14 | FlatQuant 학습 loop에 fmt_lib 플러그인: `quant_utils.py` fake_quant를 fmt_lib STE 래퍼로, `main.py --fmt_config`, α-이중클리핑 방지 규칙(§3.2) | M | E1-1..4, E2-2 | **DONE** (아래 구현 노트) |
| W15 | `gptq_utils.py` grid-agnostic화 (quantize 콜 → fmt_lib; RTN-통계 map freeze 순서 규칙) | S–M | E1-2 | 코드 배선 완료(`_fmt_w_groupsize` + configure fmt_cfg), **GPU e2e 검증 미실시** |
| W16 | 평가 사다리 glue: FlatQuant `eval_utils.py` PPL을 결과 스키마(§0.3 metrics.json)에 편입 | S | 전 셀 | 미착수 |

**W14 구현 노트 (2026-07-09, G2 18/18 통과):**
- `lossless_444/scripts/fmt_lib.py` = W1 코어 (전 포맷 §1.1–1.2 + uint4, W1 API 계약 +
  두-phase `compute_scales`/`quantize_with_plan`(GPTQ column 경로) + `fake_quant_ste`).
  MXINT4 인코드는 production `_bfp_scale_torch`(frexp, int_bits=3)와 scale·code 단위로
  동일함을 테스트로 고정 — "MXINT4 ≡ ozaki 1-digit encode" 유지.
- FlatQuant 배선: `methods/flatquant/flatquant/fmt_bridge.py`(--fmt_config JSON 파서,
  role w/a/q/k/v, bits<16 fail-fast), `quant_utils.py`(Activation/Weight 양쪽 fmt 경로,
  α/lac 클리핑은 텐서에 미분가능 clamp로 선적용 → clip factor gradient 통과 확인),
  `flat_linear.py`·`model_tools/{qwen,llama31}_utils.py` role 태깅,
  `gptq_utils.py` RTN/GPTQ, `args_utils.py`(+exp_dir에 `fmt_config.resolved.json` 기록),
  `main.py`(fake_quant_config에 fmt_config 병기). 테스트: `lossless_444/tests/test_fmt_lib.py`.
- **알려진 편차/경계 (해석 시 주의):**
  1. mixfp4의 per-block 포맷 map은 학습 중 `find_params`마다 재선택된다 — §3.2의
     "학습 전 freeze" 규칙은 아직 미구현 (E1-3 진입 전 calibration-freeze 모드 추가 필요).
  2. mxint4 code −8이 블록 max일 때 frexp scale이 한 octave 커져 재양자화가 ≤1 LSB
     드리프트 — production ozaki 재인코드와 동일한 경계. digit-collapse-safe 저장 범위는
     [−7,7]이며 **C2가 kernel 쪽에서 이 경계를 재검증해야 함**.
  3. E2M1 2nd plane support는 {+8,±12} (−8은 plane0) — 밀도가 이론 노트의 대칭 가정보다
     낮음 (theory §1에 반영됨).
  4. inference-side(저장 qmodel 재로드 후 A/KV fmt 재생) 배선은 W16/E1-1 인프라 몫 —
     현재는 main.py 내 PPL 평가까지만 fmt 경로로 돈다.

**GPU e2e 검증 결과 (2026-07-09, RTN 파이프라인 완주 + kernel nmp=4 exactness):**
- **kernel-side (nmp=4 고정, `tests/test_ozaki_nmp4_exactness.py`)**: fmt 저장 텐서의 ozaki1
  GEMM(nD=2 full, int_bits=7)이 mxint4@mxint4 / **mxint4@mxfp4(mixed-dtype)** / −8-boundary /
  uint4@mxint4 전부 **rel_err 0.0 (bit-exact)**, bf16 control 1.3e-2. nD=1의 −8 경계는
  nD=2 headroom으로 소멸 — H0 digit collapse가 커널에서 확정.
- **FlatQuant e2e (smoke 예산: 1ep×32샘플×seqlen4096, A6000)**: 파이프라인 자체는 RTN 완주
  (28층 학습→RTN→PPL→manifest). 단 이 예산에서는 **uniform INT4 control조차 PPL 2069로
  붕괴**하므로(정상 ~10) 정확도 판정 불가 — fmt-vs-uniform 정확도 비교는 현실 예산 필요
  (fp32/A6000 기준 canonical 15ep×128은 층당 ~56분 = ~26h/셀; E1 예산 재설계 필요).
- **⚠ K-cache scale 규칙 발견 (eval-only ablation, control 변환 고정)**: K=mxint4(기본
  frexp 규칙)만 PPL 2069→22273 (10.8×) 폭발, V는 무해(2138). 원인은 zero-point가 아니라
  **production frexp scale(amax/s∈[4,8))의 top-element clamp**: 블록 최대 원소에 ~12%
  오차 → K에선 그 원소들이 QK logit을 지배. `mse_scale: true`(±1 octave 탐색)로
  3023까지 회복; sym-vs-asym 손실은 부차적(int4_pt 4127). **행동 규칙: K role은
  `{"fmt": "mxint4", "mse_scale": true}`를 기본으로** (MSE-opt pow2도 정수×pow2 격자라
  digit-collapse 호환 유지). theory §5의 "A4 저장 ≡ ozaki 인코드 = 공짜"는 A에선 유효하나
  **K 저장에 raw ozaki 인코드 규칙을 쓰면 안 됨** — E/H 트랙(KV digit-plane 저장)에 직결.
- W15(GPTQ) e2e는 K-mse 수정판으로 재실행 중 (1차는 GPU 경합 OOM).

의존: E1 전 셀 ← W1(fmt_lib)·W14; E1-2 ← W15; E1-5 ← W2–W4 + G4(C2 exactness);
E2-1 ← W8(unsigned digit split); E2-2 ← W10(zero-point bias-fold pass).

## 6. Design decision log (2026-07-09 확정)

| # | 결정 | 선택 | 기각 대안·근거 |
|---|---|---|---|
| D1 | 문서 위상 | 독립 계획서 + mixed_4bit.md 인프라 참조 | 통합 개정(기존 C-번호 체계 흔들림), self-contained(중복) |
| D2 | FlatQuant 결합 | **전 셀 format-aware 재학습** | 재사용-only(변환이 INT4에 최적화돼 혼용 이득 과소평가), 2단계(비용 아끼지만 grid 전체의 공정성 손상). 재사용 arm은 E1-4 ablation으로만 |
| D3 | 혼용 granularity | R1 = per-block까지, R3 = per-tensor만 | 전면 per-block은 W6에 블로킹, per-tensor-only는 MixFP 핵심 이득 못 봄 |
| D4 | 실험 2 1차 범위 | softmax P unsigned + SwiGLU 정적 zp | lut16(W9·W11 선행), 동적 asym(rank-1 런타임) → 2차 이월 |

## 7. 실행 순서

```
G0,G1 ─► W1,W13,W16 ─► E1-0 (+E2-0 편승) ─► [후보 freeze = G3]
                              │
        W14 ─────────────► E1-1 (RTN grid) ──┐
        W14,W15 ─────────► E1-2 (GPTQ grid) ─┤
                                             ├─► E1-3 (per-block, R1) ─► E1-4 (attribution)
        W2–W5 ─► C2(G4) ─────────────────────┴─► E1-5 (R3, per-tensor)
        W8 ──────► E2-1 (softmax unsigned) ──┐
        W8,W10 ──► E2-2 (SwiGLU azp) ────────┴─► E2-3 (최종 대결, vs E1 승자)
```

- 판정·기록·통계는 전부 mixed_4bit.md §0 준수. C2(G4) 통과 전 E1-5 결론 사용 금지.
- 셀 추가는 반드시 freeze 전 (multiple-comparison guard). 경계 셀은 McNemar +
  paired bootstrap.
