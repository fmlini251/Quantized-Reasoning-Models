# Lossless W4A4KV4 — Ozaki 기반 실험 제안서

> 목표: **W-A-KV 4-4-4 수준의 저장/연산 비용**을 유지하면서, 기존 SOTA(FlatQuant 등) 4-4-4의
> accuracy drop을 **lossless(≤1pt) 수준**으로 끌어내리는 방법을 찾는다.
> 모든 후속 실험(config, 스크립트, 결과, 노트)은 이 디렉토리(`lossless_444/`) 안에서 관리한다.
>
> **개정 (2026-07-09): outlier 처리 방침 전환.** 기존의 "outlier 채널/row를 fp8·int8로 승격
> (mixed-dtype)"을 폐기하고, **FlatQuant식 learned affine 변환으로 outlier를 평탄화**해 균일
> int4에 흡수하는 방식으로 일원화한다. ozaki ALU의 dtype 무관성은 이제 **4bit 포맷 혼용
> (int4/mxfp4/nf4)** 과 digit-collapse에만 쓰고, 8bit 승격에는 쓰지 않는다. Track B는 변환
> 트랙으로 재작성, Track F(rotation)를 그 연산측 근거로 재정향. (mixed_4bit.md·notes 동반 개정.)

---

## 1. 현재 상태 — 우리가 알고 있는 것

### 1.1 기준점 (DeepSeek-R1-Distill-Qwen-7B, MATH-500 extractive_match)

| 구성 | 정확도 | 비고 |
|---|---|---|
| BF16 native | **0.948** (paper 93.9±0.7) | 이 레포 eval 파이프라인 검증됨 |
| W4 weight-only (AWQ/GPTQ, paper) | 92.5 / 93.3 | near-lossless |
| W8A8 (paper) | 93.8–94.0 | lossless |
| **W4A4 FlatQuant (paper)** | **84.1±1.3** | **−9.8pt, "risky" — 우리가 이기려는 대상** |
| KV4 (paper) | 93.4 | KV 단독 4bit은 거의 무해 |
| ozaki1_fp linear_only w4 nmp1 | 0.912 | per-GEMM relerr 23%인데도 −3.6pt뿐 |
| ozaki1_fp linear_only w4 nmp3 | 0.936 | 사실상 lossless (노이즈 범위) |
| ozaki1_fp full w8 nmp1 | 0.508 | **attention 에뮬 저정밀 → 붕괴** |
| ozaki1_fp full w4, Lin/Attn nmp≥4 | 0.92–0.94 | `outputs/sweep_nmp_grid/results_flash.md` 그리드 |

통계 노트: 500샘플 1σ ≈ 1.3pt (+temp=0.6 stochastic decode). **±2pt는 동률**, >3–4pt만 유의미.

### 1.2 왜 4-4-4가 무너지는가 (관찰 종합)

- Weight 4bit 단독은 거의 무해 (paper W4: −1pt 내외).
- KV 4bit 단독도 거의 무해 (paper KV4: −0.5pt).
- **Activation 4bit이 주범** — FlatQuant조차 W4**A4**에서 −9.8pt. 우리 쪽 증거도 일치:
  ozaki full nmp1 붕괴(0.508)는 attention 경로(=activation이 양쪽 operand)의 저정밀이 원인.
- 즉 문제는 "4bit 저장" 자체가 아니라 **activation의 outlier/dynamic range가 4bit
  균일 격자에 안 들어가는 것**. 이걸 저장·연산 어디에서 얼마의 overhead로 흡수하느냐가 관건.

### 1.3 Ozaki1 스타일 연산의 구조적 장점 (본 제안의 지렛대)

ozaki1_fp는 operand를 prealign(블록 공통 지수 정렬) 후 **4bit digit으로 잘라 정수 GEMM**을
수행하고, `nmp`(유지하는 digit-pair product 수)로 연산 정밀도를 연속적으로 조절한다.

1. **저장 정밀도와 연산 정밀도의 분리.** 값이 이미 int4/mxfp4/nf4 무엇으로 저장되어 있든,
   prealign만 거치면 같은 정수 GEMM으로 들어간다. → 기존 quant kernel처럼 "양쪽 다 int4여야
   함" 제약이 없음. **서로 다른 4bit 포맷(int4/mxfp4/nf4)을 역할·블록별로 혼용해도 커널 하나.**
   (outlier는 dtype 승격이 아니라 Track B의 변환 평탄화로 제거한다 — 8bit 승격 없음.)
2. **정밀도 다이얼(nmp).** layer/op별로 필요한 만큼만 digit product를 유지 → per-layer,
   per-op 정밀도 할당이 "다른 커널"이 아니라 "다른 파라미터"다. 인프라도 이미 있음
   (`--nmp_overrides`, `--ozaki_placement`, `--s_overrides`).
3. **4bit 저장값에 대해서는 연산이 exact해질 수 있다.** int4×int4는 digit 1개짜리 곱 —
   충분한 nmp에서 ozaki GEMM은 양자화된 값 기준 **비트 정확**. 그러면 남는 오차는 순수하게
   양자화 오차뿐이고, 우리는 그 양자화 오차를 (2)의 nmp 다이얼(연산), Track B의 변환 평탄화
   (저장 outlier 제거), (1)의 4bit 포맷 혼용으로 공략한다.

### 1.4 세 가지 오차 레짐 (실험 설계의 좌표계)

| 레짐 | 저장 | 연산 | 오차의 출처 | 현재 데이터 |
|---|---|---|---|---|
| R1: storage-only | 4bit (FlatQuant 등) | 고정밀(fake-quant bf16) | 양자화 오차만 | paper Table 1 |
| R2: compute-only | bf16 | ozaki w4 nmp | 연산 오차만 | nmp 그리드 스윕 |
| R3: **combined (목표)** | 4bit(+ε overhead) | ozaki (필요한 만큼 nmp) | 둘 다 | **없음 — 본 실험** |

R1과 R2를 이미 각각 갖고 있다는 게 이 레포의 강점. R3에서 두 오차가 어떻게 합성되는지
(additive? 상쇄? 증폭?)가 첫 번째로 답해야 할 질문이다. — Track H0의 "digit collapse"
관찰은 **상쇄** 쪽을 예측한다: 4bit 저장 operand는 digit 1개라 연산이 싼 nmp에서 exact가
되어, R2의 연산 오차가 R3에서는 구조적으로 사라질 수 있다.

---

## 2. 실험 그리드 (트랙별 제안)

### Track A — 공통 베이스라인 매트릭스 확립 (P0, 선행 필수)

모든 트랙이 같은 잣대로 비교되도록, 같은 모델/시드/프로토콜에서 재측정:

- [ ] A1. FlatQuant w4a4kv4 fake-quant 모델 생성(`scripts/quantization/flatquant.sh`) 후
      MATH-500 + GSM8K 측정 → paper 84.1 재현 확인.
- [ ] A2. QuaRot-GPTQ 4-4-4 동일 측정 (FlatQuant보다 약할 것 — 대조군).
- [ ] A3. 분해 베이스라인: FlatQuant **W4-only / A4-only / KV4-only / W4A4 / A4KV4** —
      drop의 성분 분해. §1.2 가설(A4가 주범)의 정량 확인. 이 분해표가 이후 모든 트랙의
      "어디에 overhead를 쓸지" 우선순위를 정한다.
- 산출물: `lossless_444/results/baseline_matrix.md`

### Track B — Outlier를 learned affine 변환으로 평탄화 (FlatQuant식, 본명 아이디어) (P0)

outlier를 더 높은 정밀도(fp8/int8)로 "남기는" 대신, **가역 affine 변환으로 분포를 평탄화**해
같은 균일 int4 격자에 흡수한다 (FlatQuant/QuaRot 계열). outlier 채널을 없애면 mixed-dtype
승격·ragged GEMM·채널 gather가 전부 불필요 — 저장은 순수 4bit(+변환 파라미터), 연산은 균일
int4 한 격자. §1.2에서 A4가 주범으로 확인되면 여기가 최대 레버다.

- B1. **Activation 변환 스윕**: FlatQuant식 학습 변환 P=P₁⊗P₂(Kronecker) + per-channel
      scale diag(c) + 학습형 클리핑(LWC/LAC)을 위치별로 적용/재학습 — (i) linear 입력,
      (ii) QK^T/PV의 activation operand(head_dim 변환), (iii) 둘 다. 변환 유무의 per-tensor
      κ(crest factor)와 MATH-500 drop을 대응 (변환이 κ를 어디로 옮기는지 = 격자 정합의 입력).
- B2. **KV 변환**: K는 RoPE 이후 채널 outlier가 큼(KVQuant* 관찰) — FlatQuant의 kcache/vcache
      변환(head_dim)으로 평탄화 후 int4 g128 asym. K만 vs V만 vs 둘 다의 잔여 오차 분해.
      `methods/kvquant_star`의 K outlier 통계는 "변환이 무엇을 없애야 하는지"의 진단으로 재활용.
- B3. **overhead 정산**: 변환은 **저장 비트를 늘리지 않는다**(순수 4bit) — 대신 online 변환
      FLOPs(FlatQuant 보고 +2.61%, decode 1.7×)와 변환 행렬 저장을 bits-per-value가 아닌
      **별도 열**로 병기. 목표선: **저장 ≤4.5bit + 변환 오버헤드 명시에서 ≤1pt drop**.
- 예상: A3에서 A4가 주범으로 확인되면, B1 (ii)(attention activation head_dim 변환)가 최대 레버.
  연산측 시너지(같은 변환이 ozaki relerr도 낮춤)는 #6 F에서 미시 검증.

### Track C — 4bit 포맷 혼용: int4 vs mxfp4 vs nf4 (P1)

같은 4bit라도 격자가 다르다. 분포 통계(kurtosis, outlier ratio)에 따라 tensor/layer/channel
단위로 포맷을 선택. **→ 실험 스크립트: [mixed_4bit.md](mixed_4bit.md)** (실험군 A = 표준
포맷 C0–C6, 실험군 B = op-aware custom 포맷 OP0–OP3), **이론:
[notes/mixed_4bit_theory.md](notes/mixed_4bit_theory.md)** (digit 비용 분석 — E2M1은
1 digit이 아니라 2 digit(H0 가정 정정), NVFP4 호환성 판정, lattice 설계 공간), **서베이:
[notes/mixed_4bit_survey.md](notes/mixed_4bit_survey.md)** (4bit 포맷 표, MixFP4/FlatQuant
논문 정리).

- C1. per-tensor 선택: calibration에서 int4/mxfp4 중 MSE 낮은 쪽 자동 선택 → 전층 스윕.
- C2. mxfp4는 32-elem 블록 공유 지수 = ozaki의 prealign 블록과 자연스럽게 정합(k=32와 동일
      granularity). **mxfp4 저장 + ozaki 정수 GEMM의 exact 경로**를 먼저 검증(unit test)
      후 end-to-end.
- C3. weight는 int4(GPTQ), activation은 mxfp4(동적 블록 지수) 같은 비대칭 조합.

### Track D — Per-layer/op nmp 할당 (greedy allocation) (P1)

nmp 그리드 스윕의 자연스러운 후속: 전 layer 동일 nmp가 아니라 **민감도 기반 차등 할당**.

- D1. 민감도 프로파일: layer l의 op를 저nmp로 낮췄을 때의 loss/logit-KL 변화를 calibration
      set(NuminaMath reasoning traces)에서 측정 → 민감도 랭킹.
- D2. **greedy 할당**: 전층 고nmp에서 시작, "정확도 손실 대비 비용 절감"이 최대인 layer부터
      nmp를 한 단계씩 내리며 예산(평균 nmp ≤ 목표치)까지 — 또는 반대 방향(전층 최저에서
      민감한 층만 올리기). 관찰상 첫/끝 layer와 o_proj/down_proj가 민감한 경향 예상.
- D3. R3 레짐과 결합: 4bit 저장 하에서 layer별 nmp 차등 → "평균 유효 연산 비용" 대비 정확도
      파레토 곡선. 인프라: `--nmp_overrides`는 현재 op-이름 단위(qkv_proj 등 4종) 매칭 —
      **layer index 단위 override로 확장 필요** (작은 코드 작업, `vllm_custom` 쪽).

### Track E — KV4 저장 + ozaki attention (P1)

- E1. KV cache를 int4/mxfp4로 저장하고 QK^T/PV를 ozaki로 계산. 그리드 스윕에서 attention
      에뮬 자체가 ~3–5pt 비용이었으므로, 먼저 **flash_ozaki exact 경로**로 attention 에뮬
      오버헤드와 KV 양자화 오차를 분리 측정할 것 (attention 에뮬 비용이 그대로면 KV4 효과가
      묻힌다).
- E2. K에 B2의 변환 평탄화(kcache 변환) 적용 + PV는 순수 4bit — 조합 스윕.

### Track F — Rotation/변환과 ozaki의 결합 (P2, 시너지 가설)

- F1. QuaRot Hadamard(레포에 `fast-hadamard-transform` 있음) 또는 FlatQuant 학습된 변환을
      **online으로 적용한 뒤** ozaki 4bit digit split. 가설: flatten된 분포는 블록 내
      dynamic range가 작아 prealign shift 손실이 줄고, 같은 nmp에서 relerr가 감소한다.
- F2. 검증 순서: (미시) random + 실제 activation에서 변환 유/무 per-GEMM relerr 비교 →
      효과 있으면 end-to-end. 미시 단계에서 효과 없으면 트랙 폐기 (싸게 실패하기).

### Track G — (보조) 오차 보정 소품 (P2)

- G1. digit 절단(nmp 절단)에 stochastic rounding — 절단 bias 제거.
- G2. down_proj/o_proj 출력에 저랭크 보정항(LoRC 스타일, calibration으로 fit) — overhead
      정산에 포함시켜 B3 지표로 비교.

### Track H — ozaki1 스킴 **내재적** 저오버헤드 정밀도 레버 (신규 고찰)

ozaki1의 구조 자체(prealign → w-bit digit 분해 → digit-pair 정수 GEMM → nmp 절단 → combine)
에서 나오는, 추가 비트/추가 GEMM을 거의 쓰지 않는 레버들.

#### H0. 핵심 관찰 — "digit collapse": 4bit 저장은 연산 오차를 구조적으로 소거한다 (P0)

w=4에서 **4bit로 양자화된 operand는 정확히 digit 1개**다 (int4 → signed digit 1개; mxfp4
mantissa도 블록 지수를 prealign 지수로 쓰면 동일). 따라서:

- **W4 저장 + bf16 activation**: weight 쪽 nD_W=1 → digit-pair 격자가 1×nD_A로 퇴화.
  bf16×bf16의 삼각형 비용 T_n = n(n+1)/2 대신 **선형 비용 n으로 저장값 기준 exact**.
  예: activation 3 digit 유지 = GEMM 3개 (bf16×bf16에서 같은 커버리지는 nmp=6).
- **KV4 저장**: K/V 쪽 nD=1 → QK^T는 Q를 d digit 유지 시 GEMM d개로 exact, PV는 P(softmax
  확률, [0,1] 정규화라 분포가 온순 — 2 digit면 충분할 것)×V(1 digit) = GEMM ~2개로 exact.
  → **R2 그리드에서 관측된 attention 에뮬 비용 ~3–5pt는 bf16 operand를 절단해서 생긴
  오차이므로, R3(4bit 저장)에서는 구조적으로 사라질 수 있다**는 예측. 이게 맞으면 "두 오차의
  합성"이 아니라 저장 양자화가 연산 오차를 소거하는 상쇄 관계 — P0-2 실험이 직접 검증.
- 전제 조건: 양자화 scale 격자와 prealign 블록 지수의 정합. mxfp4(32-elem 블록 지수)는
  k=32와 자연 정합(Track C2와 동일 지점); int4 per-channel scale은 combine 단계에 scale을
  접어 넣는 처리 필요.
- [ ] H0-a. unit test: int4/mxfp4 저장값에 대한 ozaki GEMM의 비트 정확성(digit 1개 경로) 검증.
- [ ] H0-b. end-to-end: KV4-as-digit + Q 2~3 digit로 attention full 측정 — R2 그리드의
      (L=BF16, A=고nmp) ≈ 0.93 대비 개선되는지.

#### H1. Phase-aware nmp — decode에서 nmp 인상은 wall-clock에 거의 공짜 (P1)

- 근거: decode는 memory-bound. ncu 프로파일에서 flash_ozaki decode도 ALU 44%/DRAM 30%로
  포화가 아니고, linear GEMV는 weight fetch가 지배 → **digit product 추가(=nmp 인상)는
  메모리 지연 뒤에 숨는다**. 반대로 prefill은 compute-bound → 낮은 nmp 유지.
- 선행 질문(싼 실험): **어느 phase의 오차가 더 아픈가?** prefill 오차는 KV 표현을 한 번
  오염시키고, decode 오차는 토큰마다 autoregressive하게 누적. prefill-only vs decode-only
  에뮬레이션 두 run으로 분리 측정 (phase별 nmp 분기는 소규모 패치 필요 — 현재 knob 없음).
- 결과에 따라 "prefill 저nmp / decode 고nmp" 또는 그 역의 비대칭 구성 — 유효 비용은
  phase별 시간 가중 평균이라 overhead 정산상 매우 유리.

#### H2. Truncation bias의 정적 보정 — 런타임 비용 0 (P1)

- 절단된 digit-pair 항 Σ_(i,j)∈dropped A_i·B_j·2^(−w(i+j))의 기댓값을 calibration으로 추정,
  **per-out-channel 상수로 bias에 접어 넣는다**. weight digit(B_j)은 정적으로 정확히 알고,
  activation digit의 채널별 평균 E[A_i]만 calibration에서 추정하면 됨 → 런타임 추가 연산 0.
- caveat: prealign 블록 지수가 data-dependent라 digit 분해가 입력마다 달라짐 — E[A_i]
  추정 분산이 큼. 먼저 (미시) 보정 유/무 per-GEMM bias/relerr 비교로 싸게 검증.
- 초저비용 변형: 마지막 유지 digit-pair에서 절단 대신 최근접 반올림(G1의 결정론 버전).

#### H3. 유의성 삼각형 대신 기여도 기반 digit-pair 선택 (P2)

- 현재 nmp는 유의성 w·(la+lb) 순서의 삼각형 선택(`oz1fp_params`) — operand 대칭적.
  하지만 실제 기여도 E[|A_i·B_j|]는 분포에 따라 비대칭 (예: activation은 MSB digit에 에너지
  집중, weight는 상대적으로 균등). **같은 GEMM 개수에서 pair 집합만 calibration 통계로
  재선택** → 오버헤드 0의 오차 감소. nmp 스칼라를 pair-mask로 일반화하는 작은 커널 작업.
- R3에서는 H0의 1×n 퇴화가 이 문제를 대부분 흡수하므로, bf16 operand가 남는 곳
  (linear 입력 activation 등)에만 해당.

#### H4. 토큰(row) 단위 outlier — 변환·per-token 스케일로 흡수 (P2)

- attention sink/특수 토큰 등 소수 토큰의 activation이 극단적으로 큼. 이 row outlier는
  **dtype 승격이 아니라** per-token 스케일(FlatQuant A는 이미 per-token) + 변환으로 흡수한다.
  변환·클리핑 후에도 잔존하면 그때 Track D의 nmp를 해당 row에만 국소 인상(연산 정밀도만,
  저장 비트 불변)하는 것이 대안 — 8bit 승격은 쓰지 않는다.

#### H5. KV를 digit-plane으로 저장 — progressive code 성질 활용 (P1)

- digit plane은 **상위 plane이 하위 plane과 독립인 embedded/progressive code** — 하위
  plane을 버리는 것 = 재양자화 없는 정밀도 강등(plane eviction).
- 제안: KV를 bf16 대신 digit-plane(+블록 scale)으로 **원본 없이** 저장.
  (i) plane 1개 = KV4와 동일 저장량인데 ozaki attention의 매 스텝 K/V 재인코딩이 사라짐
  (기존 kv_cache_prefill이 기각된 이유 — bf16 원본과 digit 캐시의 이중 보관 — 자체가 소멸);
  (ii) **age-기반 강등**: 최근 window(예: 512 tok)는 2 plane(8bit), 그 밖은 1 plane(4bit)
  — eviction만으로 구현되어 오버헤드가 거의 0, 유효 KV bits ≈ 4+ε.
- E 트랙의 구체적 구현 경로이기도 함 (E1을 이 방식으로 수행 권장).

#### H6. 기타 소품 (P2, 한 줄씩)

- softmax-aware 2-pass QK^T: 저nmp로 스코어 스캔 → top-m key만 고nmp 재계산 (softmax 뒤엔
  최대치 근방만 유효; overhead ∝ m/L). H0-b로 QK^T가 exact해지면 불필요.
- head-wise nmp: layer보다 고운 할당 입자 (retrieval head만 고nmp) — Track D의 확장.
- 민감 layer만 k=32→16: prealign 블록 내 dynamic range 반감 → shift 손실 감소, scale
  저장 2배(그래도 미미). Track F(rotation)와 동일 지점을 다른 각도에서 공략.

---

## 3. 방법별 이득/손실 분석 — ozaki ALU 활용도(변환 평탄화·포맷 혼용·digit collapse) × 기대 이득 내림차순

판단 기준이 되는 ozaki1 ALU의 구조적 장점 세 가지:

- **(i) 포맷 무관성** — prealign + digit 분해 뒤에는 int4/mxfp4/nf4/bf16이 전부 같은
  정수 GEMM으로 수렴. **4bit 포맷 조합(mxfp4 블록 × int4 블록 등)** 마다 전용 커널이 필요
  없다. (outlier는 dtype 승격이 아니라 변환으로 제거하므로 fp8/int8 경로 자체가 없다.)
- **(ii) 정밀도 = 파라미터** — 정확도 조절이 커널 교체가 아니라 nmp/digit 수라는 숫자.
- **(iii) 저장 포맷 = 연산 포맷** — digit plane 자체를 저장 포맷으로 쓰면 dequant/re-encode
  단계가 소멸.

### 3.0 순위표 (내림차순)

| # | 방법 | ALU 장점 | 기대 정확도 이득 | 속도/메모리 | 리스크·공수 |
|---|---|---|---|---|---|
| 1 | H0 digit collapse | (i)+(iii) 최대 | attention 에뮬 3–5pt 소거 예측 | W/KV fetch ~4×↓ (decode-bound 직격) | scale↔prealign 정합, 中 |
| 2 | B outlier 변환 평탄화 (FlatKV식) | 상류(변환) | A4 주범(−9.8pt) 직접 공략 | 저장 +0bit, online 변환 FLOPs +~2.6% | 변환 학습 비용, 中 |
| 3 | H5 KV digit-plane 저장 | (iii) | KV4 저장 오차를 age-강등으로 완화 | KV fetch↓ + 매스텝 재인코딩 소멸 | PagedAttention 개조, 大 |
| 4 | C int4/mxfp4 혼용 | (i) | 저장 오차↓ (bit 예산 동일) | ±0 (+0.25bit 블록지수) | ragged layout, 中小 |
| 5 | D+H1 nmp 공간·시간 할당 | (ii) | 中 ("같은 정확도를 더 싸게"에 가까움) | decode ALU 여유 활용 | 탐색 비용, 小 |
| 6 | F rotation 결합 | 간접 | 미지 (미시 검증 先) | online 변환 비용 소량 | 小 |
| 7 | H2 truncation bias 보정 | (ii) | 小·불확실 | 0 | 추정 분산, 小 |
| 8 | H3/H4/H6/G 소품 | 부분적 | 小 | 小 | 각 小 |

순위 전제: H0을 먼저 채택하면 이후 방법들의 적용 범위가 달라진다 (예: attention 연산 오차가
소거되면 H6 2-pass가 불필요해지고, H3의 적용면은 bf16 operand가 남는 곳으로 줄어든다).
아래 상세에서 ⊕ = 이득, ⊖ = 손실·비용, 🔧 = system-level(data layout·커널·통합) 고려.

### #1 H0 — digit collapse (4bit 저장 = digit 1개)

⊕ **이득**
- 정확도: R2의 attention 에뮬 비용(~3–5pt)이 저장값 기준 exact 연산으로 소거될 것으로
  예측 — 단일 항목으로 프로젝트 최대 기대 이득.
- 연산량: bf16×bf16의 삼각형 T_n개 GEMM → 1×n 선형. 같은 커버리지를 절반 이하 GEMM으로.
- 메모리 traffic: weight/KV fetch가 bf16 대비 ~4×↓. 이 워크로드는 decode-bound
  (= bandwidth-bound)이므로 **정확도 개선과 실측 속도 향상이 같은 방향** — 유일하게 둘 다
  큰 폭으로 얻는 방법.
- weight_cache 문제의 소멸: 현재 nmp=6 캐시가 ~3× weight(TP=2 강제)인 것은 bf16 원본과
  별도로 digit을 미압축 보관하기 때문. W4에서는 4bit packed plane 1개가 **곧 모델 저장**
  이라 이중 보관 자체가 없다.

⊖ **손실·비용**
- W4 자체의 저장 오차(paper 기준 ~−1pt)는 그대로 남음 — B/C가 흡수할 몫.
- activation은 여전히 bf16 → 매 forward digit 인코딩(블록 max 스캔 + shift) 비용.
  compute-bound인 prefill에서 두드러질 수 있음.
- 정합 조건: 양자화 scale 격자 ↔ prealign 블록 지수. mxfp4(32-elem 블록)는 k=32와 자연
  정합; int4 per-channel/group-128 scale은 combine 단계에 per-channel 곱을 접어 넣는 커널
  수정 필요(연산 비용은 미미).

🔧 **system-level**
- weight digit-plane layout: plane-major `[nD][N][K/2 (2×int4 packed/byte)]`, K 최내측 —
  dp4a/imma 정합 + **nmp을 내리면 하위 plane fetch 자체가 skip**되어 traffic이 정밀도에
  비례해 절감되는 layout.
- 저장 포맷 = 연산 포맷 → dequant 커널·중간 버퍼 없음. GPTQ/AWQ 체크포인트에서 digit
  plane으로 1회 오프라인 변환하는 컨버터만 필요.

### #2 B — outlier를 learned affine 변환으로 평탄화 (FlatKV식)

⊕
- A4가 −9.8pt의 주범이라는 진단 하에 가장 직접적인 공략. LLM.int8-류(fp16 별도 GEMM +
  scatter/gather)나 fp8 채널 승격이 outlier를 "더 넓은 dtype에 담는" 것과 달리, 변환은
  **outlier를 채널 간에 재분배해 없애버린다** → 저장은 순수 4bit 균일 격자, 연산은 단일
  int GEMM. mixed-dtype·ragged·gather 문제 자체가 소멸.
- FlatQuant 근거: 학습 변환만으로 L3-8B W4A4 PPL 1266→8.5, 최종 6.98 (survey §3). 저장
  비트 증가 0.
- ozaki 시너지(#6 F): 평탄화된 블록은 prealign 후 dynamic range가 작아 같은 nmp에서
  relerr↓ — **한 변환이 저장 오차와 연산 오차를 동시에** 줄인다.

⊖
- 변환 파라미터의 **학습 비용**(FlatQuant 15 epoch, 셀당 GPU·h) — mixed-dtype의 "calibration
  스캔 1회"보다 무겁다. 단 저장/런타임 오버헤드는 fp8 승격보다 작음(저장 +0bit).
- online 변환 FLOPs(FlatQuant 보고 +2.61%, decode 1.7×) — dtype 승격의 ragged 오버헤드
  대신 지불하는 비용. Kronecker 분해로 O(d)로 억제.

🔧
- FlatQuant/QuaRot 변환을 online으로 적용한 뒤 ozaki digit split (Track F와 동일 경로).
  `fast-hadamard-transform`·FlatQuant 학습 변환 레포 기존재. 변환의 절반은 인접 weight에
  오프라인 folding되나 RoPE 직전 Q/K 등은 online 잔존 → **위치별 folding 가능성 표** 필요.
- attention activation(B1-ii)은 head_dim 변환(FlatQuant kcache/vcache 형)이라 linear 입력
  변환과 별개 hook — 위치별 변환 배치표 작성.

### #3 H5 — KV digit-plane 저장 (progressive code)

⊕
- decode attention은 KV fetch가 지배 — plane 1개(4bit)로 traffic ~4×↓, 그리고 현 eager
  경로 비용의 큰 몫인 **매 스텝 K/V 전체 재인코딩이 소멸** (plane은 쓰기 시 1회 인코딩,
  이후 재사용).
- age-강등(최근 window 2 plane = 8bit / 과거 1 plane = 4bit)이 **plane eviction만으로**
  구현 — 재양자화 pass 없음. KV4 저장 오차를 가장 민감한 최근 토큰에서 완화.

⊖
- vLLM PagedAttention 개조가 본격 엔지니어링(大): KV block layout에 plane 축 추가,
  block table/copy/swap 경로 수정.
- 토큰 단위 plane 수 ragged는 다루기 어려움 → **페이지(16–32 tok) 단위 강등**으로 단순화
  필요 (같은 페이지는 같은 plane 수).
- 블록 scale(지수) 저장·fetch 추가: +~0.25 bit/val 수준.

🔧
- layout: plane-major KV pages `[plane][page][H][page_size][D]` — 상위 plane만 읽는 decode
  fast-path와, 강등 시 하위 plane 페이지의 free가 모두 자연스러운 배치.
- B2(K의 RoPE-후 채널 outlier)는 kcache 변환으로 평탄화 후 저장 — digit-plane 저장과 독립 결합.

### #4 C — int4/mxfp4 혼용

⊕
- 같은 4bit 예산에서 분포 적합 포맷 선택 — bit overhead 0. 전용 HW라면 int4/fp4 두
  datapath가 필요하지만, ozaki ALU는 mxfp4 블록 지수를 prealign 지수에 흡수하면 동일 경로.

⊖
- mxfp4 블록 지수 +0.25 bit/val. per-channel 혼용은 B와 같은 ragged 문제(같은 permutation
  folding으로 해소). 포맷 선택이 calibration 의존.

🔧
- mxfp4 블록(32) = k=32 정렬이 강제 조건. group-128 int4와 혼용 시 scale 계층 2단 설계
  필요. 선택 결과는 per-tensor 메타데이터로 고정 — 런타임 분기 없음.

### #5 D + H1 — nmp의 공간(layer/op/head)·시간(phase) 할당

⊕
- 커널 불변, 파라미터만(장점 (ii)) — 공수 최소. decode에서 weight_cache off 경로는 bf16
  fetch 고정 + ALU만 증가이므로 nmp 인상이 wall-clock ~공짜 (H0 채택 후에는 activation
  digit 수 할당에 해당).

⊖
- 이득 상한 제한: 그리드에서 균일 nmp≥4가 이미 near-lossless — 이 트랙은 "정확도 올리기"
  보다 **"같은 정확도를 더 싸게"**의 도구로 봐야 함. 민감도 프로파일링 자체가 수십 run.
- weight_cache on이면 fetch ∝ plane 수 → decode 공짜 논리가 약해짐 (4bit packing 시 완화).

🔧
- layer-index 단위 override 확장(현재는 op-이름 매칭만), phase 분기 knob 신설 — 둘 다
  소규모 패치. head-wise는 커널 grid에 head→nD 테이블 필요.

### #6 F — 변환의 ozaki 연산 시너지 (Track B의 연산측 근거)

Track B가 변환을 **저장 정확도**(균일 int4 흡수)를 위해 쓴다면, 같은 변환은 **연산측**에도
이득을 준다 — 이 절은 그 시너지의 미시 검증을 다룬다. (B와 F는 같은 변환의 두 이득 면.)

⊕
- 블록 내 dynamic range 평탄화 → prealign shift 손실 감소 → 같은 digit 수(nmp)에서 relerr↓.
  즉 B의 변환이 저장 오차와 ozaki 연산 오차를 **동시에** 줄인다 (한 파라미터, 두 이득).

⊖
- online 변환이 매 forward 추가(FlatQuant 학습 변환 = per-layer Kronecker 곱, Hadamard =
  O(d log d)). 연산측 이득 크기는 미지 — **미시(per-GEMM relerr) 검증을 반드시 먼저**.

🔧
- `fast-hadamard-transform` 커널·FlatQuant 학습 변환 기존재. QuaRot 방식대로 변환의 절반은
  weight에 오프라인 folding, activation 쪽 절반만 런타임 — B와 folding 표 공유.

### #7 H2 — truncation bias 정적 보정

⊕ 런타임 0, 저장 0(bias에 folding).
⊖ prealign 지수가 data-dependent → E[digit] 추정 분산 큼. bias 보정은 평균 오차만 잡고
분산은 못 잡으므로 기대 이득 자체가 작음. 도메인 shift 취약.
🔧 없음(오프라인 계산뿐). 미시 검증에서 relerr 평균이 안 움직이면 즉시 폐기.

### #8 소품 (H3 / H4 / H6 / G)

- **H3 (pair-mask)**: ⊕ 같은 GEMM 수에서 pair 재배치 — overhead 0. ⊖ layer별 mask →
  커널 변종 증가(`flash_oz1fp_codegen` 방식으로 억제 가능). H0 채택 시 적용면이 bf16
  operand가 남는 곳으로 축소.
- **H4 (row 승격)**: ⊕ sink 토큰 보호를 소량 overhead로. ⊖ 동적 row gather + skinny GEMM
  저효율, batch 내 ragged. B(채널축)로 흡수되는지 먼저 확인.
- **H6 (2-pass QK^T 등)**: H0-b가 성공하면 불필요 — H0 결과 대기.
- **G1 (SR)**: integer-ALU-bound 커널에 RNG 추가는 역효과 가능. **G2 (LoRC)**: rank-r GEMM
  추가 — bits-per-value 정산에 넣으면 B 대비 열위일 가능성 높음.

## 4. 평가 프로토콜 (모든 트랙 공통)

- **모델**: DeepSeek-R1-Distill-Qwen-7B (1차). lossless 달성 시 1.5B(더 취약)와
  QwQ-32B(스케일 확인)로 확장.
- **1차 스크리닝**: GSM8K — attention 에뮬 셀 기준 MATH-500보다 ~2× 빠름(trace 길이가
  절반, attn 비용 ∝ Σtok²). 단 near-ceiling이라 변별력 낮음 → **통과/탈락 필터로만** 사용.
- **본 측정**: MATH-500 (bf16 0.948 기준). 최종 후보만 AIME-120 + GPQA-Diamond.
- **판정**: ≤1pt drop = lossless, 1–3pt = fair, ≥3pt = risky (paper taxonomy).
  ±2pt는 노이즈 — 경계 판정은 McNemar exact test로.
- **공정성**: 모든 결과에 bits-per-value(W/A/KV 각각)와 overhead(%) 병기. FlatQuant 4-4-4
  (84.1)와 같은 표에서 비교.
- **재현성**: seed 42, temp 0.6, top_p 0.95, max 32k — 기존 스윕과 동일. 결과는 셀 단위로
  `lossless_444/results/`에 즉시 기록 (기존 `sweep_ozaki_nmp_grid.py`의 resumable 패턴 재사용).

## 5. 비용/자원 주의사항

- attention 에뮬 MATH-500 셀 = **~14h/셀** (eager backend, O(L²)). 스크리닝은 GSM8K로,
  attention 관련 트랙(B1-ii, E)은 flash_ozaki 경로 활용을 우선 검토.
- nmp 큰 설정의 `weight_cache`는 ~3× weight 메모리 → 2×48GB TP=2 필요 (또는 cache off,
  bitwise 동일하나 느림).
- 기존 캐노니컬 해시 run은 공짜로 재사용 가능 — 새 스윕 스크립트도 같은 패턴 유지할 것.

## 6. 로드맵 (제안 순서)

| 순서 | 항목 | 근거 |
|---|---|---|
| P0-1 | A1–A3 베이스라인 + 분해표 | 모든 의사결정의 잣대. fake-quant라 GPU 비용 낮음 |
| P0-2 | H0-a/H0-b + R3 첫 실험: 4bit 저장 + ozaki 연산 | digit collapse 예측(저장 양자화가 연산 오차를 소거) 검증 — 이후 트랙 전부의 전제 |
| P0-3 | B 변환 평탄화(activation/KV) + F 미시검증 | A4가 주범이라는 가설의 직접 공략(변환으로 outlier 제거), 기대값 최대 |
| P1 | B2(KV), C(포맷 혼용), D(nmp greedy), E=H5(KV digit-plane), H1(phase-aware), H2(bias 보정) | P0 결과가 가리키는 쪽부터. H1/H2/H5는 오버헤드 ~0이라 병렬 진행 가치 |
| P2 | G(보정 소품), H3/H4/H6 | 싸게 검증 → 되면 크게 (F는 B와 함께 P0-3로 상향) |

**첫 마일스톤**: "FlatQuant W4A4KV4(84.1) 대비, 유효 ≤4.5bit에서 MATH-500 ≥93 (−1pt 이내)"
를 달성하는 구성 1개.

## 7. 디렉토리 구조 (예정)

```
lossless_444/
├── PROPOSAL.md          # 이 문서
├── configs/             # 트랙별 실험 config (yaml)
├── scripts/             # 스윕/분석 스크립트
├── results/             # baseline_matrix.md, 트랙별 결과 표 (셀 단위 즉시 기록)
└── notes/               # 실험 노트, 실패 기록 포함
```
