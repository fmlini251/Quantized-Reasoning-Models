# 4bit number format · 관련 논문 서베이 (Track C 배경)

> 실험 절차는 [mixed_4bit.md](../mixed_4bit.md), ozaki 관점 분석은
> [mixed_4bit_theory.md](mixed_4bit_theory.md).

## 1. 4bit number format 정리

값 = (element code) × (block scale) [× (tensor scale)]. 모든 4bit 포맷은 "16개 코드 격자
+ 스케일 계층"으로 환원된다.

| 포맷 | element 격자 (×block scale) | block | scale dtype | bits/val* | 출처 |
|---|---|---|---|---|---|
| INT4 per-channel sym | 균일 {−7..7} (또는 {−8..7}) | 채널 전체 | fp16 | ≈4.004 | FlatQuant W |
| INT4 g128 | 균일 | 128 | fp16 | 4.125 | GPTQ/AWQ 관행, FlatQuant KV |
| INT4 per-token | 균일 | 행(K=수천) | fp16 | ≈4.004 | FlatQuant A |
| NVINT4 | 균일 | 16 | FP8 E4M3 | 4.5 | MixFP4 비교군 |
| MXINT4 | 균일 | 32 | E8M0 (2의 거듭제곱) | 4.25 | OCP MX 자연 확장† |
| MXFP4 | E2M1 {0,½,1,1½,2,3,4,6} | 32 | E8M0 | 4.25 | OCP MX v1.0 |
| NVFP4 | E2M1 | 16 | E4M3 (+fp32/tensor) | 4.5 | NVIDIA Blackwell |
| E1M2 | 균일 {0,½,…,3½} ≡ INT4×½ | 16 | E4M3 | 4.5 | MixFP4 제안 |
| E3M0 | 2의 거듭제곱 {…,½,1,2,4,8} | — | — | — | MixFP4에서 기각 |
| NF4 | N(0,1) 분위수 (무리수 격자) | 64 | fp32 absmax | 4.5 (DQ시 4.127) | QLoRA |
| MixFP4 | 블록별 E2M1 ∨ E1M2 | 16 | E4M3 (sign bit = 포맷 비트) | 4.5 (+0) | MixFP4 |

\* scale 오버헤드 포함 유효 비트. † OCP 스펙에는 MXINT8까지만 정의 — MXINT4는 자연 확장
(theory 노트: ozaki 1-digit encode와 동일물).

### 격자 특성 — 어떤 분포에 맞는가

- **균일 격자 (INT4-계열, E1M2)**: 상대 MSE ≈ κ²/(12·49), κ = 블록 crest factor
  max|x|/RMS. κ 작을수록(값이 고르게 클수록) 유리. outlier 하나가 블록 max를 끌어올리면
  나머지 전부의 해상도가 죽는 것이 약점.
- **E2M1 (FP 격자)**: 작은 값 근처 촘촘(0.5 간격), 큰 값 근처 성김(4→6). 상대 오차가 값
  크기에 대해 대략 일정 → heavy-tail / 고κ 블록에 유리.
- **E3M0**: dynamic range만 크고(2⁷) 레벨이 없음 — 큰 블록(g≥32)에서만 간혹 선택됨
  (MixFP4 ablation). 기각.
- **NF4**: 정규분포 저장 SNR 이론 최적. 그러나 무리수 격자라 정수 lattice에 정합 불가 —
  ozaki 경로 비호환(theory 노트 §2), R1(fake-quant) 대조군 전용.
- unsigned FP 변형 (E1M3/E2M2/E3M1): 비음수 텐서용, sign bit를 지수/mantissa에 재투자.
  정렬 후 격자와 plane 비용은 theory 노트 §6.

## 2. MixFP4 — *Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations*

arXiv 2605.31035 (2026-05, cs.AR). NVFP4 스케일 계층 안에서 **블록(16-elem)별로 E2M1 vs
E1M2(=INT 격자)를 선택**하는 포맷.

- **선택 기준**: 블록마다 두 포맷으로 quant-dequant 후 **MSE argmin** (brute force,
  Algorithm 1). RTN 호환, SmoothQuant/GPTQ/SpinQuant 파이프라인과 합성 가능.
- **이론**: 블록 crest factor κ = max|X|/RMS(X)의 닫힌꼴 QSNR 모델 → 교차점
  **κ\* = 2.2243** (QSNR 21.03dB): κ<κ\*면 INT-격자 승, 이상이면 E2M1 승. 텐서 내부의
  블록 κ 편차가 커서 per-tensor가 아닌 per-block 선택이 필요(Fig.3).
- **메타데이터 0bit — "Type-in-Scale"**: E4M3 블록 스케일은 항상 양수 → sign bit를 포맷
  비트로 재활용.
- **결과 (PTQ RTN, WikiText2 PPL, BF16/NVFP4/NVINT4/MixFP4)**: Llama-3.1-8B
  7.33/8.26/8.37/**8.06**; Qwen3-8B 12.21/12.74/12.73/**12.39**; Llama-3.2-1B
  11.57/13.90/14.79/**13.48**. RHT(random Hadamard) 하에서 격차 확대 (L3.1-8B: NVFP4 8.55
  vs MixFP4 8.10) — **회전으로 Gaussian화되면 선택이 INT-격자로 쏠리는데도 혼용이 이득**.
- **PTQ 합성 (Llama-3.2-1B PPL)**: GPTQ 13.24→12.91, SpinQuant 14.44→13.38 (NVFP4→MixFP4).
- **ablation**: 후보를 넓혀도(E3M0 등) 선택은 E2M1/E1M2 둘로 수렴(Fig.5) — 포맷 2개 +
  1bit 메타로 충분. g=16이 최적, g=32/64에서 열화.
- **HW**: 두 포맷을 MAC 내부 E2M2 통합 표현으로 디코드 — TSMC 28nm 합성 기준 tensor core
  대비 **+3.1% 면적, +1.5% 전력**.
- **적용 범위·한계 (= 우리가 채울 공백)**: linear GEMM의 W4A4만. **KV cache 양자화 없음,
  attention 내부 없음, GSM8K/MATH 등 reasoning 벤치마크 전무.** FlatQuant/QuaRot과의 직접
  비교 없음(SpinQuant가 회전 대표).

## 3. FlatQuant — *Flatness Matters for LLM Quantization*

arXiv 2410.09426 (ICML 2025), github.com/ruikangliu/FlatQuant. 우리가 이길 대상이자
Track F에서 결합할 상류 변환.

- **방법**: layer마다 학습형 affine 변환으로 W/X를 평탄화 후 **uniform INT4**.
  Θ = {P, c, α_w, α_a}: 가역 변환 P = P₁⊗P₂ (Kronecker 분해, n=4096→64×64), per-channel
  scale diag(c), sigmoid-bounded 학습형 클리핑 α. layer-wise 출력 MSE로 학습.
- **양자화 구성 (W4A4KV4)**: W = INT4 per-channel sym (RTN/GPTQ), A = INT4 **per-token**
  sym, KV = 4bit g128 asym per-head.
- **calibration**: WikiText-2 128 seq × 2048 (레포 논문은 R1-Distill에 seqlen 4096),
  bs 4, 15 epoch, AdamW lr 5e-3. LLaMA-3-8B 기준 1 GPU ~0.9h.
- **결과 (W4A4, WikiText2 PPL, FP16/QuaRot/SpinQuant/FlatQuant)**: L2-7B
  5.47/8.56/6.14/**5.79**; L3-8B 6.14/10.60/7.96/**6.98**; L3-70B 2.86/55.44/7.58/**3.78**.
  Zero-shot QA avg에서 L3-70B <1% drop.
- **online 변환 비용**: 변환+양자화를 단일 Triton 커널로 fusion — FLOPs +2.61%, 속도
  저하 0.07× (QuaRot 0.26×). E2E 속도 prefill 2.3× / decode 1.7× vs FP16.
- **ablation (L3-8B W4A4 PPL)**: RTN 1266.60 → +학습 변환 8.50 → +per-channel scale 7.95
  → +클리핑 6.98. 변환이 거의 전부.
- **레포 논문(arXiv 2504.04823)에서의 사용**: weight-activation 양자화의 leading
  algorithm으로 채택 (`scripts/quantization/flatquant.sh`, 레포는 `--a_asym` 사용 — 원
  논문과 다름). **R1-Distill-Qwen-7B W4A4KV4: MATH-500 94.6→82.8 (−11.8), GSM8K
  91.4→90.8 (−0.6), AIME-120 45.0→25.0 (−20.0), LiveCodeBench −20.9. 평균 −10.4
  ("risky").** GSM8K가 거의 안 무너진다는 것 = GSM8K는 A4 방법 변별력 없음.

## 4. 인접 논문 (한 줄씩)

- **Four over Six** (Cook et al. 2025): adaptive NVFP4 블록 스케일링 — MixFP4의 최근접
  선행 (4/6 baseline으로 비교됨).
- **MicroMix** (arXiv 2508.02343): microscaling 포맷 기반 mixed-precision.
- **KVQuant / KVQuant\***: K의 post-RoPE 채널 outlier 관찰 — B2의 **kcache 변환이 무엇을
  평탄화해야 하는지**의 진단 근거(FlatQuant kcache 변환과 동일 대상). `methods/kvquant_star`
  통계 재활용 가능.
- **QLoRA** (arXiv 2305.14314): NF4 원전 (weight-only, 저장 전용).
- **QuaRot / SpinQuant**: Hadamard/학습 회전 — **Track B(변환 평탄화, outlier 주공략)** 및
  Track F(연산 시너지), `fast-hadamard-transform` 레포 기존재.
