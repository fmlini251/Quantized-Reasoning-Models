# Track C 이론 노트 — ozaki1 ALU 관점의 4bit 포맷 분석

> 실험 절차는 [mixed_4bit.md](../mixed_4bit.md), 포맷·논문 서베이는
> [mixed_4bit_survey.md](mixed_4bit_survey.md).

## 1. exactness 판정 기준과 포맷별 digit 비용

ozaki1 prealign은 chunk(k=32, 유효 {16,32,64,128}, `ozaki_matmul.py:1317`)마다 공통 지수
2^dv로 나눠 **int_bits = w·nD−1 비트 정수**로 만든다. 저장 포맷이 "digit collapse"
(저장값 기준 exact 연산)하려면:

> 블록 내 모든 값이 **하나의 2^e 간격 균일 격자 위의 정수**여야 하고, 그 정수 범위가
> nD개 digit(w=4: nD=1이면 [−8,7], nD=2면 [−128,127])에 들어가야 한다.

| 포맷 | 정렬 후 정수값 | exact digit 수 (w=4) |
|---|---|---|
| INT4 전 변형 / MXINT4 / E1M2 / NVINT4 | {−8..7} / {0..7} | **1** |
| MXFP4 / NVFP4 (E2M1) | ×2 정렬 시 {0,±1,±2,±3,±4,±6,**±8,±12**} | **2** (2nd plane은 {−1,0,+1} ternary, 정렬 정수 **{+8,±12}에서만** non-zero → 희소. −8은 balanced digit [−8,7]의 plane0에 들어가므로 support가 비대칭 — fmt_lib G2 테스트로 확정, 이론상 밀도는 "±4/±6 코드" 대칭 가정보다 더 낮음) |
| E3M0 | {±1..±64} | 2 |
| NF4 | 무리수 — 정수 격자 없음 | **불가** (bf16 dequant 필요 → exactness·storage=compute 성질 모두 상실) |

**E2M1이 1 digit이 아닌 이유** (PROPOSAL H0의 "mxfp4 mantissa도 블록 지수를 쓰면 동일"
가정의 **정정**): E2M1은 element마다 2bit 지수를 따로 가지므로 블록 지수 정렬 후에도
dynamic range가 12:0.5 = 24배 — 최소 간격(0.5)으로 정렬하면 정수 {0..12}가 되어 5bit
필요, w=4 digit 1개([−8,7])를 초과한다.

> **같은 4bit 저장이라도 ozaki 연산 비용이 다르다: INT-격자 = 1 plane, E2M1 = 2 plane.**

## 2. 스케일/오프셋 fold — exactness는 격자만, 하드웨어 비용은 "스케일이 어디 붙느냐"

**exactness 관점**에서는 스케일 dtype이 무관하다 (§1 판정은 element 격자만 본다). 그러나
**하드웨어 비용**은 스케일이 datapath 어디에서 곱해지느냐로 갈린다 — reduction 축(k)에서
빠지는가(factor-out), 그리고 2의 거듭제곱인가.

| 스케일 종류 | reduction 축 | HW 위치 | 비용 |
|---|---|---|---|
| per-tensor / per-out-channel / per-token | **빠짐** (k 무관) | epilogue O(M·N) | **dtype 무관 공짜** — 모든 int-GEMM의 dequant 후처리 재사용 (int32 누산 → fp 스케일 곱 → 출력) |
| per-block, 2의 거듭제곱 (E8M0, MXINT4/MXFP4) | 안 빠짐 (chunk별) | in-reduction | shift / 지수-덧셈 (MAC 안) |
| per-block, mantissa (E4M3, **NVFP4**) | 안 빠짐 + mantissa | in-reduction | **진짜 곱셈 O(M·N·nchunk) — pow2 설계엔 없는 곱셈기 추가** |

- 코드 근거: `standalone_oz_gemm.py:60` `acc += cacc * sA[:,None] * sB[None,:]`가 per-chunk
  block scale을 **reduction 루프 안에서** 적용한다. 현재 sA/sB는 pow2(`_bfp_scale`)라
  하드웨어로는 shift면 되지만, E4M3 mantissa를 넣으면 이 자리가 곱셈기가 된다. NVIDIA
  Blackwell이 tensor core에 microscaling 하드웨어를 넣은 것이 정확히 이 비용의 지불.
- per-channel/token fp 스케일은 k에 무관 → `out=s_row[m]·s_col[n]·(Σ_k …)`로 인수분해되어
  출력당 1번(epilogue). group 크기가 chunk 배수(128=4×32)면 chunk마다 상수라 정확히 접힌다.
  FlatQuant의 W(per-channel)·A(per-token) 스케일이 이 행 = **공짜** (dtype 무관).
- 비대칭 zero-point: **정적(per-channel) z는 z·colsum(W)가 bias로 접혀 epilogue 덧셈 =
  공짜**. **동적 per-token z는 rowsum(상대 operand) + rank-1 보정** 필요 — epilogue O(M·N)
  + reduction 1회라 싸지만 literally 0은 아님 (1차 실험 제외).

→ exactness는 element 격자만, **하드웨어 곱셈기 추가는 per-block mantissa 스케일(=NVFP4)
  일 때만** 발생.

## 3. NVFP4 호환성 판정: **호환 (2-plane 비용)**

| NVFP4 구성 요소 | ozaki1 매핑 | 비용 |
|---|---|---|
| element E2M1 | digit 2개 (2nd ternary 희소) | plane +1 |
| block 16 | chunk=16 (`_OZ1_VALID_CHUNK`에 기존재) | 스케일 메타 2×/32elem, 블록 κ↓ 부수 이득 |
| E4M3 block scale | 2^e → prealign 지수, (1+m/8) → in-reduction 곱 | **곱셈기 추가** (pow2 MX는 shift로 족함; §2) |
| per-tensor fp32 | epilogue 스칼라 | 0 (factor-out) |

즉 NVFP4/MXFP4는 "돌아간다" — 단 **ozaki1 HW의 native cheap 포맷은 INT-격자(1 plane,
pow2)**이고 E2M1-계열은 2-plane 시민. 특히 NVFP4는 **두 겹으로 비싸다**: (i) element
E2M1 = 2 plane, (ii) E4M3 block scale = in-reduction 곱셈기(§2). MXFP4(E8M0)는 (ii)를
안 낸다. native-cheap 사다리: **MXINT4(1 plane, shift) < MXFP4(2 plane, shift) <
NVFP4(2 plane, 곱셈기)**. NVIDIA Blackwell(E2M1이 native, INT4가 특별히 쌀 이유 없음)과
**정반대의 비용 구조** — 포맷의 우열이 datapath에 상대적이라는 것 자체가 codesign 논점(§8).

## 4. outlier 대응의 두 층 — 변환(Track B) × 포맷(Track C)

**개정 (2026-07-09)**: 기존 "fp8 채널 승격(Track B) vs E2M1 블록(Track C)"의 두 승격 축
비교는 폐기. fp8/int8 8bit 승격은 **learned affine 변환(Track B, 저장 +0bit)** 으로 대체됐다.
이제 두 층은 경쟁이 아니라 순차 결합이다:

| 층 | 방식 | 저장 비용 | 연산 비용 | outlier에 하는 일 |
|---|---|---|---|---|
| Track B: 변환 평탄화 | 회전/affine (FlatQuant) | **+0bit** (변환 파라미터만) | online 변환 곱(+ozaki relerr↓) | outlier를 채널 간 재분배해 **제거** |
| Track C: 블록 → E2M1 | per-block 포맷 선택 | **+0bit** (같은 4bit!) | +1 ternary plane | 변환 후 잔여 고κ 블록의 격자 정합 |

변환이 κ를 먼저 낮추므로 E2M1의 여지는 줄지만(C5가 잔여 이득을 직접 측정), 둘 다 저장
비트를 늘리지 않는다는 점이 핵심 — 폐기된 8bit 승격과 달리 유효 비트 예산을 지킨다.

## 5. A4 저장은 ozaki에서 공짜 — 포맷 문제의 환원

ozaki가 매 forward 수행하는 activation 인코딩(블록 amax → E8M0 지수 → w-bit digit)은
**그 자체가 MXINT4 양자화다** (nD=1일 때). 따라서:

- A의 "4bit 저장"은 추가 양자화도 dequant도 없이 성립 (storage = compute format).
  **단 K에는 raw ozaki 인코드 규칙을 그대로 쓰지 말 것** (2026-07-09 e2e ablation):
  frexp scale은 amax/s∈[4,8)이라 블록 top element가 8→7 clamp로 ~12% 오차를 먹는데,
  K에선 그 원소들이 QK logit을 지배해 PPL 10× 폭발 (V·A는 둔감). K 저장은 MSE-opt
  pow2 scale(±1 octave 탐색, 여전히 정수×pow2 격자 = digit-collapse 호환) 필수.
- **A의 포맷 선택 = digit-plane 할당 문제**로 환원 (1 plane / 1+ternary / 2 plane).
  Track C(A쪽)와 Track D/H(nmp 할당)는 같은 손잡이의 다른 이름.
- 반면 **W와 KV는 진짜 저장 포맷 문제** — 체크포인트/캐시에 눕는 값의 격자 선택.

## 6. FlatQuant 대비 격자 우위 가설

FlatQuant A4는 **per-token 스케일**(행 하나, K≈3584 elem에 스케일 1개)의 균일 INT4.
MXINT4는 32-elem마다 지수 — **스케일 입자 ~112× 고움**. 가설: (i) block-32 동적 지수만
으로 FlatQuant A4보다 저장 오차↓ (C0-4 검증), (ii) 잔여 고κ 블록만 승격하면 A4 오차가
reasoning 붕괴 임계 아래 (C1/C4 검증).

## 7. digit lattice 위의 custom format 설계 공간 (실험군 B의 이론적 기초)

§1의 판정 기준을 뒤집으면 설계 규칙이 된다:

> **4bit 포맷 = digit lattice 위 정수 16개(코드)의 선택.** ozaki 비용은 격자 모양이
> 아니라 최대 코드의 비트 수(= plane 수)와 상위 plane의 희소성으로만 결정된다. 표준
> 포맷은 이 설계 공간의 특수점들일 뿐이다.

- **1-digit signed** ([−8,7]): 정수 16개가 꽉 참 — 격자 자유도 없음, INT4로 유일
  (남는 자유도는 scale/offset뿐).
- **1-digit unsigned** ([0,15]): 비음수 텐서(softmax P, σ(x), exp)에서 sign bit는 순수
  낭비. digit split을 unsigned로 바꾸면(int_bits = w·nD−1 → **w·nD**) **같은 1 plane에서
  해상도 2× = UINT4**. unsigned E1M3(denormal 포함)의 격자 {0..15}는 UINT4와 동일물.
  HW 비용은 ALU 모드 플래그 수준 (int8 곱셈기는 0..15 × [−8,7]을 그대로 소화).
- **2-digit** ([−128,127]): 진짜 custom 격자의 공간. **E2M1은 이 공간의 한 점일 뿐.**
  설계 지침: 대부분의 코드를 1-digit 범위 안에, 소수 tail 코드만 밖에 → 상위 plane이
  희소해져 실효 1+ε plane. calibration 분포의 Lloyd-Max/분위수를 lattice에 스냅한
  **"lattice-LUT" = NF4의 ozaki-호환 대체물**.
- unsigned FP 변형의 정렬 후 격자: E1M3 = {0..15} (=UINT4, 1 plane), E2M2 =
  {0..7, 8,10,12,14, 16,20,24,28} (2 plane, 상위 digit ∈{0,1,2}), E3M1 = {0..192}
  (2 plane) — dynamic range를 살수록 plane·희소성 비용.
- softmax P에의 적용: P는 top-heavy — 상단 해상도는 균일(UINT4)이, 꼬리는 FP형이 유리.
  PV 출력 오차는 |V| 가중이라 상단 지배 예상 → UINT4 우세 가설. 성공 시 H0-b의 "P
  2-digit" 가정이 1 unsigned digit로 — PV가 GEMM 1개로 exact (OP1 검증).

## 8. codesign 관점

1. **디코드 로직 0** — INT-격자 포맷은 그 자체가 digit plane, E2M1은 plane+ternary
   side-plane. 포맷 차이가 datapath가 아니라 **plane 수·희소성 패턴**으로만 나타난다.
   (MixFP4는 MAC 루프 안에 E2M2 디코더를 넣어 +3.1% 면적. 폐기된 fp8 승격은 plane 2개였음.)
2. **custom 포맷의 HW 비용 위치** — lattice-LUT의 "디코드"는 GEMM datapath가 아니라
   **양자화(인코드/쓰기) 시점의 16-entry 테이블**에만 존재. unsigned digit은 모드 플래그
   1개. → **포맷 혁신(실험군 B)이 datapath 재설계 없이 가능**하다는 것이 codesign의
   실질 이득.
3. **정밀도 = plane 수 = fetch량** — plane-major layout에서 포맷/nmp을 내리면 하위 plane
   fetch가 물리적으로 skip → bandwidth-bound decode에서 정확도-속도 tradeoff가 메모리
   traffic에 1:1 대응 (H0/H5와 동일 지점).
4. 최종 산출물 = "분포→포맷 매핑표(표준 + op-aware) + 단일 정수 ALU에서의 exact plane
   할당" — 양자화 알고리즘(SW)과 datapath(HW)가 같은 언어(digit plane)로 기술되는
   codesign 완성형. MixFP4(포맷마다 디코더)와도, FlatQuant(균일 격자 고정)와도 다른 지점.
