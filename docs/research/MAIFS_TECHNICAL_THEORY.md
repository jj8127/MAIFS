# MAIFS 기술 이론 백서

> 대상: MAIFS를 처음 접하는 연구자/개발자  
> 목적: 프로젝트에 사용된 핵심 이론, 수식, 설계 의사결정, 한계를 코드 기준으로 상세 정리

---

## 0. 문서 범위와 원칙

이 문서는 `src/` 및 `experiments/`의 실제 구현을 기준으로 작성한다.

- 범위 포함
1. 4개 포렌식 에이전트 이론
2. COBRA 합의 알고리즘(RoT/DRWA/AVGA)
3. 다중 에이전트 토론 프로토콜
4. DAAC(Disagreement-Aware Adaptive Consensus) 메타학습 이론
5. 통계 검정 및 캘리브레이션 지표
- 범위 제외
1. 외부 모델 저장소의 학습 레시피 전체 재현
2. 본 저장소에 없는 논문 구현 세부

핵심 코드 참조:

- `src/tools/`
- `src/agents/`
- `src/consensus/cobra.py`
- `src/debate/`
- `src/meta/`
- `experiments/run_phase1.py`

---

## 1. 문제 정의와 MAIFS의 기본 관점

### 1.1 문제 정의

MAIFS는 입력 이미지 \(x\)에 대해 최종 판정 \(y\in\{\text{authentic},\text{manipulated},\text{ai\_generated},\text{uncertain}\}\)을 예측한다.

중요한 점은 단일 모델이 아니라, **서로 다른 신호 도메인**을 보는 전문가 집합의 불일치까지 정보로 사용한다는 점이다.

### 1.2 왜 다중 에이전트인가

이미지 포렌식에서 단일 신호는 항상 취약점이 있다.

1. 주파수 기반은 최신 diffusion에 약할 수 있다.
2. 센서 노이즈 기반은 압축/리사이즈에 약할 수 있다.
3. 의미론 기반은 부분 조작(local forgery)에 둔감할 수 있다.
4. 공간 분할 기반은 전역 생성 이미지에 약할 수 있다.

MAIFS는 이 상보성을 이용해 **합의 + 토론 + 메타학습** 계층으로 오류를 완화한다.

---

## 2. 시스템 계층 이론

MAIFS는 아래 4계층으로 해석할 수 있다.

1. Tool Layer: 도메인별 원시 증거 추출
2. Agent Layer: 증거 해석 및 신뢰도 부여
3. Consensus/Debate Layer: 집단 의사결정
4. Meta Layer (DAAC): 불일치 패턴의 학습적 활용

수식적으로:

\[
\text{Tool outputs } z_i \rightarrow \text{Agent responses } a_i=(v_i,c_i,e_i)
\rightarrow \text{Consensus } \hat{y}_{cobra}
\rightarrow \text{Meta } \hat{y}_{daac}
\]

여기서 \(v_i\)는 판정, \(c_i\in[0,1]\)는 신뢰도, \(e_i\)는 증거다.

---

## 3. 4개 에이전트 이론

## 3.1 Compression Pillar: CAT-Net (Frequency 슬롯)

구현: `src/tools/catnet_tool.py`

### 3.1.1 포렌식 배경

JPEG 압축은 \(8\times8\) DCT + 양자화 + 부호화 과정이다.  
조작 후 재저장(특히 이중 압축) 시, DCT 계수 분포와 양자화 테이블 일관성에 흔적이 남는다.

### 3.1.2 CAT-Net 핵심 아이디어

CAT-Net은 RGB 스트림 + DCT 스트림을 결합하여 조작 마스크 \(M\in[0,1]^{H\times W}\)를 예측한다.

- 입력: RGB, DCT volume, qtable
- 출력: 픽셀 단위 조작 확률 맵

코드상 후처리:

1. softmax 후 조작 클래스 채널 추출
2. 원본 해상도로 리사이즈
3. `mask_threshold`로 이진화해 조작 비율 \(\rho\) 계산

\[
\rho = \frac{1}{HW}\sum_{p}{\mathbb{1}[M_p \ge \tau]}
\]

### 3.1.3 판정 매핑

- \(\rho < t_{auth}\): authentic
- \(\rho \ge t_{manip}\): manipulated
- 그 사이: uncertain

임계값은 `configs/tool_thresholds.json`의 `compression` 섹션으로 보정된다.

### 3.1.4 폴백 이론

CAT-Net 가중치/환경 실패 시 `FrequencyAnalysisTool`(FFT heuristic)로 폴백한다.  
즉, 이 슬롯은 "압축 기반 1순위 + 주파수 heuristic 2순위" 구조다.

---

## 3.2 Frequency Heuristic Fallback: FFT 기반 분석

구현: `src/tools/frequency_tool.py`

### 3.2.1 FFT 기반 신호

\[
F(u,v)=\mathcal{F}\{I(x,y)\},\quad S(u,v)=\log(1+|F(u,v)|)
\]

`S`에서 다음 특징을 추출한다.

1. Grid artifact score
2. GAN checkerboard score
3. Power spectrum slope
4. High/low frequency ratio

최종 AI score는 가중합:

\[
s_{ai}=0.30s_{grid}+0.30s_{chk}+0.35s_{slope}+0.05s_{hf}
\]

JPEG 의심 시 penalty를 차감한다.

### 3.2.2 JPEG vs GAN 구분 논리

JPEG의 8x8 블록 주기와 GAN 업샘플링 주기 신호를 분리하려고, 예상 피크 위치 기반 탐지를 사용한다.

### 3.2.3 구조적 한계

학습 없는 규칙 기반이므로 생성기 변화에 취약하다.  
따라서 MAIFS에서는 이 방법을 메인 축이 아닌 CAT-Net 폴백/보조 축으로 둔다.

---

## 3.3 Noise Pillar: PRNU/SRM + MVSS

구현: `src/tools/noise_tool.py`

### 3.3.1 PRNU 물리 모델

고전 모델:

\[
I = I_0(1+K) + \Theta
\]

- \(K\): 센서 고유 PRNU 패턴
- AI 생성 이미지는 일반적으로 카메라 센서 \(K\)가 없다.

### 3.3.2 PRNU/SRM 경로

코드에서 다음을 조합한다.

1. denoising residual 기반 PRNU 통계(분산, 첨도 등)
2. SRM 고주파 필터 응답
3. ELA(JPEG 재압축 차이) 맵
4. 블록 단위 이상치 비율(IQR 기반)
5. PRNU block correlation 저하 비율

이로부터:

- manipulation score
- natural diversity score
- ai generation score

를 계산하고 규칙 기반 판정을 수행한다.

### 3.3.3 MVSS 경로(권장)

`MAIFS_NOISE_BACKEND=mvss`일 때 MVSS-Net 세그멘테이션 마스크와 최대 점수로 판정한다.

- \(s_{mvss}\ge t_{high}\): manipulated
- \(s_{mvss}\le t_{low}\): authentic
- 사이 구간: uncertain

### 3.3.4 왜 두 경로를 함께 두는가

1. MVSS는 학습 기반으로 성능이 높다.
2. PRNU/SRM은 물리적 해석가능성이 높다.
3. 체크포인트/환경 실패 시 graceful degradation을 제공한다.

---

## 3.4 Semantic Pillar: FatFormer (CLIP + DWT)

구현: `src/tools/fatformer_tool.py`

### 3.4.1 핵심 아이디어

FatFormer는 CLIP ViT-L/14 표현(의미론)과 주파수 경로(DWT) 신호를 결합해 AI 생성 여부를 분류한다.

코드상 클래스는 2-way(real/fake)이며 softmax 확률:

\[
p_{fake}, p_{real} = \text{softmax}(logits)
\]

### 3.4.2 판정 규칙

- \(p_{fake}\ge t_{ai}\): ai\_generated
- \(p_{fake}\le t_{auth}\): authentic
- 사이: uncertain

임계값은 `tool_thresholds.json`의 `fatformer`에서 제어된다.

### 3.4.3 이론적 위치

이 축은 "생성모델의 의미론적 흔적"에 민감하고, 부분 조작보다는 전역 AI 생성 탐지에 강하다.

---

## 3.5 Spatial Pillar: Mesorch 기본 + OmniGuard/TruFor 대체

구현: `src/tools/spatial_tool.py`

### 3.5.1 기본 백엔드

`MAIFS_SPATIAL_BACKEND` 기본값은 `mesorch`다.

Mesorch는 조작 분할 마스크를 출력하고, MAIFS는 이를 원본 크기로 복원 후 조작 비율 \(\rho\)로 판정한다.

### 3.5.2 MVSS 마스크 융합

공간 마스크 \(M_s\)와 MVSS 마스크 \(M_n\)를 가중 융합:

\[
M = (1-w)M_s + wM_n
\]

여기서 \(w\)는 `mvss_weight`와 MVSS score 구간(`mvss_score_low/high`)에 의해 동적으로 축소될 수 있다.

### 3.5.3 판정 규칙

- \(\rho < t_{auth}\): authentic
- \(\rho > t_{ai}\): ai\_generated
- 나머지: manipulated

공간 축은 본질적으로 localization 기반이므로, 조작 위치 마스크를 근거로 제공할 수 있다.

---

## 4. Agent 해석 계층 이론

구현: `src/agents/specialist_agents.py`, `src/agents/base_agent.py`

각 Agent는 ToolResult를 받아:

1. verdict/confidence를 상속
2. 도메인 reasoning 생성
3. 토론용 arguments 추출

LLM 사용 가능 시 `SubAgentLLM`이 도메인 지식(`src/knowledge/*.md`)을 주입해 해석한다.

이론적으로 Agent는 "feature extractor"가 아니라 **evidence interpreter** 역할이다.

---

## 5. COBRA 합의 이론

구현: `src/consensus/cobra.py`

입력: \(\{(v_i,c_i,w_i)\}_{i=1}^N\)

- \(v_i\): agent verdict
- \(c_i\): agent confidence
- \(w_i\): trust score

## 5.1 RoT (Root-of-Trust)

신뢰 코호트 \(T_c\), 비신뢰 코호트 \(U_c\) 분리:

\[
S=\frac{\sum_{i\in T_c}w_ic_i + \alpha\sum_{j\in U_c}w_jc_j}
{\sum_{i\in T_c}w_i+\alpha\sum_{j\in U_c}w_j}
\]

- \(\alpha<1\): 비신뢰 코호트 감쇠
- 불일치 수준은 엔트로피 기반 정규화

## 5.2 DRWA (Dynamic Reliability Weighted Aggregation)

에이전트별 분산 프록시 \(\sigma_i\)로 동적 가중치:

\[
\omega_i = w_i + \epsilon\left(1-\frac{\sigma_i}{\sigma_{max}}\right)
\]

일관성 높은 에이전트 가중치를 올린다.

## 5.3 AVGA (Adaptive Variance-Guided Attention)

trust·confidence 점수에 softmax attention을 적용:

\[
a_i = \text{softmax}(s_i/T')
\]

\[
T' = T(1+\text{Var}(c))
\]

분산이 클수록 평탄화해 특정 에이전트 과신을 줄인다.

## 5.4 자동 선택 로직

`COBRAConsensus._select_algorithm`:

1. trust variance 큼 → RoT
2. verdict 다양성 큼(3개 이상) → AVGA
3. 기본 → DRWA

---

## 6. Debate Chamber 이론

구현: `src/debate/debate_chamber.py`, `src/debate/protocols.py`

### 6.1 토론 개시 조건

1. verdict가 2개 이상 충돌
2. confidence spread가 임계값 초과

### 6.2 프로토콜

1. Synchronous: 동시 발언
2. Asynchronous: 순차 발언(컨텍스트 반영)
3. Structured: claim/rebuttal/rejoinder/summary 단계

기본은 `AsynchronousDebate`.

### 6.3 수렴 조건

1. 만장일치
2. 신뢰도 변화량 \(<\delta\)
3. 최대 라운드 도달

토론 중 반론 강도에 따라 confidence를 미세 조정하고, 마지막에 COBRA를 재실행한다.

---

## 7. LLM 해석/토론 보조 이론

구현: `src/llm/subagent_llm.py`, `src/knowledge/__init__.py`

### 7.1 지식 주입 구조

도메인별 markdown 지식을 system prompt에 삽입한다.

- frequency/noise/fatformer/spatial

### 7.2 동작 모드

1. API 가능: structured JSON 해석 + 토론 응답
2. API 불가: 규칙 기반 fallback (`_fallback_interpret`, `_fallback_respond`)

이 설계는 해석 품질과 운영 안정성을 동시에 확보한다.

---

## 8. DAAC 메타학습 이론

## 8.1 핵심 가설

단일 에이전트 성능보다, 에이전트 간 **불일치 패턴 자체**가 조작 유형의 정보 신호다.

## 8.2 43차원 메타 특징

구현: `src/meta/features.py`

1. Per-agent (20d)
- verdict one-hot: 16d
- confidence: 4d
2. Pairwise disagreement (18d)
- binary disagree
- \(|c_i-c_j|\)
- conflict strength (불일치 시 \(c_i+c_j\))
3. Aggregate (5d)
- confidence variance
- verdict entropy
- max-min gap
- unique verdict count
- majority ratio

엔트로피:

\[
H(v)=-\sum_k p_k\log_2 p_k
\]

## 8.3 Path B 시뮬레이터 (Gaussian Copula)

구현: `src/meta/simulator.py`

1. 레이블 \(y\) 샘플링
2. 상관 정규변수 \(z\sim\mathcal{N}(0,\Sigma)\)
3. \(u=\Phi(z)\)로 균일변수 변환
4. confusion row CDF 역변환으로 agent verdict 샘플링
5. Beta 분포 quantile로 confidence 샘플링

즉, 에이전트 간 상관구조를 유지한 합성 출력을 생성한다.

## 8.4 베이스라인과 메타 분류기

구현: `src/meta/baselines.py`, `src/meta/trainer.py`

1. Majority vote
2. COBRA wrapper
3. Logistic Regression
4. Gradient Boosting / XGBoost
5. MLP (sklearn 또는 TorchMLP)

GPU 경로:

- `MAIFS_META_USE_GPU=1`
- XGBoost: `device=cuda`
- MLP: `backend=torch`, `device=cuda`

## 8.5 평가/통계 검정

구현: `src/meta/evaluate.py`

지표:

1. Macro-F1
2. Balanced Accuracy
3. AUROC(OvR)
4. ECE
5. Brier score

McNemar 검정(연속성 보정):

\[
\chi^2=\frac{(|b-c|-1)^2}{b+c}
\]

Bootstrap CI:

- 예측쌍 재표집으로 \( \Delta F1 \) 신뢰구간 추정

## 8.6 Ablation 설계

구현: `src/meta/ablation.py`

1. A1 confidence only
2. A2 verdict only
3. A3 disagreement only
4. A4 verdict+confidence
5. A5 full
6. A6 remove-one-agent

핵심 해석:

- A3가 random baseline을 충분히 넘으면 불일치 가설 지지
- A5 vs COBRA 유의 개선이면 DAAC 기여 성립

---

## 9. 임계값 보정 이론

구현 데이터: `configs/tool_thresholds.json`, 스크립트 `scripts/calibrate_tool_thresholds.py`

각 Tool은 score-to-verdict 경계값에 매우 민감하다.  
MAIFS는 캘리브레이션 결과를 JSON으로 외부화해 코드 변경 없이 임계값을 조정한다.

보정 대상:

1. CAT-Net mask ratio 경계
2. MVSS score 경계
3. FatFormer fake probability 경계
4. Spatial mask ratio 및 MVSS 융합 가중치

---

## 10. 불일치 해석 프레임

실무적으로 유용한 패턴:

1. FatFormer만 강하게 AI
- 전역 생성 의심, 압축/센서 신호는 약할 수 있음
2. Spatial만 강하게 조작
- 부분 편집/인페인팅 의심
3. Compression + Spatial 동시 조작
- JPEG 기반 스플라이싱 가능성
4. Noise는 authentic, 나머지는 조작
- 합성 강도가 낮거나 센서 흔적 일부 유지된 혼합 사례 가능

즉, MAIFS의 강점은 단일 정답보다 **불일치 구조의 설명성**에 있다.

---

## 11. 계산 복잡도와 운영 고려사항

### 11.1 병목

1. 대형 backbone 로딩(CAT-Net/FatFormer/Mesorch)
2. 고해상도 마스크 후처리
3. 다수 샘플 평가 시 I/O

### 11.2 안정성 설계

1. 모든 Tool에 fallback/uncertain 경로 존재
2. 체크포인트 미존재 시 시스템 전체 중단 방지
3. 환경변수로 백엔드/경로 강제 가능

주요 환경변수:

- `MAIFS_DEVICE`
- `MAIFS_SPATIAL_BACKEND`
- `MAIFS_NOISE_BACKEND`
- `MAIFS_MESORCH_CHECKPOINT`
- `MAIFS_MVSS_CHECKPOINT`
- `MAIFS_CATNET_CHECKPOINT`
- `MAIFS_META_USE_GPU`

---

## 12. 재현 체크리스트

1. 체크포인트 파일 배치 확인
2. 데이터셋 디렉터리 구조 확인
3. Tool re-eval (`scripts/evaluate_tools.py`)로 슬롯별 성능 확인
4. Phase1 재학습 (`experiments/run_phase1.py`) 실행
5. 결과 JSON/리포트의 runtime backend/device 확인

---

## 13. 한계와 향후 연구

1. Path B(시뮬레이션)와 Path A(실데이터) 간 도메인 갭
2. 일부 지식 문서의 레거시 용어(예: watermark 언급) 정리 필요
3. adversarial/post-processing 강건성 체계적 벤치마크 필요
4. Phase 2 Adaptive Routing(동적 가중치) 본격 실장 필요

---

## 14. 요약

MAIFS는 단순 앙상블이 아니라:

1. 서로 직교적인 4개 포렌식 축(압축/노이즈/의미/공간)
2. COBRA 합의 이론(RoT/DRWA/AVGA)
3. 토론 기반 신뢰도 조정
4. 불일치 패턴을 학습하는 DAAC 메타모델

을 결합한 **계층형 포렌식 의사결정 시스템**이다.

이 구조 덕분에 "개별 모델 정확도"뿐 아니라 "불일치의 정보성"을 실험적으로 활용할 수 있다.

