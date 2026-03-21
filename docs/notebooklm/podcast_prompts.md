# SHIELD 팟캐스트 생성 프롬프트 (4개 에피소드)

## 공통 지침
- 청중: 이미지 포렌식 분야를 전혀 모르는 AI/ML 연구자
- 언어: 한국어
- 형식: 두 진행자가 대화하는 팟캐스트
- 원칙: 아래에 명시된 모든 수치, 개념, 결과를 빠짐없이 언급할 것. 요약하거나 생략하지 말 것.

---

## 에피소드 1: MAIFS 시스템과 DAAC 연구

이것은 4부작 시리즈의 1편이다. 다음 내용을 모두 다뤄라.

### 반드시 다뤄야 할 내용 목록

**[문제 정의]**
- 이미지 위조의 두 종류: manipulation(부분 조작)과 AI-generated(AI 생성)가 탐지 방법이 근본적으로 다름
- 단일 모델이 두 문제를 동시에 잘 풀기 어려운 이유

**[MAIFS 4개 에이전트 — 각각 반드시 설명]**
- FrequencyAgent: CAT-Net 백엔드, 약 150MB, JPEG 이중 압축 흔적 탐지, DCT 계수 분석, AI-generated F1=0이라는 맹점
- NoiseAgent: MVSS-Net 백엔드, 약 120MB, 픽셀 단위 조작 마스크 예측, AI-generated F1=0이라는 맹점
- FatFormerAgent: CLIP ViT-L/14 + Forgery-Aware Adapter(FAA), 약 890MB, AI 생성 탐지 전문, CVPR 2024 논문, Manipulated F1=0이라는 맹점
- SpatialAgent: Mesorch 백엔드, 약 100MB, ViT 기반 부분 조작 영역 탐지, Mesorch 도입 후 CASIA2 mean_F1 0.55→0.91 개선
- 전체 4개 합계: 약 1.26GB, 추론 313ms

**[구조적 맹점과 핵심 통찰]**
- Frequency는 조작만, FatFormer는 AI생성만 탐지 — 서로 완전히 반대의 맹점
- 이 불일치 패턴이 오히려 탐지 신호가 된다는 DAAC의 핵심 아이디어

**[DAAC 연구]**
- COBRA(기존 합의 방법) Macro-F1=0.266의 한계
- DAAC(Disagreement-Aware Adaptive Consensus)의 핵심 가설: "에이전트 간 불일치 패턴 자체가 이미지 유형을 암시하는 신호"
- 43차원 메타 특징의 구성: 에이전트 판정(12차원), 신뢰도(4차원), 클래스별 확률(12차원), 불일치 플래그(6차원), 불일치 강도(6차원), 앙상블 통계(3차원)
- 분류기: Gradient Boosting Machine(GBM)
- DAAC-GBM Macro-F1=0.8613, COBRA 대비 +0.595, Wilcoxon p=0.00195
- Protocol-M (6개 데이터 조합, 60회): sign 60/0, p=1.63e-11
- 가장 중요한 특징: disagree_frequency_fatformer, 중요도 56.5%
- 이것의 의미: 조작이미지→Frequency 감지·FatFormer 미감지, AI생성→반대, authentic→둘 다 동의
- DAAC 합의 계층 추론시간: 0.069ms (전체의 0.02% 미만)
- 논문 제출처: KIPS 2026

**[SHIELD 연구 필요성]**
- 현재 MAIFS 문제점: 1.26GB, 313ms, RAM 4GB 이상 → 엣지 디바이스 불가
- SHIELD 목표: 크기 180MB 이하, 추론 1초 이하, Macro-F1 0.80 이상
- RPi5 선택 이유: 오픈 하드웨어, 재현성, Hailo-8L NPU 확장 가능, 학술 표준
- RPi5 사양: Cortex-A76 4코어 2.4GHz, 8GB LPDDR4X
- Galaxy S26 배제 이유: 폐쇄 플랫폼, 재현성 부족
- SHIELD 3가지 핵심 질문: ①최적 에이전트 조합, ②경량화 한계, ③배포 구조
- 다음 에피소드 예고: Phase 1 에이전트 가치 분석 (Shapley, PID, CKA, STII)

---

## 에피소드 2: 에이전트 가치 분석과 경량 백본 선정

이것은 4부작 시리즈의 2편이다. 1편에서 MAIFS와 DAAC를 다뤘다. 다음 내용을 모두 다뤄라.

### 반드시 다뤄야 할 내용 목록

**[Phase 1의 핵심 질문]**
- 단순 성능이 아닌 "고유한 정보를 제공하는가"를 기준으로 에이전트를 선택해야 하는 이유

**[Model Shapley]**
- 정의: 협력 게임 이론, N=4일 때 2⁴=16개 부분집합 전수 평가
- 계산 방법: 각 조합으로 DAAC 재학습 → Macro-F1 측정 → 한계 기여도 평균
- 결과: FrequencyAgent φ=+0.2690, FatFormerAgent φ=+0.1216, NoiseAgent φ=+0.0886, SpatialAgent φ=+0.0547
- Frequency 1위 이유: disagree_frequency_fatformer 특징이 직접 의존

**[STII: Shapley-Taylor Interaction Index]**
- 정의: 두 에이전트 시너지(양수) vs 대체재(음수) 측정
- 결과: 모든 6개 쌍이 음수
- 가장 강한 대체재: Frequency↔FatFormer = -0.1823
- 의미: 에이전트들이 서로 보완적이지 않고 일부 중복

**[PID: Partial Information Decomposition]**
- 정의: Unique/Redundant/Synergistic 정보 분해
- 결과: Frequency Unique=0.2029, FatFormer=0.0382, Noise=0.0311, Spatial=0.0000
- 핵심: SpatialAgent의 Unique information이 모든 데이터셋에서 0
- 가장 높은 시너지: Noise↔FatFormer (+0.1093)

**[Cross-Dataset 검증]**
- SpatialAgent 제거: 4/4 데이터셋에서 Unique=0 확정 → 경량화에서 제외
- Frequency 1위: 1/4 데이터셋에서만 성립 (CASIA 편향)
- FatFormer: 일부 데이터셋에서 음수 Shapley 발생
- 핵심 교훈: 무거운 모델 기준 Shapley가 경량 모델 배포 시 보장되지 않음 → 경량 백본 교체 후 재평가 필수

**[Phase 2 경량 백본 후보]**
- 딥리서치 6개 PDF 분석 기반
- FatFormer 슬롯 → MobileCLIP-S2: FAA가 backbone-agnostic 확인 (ViT-B/16, Swin-B, Swin-L 모두 동작)
- Frequency 슬롯 → ForMa (VMamba 기반, 37.3M params)
- Noise 슬롯 → MobileNetV2 dual-stream (RGB+SRM, 5.77M)
- 스크리너 → Tiny-LaDeDa (0.0013M, 1,300 params)

**[ForMa 평가 결과]**
- 4개 데이터셋 avg accuracy: 0.335 (3-class에서 낮음)
- Authentic recall 0.837~0.937 (높음), Manipulated recall 0.067~0.150 (매우 낮음), AI-gen recall 0
- CPU 추론 시간: 1,613ms → RPi5 병목
- 최종 결론: ForMa 전면 제거

**[MobileCLIP-S2 파인튜닝]**
- ft0 (Linear probe): val macro-recall 0.790
- ft4 (Last 4 blocks unfreeze): val macro 0.806, 4-DS avg 0.953
  - base=0.942, dsC=0.974, opensdi=0.953, aigen=0.943
- 크기: 99.4M params, CPU 123.8ms

**[MobileNetV2 dual-stream]**
- RGB + SRM 잔차 dual-stream 설계
- 5.77M params, GPU 18.9ms, CPU 35.9ms
- val macro 0.806, 4-DS avg 0.958
  - base=0.944, dsC=0.979, opensdi=0.949, aigen=0.961

**[Tiny-LaDeDa]**
- WildRF 학습, AI-gen recall 73~86%, Manipulated recall 0%
- Cascade Tier-1 스크리너 전용 용도

**[Phase 3.2: 경량 모델 Shapley+CKA 재분석 — 가장 중요한 발견]**
- Shapley 재결과: MNV2 φ=+0.304 ≈ MobileCLIP φ=+0.300 >> ForMa=Tiny≈+0.008
- CKA 결과: MobileCLIP↔MNV2 = 0.922 (매우 높은 중복!), 나머지 모든 쌍 < 0.02
- PID: MobileCLIP↔MNV2 Redundancy=0.599
- STII: MobileCLIP↔MNV2 = -0.584 (가장 강한 대체재)
- 3-Track 비교: in-dist Track1=0.9564, MobileCLIP 단독 Fair LOO F1=0.6386 vs Track1=0.7309
- 결론: 2-모델 최적이지만 CKA 중복 문제 → Binary Specialist로 다양성 보강 필요
- 다음 에피소드 예고: Binary Specialist 설계, SpecM v1~v5b 진화

---

## 에피소드 3: Binary Specialist 설계와 SpecM v1~v5b 진화

이것은 4부작 시리즈의 3편이다. 1편에서 MAIFS/DAAC, 2편에서 에이전트 가치 분석과 백본 선정을 다뤘다. 다음 내용을 모두 다뤄라.

### 반드시 다뤄야 할 내용 목록

**[Binary Specialist의 이론적 근거]**
- Meyen et al.(2021): Binary specialist가 generalist를 수학적으로 능가하는 조건
- 에러 Overlap 분석: MNV2+MobileCLIP avg Jaccard=0.3361, "둘 다 틀리는" 패턴 102건 중 56건이 manipulated→authentic
- MNV2와 MobileCLIP CKA=0.922 중복 문제

**[ICWMV 융합 메커니즘]**
- 수식: S(auth)=avg[MNV2,CLIP,SpecM,SpecG], S(manip)=avg[MNV2,CLIP,SpecM], S(aigen)=avg[MNV2,CLIP,SpecG]
- SpecM이 aigen에 기여하지 않는 이유, SpecG가 manip에 기여하지 않는 이유
- w_spec=1.0이 최적 가중치로 결정된 이유

**[Specialist-G 설계와 결과]**
- 아키텍처: MobileCLIP-S2 frozen(35.81M) + PiD branch(0.10M 학습)
- 총 학습 파라미터: 0.10M
- val best aigen_f1=0.981 (epoch 19)
- 4-DS: base=0.987, dsC=0.988, opensdi=0.799, aigenproxy=0.730

**[Specialist-M v1]**
- 3-stream: RGB(1280d) + SRM(1280d) + DCT(1280d) = 3840d fused
- 7.66M params, 29.3MB
- 학습 데이터: CASIA2 (Auth 7491 + Manip 5123)
- val best manip_f1=0.764 (epoch 5)
- 4-DS: base f1=0.861, dsC=0.797, opensdi auth_recall=7% (심각한 OOD 과적합)

**[Specialist-M v2: OOD 강건화]**
- 추가: IMD2020 1710장 + JPEG aug(p=0.5, q=40~95) + Gaussian Noise(p=0.4) + WeightedRandomSampler
- val best manip_f1=0.827 (+6.3%p)
- opensdi auth_recall: 7%→11% (개선 부족)

**[4-model ICWMV 첫 평가]**
- w=1.0 기준: avg macro-F1=96.40% vs MNV2 단독 95.81% (+0.59%p)
- 4-model CKA 다양성: avg CKA=0.0855 (2-model 0.9241 대비 ΔCKA=-0.8385)
- Jaccard: 4-model 0.1233 vs 2-model 0.3361
- Disagreement rate: 4-model 32.8% vs 2-model 4.4%

**[Specialist-M v3: 핵심 돌파]**
- 추가: GenImage_nature 3000장(ImageNet val 실사진) + RandomErasing(p=0.3, inpainting 시뮬)
- val best manip_f1=0.7832 (epoch 11)
- opensdi auth_recall: 11%→62% (+51%p 대폭 개선)
- ICWMV 2-model avg: 96.19% (+0.55%p vs v2)

**[Specialist-M v4: 세밀한 파인튜닝]**
- v3 resume, LR=3e-5, RandomErasing(value=random), focal_alpha=0.6
- val best manip_f1=0.7792 (epoch 10)
- openSDI manip_recall=70.3%
- ICWMV v4 avg=96.58% (서버 4-model 96.48% 초과 +0.10%p)
- base=95.86%, dsC=98.44%, opensdi=94.68%, aigenproxy=97.33%

**[Embedding CKA 분석: SpecM 근본 한계 규명]**
- 방법: Unbiased Linear CKA (Nguyen et al. NeurIPS 2021, 대각 제거 추정기), n=4200 샘플
- 브랜치별 CKA vs MNV2:
  - SpecG PiD branch(64d)=0.001, DCT branch(1280d)=0.239, SpecM RGB(1280d)=0.322, fused(3840d)=0.392, CLIP(512d)=0.420, SRM(1280d)=0.563
- 핵심 발견: mnv2_rgb↔specm_rgb=0.656 (동일 ImageNet pretrain RGB backbone), mnv2_noise↔specm_srm=0.564 (동일 SRM 필터)
- 충격적 발견: SpecM v1~v4 모두 CKA=0.725로 동일! 신호(SRM, DCT)를 아무리 추가해도 RGB backbone 공유로 중복 해소 불가

**[전 프로젝트 모델 vs MNV2 CKA 비교]**
- SpecM v1~v4: 0.725 (전부 동일)
- SpecG: 0.659
- MobileCLIP-S2-ft4: 0.028★
- MobileCLIP이 압도적 독립 이유: ViT 계열 contrastive pretrain vs ImageNet 분류 CNN → inductive bias 근본 차이

**[SpecM-v5b 설계]**
- 아키텍처: MobileCLIP-S2(frozen, 512d) + SRMLightCNN(128d) → 640d fused
- SRMLightCNN: depthwise-separable CNN (MNV2 구조 회피)
- Head: LayerNorm(640)→Linear(256)→GELU→Dropout→Linear(64)→GELU→Linear(2)
- 총 36.03M params

**[v5b frozen 학습 결과]**
- 학습 파라미터: 0.21M (MobileCLIP frozen)
- 40 epochs, lr=3e-4, batch=64
- val best manip_f1=0.7517 (epoch 37)
- 4-DS avg=0.8062: base=0.8223, dsC=0.8567, opensdi=0.6475, aigenproxy=0.8983
- ICWMV (MNV2+v5b): avg=0.9614

**[v5b_ft unfreeze + differential LR]**
- MobileCLIP 전체 unfreeze
- Differential LR: clip trunk lr=1e-5 / SRM CNN+head lr=1e-4 (10배 차이)
- 20 epochs 추가
- val best manip_f1=0.7846 (epoch 13)
- 4-DS avg=0.8347: base=0.8645, dsC=0.8932, opensdi=0.6205, aigenproxy=0.9605
- v4 대비 +1.8%p (4-DS avg 기준)
- ICWMV (MNV2+v5b_ft): avg=0.9635

**[v5b RPi5 배포 제외 결정]**
- opensdi OOD: v5b_ft=0.6205 vs v4=0.9468 → v4 압승
- 연구 기여: MobileCLIP backbone 유효성 증명, CKA=0.028 독립성 확인
- 배포: v4 유지

- 다음 에피소드 예고: ONNX 변환, Dynamic INT8 양자화, RPi5 112ms 실측 결과, WildRF 벤치마크

---

## 에피소드 4: ONNX 경량화, RPi5 배포, 최종 평가 결과

이것은 4부작 시리즈의 4편이자 마지막이다. 1편 MAIFS/DAAC, 2편 에이전트 분석, 3편 SpecM 설계를 다뤘다. 다음 내용을 모두 다뤄라.

### 반드시 다뤄야 할 내용 목록

**[ONNX 변환]**
- 변환 대상 4개 모델과 크기: MNV2=22.5MB, SpecM-v4=29.2MB, SpecG=141.5MB, MobileCLIP=141.3MB
- RPi5 200ms 예산 기준: MNV2와 SpecM만 OK, SpecG와 MobileCLIP은 CPU에서 너무 느림
- ONNX cosine similarity=0.99999 (변환 무손실 검증)

**[포렌식 신호와 양자화의 딜레마]**
- PRNU, DCT, SRM 신호는 low-magnitude, high-frequency 특성
- INT8 양자화 노이즈가 신호보다 커지면 탐지 성능 손상 우려
- 딥리서치에서 PTQ만으로는 불충분, QAT+Mixed-precision 필요하다고 예측했음

**[Dynamic INT8 실험 결과: 무손실]**
- MNV2: FP32=95.37% → Dynamic INT8=95.37% (Δ+0.00%p)
- SpecM-v4: 65.11% → 65.14% (Δ+0.03%p)
- SpecG: 87.67% → 87.84% (Δ+0.17%p)
- MobileCLIP: 95.32% → 95.38% (Δ+0.06%p)
- 결론: Dynamic INT8은 포렌식 신호에 안전

**[Static INT8 결과: FastViT 계열 붕괴]**
- MNV2: -13.42%p (심각)
- SpecG: -47.23%p (붕괴)
- MobileCLIP: -63.25%p (붕괴)
- SpecM-v4: -0.85%p (허용 범위)
- Static INT8 전면 사용 금지 결정, Dynamic으로 통일

**[RPi5 실측 환경]**
- 기기: Raspberry Pi 5 (8GB RAM)
- OS: Debian GNU/Linux 13 (Trixie), kernel 6.12.75
- Python 3.13.5, onnxruntime 1.24.4
- 모델: MNV2-Dynamic INT8 + SpecM-v4-Dynamic INT8

**[레이턴시 실측 (threads=4, warmup 제외 10회)]**
- 측정값: 108.5, 95.2, 118.4, 135.4, 106.3, 112.0, 90.3, 100.8, 127.4, 125.6 ms
- 평균: 112.0ms / 최소: 90.3ms / 최대: 135.4ms
- 예상(140ms) 대비 -20% 더 빠름

**[스레드별 레이턴시]**
- 1 thread: 192.8ms
- 2 threads: 128.6ms
- 4 threads: 114.3ms (최적)
- 1→4 스레드 감소율 40% (메모리 대역폭 한계로 이론적 4배 아님)

**[메모리 사용량]**
- Maximum resident set size: 156.3MB (두 모델 동시 로드)
- RPi5 8GB의 약 2% → 매우 여유

**[콜드스타트 vs 웜스타트]**
- 모델 로드 평균: 188.7ms
- 추론 평균: 113.8ms
- 콜드스타트 총합: 약 302ms
- 웜스타트(모델 메모리 유지 시): 112ms

**[ICWMV 충돌 케이스 처리 실증]**
- 실제 테스트 JSON 결과 설명:
  - MNV2: ai_generated 58%, manipulated 34%, authentic 8%
  - SpecM: authentic 72%, manipulated 28%
  - ICWMV 최종: ai_generated 45%, authentic 31%, manipulated 24%
  - 낮은 신뢰도(45%)가 의도된 동작 — 불확실 케이스를 정직하게 표현

**[최종 성능 비교표]**
- 서버 원본 MAIFS: ~1.26GB, ~313ms, Macro-F1=0.8613 (DAAC)
- 서버 4-model ICWMV: ~580MB, ~53ms (GPU), 96.48%
- RPi5 2-model ICWMV (MNV2+SpecM-v4 INT8): ~46MB, ~112ms (CPU), 96.58%
- RPi5가 서버보다 +0.10%p 높고, 크기는 80배 작음

**[4개 데이터셋별 최종 결과]**
- base (CASIA2+BigGAN): 95.86%
- dsC (CASIA2+IMD2020+BigGAN): 98.44%
- opensdi (OpenSDI 소셜미디어): 94.68%
- aigenproxy (AI-GenBench): 97.33%
- 평균: 96.58%

**[Phase 4.5 WildRF 벤치마크]**
- WildRF: 소셜미디어(Reddit, Twitter, Facebook) 수집 실제 딥페이크 데이터셋
- LaDeDa/Tiny-LaDeDa 논문(arXiv:2406.09398)에서 제안, Tiny-LaDeDa mAP=93.7% 기준
- 평가 방법: 3-class→binary 매핑 (authentic=real, manipulated+aigen=fake)
- OOD 테스트: 우리 시스템은 WildRF로 학습하지 않음
- 현재 데이터셋 다운로드 후 평가 예정

**[남은 연구 과제]**
- 단기: WildRF 평가 완료, ForensicHub 벤치마크
- 중기: Hailo-8L NPU 경로 (13 TOPS INT8), QAT Static INT8 성능 복구
- 장기 (SHIELD 논문 5개 Contribution):
  - C1: Model Shapley + STII 에이전트 가치 정량화 (완료)
  - C2: PID 정보 분해 (완료)
  - C3: QAT + mixed-precision 포렌식 특화 경량화 (미완료)
  - C4: MobileCLIP backbone + FAA adapter 재학습 (SpecM-v5b로 일부 달성)
  - C5: Confidence-gated cascade Tier 1→2→3 (미완료)

**[전체 연구 여정 마무리 요약]**
- Phase 1: Shapley/STII/PID/CKA로 SpatialAgent 제거 확정
- Phase 2: ForMa 제외, MNV2+MobileCLIP 2-모델 선정
- Phase 3.5: Binary Specialist + ICWMV로 서버 성능 초과
- Phase 4: RPi5에서 112ms, 96.58% 달성
- SpecM-v5b: MobileCLIP backbone으로 CKA 독립성 0.028 달성, 향후 개선 가능성 확인
