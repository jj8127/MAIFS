# SHIELD 팟캐스트 생성 스크립트 (Windows PowerShell)
# 실행 전: notebooklm login 완료 및 노트북 선택(notebooklm use e132f801) 필요

$ErrorActionPreference = "Stop"

# ──────────────────────────────────────────────────────────────────
# 에피소드 1: MAIFS 시스템과 DAAC 연구
# ──────────────────────────────────────────────────────────────────
Write-Host "=== 에피소드 1 생성 중 ===" -ForegroundColor Cyan

$ep1 = @"
이것은 4부작 시리즈의 1편이다. 청중은 이미지 포렌식을 전혀 모르는 AI/ML 연구자다. 두 진행자가 대화하는 한국어 팟캐스트로 만들어라. 아래에 나열된 모든 내용을 빠짐없이 다뤄라. 절대 요약하거나 생략하지 말라.

다뤄야 할 내용:
[1] 문제 정의: 이미지 위조의 두 종류(manipulation 부분조작, AI-generated AI생성)가 탐지 방법이 근본적으로 다른 이유.
[2] MAIFS의 4개 에이전트 각각 설명: ①FrequencyAgent(CAT-Net, 150MB, JPEG 이중압축 DCT분석, AI-generated F1=0 맹점) ②NoiseAgent(MVSS-Net, 120MB, 픽셀단위 조작마스크, AI-generated F1=0 맹점) ③FatFormerAgent(CLIP ViT-L/14+FAA, 890MB, CVPR2024, AI생성 탐지, Manipulated F1=0 맹점) ④SpatialAgent(Mesorch, 100MB, ViT 기반, Mesorch 도입 후 CASIA2 mean_F1 0.55→0.91).
[3] 전체 시스템: 4개 합계 약 1.26GB, 추론 313ms.
[4] 구조적 맹점 통찰: Frequency는 조작만, FatFormer는 AI생성만 탐지 — 이 불일치 패턴이 탐지 신호가 된다.
[5] DAAC의 핵심 아이디어: "에이전트 간 불일치 패턴 자체가 이미지 유형을 암시하는 신호".
[6] COBRA의 한계: Macro-F1=0.266.
[7] DAAC의 43차원 메타 특징 구성: 에이전트 판정 12차원, 신뢰도 4차원, 클래스별 확률 12차원, 불일치 플래그 6차원, 불일치 강도 6차원, 앙상블 통계 3차원.
[8] 분류기: Gradient Boosting Machine(GBM).
[9] DAAC 결과: Macro-F1=0.8613, COBRA 대비 +0.595, Wilcoxon p=0.00195. Protocol-M(6개 조합 60회) sign 60/0, p=1.63e-11.
[10] 가장 중요한 특징: disagree_frequency_fatformer 중요도 56.5%. 조작→Frequency 감지·FatFormer 미감지, AI생성→반대, authentic→둘 다 동의.
[11] DAAC 합의 계층 추론시간: 0.069ms (전체의 0.02% 미만). 논문: KIPS 2026.
[12] SHIELD 필요성: 1.26GB, 313ms, RAM 4GB+ → 엣지 불가. 목표: 180MB, 1초, F1>0.80.
[13] RPi5 선택 이유(오픈 하드웨어, 재현성, Hailo-8L NPU, 학술 표준)와 사양(Cortex-A76 4코어 2.4GHz, 8GB LPDDR4X). Galaxy S26 배제 이유(폐쇄 플랫폼, 재현성 부족).
[14] SHIELD 3대 핵심 질문: ①최적 에이전트 조합 ②경량화 한계 ③배포 구조.
[15] 다음 에피소드 예고: Phase 1 에이전트 가치 분석(Shapley, PID, CKA, STII).
"@

notebooklm generate audio --customization $ep1
Write-Host "에피소드 1 생성 요청 완료. 처리 대기 중..."
notebooklm artifact wait
notebooklm download audio --output "SHIELD_ep1_MAIFS와DAAC배경.mp3"
Write-Host "에피소드 1 저장 완료: SHIELD_ep1_MAIFS와DAAC배경.mp3" -ForegroundColor Green

Start-Sleep -Seconds 5

# ──────────────────────────────────────────────────────────────────
# 에피소드 2: 에이전트 가치 분석과 경량 백본 선정
# ──────────────────────────────────────────────────────────────────
Write-Host "=== 에피소드 2 생성 중 ===" -ForegroundColor Cyan

$ep2 = @"
이것은 4부작 시리즈의 2편이다. 1편에서 MAIFS 4개 에이전트 구조와 DAAC 결과를 다뤘다. 청중은 1편을 들었다고 가정하고 이어서 진행하라. 두 진행자가 대화하는 한국어 팟캐스트. 아래 내용을 빠짐없이 다뤄라.

다뤄야 할 내용:
[1] 왜 "성능이 높은 에이전트"가 아닌 "고유한 정보를 제공하는 에이전트"를 골라야 하는가.
[2] Model Shapley: 협력 게임 이론, N=4일 때 2⁴=16개 부분집합 전수 평가, 한계 기여도 계산 방법. 결과: Frequency φ=+0.2690, FatFormer φ=+0.1216, Noise φ=+0.0886, Spatial φ=+0.0547. Frequency 1위 이유.
[3] STII(Shapley-Taylor Interaction Index): 시너지(양수) vs 대체재(음수) 측정. 결과: 모든 6개 쌍 음수. 최강 대체재: Frequency↔FatFormer=-0.1823.
[4] PID(Partial Information Decomposition): Unique/Redundant/Synergistic 분해. 결과: Frequency=0.2029, FatFormer=0.0382, Noise=0.0311, Spatial=0.0000(전 데이터셋). 최고 시너지: Noise↔FatFormer=+0.1093.
[5] CKA(Centered Kernel Alignment): 특징 유사도 측정. 4개 무거운 에이전트 간 최대 CKA=0.0985(freq↔fatformer), 모두 독립적.
[6] Cross-Dataset 검증: SpatialAgent Unique=0 전 4개 데이터셋 확정→제거. Frequency 1위는 1/4만(CASIA 편향). 핵심 교훈: 무거운 모델 Shapley가 경량 배포 시 보장 안 됨.
[7] 경량 백본 후보(딥리서치 6개 PDF 기반): MobileCLIP-S2(FAA backbone-agnostic ViT-B/16·Swin-B·Swin-L 모두 동작 확인), ForMa(VMamba, 37.3M), MobileNetV2 dual-stream(5.77M), Tiny-LaDeDa(0.0013M).
[8] ForMa 평가: 4-DS avg accuracy=0.335, Authentic recall 0.837~0.937, Manipulated recall 0.067~0.150, AI-gen recall=0, CPU 1613ms→전면 제거.
[9] MobileCLIP-S2 파인튜닝: ft4(Last 4 blocks) val macro=0.806, 4-DS avg=0.953(base=0.942, dsC=0.974, opensdi=0.953, aigen=0.943). 99.4M, CPU 123.8ms.
[10] MobileNetV2 dual-stream: RGB+SRM dual-stream, 5.77M, CPU 35.9ms(RPi5 친화적), val macro=0.806, 4-DS avg=0.958(base=0.944, dsC=0.979, opensdi=0.949, aigen=0.961).
[11] Tiny-LaDeDa: AI-gen recall 73~86%, Manipulated recall=0%, Cascade Tier-1 전용.
[12] Phase 3.2 경량 모델 Shapley 재결과: MNV2 φ=+0.304 ≈ CLIP φ=+0.300 >> ForMa=Tiny≈+0.008.
[13] 경량 모델 CKA: MobileCLIP↔MNV2=0.922(충격적 중복!), 나머지 모두 <0.02. PID Redundancy=0.599. STII=-0.584.
[14] Fair LOO 평가: MobileCLIP 단독 OOD F1=0.6386 vs Track1(MNV2 포함) F1=0.7309(+0.092).
[15] 결론: 2-모델 최적(MNV2+CLIP), 단 CKA=0.922 중복 문제 → Binary Specialist로 다양성 보강 필요.
[16] 다음 에피소드 예고: Binary Specialist(SpecM, SpecG) 설계, v1~v5b 진화, ICWMV 앙상블.
"@

notebooklm generate audio --customization $ep2
Write-Host "에피소드 2 생성 요청 완료. 처리 대기 중..."
notebooklm artifact wait
notebooklm download audio --output "SHIELD_ep2_에이전트가치분석.mp3"
Write-Host "에피소드 2 저장 완료: SHIELD_ep2_에이전트가치분석.mp3" -ForegroundColor Green

Start-Sleep -Seconds 5

# ──────────────────────────────────────────────────────────────────
# 에피소드 3: Binary Specialist와 SpecM v1~v5b 진화
# ──────────────────────────────────────────────────────────────────
Write-Host "=== 에피소드 3 생성 중 ===" -ForegroundColor Cyan

$ep3 = @"
이것은 4부작 시리즈의 3편이다. 1편 MAIFS/DAAC, 2편 에이전트 가치 분석을 다뤘다. 이어서 진행하라. 두 진행자 한국어 팟캐스트. 아래 내용을 빠짐없이 다뤄라.

다뤄야 할 내용:
[1] Binary Specialist 이론 근거: Meyen et al.(2021) binary specialist가 generalist를 수학적으로 능가하는 조건. 에러 Overlap 분석: avg Jaccard=0.3361, "둘 다 틀리는" 패턴 102건 중 56건이 manipulated→authentic.
[2] ICWMV 융합 수식: S(auth)=avg[MNV2,CLIP,SpecM,SpecG], S(manip)=avg[MNV2,CLIP,SpecM], S(aigen)=avg[MNV2,CLIP,SpecG]. SpecM이 aigen에 미기여, SpecG가 manip에 미기여하는 이유.
[3] Specialist-G: MobileCLIP frozen(35.81M)+PiD branch(0.10M 학습). val best aigen_f1=0.981(epoch 19). 4-DS: base=0.987, dsC=0.988, opensdi=0.799, aigenproxy=0.730.
[4] SpecM-v1: 3-stream(RGB 1280d+SRM 1280d+DCT 1280d)=3840d fused, 7.66M/29.3MB. 학습: CASIA2(Auth 7491+Manip 5123). val best manip_f1=0.764(ep5). base f1=0.861, dsC=0.797. 심각 문제: opensdi auth_recall=7%(CASIA 과적합).
[5] SpecM-v2: +IMD2020 1710장+JPEG aug(p=0.5, q=40~95)+Gaussian Noise(p=0.4)+WeightedRandomSampler. val best=0.827(+6.3%p). opensdi auth_recall 7%→11%(개선 부족).
[6] 4-model ICWMV 첫 평가(v2 기준): avg=96.40% vs MNV2 단독 95.81%(+0.59%p). 4-model avg CKA=0.0855(2-model 0.9241 대비 -0.8385). Jaccard 0.1233 vs 0.3361. Disagreement rate 32.8% vs 4.4%.
[7] SpecM-v3 핵심 돌파: +GenImage_nature 3000장(ImageNet val 실사진)+RandomErasing(p=0.3 inpainting 시뮬). val best=0.7832(ep11). opensdi auth_recall 11%→62%(+51%p 대폭 개선). ICWMV 2-model avg=96.19%.
[8] SpecM-v4: v3 resume, LR=3e-5(↓), RandomErasing(value=random), focal_alpha=0.6. val best=0.7792(ep10). openSDI manip_recall=70.3%. ICWMV v4 avg=96.58%(서버 4-model 96.48% 초과 +0.10%p). base=95.86%, dsC=98.44%, opensdi=94.68%, aigenproxy=97.33%.
[9] Embedding CKA 분석(Nguyen et al. NeurIPS 2021 Unbiased Linear CKA, n=4200): 브랜치별 CKA vs MNV2: PiD=0.001, DCT=0.239, SpecM RGB=0.322, fused=0.392, CLIP=0.420, SRM=0.563. 핵심: mnv2_rgb↔specm_rgb=0.656(동일 ImageNet pretrain RGB backbone), mnv2_noise↔specm_srm=0.564(동일 SRM 필터).
[10] 충격적 발견: SpecM v1~v4 모두 CKA=0.725로 완전 동일. RGB backbone 공유 때문에 신호를 아무리 추가해도 중복 해소 불가.
[11] 전 프로젝트 모델 CKA: SpecM v1~v4=0.725, SpecG=0.659, MobileCLIP-S2-ft4=0.028★. MobileCLIP 독립 이유: ViT contrastive pretrain vs ImageNet 분류 CNN.
[12] SpecM-v5b 설계: MobileCLIP-S2(frozen, 512d)+SRMLightCNN(128d)→640d fused. SRMLightCNN: depthwise-separable(MNV2 구조 회피). Head: LayerNorm(640)→Linear(256)→GELU→Dropout→Linear(64)→GELU→Linear(2). 총 36.03M.
[13] v5b frozen: 0.21M 학습, 40ep, lr=3e-4. val best=0.7517(ep37). 4-DS avg=0.8062(base=0.8223, dsC=0.8567, opensdi=0.6475, aigenproxy=0.8983). ICWMV avg=0.9614.
[14] v5b_ft unfreeze+differential LR: clip lr=1e-5, SRM CNN+head lr=1e-4(10배 차이). 20ep. val best=0.7846(ep13). 4-DS avg=0.8347(base=0.8645, dsC=0.8932, opensdi=0.6205, aigenproxy=0.9605). v4 대비 +1.8%p. ICWMV avg=0.9635.
[15] v5b RPi5 배포 제외 결정: opensdi v5b_ft=0.6205 vs v4=0.9468. 연구 기여(MobileCLIP backbone 유효성, CKA=0.028)는 인정하되 배포는 v4 유지.
[16] 다음 에피소드 예고: ONNX 변환, Dynamic INT8 결과, RPi5 112ms 실측, 최종 성능 비교, 미래 연구.
"@

notebooklm generate audio --customization $ep3
Write-Host "에피소드 3 생성 요청 완료. 처리 대기 중..."
notebooklm artifact wait
notebooklm download audio --output "SHIELD_ep3_SpecM설계와진화.mp3"
Write-Host "에피소드 3 저장 완료: SHIELD_ep3_SpecM설계와진화.mp3" -ForegroundColor Green

Start-Sleep -Seconds 5

# ──────────────────────────────────────────────────────────────────
# 에피소드 4: ONNX 경량화, RPi5 배포, 최종 결과
# ──────────────────────────────────────────────────────────────────
Write-Host "=== 에피소드 4 생성 중 ===" -ForegroundColor Cyan

$ep4 = @"
이것은 4부작 시리즈의 4편이자 마지막이다. 1편 MAIFS/DAAC, 2편 에이전트 분석, 3편 SpecM 설계를 다뤘다. 두 진행자 한국어 팟캐스트. 아래 내용을 빠짐없이 다뤄라.

다뤄야 할 내용:
[1] ONNX 변환 결과: MNV2=22.5MB, SpecM-v4=29.2MB(INT8=26.4MB), SpecG=141.5MB, MobileCLIP=141.3MB. RPi5 200ms 예산 기준 MNV2와 SpecM만 OK. ONNX cosine similarity=0.99999(무손실 검증).
[2] 포렌식 신호(PRNU, DCT, SRM)의 low-magnitude high-frequency 특성 → INT8 양자화 노이즈 우려.
[3] Dynamic INT8 결과(전 모델 무손실): MNV2 Δ+0.00%p, SpecM-v4 Δ+0.03%p, SpecG Δ+0.17%p, MobileCLIP Δ+0.06%p.
[4] Static INT8 결과(FastViT 계열 붕괴): MNV2 -13.42%p, SpecG -47.23%p, MobileCLIP -63.25%p, SpecM-v4 -0.85%p. Static INT8 전면 사용 금지 결정.
[5] RPi5 실측 환경: RPi5(8GB), Debian 13 Trixie, kernel 6.12.75, Python 3.13.5, onnxruntime 1.24.4.
[6] 레이턴시 실측(threads=4, warmup 제외 10회): 108.5, 95.2, 118.4, 135.4, 106.3, 112.0, 90.3, 100.8, 127.4, 125.6ms. 평균 112.0ms / 최소 90.3ms / 최대 135.4ms. 예상 140ms 대비 -20%.
[7] 스레드별: 1T=192.8ms, 2T=128.6ms, 4T=114.3ms. 1→4 스레드 40% 감소(메모리 대역폭 한계).
[8] 메모리: Maximum resident set size 156.3MB(두 모델 동시). RPi5 8GB의 약 2%.
[9] 콜드스타트: 모델 로드 188.7ms + 추론 113.8ms = 302ms. 웜스타트(모델 상주 시) 112ms.
[10] ICWMV 충돌 케이스 실증: MNV2(ai_generated 58%, manipulated 34%) vs SpecM(authentic 72%, manipulated 28%) 불일치 → ICWMV 최종 ai_generated 45%(낮은 신뢰도). 불확실 케이스를 정직하게 표현하는 것이 의도된 동작.
[11] 최종 성능 비교: 서버 원본(1.26GB, 313ms, F1=0.8613) vs 서버 4-model ICWMV(580MB, 53ms GPU, 96.48%) vs RPi5 2-model ICWMV INT8(46MB, 112ms CPU, 96.58%). RPi5가 서버보다 +0.10%p 높고 크기 80배 작음.
[12] 4-DS 결과: base=95.86%, dsC=98.44%, opensdi=94.68%, aigenproxy=97.33%, 평균=96.58%.
[13] WildRF 벤치마크: 소셜미디어(Reddit/Twitter/Facebook) 실제 딥페이크. Tiny-LaDeDa mAP=93.7% 비교 기준. 우리 시스템은 WildRF 미학습 → 순수 OOD 테스트. 현재 데이터셋 다운로드 후 평가 예정.
[14] 남은 과제: 단기(WildRF/ForensicHub 평가), 중기(Hailo-8L NPU 13 TOPS INT8, QAT Static INT8 복구), 장기(SHIELD 논문 C1~C5 완성).
[15] SHIELD 논문 5개 Contribution 현황: C1 Model Shapley+STII(완료), C2 PID(완료), C3 QAT+mixed-precision(미완료), C4 MobileCLIP backbone+FAA adapter 재학습(SpecM-v5b로 일부 달성), C5 Confidence-gated cascade Tier1→2→3(미완료).
[16] 전체 연구 여정 마무리: Phase 1 SpatialAgent 제거 → Phase 2 ForMa 제외 MNV2+CLIP 선정 → Phase 3.5 Binary Specialist+ICWMV 서버 초과 → Phase 4 RPi5 112ms 96.58% 달성. 4부작 시리즈 마무리 인사.
"@

notebooklm generate audio --customization $ep4
Write-Host "에피소드 4 생성 요청 완료. 처리 대기 중..."
notebooklm artifact wait
notebooklm download audio --output "SHIELD_ep4_RPi5배포최종결과.mp3"
Write-Host "에피소드 4 저장 완료: SHIELD_ep4_RPi5배포최종결과.mp3" -ForegroundColor Green

Write-Host ""
Write-Host "=== 전체 완료 ===" -ForegroundColor Green
Write-Host "생성된 파일:"
Write-Host "  SHIELD_ep1_MAIFS와DAAC배경.mp3"
Write-Host "  SHIELD_ep2_에이전트가치분석.mp3"
Write-Host "  SHIELD_ep3_SpecM설계와진화.mp3"
Write-Host "  SHIELD_ep4_RPi5배포최종결과.mp3"
