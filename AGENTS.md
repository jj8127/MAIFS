# MAIFS Governance & Status Document

> **이 파일은 프로젝트의 단일 상태 문서(SSOT)입니다.**
> AI agent는 세션 시작 시 이 파일을 먼저 읽고, 세션 종료 시 변경사항을 반영해야 합니다.

---

## 1. Project Identity

**MAIFS** (Multi-Agent Image Forensic System) — 4개 전문가 AI 에이전트가 협력하여 이미지 진위를 판별하는 시스템.

- **1차 논문 (완료)**: DAAC — Disagreement-Aware Adaptive Consensus (KIPS 2026)
- **2차 논문 (진행중)**: SHIELD — Shapley-based Hardware-aware Interaction-preserving Ensemble Lightweighting for on-Device forensics
- **목표**: On-Device Image Forensics Architecture (RPi5 배포)

Primary stack: Python 3.10+ · PyTorch · NumPy/SciPy/scikit-learn · Gradio · pytest
가상환경: `.venv-qwen/bin/python`

---

## 2. Governance Rules

### 2.1 문서화 규칙

이 프로젝트는 **코드-문서 동기화 원칙**을 따릅니다.

| 규칙 | 설명 |
|------|------|
| **AGENTS.md = 1차 상태 문서** | 진행상황, 로드맵, 리스크, 우선순위가 바뀌면 이 파일을 먼저 갱신 |
| **Progress Ledger 최신 우선** | 진행 기록은 Ledger 맨 위에 최신 항목 추가 (§8) |
| **상세 분리 원칙** | AGENTS.md에는 한 줄 요약만, 깊은 내용은 별도 문서로 분리 |
| **코드-문서 동시 이동** | 기능/동작에 영향 있는 코드 변경 시 문서도 같은 작업 사이클에서 갱신 |
| **구현-규칙 일관성** | AGENTS 규칙과 코드가 어긋나면, 같은 변경셋에서 함께 수정 |

### 2.2 세션 종료 체크리스트

세션 종료 시 반드시 수행:
1. Progress Ledger에 최신 항목 추가 (§8)
2. Working Backlog 상태 갱신 (§5)
3. 리스크/블로커 변경 시 §6 갱신
4. 실험 결과 생성 시 §7 Research Tracker 갱신

### 2.3 변경 시 동기화 대상

| 변경 사항 | 갱신 대상 |
|----------|----------|
| 에이전트/Tool 추가·삭제·이름변경 | `AGENTS.md` §3, `CLAUDE.md` §2.1 |
| 실험 결과 생성 | `AGENTS.md` §7, §8 + 해당 연구 문서 |
| 경량화 모델 교체/성능 변경 | `AGENTS.md` §4.3 + `docs/research/SHIELD_RESEARCH_PLAN.md` |
| 아키텍처 변경 | `AGENTS.md` §3 + `CLAUDE.md` |
| 연구 방향/전략 변경 | `AGENTS.md` §4, §5 + 해당 연구 문서 |

### 2.4 문서 체계

| 문서 | 역할 | 상세도 |
|------|------|--------|
| `AGENTS.md` (이 파일) | 상태·로드맵·거버넌스 SSOT | 한 줄 요약 수준 |
| `CLAUDE.md` | 프로젝트 아키텍처·코딩 규칙·API 가이드 | 구현 수준 |
| `docs/research/SHIELD_RESEARCH_PLAN.md` | SHIELD 후속 연구 상세 계획 | 연구 수준 |
| `docs/research/DAAC_RESEARCH_PLAN.md` | DAAC 1차 연구 계획 (완료) | 연구 수준 |
| `docs/research/MAIFS_TECHNICAL_THEORY.md` | 이론 백서 | 수식 수준 |

---

## 3. System Architecture Snapshot

```
[Gradio UI] → [Orchestrator] → [COBRA/Debate/DAAC] → [4 Specialist Agents] → [Tool Layer]
```

### 3.1 현재 에이전트 구성 (v1.0 — 서버 기준)

| Agent | Backend Model | 크기 | 추론시간 | 역할 | 맹점 |
|-------|--------------|------|---------|------|------|
| FrequencyAgent | CAT-Net (HRNet-W48) | ~150MB | ~80ms | JPEG 이중압축 탐지 | AI-generated F1=0 |
| NoiseAgent | MVSS-Net | ~120MB | ~61ms | 픽셀 조작 마스크 | AI-generated F1=0 |
| FatFormerAgent | CLIP ViT-L/14 + FAA | ~890MB | ~57ms | AI 생성 탐지 | Manipulated F1=0 |
| SpatialAgent | Mesorch (ViT) | ~100MB | ~97ms | 부분 조작 영역 | AI-generated F1=0 |
| **합계** | | **~1.26GB** | **~313ms** | | |

### 3.2 DAAC 합의 계층 (변경 없음)

- 43-dim 메타 특징 추출 + GBM 분류기
- 추론: 0.069ms (전체의 0.02% 미만)
- Macro-F1: 0.8613 (COBRA 0.266 대비 +0.595)
- 최상위 특징: `disagree_frequency_fatformer` 56.5%

---

## 4. SHIELD 후속 연구 개요

> 상세: [docs/research/SHIELD_RESEARCH_PLAN.md](docs/research/SHIELD_RESEARCH_PLAN.md)

### 4.1 연구 목표

DAAC의 성능을 유지하면서 **RPi5에 배포 가능한 경량 포렌식 아키텍처** 설계.

- 전체 모델 크기: 1.26GB → **<500MB** (목표 ~180MB)
- 타겟 디바이스: Raspberry Pi 5 (8GB RAM, Cortex-A76, 선택적 Hailo-8L NPU)
- DAAC 43-dim 메타 특징 구조 유지 (리소스 무시 가능)

### 4.2 5대 기여 (Contributions)

| # | 기여 | 핵심 방법론 | 상태 |
|---|------|-----------|------|
| C1 | 에이전트 가치·상호작용 정량화 | Model Shapley (exact, N=4) + STII (k=2) | `NOT_STARTED` |
| C2 | 고유/중복/시너지 정보 분해 | PID (Partial Information Decomposition) | `NOT_STARTED` |
| C3 | 포렌식 특화 경량화 | QAT + mixed-precision (FP16 입력단, INT8 시맨틱) | `NOT_STARTED` |
| C4 | 백본 교체 + 어댑터 전이 | FatFormer FAA → MobileCLIP-S2 (~890→50MB) | `NOT_STARTED` |
| C5 | Confidence-gated cascade | Tier 1→2→3 조건부 추론, 평균 비용 3-5x 절감 | `NOT_STARTED` |

### 4.3 에이전트별 경량화 목표

| Agent | 현재 | 경량화 방안 | 목표 크기 | 핵심 제약 |
|-------|------|-----------|----------|----------|
| FatFormer | ~890MB | MobileCLIP-S2 + FAA 재학습 | ~50MB | FAA adapter만 재학습, 백본 freeze |
| CAT-Net | ~150MB | Structured pruning (DCT+RGB stream 유지) | ~55MB | DCT stream 절대 제거 불가 |
| MVSS-Net | ~120MB | MobileNetV3-Small + feature-level KD | ~25MB | edge supervision 유지 |
| Mesorch | ~100MB | Mesorch-P + Fast-SCNN 백본 | ~50MB | SRM 필터 CPU fallback 필요 |
| **합계** | **~1.26GB** | | **~180MB** | |

### 4.4 핵심 연구 발견 (딥리서치 종합)

> 6개 딥리서치 PDF 분석 결과 요약. 상세: [SHIELD_RESEARCH_PLAN.md](docs/research/SHIELD_RESEARCH_PLAN.md) §2

1. **포렌식 양자화 민감도**: PRNU/DCT 신호가 low-magnitude → PTQ 불충분, QAT 필수, mixed-precision 필수
2. **FatFormer FAA 백본 독립성 확인**: ViT-B/16, Swin-B, Swin-L ablation에서 FAA 동작 → MobileCLIP 교체 가능
3. **Freq↔FatFormer 시너지가 submodularity 위반**: greedy 선택 부적합 → interaction-preserving 제약 필요
4. **Cascade 선행연구**: CoE 7x 비용 절감, NoScope/BranchyNet early-exit 적용 가능
5. **RPi5 벤치마크**: 경량 CNN ~100 FPS, 하이브리드 ViT ~10 FPS (CPU 기준)
6. **Token pruning 위험**: 조작 영역 토큰 제거 가능성 → 포렌식에서는 사용 불가

---

## 5. Working Backlog

### Phase 1: Agent Valuation (이론적 근거 확립) ✅ 완료

| # | 작업 | 우선순위 | 상태 | 비고 |
|---|------|---------|------|------|
| 1.1 | Model Shapley 계산 (2⁴=16 부분집합) | P0 | `DONE` | freq=0.2690, fat=0.1216, noise=0.0886, spatial=0.0547 |
| 1.2 | STII (k=2) pairwise interaction 산출 | P0 | `DONE` | freq↔fatformer=-0.1823 (모든 쌍 음수, 대체재 관계) |
| 1.3 | CKA 분석 (feature map 유사도) | P1 | `DONE` | 최대 CKA=0.0985 (freq↔fatformer), 모두 독립적 |
| 1.4 | PID 정보 분해 | P1 | `DONE` | Unique: freq=0.2029 > fat=0.0382 > noise=0.0311 > spatial=0.0000. 최고시너지: noise↔fatformer(+0.1093) |
| 1.5 | 에이전트 조합 최적 부분집합 결정 | P1 | `DONE` | freq+noise+fatformer(3개) 1차 권고. 단, cross-dataset 검증으로 수정 필요 |
| 1.6 | Cross-Dataset 검증 (4개 데이터셋) | P0 | `DONE` | **Spatial Unique=0 전 데이터셋 확정. 나머지 순위는 데이터셋 의존적** → 3-Track 실험으로 전환 |

> **Phase 1 핵심 발견**: Spatial 제거는 4/4 데이터셋에서 확정. 그러나 freq vs noise 1위, fatformer 가치는 데이터셋 편향. **무거운 모델 기준 Shapley는 경량 모델 배포 시 보장 안 됨** → 경량 백본 교체 후 재평가 필수.

---

### Phase 2: Backbone 수급 및 단독 평가 (Step 1) ← **현재 단계**

> **목적**: 3-Track 비교를 위한 경량 백본 추론 JSONL 생성. 4개 데이터셋 × 백본별 평가.

| # | 작업 | 우선순위 | 상태 | 비고 |
|---|------|---------|------|------|
| 2.1 | ForMa 수급 (VMamba 기반, freq+noise 통합 가능) | P0 | `DONE` | 코드+가중치 연동 완료(37.3M). 4개 데이터셋 단독 평가 완료, 평균 acc=0.3347 |
| 2.2 | MobileCLIP-S2 수급 (FatFormer 대체) | P0 | `DONE` | open_clip datacompdr 다운로드 완료. **Linear probe 파인튜닝 완료**: val macro_recall=0.790, 4개 데이터셋 평균 acc=0.932 |
| 2.3 | Tiny-LaDeDa 수급 (Cascade Tier-1 스크리너) | P1 | `DONE` | WildRF 가중치 repo 포함. ai_gen recall 73-86%, manip=0%(binary) |
| 2.4 | MobileNetV2 dual-stream 구현 (noise 대체) | P1 | `DONE` | 15ep 재학습 완료(5.77M). best val macro=0.806, 4개 데이터셋 avg acc=0.958 |
| 2.5 | 백본별 단독 평가 JSONL 생성 (4개 데이터셋) | P0 | `DONE` | ForMa + MobileCLIP-S2 + Tiny-LaDeDa 전체 JSONL/summary 생성 완료 |
| 2.6 | ForMa 가중치 수동 다운로드 | P0 | `DONE` | 사용자 제공 `ForMa_weights.pth` 확보. repo-root fallback 경로로 평가 스크립트 연동 완료 |
| 2.7 | 백본별 추론 시간 + 모델 크기 실측 | P1 | `DONE` | ForMa/MobileCLIP-ft4/Tiny-LaDeDa/MobileNetV2-dualstream GPU+CPU 실측 완료 |

---

### Phase 3: 3-Track 비교 실험

> **3개 아키텍처 전략**을 동일 4개 데이터셋에서 비교. 경량 백본 기반 Shapley/PID 재실행.

| Track | 구성 | 에이전트 수 | 예상 크기 |
|-------|------|-----------|---------|
| **Track 1** | ForMa + MobileNetV2-noise + MobileCLIP-S2 | 3개 분리 | ~150MB |
| **Track 2** | ForMa(freq+noise 통합) + MobileCLIP-S2 | 2개 통합 | ~100MB |
| **Track 3** | Tiny-LaDeDa(Tier1) → ForMa+MobileCLIP(Tier2) → DAAC(Tier3) | 적응형 cascade | ~100MB |

| # | 작업 | 우선순위 | 상태 | 비고 |
|---|------|---------|------|------|
| 3.1 | Track별 JSONL 구성 (Phase 2 출력 조합) | P0 | `DONE` | Track1/2/3 JSONL 12개 생성 + Track1 재평가 완료(combined 10-seed best=0.9564) |
| 3.2 | Track별 Shapley + PID 재실행 (경량 모델 기준) | P0 | `DONE` | MNV2 φ=+0.304 ≈ CLIP φ=+0.300 >> ForMa=Tiny≈0.008. CLIP↔MNV2 CKA=0.92(고중복). ForMa/Tiny Unique≈0 → 2모델 조합으로 충분 |
| 3.3 | 다기준 비교 (F1 × 크기 × 속도 × cross-DS 일관성) | P0 | `DONE` | **RPi5 권고: MNV2-only**(22.5MiB/35.9ms/score=0.730). **GPU 권고: CLIP+MNV2**(402.8MiB/34.4ms GPU/OOD=0.731). ForMa 전면 제거 확정(CPU=1613ms 병목, Shapley≈0) |
| 3.4 | DAAC 메타 분류기 재학습 (선정된 Track 기준) | P0 | `DONE` | 경량 25-dim(MNV2+CLIP only, label leakage 방지로 specialist 제외). GBM base=99.01%(원본 86.13% 대비 +12.88%p), avg 4-DS=96.25%. Top-feature: mnv2_aigen(29.8%) |

---

### Phase 3.5: Binary Specialist + ICWMV Ensemble ← **현재 단계**

> **목적**: MNV2+CLIP의 에러 독립성을 활용하여 Binary Specialist 전문가 추가 → ICWMV 4-model 합의로 OOD 강건성 개선.
>
> **핵심 발견 (딥리서치 종합)**: Meyen et al.(2021) 이론 — Binary specialist가 generalist 수학적으로 능가. ICWMV = 개별 confidence 가중 다수결. DREP = 에러 집합 기반 직접 학습.

**4-Model 아키텍처:**

| 모델 | 역할 | 크기 | 학습 데이터 | 특수 신호 |
|------|------|------|-----------|---------|
| MobileNetV2 dual-stream | 3-class generalist | 5.77M / 22.5MiB | CASIA2+BigGAN | SRM 노이즈 |
| MobileCLIP-ft4 | 3-class generalist | 99.4M / 380.3MiB | 동일 | CLIP 시맨틱 |
| Specialist-M | binary: auth vs manip | 7.66M / 29.3MiB | CASIA2 Au+Tp | SRM+DCT 3-stream |
| Specialist-G | binary: auth vs aigen | 35.91M (0.10M trainable) | CASIA2+BigGAN+AIGenBench | MobileCLIP frozen + PiD |

| # | 작업 | 우선순위 | 상태 | 비고 |
|---|------|---------|------|------|
| 3.5.0 | 에러 Overlap 분석 (MNV2 vs CLIP) | P0 | `DONE` | avg Jaccard=0.3361 → 에러 독립적, ASSIST 아이디어 유효. "둘 다 틀림" 지배 패턴: manipulated→authentic |
| 3.5.1 | Specialist-M 학습 (v1, CASIA2 only) | P0 | `DONE` | best manip_f1=0.764(Ep.5), 4-DS eval: base manip_recall=0.962/f1=0.861, dsC=0.797. **OOD 한계**: opensdi auth_recall=0.07 (CASIA2 과적합) |
| 3.5.2 | Specialist-G 학습 | P0 | `DONE` | best aigen_f1=0.981(Ep.19), 4-DS eval: base=0.987, dsC=0.988, opensdi=0.799, aigenproxy=0.730 |
| 3.5.3 | ICWMV 4-model 합의 평가 | P0 | `DONE` | w=1.0: avg macro_F1=96.40% vs MNV2 95.81%(+0.59%p). base+1.0, dsC+0.9, opensdi+0.9%p. **aigenproxy −0.4%p** (SpecM OOD 약점) |
| 3.5.4 | Specialist-M v2 학습 (OOD 강건화) | P0 | `DONE` | +IMD2020 1710장 + JPEG/Noise aug + WeightedSampler. best manip_f1=0.827(v1 0.764 대비 +6.3%p). opensdi auth_recall 7%→11%, aigenproxy 17%→25% |
| 3.5.5 | ICWMV 4-model 재평가 (SpecM-v2 기준) | P1 | `DONE` | SpecM-v2 기준 w=1.0: avg macro_F1=**96.48%**(v1 96.40% 대비 +0.08%p). base=95.73/dsC=99.11/opensdi=95.31/aigenproxy=95.77%. aigenproxy 약점 개선 확인 |
| 3.5.6 | CKA 다양성 재분석 (4-model) | P1 | `DONE` | 4-model avg CKA=0.0855 vs 2-model 0.9241(ΔCKA=-0.8385). 4-model avg Jaccard=0.1233 vs 2-model 0.3361. disagreement rate: 4-model 32.8% vs 2-model 4.4% — binary specialist 추가로 다양성 대폭 향상 확인 |

---

### Phase 4: Edge Deployment & Evaluation

| # | 작업 | 우선순위 | 상태 | 비고 |
|---|------|---------|------|------|
| 4.1 | ONNX 변환 + CPU 벤치마크 | P1 | `DONE` | MNV2=22.5MB/14ms, SpecM=30MB/20.6ms, SpecG=141.5MB/200ms, CLIP=141.3MB/197.7ms (1-thread). RPi5 예산(200ms): MNV2+SpecM만 OK |
| 4.2 | PTQ INT8 양자화 + 정확도 평가 + SpecM-v3/v4 재학습 | P1 | `DONE` | Dynamic INT8 전 모델 무손실(Δ≤+0.17%p). Static: FastViT 계열 붕괴. **SpecM-v3**: opensdi auth_recall 11%→62%. **SpecM-v4**: v3 resume(LR=3e-5)+RandomErasing(value=random), val manip_f1=0.7792(v3 0.7832 대비 -0.4%p). **ICWMV avg**: MNV2 단독 95.81% < v3 96.43%(+0.62%p) < **v4 96.58%(+0.77%p)**. RPi5 최종: **MNV2-Dynamic + SpecM-v4-Dynamic = ~140ms** |
| 4.3 | Hailo-8L HEF 변환 (선택) | P2 | `NOT_STARTED` | NPU 경로 |
| 4.4 | RPi5 end-to-end 벤치마크 | P0 | `DONE` | **실측 완료(2026-03-21)**. 추론 avg **112ms**(예상 140ms 대비 -20%). 메모리 156MB. 콜드스타트 189ms(로드)+114ms(추론). threads=4 최적. 상세: §7.2 참조 |
| 4.5 | ForensicHub/WildRF 벤치마크 | P1 | `NOT_STARTED` | 논문 비교 실험 |
| 4.6 | Embedding-level CKA 분석 (MNV2 vs 전 모델/브랜치) | P0 | `DONE` | ①브랜치별: PiD=0.001★, DCT=0.239, CLIP=0.420. RGB branch 중복 주범(CKA=0.656 vs mnv2_rgb). ②전 프로젝트 모델 비교: **MobileCLIP-S2-ft4=0.028★**, SpecG=0.659, SpecM v1~v4=0.725(전부 동일 — ImageNet pretrain RGB 공유가 원인). → SpecM-v5b backbone 확정 |
| 4.7 | SpecM-v5b (MobileCLIP+SRMLightCNN) 설계·학습·평가 | P1 | `DONE` | frozen(0.21M 학습): val f1=0.7517, 4-DS avg=0.8062. **unfreeze/diff-LR ft**: val f1=0.7846, 4-DS avg=**0.8347**(v4 대비 +0.018). ICWMV v5b_ft avg=**0.9635**(MNV2+v5b_ft). RPi5 배포는 v4 유지(opensdi OOD 약점) |
| 4.8 | Coral USB / Edge TPU 배포 경로 정비 | P0 | `DONE` | **툴링 + export 파이프라인 정비 완료(2026-03-21)**. Python 3.9 venv, ONNX→full INT8 TFLite 스크립트, `edgetpu_compiler` wrapper, multi-backend 추론 추가. 초기 dry-run에서 `int16_act` 산출물 선택 버그와 `specm_v4` Resize 변환 문제를 확인했고, 후속 4.9에서 해결 |
| 4.9 | Coral-friendly 모델 리팩터링 (LayerNorm/GELU/Resize 정리) | P0 | `DONE` | **완료(2026-03-21)**. export 전용 `mnv2_coral`/`specm_v4_coral` ONNX 추가. `mnv2`: LayerNorm 제거 + ReLU head, `specm_v4`: 정적 `scale_factor=8` DCT + ReLU head. `run_edgetpu_export.py`가 full INT8(`*_full_integer_quant.tflite`)만 선택하도록 수정 후 **두 모델 모두 Edge TPU 100% 매핑 compile 성공** |
| 4.10 | Coral variant 정확도 재평가 + RPi5 Coral 실측 | P0 | `IN_PROGRESS` | **정확도 재평가 완료(2026-03-21), MNV2 PTQ sweep + SpecM coral-native fine-tune 추가(2026-03-22)**. `mnv2`는 **per-channel + calib 64 + int8 IO**가 best(`0.9173`, baseline coral 대비 `+0.0077`). `specm_v4_coral` 구조를 직접 학습한 `specm_v4_coral_ft`는 TFLite standalone avg manip-F1=`0.8360`으로 current ONNX `0.8392`에 근접(Δ`-0.0032`). pair는 `w_spec=1.0`에서 과가중되지만 `w_spec=0.2`로 재튜닝 시 avg macro-F1=`0.9308`까지 회복(old coral pair `0.9051` 대비 `+0.0257`, current ONNX 대비 `-0.0226`). 실제 Coral USB latency 실측만 남음 |

---

## 6. Risks & Blockers

| ID | 리스크 | 심각도 | 완화 전략 | 상태 |
|----|--------|--------|----------|------|
| R1 | FatFormer→MobileCLIP 교체 시 FAA 재학습 실패 | HIGH | FAA가 backbone-agnostic임을 논문 ablation으로 확인. 실패 시 TinyCLIP 대안 | `OPEN` |
| R2 | 양자화로 포렌식 신호 손실 (PRNU/DCT) | MEDIUM | Dynamic INT8: MNV2/SpecM 무손실 확인(Δ≤+0.03%p). FastViT계(SpecG/CLIP)는 Static PTQ 붕괴 → Dynamic 사용 확정. RPi5 조합(MNV2+SpecM Dynamic)은 PTQ로 충분, QAT 불필요 | `MITIGATED` |
| R3 | 경량화 후 DAAC 43-dim 특징 분포 변화 | MEDIUM | 경량 모델 기준 재학습으로 대응 | `OPEN` |
| R4 | RPi5 메모리 부족 (8GB 내 4모델 동시 로드) | MEDIUM | Cascade로 동시 로드 회피 + 모델 swap 전략 | `OPEN` |
| R5 | SRM/DWT custom ops NPU 미지원 | LOW | CPU fallback 경로 유지 (Mesorch에만 해당) | `OPEN` |
| R6 | Token pruning이 조작 영역 토큰 제거 | HIGH | Token pruning 사용 금지 (연구에서 확인) | `MITIGATED` |
| R7 | 무거운 모델 Shapley가 경량 모델과 다름 (데이터셋 편향) | HIGH | Cross-dataset 검증으로 발견. 경량 백본 교체 후 Shapley 재실행으로 대응 | `MITIGATED` |
| R8 | ForMa/Tiny-LaDeDa 공개 가중치 부재 또는 호환성 문제 | MEDIUM | 사용자 제공 ForMa 가중치 연동 및 Tiny-LaDeDa 호환성 확인 완료. 미확보 시 직접 학습 or 대안 백본(MNVFusion, BNN) 준비 | `MITIGATED` |
| R9 | fatformer(MobileCLIP)가 다양한 AI 생성기에서 음수 Shapley | MEDIUM | Track 2(통합)에서 ForMa가 흡수 → fatformer 의존도 낮춤 | `OPEN` |
| R10 | Specialist-M OOD 일반화 실패 (openSDI 도메인) | HIGH | v2 완료했으나 opensdi F1=41%, aigenproxy 53% 여전히 낮음. 2-model ICWMV에서 opensdi -1.56%p 역효과 발생. **대응: openSDI/COVERAGE/NIST16 데이터 추가 후 v3 재학습 필요** | `OPEN` |
| R11 | ICWMV w_spec 과도하면 OOD 성능 역전 | MEDIUM | w_spec=1.0이 균형점. Specialist-M v2로 OOD 개선 후 재튜닝 | `MITIGATED` |
| R12 | Python 3.13에서 `tflite-runtime` 미지원 | HIGH | Coral 경로를 Python 3.9 uv venv로 분리하고(`deploy/setup_rpi5_coral_env.sh`), ONNX 경로와 별도 requirements 유지 | `MITIGATED` |
| R13 | 원본 MNV2/SpecM-v4 ONNX가 Coral Edge TPU와 직접 호환되지 않음 | HIGH | 원본 `mnv2.onnx`는 `FlexErf`/GELU로 compile 불가, 원본 `specm_v4.onnx`는 DCT `Resize` 변환 실패. **대응**: export 전용 `mnv2_coral`/`specm_v4_coral` 변형으로 우회 완료 | `MITIGATED` |
| R14 | Coral-friendly export variant의 정확도 드리프트 가능성 | HIGH | **완화 진전(2026-03-22)**. old export-only coral pair는 avg macro-F1=`0.9051` vs current ONNX=`0.9535`(Δ`-0.0483`)였으나, `specm_v4_coral` 구조 직접 학습 + `w_spec=0.2` 재튜닝 후 explicit Coral pair best는 `0.9308`까지 회복(Δ`-0.0226`). `specm_v4_coral_ft` standalone avg manip-F1=`0.8360`으로 current `0.8392`에 근접. 다만 여전히 current ONNX 대비 격차가 남고 실제 Coral USB 실측이 필요 | `OPEN` |
| R15 | Edge TPU용 full INT8 TFLite PTQ가 MNV2 성능을 크게 저하시킴 | HIGH | `mnv2_coral` PyTorch avg macro-F1=`0.9530`이지만 기존 full INT8 TFLite=`0.9096`(Δ`-0.0434`). 2026-03-22 sweep에서 **per-channel + calib 64 + int8 IO**가 best로 `0.9173`까지 회복, tuned pair ICWMV도 `0.9160`(+`0.0108`)으로 개선됐지만 여전히 current ONNX 대비 `-0.0375`. 즉 대표 데이터/quant setting으로 일부 완화는 가능하지만, 근본 해결에는 QAT 검토가 필요 | `OPEN` |

---

## 7. Research Tracker

### 7.1 완료된 연구

| 항목 | 결과 | 문서 |
|------|------|------|
| DAAC Phase 1 (Path B 시뮬레이션) | GBM F1=0.9949, Go/No-Go PASS | `experiments/results/phase1/` |
| DAAC Phase 2 (Path A 실데이터) | GBM F1=0.8613, COBRA 대비 +0.595 | `experiments/results/paper_final/` |
| DAAC 논문 (KIPS 2026) | 초안 완성 | `docs/research/MAIFS_PAPER_DRAFT2_20260306.md` |
| 딥리서치 Prompt 1 (On-device SOTA) | 6개 PDF 종합 완료 | `SHIELD_RESEARCH_PLAN.md` §2.1 |
| 딥리서치 Prompt 2 (Agent selection theory) | 6개 PDF 종합 완료 | `SHIELD_RESEARCH_PLAN.md` §2.2 |
| 딥리서치 Prompt 3 (Compression for forensics) | 6개 PDF 종합 완료 | `SHIELD_RESEARCH_PLAN.md` §2.3 |

### 7.2 완료된 실험

| 실험 | 결과 | 파일 |
|------|------|------|
| Phase 1.1 Model Shapley (N=4) | frequency φ=+0.2690 > fatformer φ=+0.1216 > noise φ=+0.0886 > spatial φ=+0.0547 | `experiments/results/shapley_phase1/shapley_phase1_20260318_161543.json` |
| Phase 1.2 STII k=2 | freq↔fatformer: -0.1823 (최강, 음수=대체재), 모든 쌍 음수 | 동일 파일 |
| Phase 1.3 CKA | 모든 쌍 CKA < 0.1 (에이전트 간 특징 독립적) | 동일 파일 |
| Phase 1.4 PID | Unique: freq=0.2029, fatformer=0.0382, noise=0.0311, **spatial=0.0000** | `experiments/results/shapley_phase1/pid_phase1_20260318_162008.json` |
| Phase 1.6 Cross-Dataset | **spatial Unique=0: 4/4 확정.** freq 1위: 1/4만(CASIA 편향). fatformer 음수 Shapley(aigenproxy) | `experiments/results/shapley_phase1/cross_dataset_validation_20260319_041001.json` |
| 백본 후보 리서치 | 에이전트별 Top 후보: ForMa(freq+noise), MobileNetV2(noise), MobileCLIP-S2(fatformer), RelayFormer(spatial) | 6개 딥리서치 PDF 종합 |
| Phase 2 백본 단독 평가 (ForMa) | acc: base=0.349, dsC=0.332, opensdi=0.334, aigen=0.323. authentic recall 0.837-0.937, manipulated recall 0.067-0.150, **ai_gen recall 0** (3-class 단독 한계) | `experiments/results/backbone_eval/backbone_eval_summary_20260319_055541.json` |
| Phase 2 백본 단독 평가 (MobileCLIP-S2) | **Zero-shot**: acc 33-35%, authentic recall 7-10%. **Linear probe(ft0)**: val macro=0.790, 전체eval avg=0.932. **Last-4-block FT(ft4)**: val macro=**0.806**, 전체eval base=0.942/dsC=0.974/opensdi=0.953/aigen=0.943(avg=**0.953**). 체크포인트: `weights/mobileclip_forensics/mobileclip_s2_forensics_ft4.pth` | `experiments/results/backbone_eval/` |
| Phase 2 백본 단독 평가 (Tiny-LaDeDa) | WildRF: acc: base=0.372, dsC=0.367, opensdi=0.379, aigen=0.316. ai_gen recall 73-86%, **manip recall 0%** (binary 한계 → Cascade Tier-1 전용) | `experiments/results/backbone_eval/` |
| Phase 2 백본 단독 평가 (MobileNetV2 dual-stream) | RGB + SRM residual dual-stream 구현 후 15epoch 재학습. best val macro=**0.806**, 전체eval base=0.944/dsC=0.979/opensdi=0.949/aigen=0.961(avg=**0.958**). Track 1용 noise 축이 단독 strong baseline으로 상승 | `experiments/results/backbone_eval/mobilenetv2_dualstream_summary_20260319_070725.json` |
| Phase 2.7 백본 latency/size benchmark | H200/EPYC 실측. ForMa: 37.3M, GPU 16.8ms / CPU 1613ms. MobileCLIP-ft4: 99.4M, GPU 15.5ms / CPU 123.8ms. Tiny-LaDeDa: 0.0013M, GPU 5.8ms / CPU 2.5ms. **MobileNetV2 dual-stream**: 5.77M, GPU 18.9ms / CPU 35.9ms | `experiments/results/backbone_benchmark/backbone_benchmark_20260319_065356.json`, `.md` |
| ForMa 코드 수급 (arXiv 2502.09941) | 37.3M params, repo-root 가중치 fallback + CUDA mamba 배치 평가 경로 연결. 사용자 제공 가중치로 로드/재평가 완료 | `ForMa-main/`, `experiments/run_backbone_eval.py` |
| Phase 3.1 Track 1/2/3 앙상블 평가 (combined) | combined 10-seed: Track1-LR=0.9512★, MobileCLIP=0.9500, Track3-LR=0.9496. in-dist에서 Track1이 소폭 우세 | `experiments/results/phase3_tracks/` |
| Phase 3.1 Fair LOO 재파인튜닝 평가 | **핵심 결과**: clip_loo avg=0.6386 vs Track1-GBM avg=**0.7309** (Δ+0.092). OOD 분포 이동 시 Track1 앙상블이 MobileCLIP 단독 대비 유의미하게 우수. opensdi: clip=0.497→T1=0.697(+0.20), aigenproxy: clip=0.554→T1=0.680(+0.13). Track2/3는 Track1 대비 약함(MobileNetV2 noise 신호가 핵심 기여) | `experiments/results/fair_cross/fair_cross_20260319_070940.json` |
| Phase 3.2 경량 모델 Shapley+STII+CKA+PID | **Shapley**: MNV2 φ=+0.304 ≈ CLIP φ=+0.300 >> ForMa=Tiny=+0.008. **CKA**: CLIP↔MNV2=0.922(고중복), 나머지 모두 <0.02(독립). **STII**: CLIP↔MNV2=-0.584(최대 중복). **PID**: CLIP↔MNV2 Redundancy=0.599. **최적 2-조합**: CLIP+MNV2(F1=0.953). ForMa/Tiny는 in-dist 기여 거의 없으나 OOD 강건성(Fair LOO)에서 MNV2 단독이 결정적. 결론: CLIP+MNV2 2-모델이 in-dist 최적, OOD에는 MNV2 필수 | `experiments/results/shapley_phase3/shapley_phase3_20260319_074635.json` |
| Phase 3.5.0 에러 Overlap 분석 | avg Jaccard=0.3361 (에러 독립적). "둘 다 틀림" 패턴: manipulated→authentic 지배(56/102건). Binary specialist 설계 근거 확립 | `experiments/results/error_overlap_analysis.json`, `run_error_overlap_analysis.py` |
| Phase 3.5.1 Specialist-M v1 학습 | CASIA2(7491 auth + 5123 manip), 3-stream(RGB+SRM+DCT), 7.66M. best manip_f1=0.764(Ep.5). 4-DS: base f1=0.861/dsC=0.797/opensdi=0.584/aigenproxy=0.631. **OOD 한계**: opensdi auth_recall=0.07 | `weights/specialist_m/specialist_m_best.pth`, `experiments/results/specialist_eval/` |
| Phase 3.5.2 Specialist-G 학습 | MobileCLIP frozen(35.81M) + PiD branch(0.10M trainable). best aigen_f1=0.981(Ep.19). 4-DS: base=0.987/dsC=0.988/opensdi=0.799/aigenproxy=0.730 | `weights/specialist_g/specialist_g_best.pth`, `experiments/results/specialist_eval/` |
| Phase 3.5.3 ICWMV 4-model 합의 (w=1.0) | MNV2+CLIP+SpecM+SpecG. avg macro_F1=**96.40%** vs MNV2 95.81%(+0.59%p) vs 2-model 96.23%(+0.17%p). base+1.0, dsC+0.9, opensdi+0.9%p. aigenproxy −0.4%p(SpecM OOD 약점). "둘 다 틀림" fix rate: base 2.1%, dsC 12.5% | `experiments/results/icwmv/`, `run_icwmv_consensus.py` |
| Phase 3.5.4 Specialist-M v2 (OOD 강건화) | +IMD2020 1710장(non-eval) + JPEG압축(p=0.5, q=40~95) + GaussianNoise(p=0.4) + WeightedRandomSampler. best manip_f1=0.827(v1 0.764 대비 +6.3%p). opensdi auth_recall 7%→11%, aigenproxy 17%→25% 개선 | `weights/specialist_m_v2/specialist_m_v2_best.pth`, `experiments/results/specialist_eval/specialist_m_v2_*` |
| Phase 3.5.5 ICWMV SpecM-v2 재평가 | w=1.0: avg macro_F1=**96.48%**(v1 96.40% 대비 +0.08%p). base=95.73/dsC=99.11/opensdi=95.31/aigenproxy=**95.77%**(v1 aigenproxy 약점 개선). GBM DAAC 96.25% avg와 동등 수준 | `experiments/results/icwmv/icwmv_4model_wspec1.0_20260319_123542.json` |
| Phase 3.5.6 CKA 다양성 재분석 (4-model) | 4-model avg CKA=**0.0855** vs 2-model(MNV2+CLIP) 0.9241(ΔCKA=-0.8385). Jaccard: 4-model 0.1233 vs 2-model 0.3361. disagreement rate: 4-model 32.8% vs 2-model 4.4%. Binary specialist 추가로 출력 공간 다양성 대폭 향상 — 앙상블 설계 이론적 정당화 | `experiments/results/cka_diversity/cka_diversity_4model_20260319_124909.json` |
| Phase 3.4 경량 DAAC 메타 분류기 재학습 | 25-dim 메타 특징(MNV2+CLIP 기반, specialist 제외-label leakage 방지). GBM: base=**99.01%**(원본 무거운 DAAC 86.13% 대비 +12.88%p), avg 4-DS=96.25% ≈ ICWMV 96.48%(Δ-0.23%p). Top-feature: mnv2_aigen(29.8%), mnv2_auth(20.9%) | `experiments/results/daac_retrain/daac_retrain_lightweight_20260319_125524.json` |
| Phase 4.4 RPi5 end-to-end 벤치마크 | **환경**: RPi5 / Debian 13 / Python 3.13.5 / onnxruntime 1.24.4. **레이턴시(threads=4, 10회)**: avg **112.0ms** / min 90.3ms / max 135.4ms (예상 140ms 대비 -20%). **스레드별**: 1T=192.8ms / 2T=128.6ms / 4T=114.3ms (4T 최적). **메모리**: 156.3MB (2모델 동시 로드). **콜드스타트**: 로드 188.7ms + 추론 113.8ms. **정확도**: 서버 ICWMV와 동일 (ONNX cosine=0.99999, Dynamic INT8 무손실 확인). ICWMV 작동 확인: MNV2(aigen 58%)↔SpecM(auth 72%) 충돌 시 최종 판정 ai_generated(45%) — 불확실 케이스 정상 처리 | `inference_rpi5.py`, `weights/onnx_quant/mnv2_int8_dynamic.onnx`, `weights/onnx_quant/specm_v4_int8_dynamic.onnx` |
| Phase 4.8 Coral export dry-run | **서버 환경 구축 완료**: `.venv-edgetpu-export`(Python 3.10), `onnx2tf` 1.28.8, TensorFlow/tf-keras 2.19.0, `edgetpu_compiler` 16.0 설치. **결과**: `mnv2`는 full INT8 TFLite 생성 성공(`mnv2_int8_full.tflite`, 6.9MB) but compiler가 `Rsqrt` INT16 거부. `specm_v4`는 `wa/dct_extractor/Resize`에서 onnx2tf 실패. Coral용 현재 모델 비호환 확인 | `experiments/results/edgetpu_export/edgetpu_export_20260321_145840.json`, `weights/tflite/mnv2_int8_full.tflite`, `experiments/run_edgetpu_export.py` |
| Phase 4.9 Coral-friendly export variants | **최종 성공(2026-03-21)**. `mnv2_coral.onnx(22MB)` / `specm_v4_coral.onnx(30MB)` export 추가. `run_edgetpu_export.py`가 `*_full_integer_quant.tflite`를 우선 선택하도록 수정 후 `mnv2_coral_int8_full.tflite(6.9MB)`·`specm_v4_coral_int8_full.tflite(9.3MB)`와 `_edgetpu.tflite` 모두 생성. compiler report: **MNV2 151/151 ops**, **SpecM 227/227 ops** 전부 Edge TPU 매핑. quick sanity: MNV2 96샘플 agreement 98.96%, SpecM 64샘플 agreement 46.88%(acc 75.0%→71.9%) | `experiments/results/edgetpu_export/edgetpu_export_20260321_153333.json`, `weights/tflite/*_coral_int8_full.tflite`, `weights/tflite_edgetpu/*_coral_int8_full_edgetpu.tflite`, `experiments/coral_export_models.py` |
| Phase 4.10 Coral deployment accuracy reevaluation | **배포 경로 기준 재평가 완료(2026-03-21)**. `current_onnx` avg macro-F1=`0.9535`, `coral_tflite`=`0.9051`(Δ`-0.0483`). dataset별 ICWMV delta: base `-0.0652`, dsC `-0.0665`, opensdi `-0.0519`, aigenproxy `-0.0098`. 원인 분해: `mnv2_coral` PyTorch avg=`0.9530`이지만 TFLite=`0.9096`; `specm_v4_coral` PyTorch avg manip-F1=`0.6066`, TFLite=`0.5827`. 추가 검증: `specm_v4`에서 DCT resize만 정적으로 바꾸고 원래 head를 유지하면 compiler가 `FlexErf`로 실패 → head 교체는 불가피 | `experiments/results/coral_eval/coral_eval_compare_20260321_154854.json`, `experiments/eval_coral_rpi5_variants.py` |
| Phase 4.10b MNV2 Coral PTQ sweep | **대표 데이터/quant setting sweep 완료(2026-03-22)**. grid: `per-channel/per-tensor × calib 64/128/256 × int8 IO`. best는 **per-channel + calib 64**로 `mnv2` avg macro-F1=`0.9173`(baseline coral `0.9096` 대비 `+0.0077`). same tuned MNV2 + existing `specm_v4_coral` pair는 ICWMV avg=`0.9160`으로 baseline coral pair `0.9051` 대비 `+0.0108`, current ONNX 대비는 여전히 `-0.0375`. best candidate도 Edge TPU compile **151/151 ops** 유지 | `experiments/results/coral_quant_sweep/mnv2_coral_quant_sweep_20260322_124316.json`, `weights/tflite_sweep/mnv2_coral_qsweep_qtpc_cal064_ioint8.tflite`, `weights/tflite_edgetpu_sweep/mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite`, `experiments/run_mnv2_coral_quant_sweep.py`, `experiments/results/coral_eval/coral_tflite_mnv2_qtpc64_20260322_124423/coral_tflite_mnv2_qtpc64_summary.json` |
| Phase 4.10c SpecM coral-native fine-tune + pair retune | **핵심 돌파(2026-03-22)**. `train_specialist_m_v4_coral.py`로 `specm_v4_coral` 구조를 직접 fine-tune한 뒤 `specm_v4_coral_ft_int8_full(.tflite)` / `_edgetpu.tflite` 생성. standalone TFLite avg manip-F1=`0.8360`(old coral `0.5827` 대비 `+0.2533`, current ONNX `0.8392` 대비 `-0.0032`). tuned MNV2와의 pair는 `w_spec=1.0`에서 `0.8891`이지만, JSONL 기반 `w_spec` sweep에서 best=`0.2`, avg macro-F1=`0.9308`까지 회복(old coral pair `0.9051` 대비 `+0.0257`, current ONNX 대비 `-0.0226`) | `weights/specialist_m_v4_coral/specialist_m_v4_coral_best.pth`, `weights/tflite/specm_v4_coral_ft_int8_full.tflite`, `weights/tflite_edgetpu/specm_v4_coral_ft_int8_full_edgetpu.tflite`, `experiments/results/specialist_eval/specialist_m_v4_coral_summary_20260322_130007.json`, `experiments/results/coral_eval/coral_tflite_ft_pair_20260322_130431/`, `experiments/sweep_icwmv_wspec_from_jsonl.py` |
| Phase 4.2 PTQ INT8 양자화 정확도 평가 (재평가) | **이중정규화 버그 수정 후 재평가.** Dynamic INT8: MNV2=**95.37%**(Δ+0.00%p)/SpecM=65.14%(Δ+0.03%p)/SpecG=87.84%(Δ+0.17%p)/CLIP=95.38%(Δ+0.06%p) — 전 모델 무손실. Static: MNV2 -13.42%p/SpecG -47.23%p/CLIP -63.25%p(FastViT PTQ 붕괴)/SpecM -0.85%p. **2-model ICWMV (MNV2+SpecM Dynamic): avg 95.64%** vs 4-model 96.48%(Δ-0.84%p) — SpecM openSDI OOD 약점이 -1.56%p. RPi5 확정: **MNV2-Dynamic(14ms) + SpecM-Dynamic(21ms) = ~140ms** | `experiments/results/onnx_benchmark/quant_accuracy_20260319_163814.json` |
| Phase 4.2.1 SpecM-v3 재학습 + ONNX 재배포 | **Authentic OOD 강건화**: +GenImage_nature 3000장(ImageNet val 실사진) + RandomErasing(p=0.3, inpainting 시뮬레이션). best manip_f1=0.7832(ep.11). **opensdi auth_recall 11%→62%(+51%p)**. 2-model ICWMV(MNV2+SpecM-v3 Dynamic): avg **96.19%** (+0.55%p vs v2, 4-model 서버 대비 **Δ-0.29%p**). RPi5 최종 확정: **MNV2-Dynamic + SpecM-v3-Dynamic** | `weights/specialist_m_v3/`, `weights/onnx/specm_v3.onnx`, `weights/onnx_quant/specm_v3_int8_dynamic.onnx` |
| Phase 4.2.2 SpecM-v4 fine-tuning + ONNX + ICWMV 재평가 | **v3 resume fine-tuning**: LR=3e-5(v3 1e-4 대비 ↓), RandomErasing(value=random, inpainting fill 노이즈 시뮬), focal_alpha=0.6, 20 epochs. best val manip_f1=0.7792(ep.10). openSDI manip_recall 70.3%(v3 69.7% +0.6%p). **ICWMV v4 avg: 96.58%**(v3 96.43% 대비 +0.15%p, 서버 4-model 96.48% 대비 **+0.10%p 초과**). base=95.86%/dsC=98.44%/opensdi=94.68%/aigenproxy=97.33%. **RPi5 배포 기준 모델 v4로 업그레이드** | `weights/specialist_m_v4/`, `weights/onnx/specm_v4.onnx(29.2MB)`, `weights/onnx_quant/specm_v4_int8_dynamic.onnx(26.4MB)` |
| Phase 4.6 Embedding CKA 분석 (MNV2 vs 전 모델/브랜치) | **Unbiased Linear CKA** (Nguyen et al. 2021, 대각 제거 추정기). 4200 샘플(4-DS 통합). **브랜치별 독립성 순위**: ①PiD(64d)=0.001★ ②DCT(1280d)=0.239 ③SpecM-RGB(1280d)=0.322 ④SpecM-fused(3840d)=0.392 ⑤CLIP(512d)=0.420. **핵심 발견**: 중복 주범은 `mnv2_rgb↔specm_rgb=0.656`. SRM↔mnv2_noise=0.564. DCT와 PiD는 MNV2와 거의 독립. **전 프로젝트 모델 비교**: MobileCLIP-S2-ft4=**0.028★**(압도적 독립), SpecG=0.659, SpecM v1~v4=**0.725(전부 동일)** — RGB branch ImageNet pretrain 공유가 원인. **SpecM-v5b 설계 근거 확정**: MobileCLIP을 backbone으로 선정 | `experiments/results/cka_embedding/cka_mnv2_specm_v4_20260321_*.json`, `experiments/results/cka_embedding/cka_all_vs_mnv2_20260321_*.json` |
| Phase 4.7 SpecM-v5b frozen 학습 | MobileCLIP-S2(frozen,512d) + SRMLightCNN(128d) → 640d fused. 총 36.03M, 학습 0.21M. 40ep, lr=3e-4, batch=64. **val best manip_f1=0.7517**(ep.37). **4-DS avg manip_f1=0.8062**: base=0.8223/dsC=0.8567/opensdi=0.6475/aigenproxy=0.8983. 비교: SpecM-v4 avg=0.8165 대비 -0.010(frozen 한계). ICWMV(MNV2+v5b) avg=0.9614 | `weights/specialist_m_v5b/specialist_m_v5b_best.pth`, `experiments/results/specialist_eval/specialist_m_v5b_20260321_*.json` |
| Phase 4.7 SpecM-v5b_ft unfreeze 학습 | v5b best resume. MobileCLIP 전체 unfreeze, differential LR: clip trunk lr=1e-5 / SRM CNN+head lr=1e-4(10배 차이). 20ep 추가 학습. **val best manip_f1=0.7846**(ep.13). **4-DS avg manip_f1=0.8347**: base=0.8645/dsC=0.8932/opensdi=0.6205/aigenproxy=**0.9605**. v5b frozen 대비 +0.028, SpecM-v4 대비 **+0.018**. ICWMV(MNV2+v5b_ft) avg=**0.9635**. 약점: opensdi(OOD) 0.6205(v4 0.9468 대비 낮음) → RPi5 배포는 v4 유지 | `weights/specialist_m_v5b_ft/specialist_m_v5b_ft_best.pth`, `experiments/results/specialist_eval/specialist_m_v5b_ft_20260321_*.json` |
| **C6-A Phase 1: MNV2 backbone_eval 분석 + w(x) 설계** | **MNV2 confident-but-wrong 패턴 확인(2026-03-23)**. 오분류 케이스 confidence 분포: base mean=0.770/median=0.801, dsC=0.702, opensdi=0.722, aigenproxy=0.741. `>0.9` 비율: 10~35%. 핵심 발견: **MNV2는 틀릴 때도 high-confidence(≈0.77)**로 단순 threshold cascade 비효율 실증. Temperature Scaling: T=1.2263 (MNV2 약간 과신). w(x) 캐시 생성 완료(19324개). γ=1: mean=0.318/median=0.177, γ=2: mean=0.200/median=0.031, γ=3: mean=0.153/median=0.006 | `experiments/results/phase1_ewct/mnv2_training_weights_cache.jsonl` |
| **C6-A Phase 2a: SpecM-Comp γ ablation (w_max=10)** | **γ=1 최적 확정(2026-03-23)**. γ=1: val best **0.8385**(E2), γ=2: 0.7727(E1→급락), γ=3: 0.7578(E1→급락). w(x) median이 극단적일수록(γ↑) val 불안정. **eval 4-DS 결과** — γ=1: avg manip_f1=0.7918(base=0.8015/dsC=0.8250/opensdi=0.5870/aigenproxy=0.9600), MNV2 교정률 avg=0.738(base=80.5%/dsC=64.7%/opensdi=44.4%/aigenproxy=88.9%). γ=2: avg=0.7629, γ=3: avg=0.7514. **HEMA PoC 대비 교정률 대폭 개선**: base 50%→80.5%, aigenproxy 34.3%→88.9% | `weights/specialist_m_complementary/gamma{1,2,3}.0_wmax10/best.pth`, `experiments/results/specm_complementary_eval/specm_comp_gamma*.json` |
| **C6-A Phase 2b: w_max ablation (완전 완결)** | **모든 γ×w_max 조합에서 w_max 클리핑 무효 완전 확인(2026-03-23)**. w(x)=(1-c(x))^γ ∈ [0,1]이므로 w_max>1은 이론+실험으로 검증. γ=1: w_max=1 vs 10 완전 동일. γ=2: w_max=5 vs 10 vs **20** 완전 동일(base=0.8122/dsC=0.8139/opensdi=0.4781/aigenproxy=0.9472 비트 단위 일치). **결론: w_max 하이퍼파라미터 제거 가능 — 논문 명시** | `weights/specialist_m_complementary/gamma1.0_wmax{1,10}/best.pth`, `weights/specialist_m_complementary/gamma2.0_wmax{5,10,20}/best.pth` |
| **C6-A Phase 3: Fuser 비교 (MNV2 + SpecM-Comp γ=1)** | **HEMA-XGBoost 최적 확인(2026-03-23)**. 5가지 Fuser 비교(MNV2 단독/SpecM 단독/ICWMV/Cascade/HEMA-XGBoost/FoE-MLP). **avg macro-F1**: HEMA-XGBoost=**0.9617**(+0.011 vs MNV2 0.9503), ICWMV=0.9546, Cascade(τ=0.6)=0.9537, FoE-MLP=0.9450. **avg MNV2 교정률**: SpecM 단독=0.696(최고), HEMA-XGBoost=**0.598**(F1+교정률 균형 최적), ICWMV=0.433. dsC에서 HEMA 교정률=100%. Cascade: τ↑일수록 교정률↑(τ=0.95: 66.2%) — MNV2 오분류가 high-confidence라서 낮은 τ에선 SpecM 개입이 불가. **HEMA-XGBoost** = 논문 제안 Fuser로 확정 | `experiments/results/fuser_comparison/fuser_comp_g1_20260322_160619.json`, `experiments/run_fuser_comparison.py` |
| **C6-A Phase 3b: HEMA LOO-CD + Feature Ablation + Model Comparison** | **LOO-CD 완료(2026-03-23)**. in-domain 70/30 대비 LOO-CD의 실제 교차 도메인 성능 측정. **LOO-CD avg**: F1=0.9452/교정률=38.2% (in-domain 0.9617/59.8% 대비 **-1.7%p/-21.6%p**). **All-In 5-fold**: F1=0.9514/38.9%. **Feature Ablation(LOO-CD)**: Cross(3-dim)=**F1 0.9525 > Full(13-dim) 0.9452** — cross-modal 특징 3개가 도메인 불변 신호 담당. MNV2-only(6-dim)=교정률 18.1%(매우 낮음). **Model Comparison(LOO-CD)**: comp_noTS=F1 0.9546/교정률 41.4%(최고) > v4=0.9528/36.9% > comp_g1=0.9452/38.2%. **in-domain 역전**: comp_g1이 LOO-CD에서 최하(in-domain 최고)였던 이유: TS+EWCT가 도메인 특이적 패턴 학습. comp_noTS가 LOO-CD 기준 최적 | `experiments/results/hema_crossdataset/hema_cd_comp_g1_20260323_012431.json`, `experiments/run_hema_xgb_crossdataset.py` |
| **C6-A Phase 3c: HEMA 복구 시도 (full-coverage fair eval)** | **Scalar-level learned fusion 복구 실험 완료(2026-03-23)**. 기존 `comp_noTS/comp_g1`는 auth/manip subset JSONL만 있던 상태라, 먼저 `run_specm_eval.py`로 **full-coverage** 4-DS JSONL을 새로 생성한 뒤 공정 비교. 신규 `HEMA action-gate`(keep/override/fallback)와 `ICWMV-veto`(ICWMV override만 선택적 되돌림) 모두 strong/weak MNV2에서 ICWMV를 **동시에 F1+교정률로 넘지 못함**. best 근접 사례: strong+comp_noTS action-gate F1=`0.9628` vs ICWMV `0.9630`(Δ`-0.0002`), corr=`0.423` vs `0.492`; weak+comp_noTS는 F1만 `+0.0142` 또는 `+0.0131` 이겼지만 corr이 각각 `-0.229`/`-0.360` 급락. 반면 **oracle veto**는 strong+comp_noTS `0.9726`, weak+comp_noTS `0.8935`로 headroom이 큼. 결론: 현재의 **scalar feature 기반 learned gate/veto는 ICWMV의 유익한 override와 해로운 override를 충분히 분리하지 못함**. HEMA를 살리려면 score-level 메타분류기보다 representation-level 신호나 richer disagreement evidence가 필요 | `experiments/run_specm_eval.py`, `experiments/run_hema_action_gate_loo_cd.py`, `experiments/run_hema_icwmv_veto_loo_cd.py`, `experiments/results/specm_eval/specm_comp_noTS_summary_20260323_112912.json`, `experiments/results/specm_eval/specm_comp_g1_summary_20260323_112928.json`, `experiments/results/hema_action_gate/hema_action_gate_loo_cd_20260323_113100.json`, `experiments/results/hema_icwmv_veto/hema_icwmv_veto_loo_cd_20260323_114321.json` |
| **Paper support: ICWMV backbone transfer / strength ladder** | **ICWMV 일반성 실험 확장 완료(2026-03-23)**. 동일한 `SpecM-v4`를 고정하고 generalist backbone만 교체: `MNV2 no-finetune(ImageNet pretrained branch + random forensic head)`, `MNV2 strong`, `MNV2 weak`, `MobileCLIP-ft4 strong`, `MobileCLIP zero-shot weak(no fine-tuning)`. 결과: `MNV2 no-ft 0.3658→0.5556 (+0.1898), corr=0.8080`; `MNV2 strong 0.9581→0.9652 (+0.0071), corr=0.3545`; `MNV2 weak 0.8414→0.8637 (+0.0223), corr=0.4977`; `CLIP-ft4 strong 0.9532→0.9569 (+0.0037), corr=0.1973`; `CLIP zero-shot weak 0.3005→0.4320 (+0.1314), corr=0.7708`. 결론: **ICWMV는 MNV2 전용 트릭이 아니라 backbone family를 넘어 전이되며, gain은 backbone family보다 strength/error volume에 더 크게 비례**. weak CLIP과 no-ft MNV2는 기존 결과에 full `scores`가 없어 신규 스크립트가 cache를 자동 생성하도록 구현 | `experiments/run_icwmv_backbone_transfer.py`, `experiments/results/icwmv_backbone_transfer/icwmv_backbone_transfer_20260323_142828.json`, `experiments/results/backbone_eval/mobileclip_s2_zeroshot_scored_*_20260323_141929.jsonl`, `experiments/results/backbone_eval/mobilenetv2_dualstream_nofinetune_*_20260323_142821.jsonl`, `docs/research/PAPER_DRAFT_v1.md` |
| **C6-A Phase 4: 3-layer 평가 (MNV2/SpecM-v4/SpecM-Comp/Cascade)** | **3층 종합 평가 완료(2026-03-23)**. Layer1(macro-F1): MNV2=0.9581, SpecM-v4=0.8829, SpecM-Comp=0.7959, Cascade(τ=0.6)=0.9547. Layer2(교정률): SpecM-Comp=**0.696**(최고), SpecM-v4=0.610, Cascade=0.186. Layer3(ECE): MNV2=0.028, SpecM-Comp=0.092(캘리브레이션 불량). AURC: MNV2=0.007(최고), SpecM-Comp=0.117. **Brier**: MNV2=0.021, SpecM-Comp=0.097. 해석: SpecM-Comp 단독은 3-class 공간에서 낮지만 MNV2 오분류 교정에 특화. HEMA-XGBoost(Phase 3)가 F1+교정률 동시 최적 | `experiments/results/phase4_3layer/phase4_3layer_20260323_005459.json`, `experiments/run_phase4_3layer_eval.py` |

---

## 8. Progress Ledger
> 형식: `YYYY-MM-DD | Scope | Change | Key Files | Verification | Next`
> 최신 항목이 맨 위.

- `2026-03-24 | docs/refresh-roundup | 문서 허브를 현재 연구 결론 기준으로 전면 정리. `README.md`를 DAAC / SHIELD / ICWMV / EWCT 중심의 최신 프로젝트 허브로 재작성했고, 오래된 `docs/PROGRESS_REPORT.md`를 2026-03-24 기준 연구 상태 요약판으로 교체. `deploy/README.md`와 `docs/research/RPi5_EXPERIMENT_GUIDE.md`는 현재 저장소에 실제 포함된 배포 진입점(`deploy/rpi5_infer.py`)과 최신 RPi5 실측 수치(ONNX `88.3ms`, Coral `63.4ms`) 기준으로 갱신. `PAPER_DRAFT_NOTES.md`는 draft v3.1 기준 논문 주장/수치/제외 claim을 재정리했고, `EWCT_EXPERIMENT_REPORT.md` 상단에는 현재 해석(HEMA는 exploratory, ICWMV가 메인 baseline)과 chronology 주의문을 추가 | README.md, docs/PROGRESS_REPORT.md, deploy/README.md, docs/research/PAPER_DRAFT_NOTES.md, docs/research/RPi5_EXPERIMENT_GUIDE.md, docs/research/EWCT_EXPERIMENT_REPORT.md, AGENTS.md | 주요 문서 대상 `rg` 재검색으로 stale 표현/구 경로 제거 확인 | 참고문헌/표 캡션 최종 마감 후 문서 표현 추가 압축`
- `2026-03-24 | paper/draft-v1-structure-cleanup | 사용자가 진행한 본문 축소 이후 남아 있던 정합성 후처리를 마무리. `PAPER_DRAFT_v1.md`에서 제목을 `역신뢰도 가중 다수결과 엣지 배포`로 다듬고, 키워드를 `역신뢰도 가중 다수결`로 통일. 삭제된 섹션 기준으로 표 번호를 재정렬해 Table 4(백본 일반화), Table 5(RPi5 레이턴시), Table 6(배포 경로 종합)로 맞췄고, `선행 연구 [10]는 -> [10]은` 문장 교정 및 `MNV2 no-finetuning` 각주 표현도 자연스럽게 수정 | docs/research/PAPER_DRAFT_v1.md, AGENTS.md | 표 번호/용어 재검색으로 잔여 불일치 없음 확인 | 참고문헌 형식/캡션 스타일 최종 통일`
- `2026-03-23 | paper/draft-v1-polish | `PAPER_DRAFT_v1.md`를 v2.3으로 다듬음. `ICWMV`의 한국어 정식 명칭을 **역신뢰도 가중 다수결**로 통일하고 제목/초록/서론/제안방법/결론에 반영. Section 6.5에 `MNV2 no-finetuning` 결과(`0.3658→0.5556`, Δ`+0.1898`, corr=`0.8080`)를 보조 stress-test로 추가하고, MobileCLIP zero-shot과의 비대칭성(`random forensic head`)을 각주로 명시. 아울러 Table 2의 dataset별 ΔF1 표기 오류(`+0.128` 등)를 `+0.0128` 형식으로 수정해 수치 일관성 정리 | docs/research/PAPER_DRAFT_v1.md, AGENTS.md | 문서 수치/용어 재검토 완료 | 표/참고문헌/국문 용어 최종 선택 마감`
- `2026-03-23 | paper/icwmv-transfer-mnv2-nofinetune | 사용자가 요청한 `MNV2 no-finetuning` 조건을 추가로 실험. 정의는 `ImageNet-pretrained dual-stream branch + random forensic head(no task training)`로 두고, `run_icwmv_backbone_transfer.py`에 `mnv2_nofinetune` cache 생성 경로를 추가. 결과: baseline avg F1=`0.3658`, ICWMV+SpecM-v4=`0.5556`(Δ`+0.1898`), 교정률=`0.8080`; dataset별로 base `0.4507→0.6316`, dsC `0.4610→0.6427`, opensdi `0.2377→0.3496`, aigenproxy `0.3139→0.5985`. 해석: zero-shot MobileCLIP과 마찬가지로 no-ft MNV2는 절대 성능은 낮지만, ICWMV가 극약한 operating point에서도 큰 교정 이득을 준다는 stress-test 증거가 추가됨 | experiments/run_icwmv_backbone_transfer.py, experiments/results/icwmv_backbone_transfer/icwmv_backbone_transfer_20260323_142828.json, experiments/results/backbone_eval/mobilenetv2_dualstream_nofinetune_*_20260323_142821.jsonl, AGENTS.md | py_compile + cache 생성 + full rerun 완료 | 논문 본문에 no-ft MNV2를 메인 표에 넣을지 appendix로 보낼지 결정`
- `2026-03-23 | paper/icwmv-transfer | ICWMV를 논문 제안 방법으로 밀기 위한 backbone transfer/strength ladder 실험 추가. 신규 `run_icwmv_backbone_transfer.py`를 구현해 strong/weak `MNV2`, strong `MobileCLIP-ft4`, weak `MobileCLIP zero-shot`을 동일 `SpecM-v4` 기준으로 비교했고, weak CLIP용 full-score JSONL cache(`mobileclip_s2_zeroshot_scored_*`)를 자동 생성. 결과: strong MNV2 `+0.71%p`, weak MNV2 `+2.23%p`, strong CLIP `+0.37%p`, weak CLIP `+13.14%p` — ICWMV가 MNV2 전용이 아니라 backbone family를 넘어 전이되며 gain은 weak operating point에서 크게 확대됨을 확인. 이를 반영해 `PAPER_DRAFT_v1.md` 초록/서론/6.1/6.5/토의/결론을 `family+strength ladder` 서술로 수정 | experiments/run_icwmv_backbone_transfer.py, experiments/results/icwmv_backbone_transfer/icwmv_backbone_transfer_20260323_141939.json, experiments/results/backbone_eval/mobileclip_s2_zeroshot_scored_*_20260323_141929.jsonl, docs/research/PAPER_DRAFT_v1.md, AGENTS.md | py_compile + full experiment run 완료 | 표/참고문헌/ICWMV-v4 LOO-CD 서술 최종 교정`
- `2026-03-23 | paper/draft-v1 | docs/research/PAPER_DRAFT_v1.md 전면 구조 수정. 프레이밍을 `DAAC 4-agent+meta-classifier는 엣지에 과대 -> 여러 경량 백본 비교 후 MNV2 선택 -> COBRA/DAAC 합의 아이디어에서 영감을 받은 ICWMV 제안 -> MAIFS-Edge`로 교정. ICWMV를 **본 논문 제안 엣지 합의**로 명시하고, DAAC 연계·공유 LOO-CD 프로토콜·MAIFS 4-에이전트의 엣지 경량 버전 포지션을 전반에 반영 | docs/research/PAPER_DRAFT_v1.md, AGENTS.md | 문서 구조/용어 정합성 확인 | 표/참고문헌/수치 근거 최종 교정`
- `2026-03-23 | ewct/final | 모든 ablation 완료. γ=2 w_max=20 → w_max=5/10과 비트 단위 동일(이론 완전 검증). ICWMV v4 버그 수정(ai_gen fallback 추가) → 재평가: ICWMV-v4 avg_F1=0.9652(이전 0.5173 broken → 수정). Phase 3 v4 정상 기준선 확보. EWCT_EXPERIMENT_REPORT.md 13개 섹션 완성(표지 요약/Phase1-4/LOO-CD/논문용 테이블). SHIELD_RESEARCH_PLAN.md Section 6.2 추가 | experiments/results/fuser_comparison/fuser_v4_20260323_015411.json, experiments/results/specm_complementary_eval/specm_comp_gamma2.0_wmax20_summary_*.json, docs/research/EWCT_EXPERIMENT_REPORT.md | 파일 생성 확인 | LOO-CD 기반 comp_noTS 재검토`
- `2026-03-23 | ewct/hema-repair | HEMA 복구 시도 2종을 full-coverage 공정 조건에서 재평가. 먼저 `run_specm_eval.py --model comp_noTS/comp_g1`로 전체 4-DS JSONL을 다시 생성해 이전 auth/manip subset 편향을 제거. 그 후 `run_hema_action_gate_loo_cd.py`(disagreement-only keep/override/fallback)와 `run_hema_icwmv_veto_loo_cd.py`(ICWMV override selective veto)를 strong/weak MNV2 모두에서 실행. 결과: 둘 다 ICWMV를 F1+교정률로 동시에 넘지 못함. strong+comp_noTS action-gate만 `0.9628 vs 0.9630`으로 근접했지만 corr는 `0.423 vs 0.492`. weak+comp_noTS는 F1만 `+1.3~1.4%p` 올렸으나 corr가 `-22.9~-36.0%p` 급락. 반면 oracle veto는 strong+comp_noTS `0.9726`, weak+comp_noTS `0.8935`로 headroom이 큼. 해석: current scalar/meta-score 수준에선 ICWMV의 beneficial override와 harmful override를 안정적으로 분리하지 못하며, learned fusion을 살리려면 representation-level evidence가 필요 | experiments/results/specm_eval/specm_comp_noTS_summary_20260323_112912.json, experiments/results/specm_eval/specm_comp_g1_summary_20260323_112928.json, experiments/results/hema_action_gate/hema_action_gate_loo_cd_20260323_113100.json, experiments/results/hema_icwmv_veto/hema_icwmv_veto_loo_cd_20260323_114321.json, experiments/run_hema_action_gate_loo_cd.py, experiments/run_hema_icwmv_veto_loo_cd.py | py_compile + full-coverage eval + strong/weak LOO-CD 실행 완료 | ICWMV-v4 동일 프로토콜 고정 baseline 유지 + representation-level learned fusion 가능성 검토`
- `2026-03-23 | ewct/hema-loo-cd | HEMA LOO-CD + Feature Ablation + Model Comparison 완료. 핵심 발견: (1) in-domain 교정률 59.8% → LOO-CD 38.2% (-21.6%p), (2) Cross 3-dim 특징이 Full 13-dim보다 LOO-CD에서 더 좋음(도메인 불변 신호), (3) comp_noTS가 LOO-CD 기준 최적(F1 0.9546, 교정률 41.4%). in-domain 최고인 comp_g1이 LOO-CD 최하로 역전. 논문에서 LOO-CD 기준 사용 권장 | experiments/results/hema_crossdataset/hema_cd_comp_g1_20260323_012431.json | 파일 생성 확인 | Cross 특징 도메인 불변성 추가 분석 + comp_noTS LOO-CD 재검토`
- `2026-03-23 | ewct/phase2b-extended | w_max ablation γ=2 완결. γ=2 w_max=5 vs w_max=10 완전 동일(비트 단위) — w(x)=(1-c(x))^γ∈[0,1] 이론이 모든 γ에서 검증됨. w_max는 논문 하이퍼파라미터에서 제거 가능. no-TS ablation 완료(avg_f1=0.7632, avg_corr=0.7718 — TS 제거 시 교정률 상승/F1 하락). HEMA LOO-CD cross-dataset 평가 실행 중(PID 139692) | experiments/results/specm_complementary_eval/specm_comp_gamma2.0_wmax5_summary_*.json, specm_comp_gamma1.0_wmax10_noTS_summary_*.json | γ=2 w_max=5 eval 완료 | γ=2 w_max=20 + HEMA LOO-CD 완료 후 문서 최종화`
- `2026-03-23 | ewct/phase3-4 | Phase 3 Fuser 비교 + Phase 4 3-layer eval 완료. HEMA-XGBoost avg_F1=0.9617(MNV2 0.9503 대비 +0.011, 교정률=0.598). SpecM-Comp 단독 교정률=0.696(최고)이지만 macro-F1=0.7959(3-class 공간에서 낮음). Cascade(τ=0.6) 교정률=0.186 — MNV2 오분류가 high-confidence(mean≈0.77)라서 낮은 τ로는 개입 불가 → τ=0.95에서 66.2%까지 개선. Phase 4 3-layer: ECE(SpecM-Comp=0.092 불량, MNV2=0.028 양호), AURC(SpecM-Comp=0.117, MNV2=0.007). 신규 스크립트: run_fuser_comparison.py, run_phase4_3layer_eval.py, run_specm_eval.py | experiments/results/fuser_comparison/, experiments/results/phase4_3layer/, experiments/results/specm_eval/ | Phase 3/4 eval 실행 완료 | γ=2 w_max ablation + no-TS ablation + Phase3-v4 기준선`
- `2026-03-23 | ewct/phase2 | SpecM-Comp γ ablation (γ=1,2,3 w_max=10) + w_max ablation (γ=1: w_max=1.0 vs 10.0) 완료. γ=1 최적 확정: val best_f1=0.8385, eval avg manip_f1=0.7918. γ↑→w(x) median 극단화(0.177→0.031→0.006)→val 불안정. γ=1에서 w_max 클리핑 무효 확인(w(x)∈[0,1]). MNV2 오분류 교정률 대폭 개선: base 50%→80.5%, aigenproxy 34.3%→88.9%. T=1.2263(MNV2 약간 과신). 진행 중: no-TS ablation + γ=2 w_max=5/20 | weights/specialist_m_complementary/gamma{1,2,3}.0_wmax{1,10}/, experiments/results/specm_complementary_eval/ | 4-DS eval 완료 | no-TS + γ=2 w_max 변형 완료 후 Phase 4 최종`
- `2026-03-23 | ewct/phase1 | MNV2 backbone_eval 분석 + DeepResearch 3종 PDF 분석 완료. confident-but-wrong 패턴 확인(오분류 mean conf≈0.77). Temperature Scaling 구현(T=1.2263). w(x) 캐시 생성(19324개). SpecM-v4 eval 전 4-DS 생성(specm_eval/). 설계 결정: Dustbin(ai_gen→manip 병합), 정답확률 c(x)=P(y_true|x), TS ablation, Medium contribution | experiments/results/phase1_ewct/, experiments/results/specm_eval/ | 캐시+eval 완료 | Phase 2 학습`
- `2026-03-22 | shield/phase4.10 | SpecM coral-native fine-tune + pair retune 완료. `train_specialist_m_v4_coral.py`로 `specm_v4_coral` 구조를 직접 fine-tune해 PyTorch 4-DS avg manip-F1=`0.8429`까지 회복(v4 `0.8444`와 거의 동일). 이어 `specm_v4_coral_ft`를 ONNX→full INT8 TFLite→Edge TPU compile까지 연결했고, TFLite standalone avg manip-F1=`0.8360`으로 current ONNX `0.8392`에 근접. tuned MNV2와 조합 시 `w_spec=1.0`은 과가중으로 `0.8891`이지만, 신규 `sweep_icwmv_wspec_from_jsonl.py`로 sweep한 결과 best=`w_spec=0.2`, avg macro-F1=`0.9308` 확보. 조치: explicit `tflite|edgetpu` 기본 SpecM 후보를 `specm_v4_coral_ft*`로 올리고, 이 조합에서는 `w_spec=0.2`를 자동 기본값으로 설정 | experiments/train_specialist_m_v4_coral.py, weights/specialist_m_v4_coral/specialist_m_v4_coral_best.pth, weights/tflite/specm_v4_coral_ft_int8_full.tflite, weights/tflite_edgetpu/specm_v4_coral_ft_int8_full_edgetpu.tflite, experiments/results/coral_eval/coral_tflite_ft_pair_20260322_130431/, experiments/sweep_icwmv_wspec_from_jsonl.py, inference_rpi5.py, README.md, deploy/README.md, AGENTS.md | py_compile + coral fine-tune + export/compile + 4-DS eval + w_spec sweep 완료 | 실제 Coral USB latency 실측 또는 마지막 `~2.3%p` 격차 추가 축소`
- `2026-03-22 | shield/phase4.10 | MNV2 Coral PTQ sweep 완료. `run_edgetpu_export.py`에 output path / IO dtype 훅을 추가하고, 신규 스크립트 `run_mnv2_coral_quant_sweep.py`로 `per-channel/per-tensor × calib 64/128/256 × int8 IO` 1차 grid를 4-DS에서 평가. best는 `per-channel + calib64`로 `mnv2` avg macro-F1=`0.9173`(baseline coral `0.9096` 대비 +0.0077). tuned MNV2 + 기존 `specm_v4_coral` pair도 ICWMV avg=`0.9160`으로 baseline coral pair `0.9051` 대비 +0.0108까지 회복. Edge TPU compile은 151/151 ops 유지. 조치: `inference_rpi5.py` explicit `tflite|edgetpu` 경로가 tuned MNV2 후보를 우선 사용하도록 갱신. 결론: MNV2는 부분 회복됐지만 전체 병목은 여전히 `specm_v4_coral` | experiments/run_mnv2_coral_quant_sweep.py, experiments/results/coral_quant_sweep/mnv2_coral_quant_sweep_20260322_124316.json, experiments/results/coral_eval/coral_tflite_mnv2_qtpc64_20260322_124423/coral_tflite_mnv2_qtpc64_summary.json, inference_rpi5.py, README.md, deploy/README.md, AGENTS.md | py_compile + 4-DS sweep + Edge TPU compile + tuned pair eval 완료 | SpecM Coral 대체안 설계 또는 실제 Coral USB latency 실측`
- `2026-03-21 | shield/phase4.10 | Coral deployment-path 정확도 재평가 완료. 신규 스크립트 `eval_coral_rpi5_variants.py`로 current ONNX vs coral TFLite를 동일한 `inference_rpi5.py` 전처리/백엔드로 4-DS 비교. 결과: current ONNX avg macro-F1=`0.9535`, coral TFLite=`0.9051`(Δ`-0.0483`). 원인 분해: `mnv2_coral`은 PyTorch avg=`0.9530`으로 유지되지만 full INT8 TFLite=`0.9096`으로 하락 → PTQ 병목. `specm_v4_coral`은 PyTorch 단계부터 avg manip-F1=`0.6066`, TFLite=`0.5827` → head 교체 영향이 더 큼. 추가 확인: DCT resize만 정적으로 바꾸고 원래 `LayerNorm+GELU` head를 유지한 `SpecM`은 compiler가 `FlexErf`로 실패. 조치: `inference_rpi5.py` auto를 ONNX 우선으로 되돌려 silent regression 방지 | experiments/eval_coral_rpi5_variants.py, experiments/results/coral_eval/coral_eval_compare_20260321_154854.json, inference_rpi5.py, README.md, deploy/README.md, AGENTS.md | 4-DS 재평가 + 원인 분리 완료 | Coral용 QAT/quant 재탐색 또는 SpecM 전용 대체안 검토`
- `2026-03-21 | shield/phase4.9 | Coral-friendly export variants 완료. `coral_export_models.py` + `export_coral_onnx.py` 추가, `run_edgetpu_export.py`가 `full_integer_quant.tflite`만 선택하도록 수정. `mnv2_coral`/`specm_v4_coral` ONNX→full INT8 TFLite→`_edgetpu.tflite` end-to-end 성공. compiler report: MNV2 151/151 ops, SpecM 227/227 ops 전부 Edge TPU 매핑. `inference_rpi5.py`는 `*_coral` 산출물을 자동 우선 사용하도록 갱신. quick sanity: MNV2 96샘플 agreement 98.96%, SpecM 64샘플 agreement 46.88%(acc 75.0%→71.9%) → full 4-DS 재평가 필요 | experiments/coral_export_models.py, experiments/export_coral_onnx.py, experiments/run_edgetpu_export.py, weights/onnx/mnv2_coral.onnx, weights/onnx/specm_v4_coral.onnx, weights/tflite/*_coral_int8_full.tflite, weights/tflite_edgetpu/*_coral_int8_full_edgetpu.tflite, inference_rpi5.py, README.md, deploy/README.md, AGENTS.md | 실제 export/compile + 샘플 sanity 완료 | Phase 4.10 정확도 재평가 + RPi5 Coral 실측`
- `2026-03-21 | shield/phase4.8 | Coral export 실변환 수행. 서버에 `edgetpu_compiler` 16.0 + `.venv-edgetpu-export`(onnx2tf 1.28.8 / tensorflow 2.19 / tf-keras 2.19 / onnx 1.19) 구성. `mnv2`: full INT8 TFLite 생성 성공(6.9MB) but compiler가 `Rsqrt` INT16 unsupported로 실패. `specm_v4`: onnx2tf가 `wa/dct_extractor/Resize`에서 실패. 결론: 현재 ONNX 모델은 Coral-friendly 아키텍처가 아니며 head/resize 리팩터링 필요. `run_edgetpu_export.py`는 실패 JSON 저장 + venv PATH 보정까지 반영 | experiments/run_edgetpu_export.py, experiments/results/edgetpu_export/edgetpu_export_20260321_145840.json, weights/tflite/mnv2_int8_full.tflite, README.md, deploy/README.md, AGENTS.md | 실제 export/compile 실행 | Phase 4.9 Coral-friendly 모델 리팩터링`
- `2026-03-21 | shield/phase4.8 | Coral USB 배포 경로 정비 착수. Python 3.9 전용 venv bootstrap(`deploy/setup_rpi5_coral_env.sh`) 추가, ONNX→full INT8 TFLite→Edge TPU compile wrapper(`experiments/run_edgetpu_export.py`) 추가, `inference_rpi5.py`를 multi-backend(auto/onnx/tflite/edgetpu)로 확장. README/deploy 문서도 Coral 경로 기준으로 갱신. 실제 TFLite 산출물 생성 및 RPi5 Coral 실측은 후속 필요 | inference_rpi5.py, experiments/run_edgetpu_export.py, deploy/setup_rpi5_coral_env.sh, deploy/requirements_rpi5_coral.txt, README.md, deploy/README.md, AGENTS.md | py_compile + bash -n | 서버에서 run_edgetpu_export.py 실행 후 Coral 실측`
- `2026-03-21 | shield/phase4.4 | RPi5 end-to-end 벤치마크 완료. 환경: Debian 13/Python 3.13.5/onnxruntime 1.24.4. 추론 avg 112ms(예상 140ms 대비 -20%), threads=4 최적(1T=192ms→4T=114ms). 메모리 156MB(2모델 동시). 콜드스타트 189ms+114ms. ICWMV 충돌 처리(MNV2 aigen↔SpecM auth) 정상 동작 확인. RPi5 배포 완전 검증 완료. | inference_rpi5.py, weights/onnx_quant/ | 실측 완료 | Phase 4.5 ForensicHub/WildRF 벤치마크`
- `2026-03-21 | shield/phase4.7 | SpecM-v5b_ft ICWMV 평가 완료. MNV2+SpecM-v5b_ft ICWMV avg=0.9635 (v4 기준 0.9658 대비 -0.0023). 4-DS avg manip_f1=0.8347(v4 0.8165 대비 +0.018). opensdi OOD 약점(manip_f1=0.6205)으로 RPi5 배포는 v4 유지 결정. 신규 스크립트: run_icwmv_v5b.py | experiments/results/icwmv/ | 평가 완료 | Phase 4.4 RPi5 실측`
- `2026-03-21 | shield/phase4.7 | SpecM-v5b_ft (MobileCLIP unfreeze + differential LR) 학습 완료. v5b best resume, clip lr=1e-5/head lr=1e-4, 20ep. val best manip_f1=0.7846(ep.13). 4-DS avg=0.8347: base=0.8645/dsC=0.8932/opensdi=0.6205/aigenproxy=0.9605. v5b frozen 대비 +0.028, SpecM-v4 대비 +0.018. 신규 스크립트: train_specialist_m_v5b_ft.py | weights/specialist_m_v5b_ft/specialist_m_v5b_ft_best.pth | 학습 완료 | ICWMV 평가`
- `2026-03-21 | shield/phase4.7 | SpecM-v5b frozen 학습 완료. MobileCLIP-S2(frozen,512d)+SRMLightCNN(128d)=640d, 0.21M 학습파라미터. 40ep, lr=3e-4. val best manip_f1=0.7517(ep.37). 4-DS avg=0.8062: base=0.8223/dsC=0.8567/opensdi=0.6475/aigenproxy=0.8983. 신규 스크립트: train_specialist_m_v5b.py | weights/specialist_m_v5b/specialist_m_v5b_best.pth | 학습 완료 | v5b_ft unfreeze 학습`
- `2026-03-21 | shield/phase4.6 | 전 프로젝트 모델 vs MNV2 Unbiased CKA 비교 완료. **MobileCLIP-S2-ft4=0.028★**(압도적 독립), SpecG=0.659, SpecM v1~v4=0.725(전부 동일 — RGB branch ImageNet pretrain 공유). 핵심 발견: SpecM에 신호를 추가해도 CKA는 변하지 않음(RGB가 지배). SpecM-v5b backbone=MobileCLIP으로 확정. 신규 스크립트: run_cka_all_models_vs_mnv2.py | experiments/results/cka_embedding/ | 분석 완료 | Phase 4.7 SpecM-v5b 설계·학습`
- `2026-03-21 | shield/phase4.6 | Embedding-level Unbiased CKA 분석 완료 (MNV2 vs 전 모델/브랜치, n=4200). 독립성 순위: PiD=0.001★ > DCT=0.239 > SpecM-RGB=0.322 > SpecM-fused=0.392 > CLIP=0.420 > SRM=0.563. 핵심 발견: 중복 주범=mnv2_rgb↔specm_rgb(CKA=0.656), SRM↔mnv2_noise(0.564)도 중복. DCT와 PiD는 MNV2와 거의 독립(CKA<0.25). manipulated 클래스에서 중복 최심(0.706). **SpecM-v5b 설계 방향 전환: RGB/SRM 제거, MobileCLIP backbone 채택**. 신규 스크립트: run_cka_embedding_mnv2_specm.py, run_cka_all_vs_mnv2.py | experiments/results/cka_embedding/ | 분석 완료 | Phase 4.7 SpecM-v5b 학습`
- `2026-03-20 | shield/phase4.2.2 | SpecM-v4 fine-tuning + ONNX export + ICWMV 재평가 완료. v3 best checkpoint resume, LR=3e-5, RandomErasing(value=random). best val manip_f1=0.7792(ep.10). openSDI manip_recall 70.3%(v3 +0.6%p). ICWMV v4 avg=96.58%(v3 96.43% 대비 +0.15%p). **서버 4-model 96.48% 초과(+0.10%p)**. base=95.86/dsC=98.44/opensdi=94.68/aigenproxy=97.33%. ONNX cosine=0.99999(무손실). RPi5 배포 모델 v3→v4 업그레이드. | weights/specialist_m_v4/, weights/onnx/specm_v4.onnx(29.2MB), weights/onnx_quant/specm_v4_int8_dynamic.onnx(26.4MB) | ONNX cosine=0.99999 | Phase 4.4 RPi5 실측`
- `2026-03-20 | shield/phase4.2 | SpecM-v3 재학습 + ONNX 재배포 완료. GenImage_nature 3000장 + RandomErasing(p=0.3) 추가. best manip_f1=0.7832(ep.11). opensdi auth_recall 11%→62%(+51%p). 2-model ICWMV (MNV2+SpecM-v3 Dynamic): avg 96.19%(+0.55%p vs v2). 서버 4-model 96.48% 대비 Δ-0.29%p — RPi5 2-model이 서버 수준에 근접. ONNX: weights/onnx/specm_v3.onnx(30.6MB), INT8: weights/onnx_quant/specm_v3_int8_dynamic.onnx(27.7MB). 추론 스크립트: inference_rpi5.py | weights/specialist_m_v3/, weights/onnx/specm_v3.onnx, weights/onnx_quant/specm_v3_int8_dynamic.onnx, inference_rpi5.py | 완료 | Phase 4.4 RPi5 실측`
- `2026-03-20 | shield/phase4.2 | PTQ 정확도 재평가 완료(이중정규화 버그 수정). MNV2-Dynamic: 76.11%→95.37%(+19.26%p 회복), FP32 대비 Δ=0.00%p. SpecM-Dynamic: 65.54%→65.14%(Δ+0.03%p 무손실). Static INT8: SpecG/CLIP 붕괴 확정(FastViT PTQ 한계), SpecM -0.85%p. 2-model ICWMV (MNV2+SpecM Dynamic) 하이브리드 조합: avg 95.64% — 4-model 서버 대비 Δ-0.84%p. RPi5 배포 확정: MNV2-Dynamic(14ms) + SpecM-Dynamic(21ms) = ~140ms | experiments/results/onnx_benchmark/quant_accuracy_20260319_163814.json | 재평가 완료 | SpecM-v3 재학습`
- `2026-03-19 | shield/phase4.2 | PTQ INT8 양자화 정확도 평가 완료(4모델 × 3variants × 4-DS). Dynamic INT8: 전 모델 Δ≤+0.17%p(무손실 확인). Static INT8: SpecM만 성공(Δ+0.01%p), MNV2 Δ-17.97%p(파국적), SpecG Δ-47.23%p, CLIP Δ-63.25%p 붕괴. RPi5 최적 배포 확정: MNV2-FP32(76.11%/14ms) + SpecM-Dynamic(65.54%/21ms). MNV2 Static은 cosine=0.69에서 F1 -18%p로 실용 불가 실증 | experiments/results/onnx_benchmark/quant_accuracy_20260319_145436.json | 4모델 정확도 평가 완료 | Phase 4.4 RPi5 실측 또는 논문 실험 섹션 작성`
- `2026-03-19 | shield/phase4.2 | PTQ INT8 양자화 완료(4모델). Dynamic(Gemm): 속도개선 <1.05×(Conv 미포함). Static(QDQ): SpecM+1.25×/cos=1.000(유효), MNV2+1.05×/cos=0.69(정확도 저하), SpecG/CLIP 정확도 붕괴(cos<0.2, FastViT attention 양자화 불안정). 최종 RPi5 배포 결정: MNV2-FP32(57ms)+SpecM-INT8(67ms)=124ms. AI-gen 탐지(SpecG)는 서버 전용 확정. 논문 기여: "FastViT 백본은 표준 PTQ로 edge 배포 불가 → 경량 binary AI-gen detector 별도 설계 필요"로 프레이밍 | weights/onnx_quant/, experiments/results/onnx_benchmark/quantization_*.json, experiments/run_quantization.py | 양자화+벤치마크 완료 | Phase 4.4 RPi5 실측 또는 논문 실험 섹션 작성`
- `2026-03-19 | shield/phase4.1 | ONNX 변환 + CPU 벤치마크 완료(4모델). MNV2(22.5MB, 14.0ms/56ms RPi5 ✓), SpecM(30MB, 20.6ms/82ms ✓), SpecG(141.5MB, 200ms/800ms ✗ OVER), CLIP(141.3MB, 198ms/791ms ✗ OVER). RPi5 예산(200ms) 내 배포 가능: MNV2+SpecM(138ms, 3-class+manip) 조합만 가능. AI-gen 탐지(SpecG)는 서버 전용. 다음: QAT/INT8 양자화로 SpecG RPi5 지연 개선 시도 | weights/onnx/, experiments/results/onnx_benchmark/, experiments/run_onnx_export.py | 4모델 ONNX+벤치마크 완료 | Phase 4.2 QAT 양자화`
- `2026-03-19 | shield/phase3.4+3.5 | Phase 3.5 전체 완료(3.5.4~3.5.6) + Phase 3.4 DAAC 재학습 완료. SpecM-v2 best manip_f1=0.827. ICWMV SpecM-v2 avg 96.48%(v1 96.40% 대비 +0.08%p). CKA 4-model avg=0.0855 vs 2-model 0.9241(ΔCKA=-0.8385 — specialist로 다양성 대폭 향상). DAAC-GBM 경량화(25-dim): base=99.01%(원본 86.13% 대비 +12.88%p), avg 4-DS=96.25% ≈ ICWMV. Label leakage 문제(specm/specg_avail flag) 발견 및 수정 — specialist 특징 제거 후 generalist 2개 기반으로 공정 비교 | experiments/results/specialist_eval/specialist_m_v2_*, experiments/results/icwmv/icwmv_4model_wspec1.0_20260319_123542.json, experiments/results/cka_diversity/cka_diversity_4model_20260319_124909.json, experiments/results/daac_retrain/daac_retrain_lightweight_20260319_125524.json | 4-DS 모두 재평가 완료 | Phase 3.5 AGENTS.md 업데이트 완료. 다음: 논문 실험 섹션 작성`
- `2026-03-19 | shield/phase3.5 | Specialist-M v2 학습 시작(OOD 강건화). +IMD2020 1710장(non-eval) + JPEG압축/GaussianNoise augmentation + WeightedRandomSampler. Ep.10 기준 manip_f1=0.825(v1 0.764 대비 +6.1%p, auth_recall=0.722). 완료 후 ICWMV 4-model 재평가 예정 | experiments/train_specialist_m_v2.py, weights/specialist_m_v2/ | 학습 진행중(PID:2253321) | SpecM-v2 완료 후 run_icwmv_consensus.py 재실행`
- `2026-03-19 | shield/phase3.5 | ICWMV 4-model 합의 평가 완료. w=1.0: avg macro_F1=96.40%(MNV2 95.81% 대비 +0.59%p). base+1.01, dsC+0.89, opensdi+0.87%p. aigenproxy −0.44%p(SpecM OOD 약점). "둘 다 틀림" fix rate 저조(base 2.1%) — manipulated→authentic 패턴은 binary specialist만으로 해결 어려움 확인 | experiments/run_icwmv_consensus.py, experiments/results/icwmv/ | 4-DS 평가 완료 | SpecM-v2 완료 후 재평가`
- `2026-03-19 | shield/phase3.5 | Specialist-M v1(CASIA2 only) + Specialist-G(MobileCLIP+PiD) 학습 완료. SpecM: 3-stream(RGB+SRM+DCT) 7.66M, best manip_f1=0.764(Ep.5 조기수렴). SpecG: 35.91M(0.10M trainable), best aigen_f1=0.981(Ep.19). 에러 Overlap 분석: avg Jaccard=0.3361, "둘 다 틀림" manipulated→authentic 지배 → Binary Specialist 설계 확정 | experiments/train_specialist_m.py, experiments/train_specialist_g.py, weights/specialist_m/, weights/specialist_g/, experiments/results/specialist_eval/ | v1 학습완료/4-DS eval 완료 | ICWMV 합의 평가`
- `2026-03-19 | shield/phase3.3 | 다기준 비교 완료. RPi5 권고: MNV2-only(22.5MiB, CPU 35.9ms, InDist=0.956, OOD≈0.669). GPU 권고: CLIP+MNV2(402.8MiB, CPU 159.7ms, InDist=0.953, OOD=0.731). ForMa 전면 제거 확정(CPU 1613ms 병목, Shapley=0.008). Track1/2/3-cascade 모두 RPi5 배포 불가. 핵심 발견: CLIP↔MNV2 중복(CKA=0.92) → 경량화 시 interaction 붕괴, SHIELD C2 기여 논거로 활용 | experiments/results/phase3_comparison/phase3_comparison_20260319_080903.json | 완료 | Phase 3.4 메타 분류기 재학습`
- `2026-03-19 | shield/phase3.2 | 경량 모델 Shapley+STII+CKA+PID 분석 완료(4157개, 15 subset, 10 seeds). MNV2 φ=+0.304 ≈ CLIP φ=+0.300 >> ForMa=Tiny=+0.008. CLIP↔MNV2 CKA=0.922 — in-dist에서는 두 모델이 거의 동일 특징 공간(Redundancy=0.60). ForMa/Tiny는 Unique≈0. 논문 핵심 발견: Phase1(무거운 freq↔fat 대체재, STII=-0.18)과 달리 경량 모델에서는 CLIP과 MNV2가 강한 중복(STII=-0.584) → 경량화 시 아키텍처 재설계 필요. OOD 강건성(Fair LOO +0.092)을 위해서는 MNV2 유지가 여전히 핵심 | experiments/results/shapley_phase3/shapley_phase3_20260319_074635.json | 완료 | Phase 3.3 다기준 비교`
- `2026-03-19 | shield/phase2.4+3.1 | MobileNetV2 dual-stream 15epoch 재학습 후 Track1 재평가. best val macro=0.806. 단독 eval은 base=0.944/dsC=0.979/opensdi=0.949/aigen=0.961(avg=0.958)까지 상승. combined 10-seed: T1-GBM=0.9547, T1-LR=0.9564, Triad-rule=0.9563, MobileNetV2 single=0.9557, MobileCLIP=0.9500. 이전과 달리 Track1 feature importance top-3가 모두 MobileNet score로 바뀌어 noise 축이 실제 주도 신호로 전환 | experiments/train_mobilenetv2_dualstream.py, experiments/results/backbone_eval/mobilenetv2_dualstream_summary_20260319_070725.json, experiments/run_phase3_tracks.py, experiments/results/phase3_tracks/phase3_tracks_combined_20260319_070740.json | 15epoch 재학습 + combined 10-seed 재평가 완료 | cross-dataset Track1 검증 또는 Track 3.3 다기준 비교`
- `2026-03-19 | shield/phase2.4+3.1 | MobileNetV2 dual-stream 구현 및 Track1/2/3 JSONL 12개 생성 완료. dual-stream 단독 eval: base=0.826/dsC=0.891/opensdi=0.789/aigen=0.866(avg=0.843), val macro=0.775. combined 10-seed: T1-GBM=0.9486, T1-LR=0.9512, T2-GBM=0.9448, T3-GBM=0.9469 vs MobileCLIP=0.9500. Track1 importance에서 `mobilenet_manip_s`가 top-5로 들어와 noise 축이 완전히 무의미하진 않지만, 전체 우세 신호는 여전히 CLIP score | experiments/train_mobilenetv2_dualstream.py, experiments/run_phase3_tracks.py, experiments/results/backbone_eval/mobilenetv2_dualstream_summary_20260319_064748.json, experiments/results/phase3_tracks/phase3_tracks_combined_20260319_065436.json | dual-stream 학습/eval + combined 10-seed 완료 | cross-dataset 기준 Track1 재검증 또는 ForMa manip_ratio 연속 특징 추가`
- `2026-03-19 | shield/phase2.7 | Backbone benchmark를 실제 Track1 모델 기준으로 갱신. H200/EPYC 기준: ForMa 16.8ms/1613ms, MobileCLIP-ft4 15.5ms/123.8ms, Tiny 5.8ms/2.5ms, MobileNetV2 dual-stream 18.9ms/35.9ms. params/ckpt: ForMa 37.3M/422.6MiB, CLIP 99.4M/380.3MiB, Tiny 0.0013M/0.011MiB, MNV2-DS 5.77M/22.5MiB | experiments/run_backbone_latency_benchmark.py, experiments/results/backbone_benchmark/backbone_benchmark_20260319_065356.json, experiments/results/backbone_benchmark/backbone_benchmark_20260319_065356.md | 4개 백본 GPU+CPU 재실측 완료 | Track 선택용 다기준 비교(3.3) 또는 cross-dataset Track1`
- `2026-03-19 | shield/phase3.1 | Track2(ForMa+CLIP) / Track3(Tiny+ForMa+CLIP) 앙상블 평가. combined 10-seed: T2-GBM=0.9462, T3-GBM=0.9494 vs MobileCLIP=0.9518(best). cross-dataset: 4/4 데이터셋 중 3/4에서 MobileCLIP 단독이 best. GBM feature importance: MobileCLIP scores가 ~95% → ForMa binary verdict 기여 거의 0. 핵심 원인: ForMa manip_ratio 연속값 미저장. 다음: manip_ratio feature 추가 후 재평가 또는 논문 전략 재검토 | experiments/run_phase3_tracks.py, experiments/results/phase3_tracks/ | combined/per_ds/cross-dataset 3개 프로토콜 완료 | ForMa manip_ratio 연속 특징 추가`
- `2026-03-19 | shield/phase2.1+2.5 | 사용자 제공 ForMa_weights.pth 연동 후 4개 데이터셋 재평가 완료. ForMa acc: base=0.349/dsC=0.332/opensdi=0.334/aigen=0.323, authentic recall 0.84~0.94, manip recall 0.07~0.15, ai_gen recall 0. run_backbone_eval.py에 repo-root weight fallback + CUDA mamba batch inference 추가 | experiments/run_backbone_eval.py, experiments/results/backbone_eval/backbone_eval_summary_20260319_055541.json | 4dataset × 3backbone JSONL/summary 생성 확인 | 2.7 latency/size 실측 및 Phase 3.1 Track JSONL 조합`
- `2026-03-19 | shield/phase2.2 | MobileCLIP-S2 last-4-block fine-tuning 완료(ft4, 20ep, lr=2e-5, batch=32, 33.28M trainable). val macro=0.806(ft0 0.790 대비 +1.6%p). 전체eval avg=0.953(base=0.942/dsC=0.974/opensdi=0.953/aigen=0.943). authentic recall ~93%, manip ~97%, aigen ~97%. 체크포인트: weights/mobileclip_forensics/mobileclip_s2_forensics_ft4.pth | experiments/finetune_mobileclip.py | 4dataset × finetuned JSONL 생성 확인 | MobileNetV2 dual-stream(2.4)`
- `2026-03-19 | shield/phase2.2 | MobileCLIP-S2 linear probe 파인튜닝 완료(30ep, lr=1e-3, batch=64). val macro_recall=0.790. 4데이터셋 전체 평균 acc=0.932(base=0.924/dsC=0.964/opensdi=0.923/aigen=0.918). zero-shot 34%→92%+ 대폭 개선. 체크포인트: weights/mobileclip_forensics/ft0.pth | experiments/finetune_mobileclip.py, experiments/results/backbone_eval/mobileclip_s2_finetuned_* | 4dataset × finetuned JSONL 생성 확인 | MobileNetV2 dual-stream(2.4) 또는 ForMa 가중치 확보`
- `2026-03-19 | shield/phase2.5 | 백본 단독 평가 완료(MobileCLIP-S2 + Tiny-LaDeDa, 4개 데이터셋). MobileCLIP zero-shot acc~34%(파인튜닝 필요). Tiny-LaDeDa ai_gen recall 73-86%, manip 0%(binary한계). ForMa 가중치 Google Drive 수동 다운로드 BLOCKED | experiments/results/backbone_eval/ | 4dataset × 2backbone JSONL | ForMa 가중치 확보 후 재평가`
- `2026-03-19 | shield/phase2 | AGENTS.md 3-Track 실험 계획 수립. Phase 2 백본 수급(ForMa/MobileCLIP-S2/Tiny-LaDeDa) Step 1 착수. ForMa 코드 패치(config 경로, backend="torch", debug print 제거) | AGENTS.md, ForMa-main/models/ | 수동 검증 | ForMa/MobileCLIP-S2/Tiny-LaDeDa 수급 및 단독 평가`
- `2026-03-19 | shield/phase1.6 | Cross-Dataset 검증(4개 데이터셋). Spatial Unique=0 4/4확정. freq 1위는 CASIA편향(1/4). fatformer aigenproxy에서 음수Shapley 발견 → 3-Track 전략으로 전환 | cross_dataset_validation_20260319_041001.json | 4/4 일관성 분석 | Phase 2 백본 수급`
- `2026-03-18 | shield/phase1.5 | 최적 조합 결정: freq+noise+fatformer(3개), F1=0.8478(97.8%), 경량화 후 130MB. Spatial 제거(Unique=0) 이론 정당화 완료 | experiments/results/shapley_phase1/optimal_combination_phase1.json | 수동 검증 | Phase 2.1 FatFormer→MobileCLIP 착수`
- `2026-03-18 | shield/phase1.4 | PID(Williams & Beer Imin) 완료. Spatial Unique=0, Freq Unique=0.2029(최고). noise↔fatformer 시너지 최강(+0.1093) | experiments/results/shapley_phase1/pid_phase1_20260318_162008.json | 수동 검증 | Phase 1.5 최적 조합 결정`
- `2026-03-18 | shield/phase1 | Model Shapley(16 subset) + STII(k=2) + CKA 계산 완료. Freq φ=+0.2690 최고, Freq↔FatFormer STII=-0.1823 최강 상호작용 | experiments/results/shapley_phase1/shapley_phase1_20260318_161543.json | 10 seed 반복 평균 | Phase 1.4 PID 분해 또는 Phase 2 경량화 설계`
- `2026-03-18 | docs/governance | AGENTS.md 후속연구(SHIELD) 기준으로 전면 재구성 + SHIELD_RESEARCH_PLAN.md 생성 | AGENTS.md, docs/research/SHIELD_RESEARCH_PLAN.md | 수동 검증 | Phase 1 Model Shapley 실험 설계`
- `2026-03-07 | docs | CLAUDE.md 전면 정합성 수정 | CLAUDE.md | 수동 검증 | 후속 연구 방향 설계`
- `2026-03-06 | paper | DAAC 논문 초안 완성 + 추론 속도 실측 | docs/research/MAIFS_PAPER_DRAFT2_20260306.md | benchmark 결과 확인 | 논문 제출`
- `2026-03-04 | experiments | DAAC 최종 실험(Protocol-P/M) 완료 | experiments/results/paper_final/ | 통계 검정 PASS | 논문 작성`
- `2026-03-03 | runtime | trust 이중반영 제거 + COBRA baseline 정렬 + case3 평가 추가 | src/consensus/cobra.py, src/meta/baselines.py | pytest 28 passed | 문서 동기화`

---

## 9. Completed Milestones (Archive)

> DAAC 1차 논문까지의 상세 이력. 일상 작업에서는 참조 불필요.

<details>
<summary>DAAC Phase 1~2 완료 이력 (2026-01 ~ 2026-03)</summary>

- Phase 1~4 runtime core 완료 (tools, agents, consensus, debate)
- Phase 5 Qwen vLLM 통합 완료
- Watermark → FatFormer 전면 교체 (2026-02-12)
- DAAC Phase 1 (Path B) 구현·검증 (43-dim meta features)
- CAT-Net frequency slot 통합
- Mesorch spatial backend 통합·A/B 검증
- Meta trainer GPU 경로 (torch/xgboost) 추가
- Specialist trust flow 수정 (합의 단계 단일 반영)
- ManagerAgent consensus/debate 경로 MAIFS 런타임 정렬
- DAAC Phase 2 Path A 실데이터 60회 반복 실험 완료
- 논문 초안 완성 (KIPS 2026)
- COBRA DRWA 다중신호 보강 + COBRABaseline 런타임 정렬

상세 Status Log는 git history 참조.

</details>

---

## 10. Golden Rules

### Immutable
- Verdict contract 유지: `authentic`, `manipulated`, `ai_generated`, `uncertain`
- `FATFORMER` 네이밍 사용 (WATERMARK 절대 재도입 금지)
- Graceful degradation 필수 (모델/체크포인트/API 부재 시 crash 금지)
- API 키·시크릿·절대경로 하드코딩 금지

### Do
- 규칙/동작 변경 시 문서 동시 갱신
- 모델 경로·임계값은 `configs/` 경유 설정화
- `src/` 내부는 상대 import
- confidence 값 `[0.0, 1.0]` 정규화 유지
- 동작 변경 시 테스트 추가/갱신

### Don't
- vendored 외부 모델 리포(`CAT-Net-main`, `MVSS-Net-master`, `Mesorch-main` 등) 직접 수정 금지
- 대용량 바이너리(체크포인트, 데이터셋) 커밋 금지
- fallback 동작을 hard failure로 대체 금지
- Token pruning을 포렌식 모델에 적용 금지 (R6)

---

## 11. Context Map

| 영역 | 파일 | 설명 |
|------|------|------|
| Runtime orchestration | `src/AGENTS.md` | maifs.py, 패키지 아키텍처 |
| Forensic tools | `src/tools/AGENTS.md` | CAT-Net, MVSS, FatFormer, Mesorch |
| Agent behavior | `src/agents/AGENTS.md` | specialist/manager, trust, debate |
| Meta learning / DAAC | `src/meta/AGENTS.md` | 43-dim features, trainer, router |
| Experiments | `experiments/AGENTS.md` | phase configs, run scripts, outputs |
| Config & thresholds | `configs/AGENTS.md` | settings, backend toggles, trust |
| Scripts & utils | `scripts/AGENTS.md` | CLI, calibration, evaluation |
| Tests | `tests/AGENTS.md` | pytest scope, skip policy |
| Research docs | `docs/research/` | SHIELD/DAAC 연구 계획, 이론 백서 |

---

## 12. Key References

| 참조 | 위치 |
|------|------|
| DAAC 논문 최종본 | `/data/jj812_files/DAAC_최종.pdf` |
| 딥리서치 Prompt 1 (Gemini) | `/data/jj812_files/DeepResearch_Prompt1_Gemini.pdf` |
| 딥리서치 Prompt 1 (GPT) | `/data/jj812_files/DeepResearch_Prompt1_GPT.pdf` |
| 딥리서치 Prompt 2 (Gemini) | `/data/jj812_files/DeepResearch_Prompt2_Gemini.pdf` |
| 딥리서치 Prompt 2 (GPT) | `/data/jj812_files/DeepResearch_Prompt2_GPT.pdf` |
| 딥리서치 Prompt 3 (Gemini) | `/data/jj812_files/DeepResearch_Prompt3_Gemini.pdf` |
| 딥리서치 Prompt 3 (GPT) | `/data/jj812_files/DeepResearch_Prompt3_GPT.pdf` |
| Phase 2 최종 결과 | `experiments/results/paper_final/` |
| 기존 모델 체크포인트 | `CLAUDE.md` §7.5 참조 |
| 딥리서치 Prompt 4 (이미지 위변조 탐지 전문가 앙상블) | `/data/jj812_files/이미지 위변조 탐지 전문가 앙상블 연구.pdf` |
| 딥리서치 Prompt 5 (CKA 표현 다양성 정규화) | `/data/jj812_files/CKA 기반 표현 다양성 정규화를 이용한 이미지 포렌식 이중 경량 모델 앙상블 설계 심층 조사.pdf` |
| 딥리서치 Prompt 6 (앙상블 다양성 학습 방법론) | `/data/jj812_files/이미지 포렌식 앙상블 다양성 학습 연구.pdf` |
| 딥리서치 Prompt 4 (Error-Driven Specialist Ensemble) | `/data/jj812_files/Error-Driven Specialist Ensemble for Image Forensics 최신 연구 동향과 적용 전략.pdf` |
