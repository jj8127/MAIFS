# MAIFS 진행 보고서

> 최종 업데이트: 2026-03-24
> 이 문서는 과거 시스템 통합 로그가 아니라, **현재 연구 진행상황의 요약판**입니다.
> 상세 SSOT는 [AGENTS.md](../AGENTS.md)를 우선 참고합니다.

---

## 1. Executive Summary

MAIFS는 현재 두 단계의 연구를 지나왔습니다.

- **DAAC**: 4-agent 서버형 합의 시스템
- **SHIELD / ICWMV**: DAAC의 합의 아이디어를 경량 백본과 엣지 환경으로 옮긴 후속 연구

2026-03-24 기준으로 가장 중요한 결론은 다음과 같습니다.

1. `MobileNetV2`는 강한 generalist지만, **오분류의 74.3%가 고확신(confidence > 0.6)** 인 `confident-but-wrong` 패턴을 보인다.
2. 이 때문에 confidence threshold 기반 cascade는 오분류 대부분에 개입하지 못한다.
3. 이를 교정하기 위해 제안한 **ICWMV(역신뢰도 가중 다수결)** 가 현재 가장 안정적인 fixed-rule fusion이다.
4. `EWCT`는 교정률을 높이는 보완 학습으로 유효하지만, learned scalar fusion(`HEMA`, action-gate, veto)은 아직 ICWMV를 안정적으로 넘지 못했다.

---

## 2. 트랙별 상태

| 트랙 | 목표 | 현재 상태 |
|------|------|----------|
| **DAAC** | 4-agent disagreement-aware consensus | 실험 완료, 논문화 단계 |
| **SHIELD** | on-device image forensics architecture | 핵심 실험 완료 |
| **ICWMV 논문선** | confident-but-wrong 분석 + fixed-rule consensus | 초안 v3.1 정리 중 |
| **EWCT 논문 보조선** | complementary specialist 학습과 한계 분석 | 보고서 정리 완료 |
| **RPi5 / Coral 실험선** | 실제 배포 경로 검증 | CPU/Coral 실측 완료 |

---

## 3. 지금까지 확정된 핵심 결과

### 3.1 DAAC

| 항목 | 결과 |
|------|------|
| 시스템 | 4-agent + meta-classifier |
| Protocol-P | `DAAC-GBM = 0.861 ± 0.016` |
| 비교 기준 | `COBRA = 0.266` |
| 핵심 통찰 | disagreement 자체가 탐지 신호 |

### 3.2 ICWMV 주력선

강한 MNV2, 4-DS LOO-CD 기준:

| 방법 | avg macro-F1 | 교정률 |
|------|-------------|--------|
| `MNV2 only` | 0.9581 | 0.0% |
| `Cascade + SpecM-v4` | 0.9612 | 19.3% |
| **`ICWMV + SpecM-v4`** | **0.9652** | **35.4%** |
| `ICWMV + EWCT-noTS` | 0.9575 | 49.2% |

해석:

- **F1 기준 주 운영점**: `ICWMV + SpecM-v4`
- **교정률 기준 운영점**: `ICWMV + EWCT-noTS`

### 3.3 백본 전이 / 강도 실험

| Generalist | 설정 | Backbone F1 | ICWMV F1 | ΔF1 | 교정률 |
|------------|------|-------------|----------|-----|--------|
| MNV2 | strong | 0.9581 | 0.9652 | +0.71%p | 35.4% |
| MNV2 | weak | 0.8414 | 0.8637 | +2.23%p | 49.8% |
| MobileCLIP | fine-tuned | 0.9532 | 0.9569 | +0.37%p | 19.7% |
| MobileCLIP | zero-shot | 0.3005 | 0.4320 | +13.14%p | 77.1% |
| MNV2 | no-finetuning | 0.3658 | 0.5556 | +18.98%p | 80.8% |

핵심 해석:

- ICWMV는 `MNV2` 전용 규칙이 아니다.
- 다만 이득 크기는 backbone family보다 **baseline 오류량**에 더 민감하다.

### 3.4 EWCT / learned fusion

현재까지의 결론:

- `w_max`는 이론적으로 제거 가능하며 실험도 동일 결과를 확인
- `confident-but-wrong` 분석은 논문에서 가장 clean한 supporting evidence
- learned scalar fusion(`HEMA`, action-gate, ICWMV-veto)은 아직 ICWMV를 안정적으로 넘지 못함

따라서 EWCT에서 논문으로 가장 강한 부분은:

1. `w_max` 제거
2. `confident-but-wrong` 정량화
3. in-domain 평가가 learned corrective fusion을 과대평가할 수 있다는 점

---

## 4. RPi5 / Coral 상태

| 경로 | avg macro-F1 | RPi5 평균 레이턴시 | 상태 |
|------|-------------|-------------------|------|
| ONNX INT8 CPU | 0.9535 | 88.3ms | 현재 주력 |
| Coral Edge TPU | 0.9308 | 63.4ms | 속도 우위, 정확도 gap 존재 |

추가 메모:

- Coral 경로는 `MNV2`에서 가속이 크고(`1.85x`), `SpecM`은 CPU fallback 영향이 남음
- 실제 연구 포지션에서는 Coral이 본문 메인보다 **부록 / 후속연구**에 더 적합함

---

## 5. 현재 문서 상태

| 문서 | 상태 | 비고 |
|------|------|------|
| `AGENTS.md` | 최신 | SSOT |
| `README.md` | 최신 | 프로젝트 허브 |
| `docs/research/PAPER_DRAFT_v1.md` | 최신 | KIPS 초안 |
| `docs/research/PAPER_DRAFT_NOTES.md` | 최신 | 논문 수치/테이블 메모 |
| `docs/research/EWCT_EXPERIMENT_REPORT.md` | 최신 | chronology + final takeaway |
| `deploy/README.md` | 최신 | 배포 요약 |
| `docs/research/RPi5_EXPERIMENT_GUIDE.md` | 최신 | 실측 절차 |

---

## 6. 지금 가장 중요한 다음 작업

1. `PAPER_DRAFT_v1.md`의 참고문헌과 표 캡션 마감
2. `ICWMV`를 메인 contribution으로, `EWCT`는 supporting study로 정리
3. 필요 시 backbone transfer / no-finetuning 결과를 appendix로 분리

---

## 7. 참고 문서

- [AGENTS.md](../AGENTS.md)
- [README.md](../README.md)
- [docs/research/PAPER_DRAFT_v1.md](research/PAPER_DRAFT_v1.md)
- [docs/research/PAPER_DRAFT_NOTES.md](research/PAPER_DRAFT_NOTES.md)
- [docs/research/EWCT_EXPERIMENT_REPORT.md](research/EWCT_EXPERIMENT_REPORT.md)
- [deploy/README.md](../deploy/README.md)

