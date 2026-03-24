# 논문 초안 준비 노트

> 작성: 2026-03-23
> 최종 수정: 2026-03-24
> 대상 초안: [docs/research/PAPER_DRAFT_v1.md](PAPER_DRAFT_v1.md)
> 현재 상태: **draft v3.1 기준 메모 정리 완료**

---

## 1. 현재 논문 프레이밍

현재 논문은 아래 흐름으로 가는 것이 가장 안정적입니다.

1. **DAAC 동기**
   - DAAC의 4-agent + 메타 분류기 스택은 서버에서는 강력하지만 엣지에는 과중함
2. **경량 백본 선택**
   - 여러 경량 백본을 비교한 뒤 `MNV2`를 주 generalist로 선택
3. **문제 분석**
   - `MNV2`는 `confident-but-wrong` 오류가 많아 naive cascade가 구조적으로 불리함
4. **제안**
   - 이를 해결하는 fixed-rule consensus로 **ICWMV(역신뢰도 가중 다수결)** 제안
5. **보조 실험**
   - `EWCT`는 specialist의 교정률 operating point를 확장하는 supporting study

즉, 현재 paper-ready contribution은 **`confident-but-wrong + ICWMV + EWCT의 w_max 제거`** 입니다.

---

## 2. 지금 주장해야 할 것 / 낮춰야 할 것

### 2.1 메인 claim

1. `MNV2`의 오분류 중 `74.3%`가 `conf > 0.6`인 고확신 오류다.
2. 이 때문에 confidence threshold 기반 cascade는 오분류 대부분에 개입하지 못한다.
3. `ICWMV + SpecM-v4`는 strong MNV2 기준 4-DS LOO-CD에서 `0.9652`, `+0.71%p`, 교정률 `35.4%`를 달성한다.
4. ICWMV는 `MNV2` 하나의 체크포인트 전용 트릭이 아니라, backbone strength가 달라져도 일관된 이득을 준다.
5. EWCT에서 `w_max`는 제거 가능한 하이퍼파라미터다.

### 2.2 보조 claim

- `EWCT-noTS`는 `ICWMV + v4`보다 F1은 낮지만 교정률은 더 높다.
- weak / zero-shot / no-finetuning operating point에서 ICWMV 이득이 더 커진다.
- Coral 경로는 속도는 좋지만 정확도 기준으로는 아직 ONNX보다 불리하다.

### 2.3 논문 본문에서 낮춰야 할 것

- `HEMA`를 주 방법론처럼 쓰지 않기
- `Cross 3-dim`을 domain-invariant 증거처럼 과장하지 않기
- learned fusion이 ICWMV보다 낫다는 식의 표현은 피하기

---

## 3. 핵심 수치

### 3.1 Confident-but-Wrong

| 지표 | 값 |
|------|----|
| binary(auth/manip) 오분류 수 | 144 |
| 오분류 평균 confidence | **0.7538** |
| 정분류 평균 confidence | 0.9502 |
| 오분류 중 `conf > 0.6` | **74.3%** (107/144) |
| 오분류 중 `conf > 0.8` | **46.5%** (67/144) |

### 3.2 메인 결과

강한 MNV2, 4-DS LOO-CD 기준:

| Method | avg F1 | ΔF1 | corr rate |
|--------|--------|-----|-----------|
| MNV2 only | 0.9581 | — | 0.0% |
| Cascade `τ=0.6` + v4 | 0.9612 | +0.31%p | 19.3% |
| **ICWMV + v4** | **0.9652** | **+0.71%p** | **35.4%** |
| ICWMV + EWCT-noTS | 0.9575 | -0.06%p | **49.2%** |

### 3.3 백본 전이 / 강도 실험

| Generalist | 설정 | Backbone F1 | ICWMV F1 | ΔF1 | corr rate |
|------------|------|-------------|----------|-----|-----------|
| MNV2 | strong | 0.9581 | 0.9652 | +0.71%p | 35.4% |
| MNV2 | weak | 0.8414 | 0.8637 | +2.23%p | 49.8% |
| MobileCLIP | fine-tuned | 0.9532 | 0.9569 | +0.37%p | 19.7% |
| MobileCLIP | zero-shot | 0.3005 | 0.4320 | +13.14%p | 77.1% |
| MNV2 | no-finetuning | 0.3658 | 0.5556 | +18.98%p | 80.8% |

해석:

- `ICWMV`는 특정 MNV2 체크포인트에만 맞춘 규칙이 아니다.
- gain은 backbone family보다 **backbone strength와 baseline error volume**에 더 민감하다.

### 3.4 EWCT

| SpecM variant | standalone manip-F1 | ICWMV corr rate | ICWMV F1 |
|---------------|---------------------|-----------------|----------|
| v4 baseline | 0.7438 | 35.4% | **0.9652** |
| comp_g1 | — | 39.4% | 0.9563 |
| comp_noTS | **0.7632** | **49.2%** | 0.9575 |

### 3.5 RPi5 / Coral

| 경로 | avg macro-F1 | 평균 레이턴시 |
|------|-------------|--------------|
| ONNX CPU | 0.9535 | 88.3ms |
| Coral Edge TPU | 0.9308 | 63.4ms |

---

## 4. 논문에서 권장하는 메시지

### 한 줄 요약

> `MNV2`의 confident-but-wrong 오류 때문에 threshold cascade는 구조적으로 한계가 있고, 이를 보완하기 위해 제안한 `ICWMV`가 fixed-rule edge consensus로 가장 안정적인 개선을 제공한다.

### 초록 / 서론용 핵심 문장

- `MNV2`의 binary 오분류 중 `74.3%`가 `confidence > 0.6`이다.
- 따라서 low-confidence sample에만 개입하는 cascade는 오분류 대부분을 놓친다.
- `ICWMV + SpecM-v4`는 4-DS LOO-CD에서 `+0.71%p`, 교정률 `35.4%`를 달성한다.
- `EWCT`는 교정률을 `49.2%`까지 확장하는 보조 operating point를 제공한다.

### 결론용 핵심 문장

- ICWMV는 strong backbone뿐 아니라 weak/zero-shot/no-ft operating point에서도 일관된 이득을 보인다.
- EWCT에서 `w_max`는 이론·실험 모두 제거 가능하다.
- learned scalar fusion은 아직 ICWMV를 안정적으로 넘지 못했으므로, 현재 가장 신뢰할 수 있는 합의기는 ICWMV다.

---

## 5. 표 구성 메모

현재 draft v3.1 축약판 기준으로는 아래 구성이 적절합니다.

### 본문에 남길 것

1. `Table 1` confident-but-wrong 분포
2. `Table 2` 메인 성능 비교 (`MNV2 / Cascade / ICWMV`)
3. `Table 3` EWCT 효과
4. `Table 4` backbone family + strength ladder
5. `Table 5~6` 배포 실측 요약

### appendix 또는 생략 후보

- HEMA 상세 비교
- Cross 3-dim feature ablation
- 과도한 learned fusion 복구 시도 로그

---

## 6. 현재 제외하는 주장

- “HEMA가 ICWMV보다 낫다”
- “Cross 3-dim이 domain-invariant signal임이 증명됐다”
- “Coral 경로가 현재 주력 배포선이다”

이 세 가지는 현재 증거 수준상 본문 contribution으로 밀지 않는 편이 안전합니다.

---

## 7. 남은 체크리스트

- [ ] 참고문헌 `[10]` DAAC 서지 정보 최종 입력
- [ ] 표 캡션 스타일 통일
- [ ] no-finetuning 결과를 본문 표에 둘지 appendix로 돌릴지 최종 결정
- [ ] Coral 단락을 본문보다 부록으로 낮출지 결정

---

## 8. 결과 파일 메모

| 결과 | 파일 |
|------|------|
| strong MNV2 LOO-CD | `experiments/results/fuser_loo_cd/fuser_loo_cd_20260323_022122.json` |
| weak MNV2 LOO-CD | `experiments/results/fuser_loo_cd/fuser_loo_cd_20260323_032450.json` |
| ICWMV + SpecM 비교 | `experiments/results/icwmv_specm_compare/icwmv_specm_20260323_032558.json` |
| backbone transfer / no-ft | `experiments/results/icwmv_backbone_transfer/icwmv_backbone_transfer_20260323_142828.json` |
| EWCT report | `docs/research/EWCT_EXPERIMENT_REPORT.md` |
| 논문 초안 | `docs/research/PAPER_DRAFT_v1.md` |

