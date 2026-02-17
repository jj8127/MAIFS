# MAIFS 진행상황 정리 (2026-02-17)

## 1) 목적
- Phase2 Path A 신뢰성/재현성 이슈(P0/P1)를 코드 레벨에서 정리하고,
- 성능 개선 후보(FatFormer threshold, evidence_2ch)를 동일 프로토콜로 검증했습니다.

## 2) 구현 완료 사항

### A. 신뢰성/구조 정비
- `configs/trust.py` 신설
  - Trust SSOT 분리 (`resolve_trust`, deprecated key 처리 포함)
- 런타임 trust 참조 통일
  - `src/maifs.py`
  - `src/agents/manager_agent.py`
  - `src/meta/baselines.py`
- 수집 파이프라인 assert 제거
  - `src/meta/collector.py`에서 `assert` -> `RuntimeError` 전환

### B. Fallback/증거 체계 정비
- CAT-Net fallback cap + 원본 보존
  - `src/tools/catnet_tool.py`
  - fallback 시 `UNCERTAIN`, confidence cap(<=0.35), `fallback_raw_*` evidence 보존
- evidence 스키마 표준화
  - `src/meta/collector.py`에 `EVIDENCE_SCHEMA`, numeric key 처리 추가
- `evidence_2ch` feature 구현 및 정합성 수정
  - `src/meta/collector.py`
  - 중첩 evidence를 flat-merge 후 `(value, is_present)` 2채널(총 14-dim)로 반영

### C. Split/Seed 분리 및 실험 재현성
- split index 정합성 확보
  - `src/meta/collector.py`: `stratified_split/stratified_kfold_split(return_indices=True)` 지원
  - `experiments/run_phase2_patha.py`: samples/records 인덱스 동기화
- seed decoupling 반영
  - `experiments/run_phase2_patha.py`: split seed legacy fallback 경고 + 메타 기록
  - `experiments/run_phase2_patha_multiseed.py`: split/router seed 루프 외 고정
- Stage 2 config 반영
  - `experiments/configs/phase2_patha*.yaml` 전반 `split.seed: 300` 명시

### D. Threshold/실험 구성
- FatFormer threshold 분리
  - `configs/tool_thresholds.json`: `ai_threshold=0.5`, `auth_threshold=0.1`
- evidence_2ch 실험 config 추가
  - `experiments/configs/phase2_patha_scale120_feat_evidence51_oracle_lossaverse_guard.yaml`

## 3) 테스트 결과
- 실행: `.venv-qwen/bin/pytest tests/ -q --tb=short`
- 결과: `213 passed, 10 skipped, 0 failed`

## 4) 성능 검증 (live recollect + 동일 프로토콜 A/B)

### 실험 조건
- 공통 config 기반: `phase2_patha_scale120_feat_evidence51_oracle_lossaverse_guard.yaml`
- 공통 조건: `split.seed=300`, 동일 router/guard/protocol
- A (old): FatFormer `ai=1e-5`, `auth=1e-5`
- B (new): FatFormer `ai=0.5`, `auth=0.1`
- 모두 precollected 미사용 (live recollect)

### 결과 파일
- A: `experiments/results/ab_live_evidence51_old/phase2_patha_results_20260217_112910.json`
- B: `experiments/results/ab_live_evidence51_new/phase2_patha_results_20260217_113029.json`

### 핵심 지표
- `phase2_vs_phase1_best.f1_diff`
  - A: `0.0`
  - B: `0.0`
- `phase2_best_f1`
  - A: `0.7513375928523193`
  - B: `0.7671265615109354` (+0.01579)
- Baseline (`macro_f1`)
  - Majority: A `0.3790849673` -> B `0.3326129464`
  - COBRA: A `0.3460606061` -> B `0.3118787080`

### FatFormer verdict 분포 (agent outputs)
- A: `authentic=195`, `ai_generated=165`, `uncertain=0`, `manipulated=0`
- B: `authentic=319`, `ai_generated=34`, `uncertain=7`, `manipulated=0`
- 해석
  - threshold 분리로 UNCERTAIN 분기는 실제 활성화됨
  - 다만 현재 조건에서 최종 `f1_diff` 개선은 확인되지 않음

## 5) 남은 과제
- FatFormer calibration 재설계
  - threshold 단일 고정값보다 ROC/PR 기반 재보정 필요
- evidence_2ch 다회 반복 실험
  - single-run이 아닌 multiseed/kfold 통계 검증 필요
- gate profile과 문서 동기화
  - 실운영 profile 기준을 문서/설정 모두 일치시킬 것

