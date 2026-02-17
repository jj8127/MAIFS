# Phase 2 Loss-Averse Runbook (2026-02-17)

## 1) 운영 기준

- 목적: 평균 개선 최대화가 아니라 downside-risk 제어
- 현재 운영 프로파일: `loss_averse_sparse_v2`
- 기준 config:
  - `experiments/configs/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec.yaml`

핵심 gate 조건:

- `min_runs=30`
- `min_f1_diff_mean=-0.001`
- `min_improvement_over_baseline=-0.005`
- `max_negative_rate=0.10`
- `max_downside_mean=0.005`
- `max_cvar_downside=0.02`
- `max_worst_case_loss=0.03`

## 2) 재현 명령

작업 디렉터리: `MAIFS/`

```bash
.venv-qwen/bin/python experiments/run_phase2_patha_repeated.py \
  experiments/configs/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec.yaml \
  --precollected-jsonl experiments/results/phase2_patha_scale120_feat_enhanced36_ridge/patha_agent_outputs_20260216_141946.jsonl \
  --split-strategy kfold \
  --k-folds 5 \
  --kfold-split-seeds 300,301,302,303,304,305 \
  --summary-out experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217.json
```

```bash
.venv-qwen/bin/python experiments/evaluate_phase2_gate_profiles.py \
  experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217.json \
  --config experiments/configs/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec.yaml \
  --profiles auto \
  --out experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217_gate_auto.json
```

## 3) 기준 아티팩트

- Summary:
  - `experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217.json`
- Gate(auto):
  - `experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217_gate_auto.json`
- Guard sensitivity:
  - `experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/guard_sensitivity_30runs_20260217.json`

## 4) 해석 규칙

- `gate_pass=true` and downside 제약 통과:
  - 운영 기준에서 채택 가능
- `sign_test_pvalue`는 참고 지표:
  - sparse routing에서는 0-diff run이 많아 `p=1.0`이 자주 발생
  - 이 경우 평균 개선 주장 대신 downside-control 관점으로 해석
- guard sensitivity 진단:
  - 음수 run이 `raw_phase2_gate_pass=true` + 높은 `route_rate_test`에서만 발생하면,
    oracle 교체보다 route-rate 제어/조건부 veto를 먼저 조정

## 5) 민감도 분석 명령

```bash
.venv-qwen/bin/python experiments/analyze_patha_guard_sensitivity.py \
  experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/fixed_kfold_summary_30runs_6seeds_risk52_oracle_lossaverse_guard_valselect_tunec_20260217.json \
  --out experiments/results/phase2_patha_scale120_feat_risk52_oracle_lossaverse_guard_valselect_tunec/guard_sensitivity_30runs_20260217.json
```

## 6) 다음 실험 우선순위

1. 외부 홀드아웃(시간축/생성기 축)에서 `loss_averse_sparse_v2` 유지 여부 검증
2. `min_phase2_val_gain`과 `max_route_rate`의 subtype별 민감도 분석
3. worst-case run의 공통 패턴(특정 fold/sub_type) 역추적
