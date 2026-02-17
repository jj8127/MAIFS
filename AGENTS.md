# MAIFS Governance Guide

## Project Context & Operations

MAIFS is a multi-agent image forensic system. The runtime goal is to classify each input image as `authentic`, `manipulated`, `ai_generated`, or `uncertain` by combining four specialist agents with consensus and debate.

Primary stack:
- Python 3.10+
- PyTorch (model inference)
- NumPy/SciPy/scikit-learn (analysis and meta learning)
- Gradio (UI)
- pytest (tests)

Operational commands (repo root):
- Create/activate env:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `python -m pip install --upgrade pip setuptools wheel`
  - `pip install -r requirements.txt`
- Verify install:
  - `python main.py version`
- CLI analysis:
  - `python main.py analyze /path/to/image.jpg --algorithm drwa --device cuda`
- Web UI:
  - `python main.py server --host 0.0.0.0 --port 7860`
- Tool-level evaluation:
  - `python scripts/evaluate_tools.py --max-samples 20 --out outputs/tool_reeval_smoke_20.json`
- Phase 1 meta experiment:
  - `python experiments/run_phase1.py experiments/configs/phase1_mesorch_retrain.yaml`
- Phase 2 meta experiment:
  - `python experiments/run_phase2.py experiments/configs/phase2.yaml`
- Tests:
  - `.venv-qwen/bin/python -m pytest tests/ -v --tb=short`

## Current Status (Single-File Tracker)

This section is the project-status source of truth for day-to-day work.  
When a change is made, update this section first in the same commit cycle.

Snapshot date:
- `2026-02-16`

Current branch policy:
- Primary integration branch: `feat/catnet-integration`
- Keep `main` stable; merge only after verification.

Completed milestones:
- Phase 1~4 runtime core completed (tools, agents, consensus, debate).
- Phase 5 Qwen vLLM integration completed.
- Watermark legacy removed and replaced with FatFormer naming/flow.
- DAAC Phase 1 (Path B) implemented and validated (43-dim meta features).
- CAT-Net integrated into frequency/compression slot.
- Spatial Mesorch backend integrated and validated via A/B evaluation.
- Meta trainer supports optional GPU paths (torch/xgboost) with CPU fallback.

In progress:
- DAAC Phase 2 adaptive routing stabilization (Path B synthetic reproducibility verified; Path A collector pipeline integrated).
- Cross-checking experiment outputs and syncing operational documentation.
- Path A router-oracle redesign(oracle entropy/confidence + enhanced36 + router regressor 비교) 후속 안정화.
- Path A `enhanced36+ridge` seed10 검증 결과를 기반으로 유의성 부족 원인 진단 및 다음 실험 설계.
- Path A 게이트 기준을 seed-level McNemar count 단일 기준에서 sign-test/pooled McNemar 보완 기준으로 확장.
- Path A 게이트 운영 정책을 `active_gate_profile` 기반으로 고정 적용하고 후속 검증/문서 동기화.

Latest implemented items (code-level):
- `experiments/run_phase2.py` added.
- `experiments/configs/phase2.yaml` added.
- `src/meta/router.py` added.
- `src/meta/trainer.py` hardened for environments without torch (import-time safety/fallback).
- `src/meta/router.py` extended with entropy/confidence-aware oracle weighting and router regressor options (`mlp`/`ridge`/`gbrt`).
- `src/meta/collector.py` extended with `enhanced36` proxy feature profile.
- `experiments/run_phase2_patha.py` extended with subgroup metrics, protocol snapshot, and result path export.
- `experiments/run_phase2_patha_multiseed.py` extended to persist per-run `result_path`.
- `experiments/evaluate_phase2_gate.py` and `experiments/summarize_patha_subgroups.py` added for gate/subgroup post-analysis.
- `experiments/configs/phase2_patha_scale120_*` redesign configs added (oracle entropy v1, enhanced36+MLP/Ridge/GBRT).
- `src/meta/evaluate.py` now records McNemar discordant counts (`b`,`c`) in comparison outputs.
- `experiments/run_phase2_patha.py` now stores per-run McNemar components (`mcnemar_statistic`,`mcnemar_b`,`mcnemar_c`).
- `experiments/run_phase2_patha_multiseed.py` now aggregates sign-test and pooled-McNemar diagnostics.
- `experiments/evaluate_phase2_gate.py` now supports sign-test/pooled-McNemar based gate checks.
- `experiments/evaluate_phase2_gate_profiles.py` added to compare strict/sign-driven gate policies in one run.
- `tests/test_phase2_gate_stats.py` added for sign/pooled-McNemar gate regression coverage.
- `experiments/evaluate_phase2_gate_profiles.py` extended to read protocol gate profiles from config yaml.
- `experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml` now includes `protocol.gate_profiles` presets.
- `experiments/evaluate_phase2_gate_profiles.py` now supports `--profiles auto` with `protocol.active_gate_profile` resolution.
- `experiments/run_phase2_patha_multiseed.py` now auto-evaluates configured `protocol.active_gate_profile` after summary write.
- `experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml` now sets `protocol.active_gate_profile: scale120_conservative`.
- `experiments/analyze_patha_seed_drift.py` added to compare two seed-block summaries (aggregate/sign/model-pair/subgroup sign-flip).
- `src/meta/collector.py` now supports JSONL roundtrip load (`load_jsonl`) for fixed-dataset reruns.
- `experiments/run_phase2_patha.py` now supports `collector.precollected_jsonl` and decoupled `split.seed`.
- `experiments/run_phase2_patha_repeated.py` added for fixed-dataset repeated-split protocol with active-gate output.
- `src/meta/collector.py` now supports `stratified_kfold_split` for fixed-fold train/val/test protocol.
- `experiments/run_phase2_patha_repeated.py` now supports `--split-strategy kfold` (fold sweep on fixed dataset).
- `experiments/run_phase2_patha_repeated.py` now supports `--kfold-split-seeds` for multi-seed kfold coverage in one run.
- `experiments/select_patha_split_protocol.py` added to rank split protocol candidates and recommend a default.
- `experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml` now includes `repeated_split_defaults` (kfold25 standard).
- `experiments/tune_phase2_gate_profile.py` added for scale-aware gate threshold search using labeled pass/fail targets.
- `experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml` now includes `gate_profiles.scale120_tuned`.
- `experiments/configs/phase2_patha_scale120_feat_enhanced36_ridge.yaml` now includes `gate_profiles.scale120_conservative` (expanded independent-block 기준).
- `experiments/analyze_patha_kfold_variance.py` added for split_seed/test_fold variance decomposition across multiple fixed-kfold summaries.

Known blockers and risks:
- GitHub push may fail without local credential/SSH setup.
- Some tests are environment-dependent (checkpoint availability).
- Domain mismatch risk remains for specific datasets (notably spatial on IMD-like distributions).
- Optional dependencies (`yacs`, `pytorch_wavelets`, `timm`) are not guaranteed in clean environments, which can force fallback backends and reduce tool-level quality.
- Dependency drift (`opencv-python` vs `opencv-python-headless`, `numpy` ABI) can silently reintroduce CAT-Net runtime failures if optional requirements are not pinned.

Immediate next actions:
- Keep optional tool dependencies pinned via `requirements-optional-tools.txt` and verify clean-environment install.
- Preserve Path A `scale120` baseline seed10 summary as reference and keep future runs comparable.
- Keep `enhanced36+ridge` as current best candidate and investigate why McNemar significance remains 0/10 despite positive mean gain.
- Design next Path A protocol iteration for significance power (sample-size/fold design, test strategy, and threshold sanity check).
- Keep `scale120_conservative` as active production gate profile and treat `scale120_tuned` as analysis-only profile.
- Investigate seed-block drift (42~51 vs 52~61) before finalizing merge/rollout confidence.
- Expand fixed-kfold coverage (multiple shuffle seeds / repeated fold cycles) to stabilize fold-wise variance estimates.
- Keep `kfold25` as split-protocol default for variance diagnostics and monitor drift against live random reruns.
- Keep active gate threshold strictness (`min_f1_diff_mean=0.01`, `max_sign_test_pvalue=0.2`, `max_pooled_mcnemar_pvalue=0.1`) and investigate metric-target mismatch under current data scale.
- Analyze why kfold25 independent blocks(305~309, 310~314) are consistently negative while the initial kfold25 block(300~304) was positive.
- Preserve kfold75(300~314) summary as current variance baseline and avoid overfitting policy to single 25-run block.
- Shift next tuning axis from gate-threshold to model-side changes (router feature/oracle/regressor) because kfold75 mean is near zero.
- Sync docs for latest scale-up results and reproducibility scripts.
- Keep docs and thresholds aligned with runtime behavior before merge to `main`.

## Roadmap (Rolling Plan)

Now:
- Stabilize Phase 2 pipeline behavior and reproducibility.
- Keep CAT-Net + Mesorch default paths production-safe with fallback intact.
- Remove avoidable fallback activation in validation environments by aligning optional backend dependencies.
- Validate Path A collector results across multiple runs before merge readiness decision.

Next:
- Real-data validation track (Path A) with the same feature/consensus assumptions.
- DAAC comparison report: baseline COBRA vs Phase 1 vs Phase 2.

Later:
- Benchmark packaging for publication-style reporting.
- Robustness track (post-processing/adversarial stress tests).
- Merge strategy to `main` after checklist and reproducibility gates pass.

## Status Update Protocol

Required when any meaningful change is made:
- Update `Current Status` and `Roadmap` in this file.
- Add or adjust only facts that are verified by code/results.
- Keep updates concise and timestamped when relevant.

Minimum checklist for each update:
- What changed (module/script/config).
- Why it changed (bugfix, feature, calibration, refactor).
- What was validated (tests/experiment command).
- What remains (open risk, blocker, next action).

Entry format guideline:
- `YYYY-MM-DD | scope | change | validation | next`
- Example:
  - `2026-02-16 | src/meta | add phase2 router path | run_phase2 config smoke | compare with phase1 baseline`

## Status Log (Recent Verified Changes)

- `2026-02-12 | runtime/core | Watermark flow removed, FatFormer flow standardized | test and integration updates landed | keep enum naming consistent`
- `2026-02-12 | daac/phase1 | 43-dim meta pipeline implemented | phase1 runner + ablation/evaluation outputs | prepare adaptive routing`
- `2026-02-13 | tools/frequency | CAT-Net integrated into frequency/compression slot | tool re-evaluation outputs generated | continue threshold hardening`
- `2026-02-13 | tools/spatial | Mesorch backend integrated and A/B evaluated | 20/100 sample reports generated | maintain fallback and dataset checks`
- `2026-02-13 | meta/training | GPU-capable meta trainer path finalized | retrain config run completed | stabilize reproducibility`
- `2026-02-15 | daac/phase2 | adaptive routing runner/config/router module added | phase2 run output generated | compare phase2 vs phase1 baselines`
- `2026-02-16 | governance | AGENTS promoted to single-file status+roadmap tracker | section-level manual verification | keep this log updated per change`
- `2026-02-16 | daac/phase2 | phase2 rerun consistency verified (schema/type stable across runs) | run_phase2 twice with phase2.yaml | extend check to Path A real-data track`
- `2026-02-16 | tests/regression | full pytest regression rerun passed after environment dependency sync | .venv-qwen/bin/python -m pytest tests/ -v --tb=short (161 passed, 10 skipped) | monitor cobra zero-weight warning`
- `2026-02-16 | tools/eval | tool smoke output regenerated; fallback observed for CAT-Net/FatFormer/Mesorch without optional deps | scripts/evaluate_tools.py --max-samples 20 --out outputs/tool_reeval_smoke_20.json | install/lock yacs+pytorch_wavelets+timm and rerun`
- `2026-02-16 | tools/deps | optional backend deps installed and backend loading restored (CAT-Net/FatFormer/Mesorch) | scripts/evaluate_tools.py --max-samples 20 --out outputs/tool_reeval_smoke_20_after_all_optional_deps.json | calibrate CAT-Net thresholds due high uncertain rate`
- `2026-02-16 | daac/pathA-proxy | real-data proxy Phase1/Phase2 comparison executed (CASIA Au/Tp + BigGAN ai, n=180) | experiments/results/phase2_pathA/phase2_patha_proxy_results_20260216_095925.json | implement reusable collector pipeline for full Path A`
- `2026-02-16 | tools/env-fix | CAT-Net runtime dependency chain fixed (`jpegio`, `torch-dct`, headless OpenCV, numpy ABI) and frequency-slot uncertainty issue resolved | outputs/tool_reeval_smoke_20_after_catnet_env_fix.json (frequency f1=0.6667, uncertain=0) | keep optional deps pinned`
- `2026-02-16 | daac/pathA | collector module and dedicated Path A phase2 runner/config added | python experiments/run_phase2_patha.py experiments/configs/phase2_patha.yaml | scale sample plan and compare multi-seed stability`
- `2026-02-16 | tests/meta | collector utility unit tests added/passed | .venv-qwen/bin/python -m pytest tests/test_meta_collector.py -v --tb=short | include in regular regression scope`
- `2026-02-16 | tests/regression | full-suite regression rerun passed after collector/env updates | .venv-qwen/bin/python -m pytest tests/ -v --tb=short (164 passed, 10 skipped) | monitor warning debt in external backends`
- `2026-02-16 | daac/pathA-multiseed | pathA collector pipeline multi-seed pilot completed (5 runs) | experiments/results/phase2_patha/phase2_patha_multiseed_summary_20260216_101723.json | expand sample scale and rerun significance checks`
- `2026-02-16 | docs/sync | README + research docs synchronized to collector/pathA runtime and multi-seed findings | README.md + docs/research updates | keep docs aligned with subsequent phase2 reruns`
- `2026-02-16 | daac/pathA-scale120 | pathA scale-up multi-seed completed (120/class, 5 runs) and multiseed runner direct-exec import path fixed | .venv-qwen/bin/python experiments/run_phase2_patha_multiseed.py experiments/configs/phase2_patha_scale120.yaml --seeds 42,43,44,45,46 | extend seeds and tune router for significance`
- `2026-02-16 | daac/pathA-scale120-seed10 | pathA baseline seed expansion completed (120/class, 10 runs) | .venv-qwen/bin/python experiments/run_phase2_patha_multiseed.py experiments/configs/phase2_patha_scale120.yaml --seeds 42,43,44,45,46,47,48,49,50,51 | keep baseline and continue routing design`
- `2026-02-16 | daac/pathA-router-tuned | router regularization/size tuned pilot added and evaluated (5 runs) | .venv-qwen/bin/python experiments/run_phase2_patha_multiseed.py experiments/configs/phase2_patha_scale120_router_tuned.yaml --seeds 42,43,44,45,46 | tuned config underperformed; pivot tuning strategy`
- `2026-02-16 | daac/pathA-oracle-grid | oracle power/label_smoothing grid support added and pathA 5-seed grid executed | run_phase2_patha_multiseed with phase2_patha_scale120_oracle_p{15,20,25}_ls005.yaml | p15 looked best on 5 seeds but no significance`
- `2026-02-16 | daac/pathA-oracle-seed10 | best oracle candidate(`power=1.5`,`ls=0.05`) validated on seed10 and compared to baseline seed10 | experiments/results/phase2_patha_scale120_oracle_p15_ls005/summary_10seeds_42_51.json | candidate tied baseline; shift to oracle/feature redesign`
- `2026-02-16 | daac/pathA-router-redesign | router/oracle redesign code path implemented (entropy-aware oracle, enhanced36 features, router regressor variants, gate/subgroup scripts) | python -m compileall src/meta/router.py src/meta/collector.py experiments/run_phase2_patha.py experiments/run_phase2_patha_multiseed.py experiments/evaluate_phase2_gate.py experiments/summarize_patha_subgroups.py | execute sequential multiseed comparison`
- `2026-02-16 | daac/pathA-router-seq | sequential comparison completed (oracle_entropy_v1, enhanced36+MLP/Ridge/GBRT; 5 seeds each) and Ridge selected | experiments/results/phase2_patha_scale120_*/*summary_5seeds_42_46.json | validate selected candidate on seed10 protocol gate`
- `2026-02-16 | daac/pathA-ridge-seed10 | enhanced36+ridge seed10 completed with improved mean ΔF1 but significance gate fail(0/10) | summary_10seeds_42_51.json + gate_report_10seeds_20260216.json + subgroup_summary_10seeds_20260216.json | redesign significance-power plan before merge decision`
- `2026-02-16 | daac/pathA-stats-gate-v2 | multiseed 통계 판단 로직 강화(sign-test + pooled McNemar 지원, run-level b/c 저장) | compileall + gate_report_10seeds_sign_driven_20260216.json | re-run with new schema and compare gate policy`
- `2026-02-16 | tests/meta-gate | phase2 gate 통계 보강 로직 회귀 테스트 추가(7 passed) | .venv-qwen/bin/python -m pytest tests/test_phase2_gate_stats.py -v --tb=short | keep policy-change test coverage`
- `2026-02-16 | daac/pathA-gate-profiles | strict/sign-driven 게이트 프로파일 일괄 비교 스크립트 추가 및 실행 | gate_profiles_10seeds_20260216.json 생성 | integrate profile choice into protocol docs`
- `2026-02-16 | daac/pathA-gate-profiles-v2 | gate profile evaluator를 config 연동으로 확장하고 pooled_relaxed 프로파일 추가 | gate_profiles_10seeds_from_config_20260216.json + tests/test_phase2_gate_stats.py(9 passed) | apply same policy config to future candidates`
- `2026-02-16 | daac/pathA-ridge-seed10-statsv2 | enhanced36+ridge seed10 재실행으로 sign/pooled 통계 포함 summary 재생성 | summary_10seeds_42_51_statsv2_20260216.json + gate_profiles_10seeds_statsv2_all_20260216.json | strict/sign-driven fail, pooled_relaxed pass 상태에서 정책 확정 필요`
- `2026-02-16 | daac/pathA-active-gate | active gate profile 자동 적용 경로를 multiseed/evaluator에 반영하고 운영 프로파일을 pooled_relaxed로 고정 | .venv-qwen/bin/python -m pytest tests/test_phase2_gate_stats.py -q (12 passed) + evaluate_phase2_gate_profiles auto smoke | use active gate report output in next seed reruns`
- `2026-02-16 | daac/pathA-ridge-seed10-block2 | 신규 seed block(52~61) multiseed 실행 및 active gate 재검증 완료 | summary_10seeds_52_61_statsv2_20260216.json + summary_10seeds_52_61_statsv2_20260216_gate_pooled_relaxed.json + gate_profiles_10seeds_52_61_statsv2_all_20260216.json | pooled_relaxed 포함 모든 gate fail로 seed-block drift 원인 분석 필요`
- `2026-02-16 | daac/pathA-seed-drift-analysis | seed-block drift 분석 스크립트/테스트 추가 및 42~51 vs 52~61 비교 리포트 생성 | .venv-qwen/bin/python -m pytest tests/test_patha_seed_drift.py tests/test_phase2_gate_stats.py -q (15 passed) + seed_drift_42_51_vs_52_61_20260216.json | design split-variance control protocol (repeated split / fixed-fold)`
- `2026-02-16 | daac/pathA-repeated-split | fixed dataset(`precollected_jsonl`) 기반 repeated split 프로토콜 추가 및 10-run 실행(split_seed=300~309) | tests/test_meta_collector.py + tests/test_patha_repeated_split.py + tests/test_patha_seed_drift.py + tests/test_phase2_gate_stats.py (25 passed) + repeated_split_summary_10runs_300_309_20260216.json | active gate fail 유지, fixed-fold 변형으로 추가 분산 분해 필요`
- `2026-02-16 | daac/pathA-fixed-kfold | fixed dataset 기반 k-fold(5) 프로토콜 추가/실행 및 random-repeated 대비 비교 리포트 생성 | tests 25 passed + fixed_kfold_summary_5fold_20260216.json + seed_drift_repeated_random_vs_fixed_kfold_20260216.json | gate fail 유지, kfold 확장(coverage/repeats) 후 기준값 재설정 필요`
- `2026-02-16 | daac/pathA-fixed-kfold-coverage | kfold 다중 split seed(300,301) 확장으로 10-run coverage 실행 및 gate/profile 재검증 | tests 27 passed + fixed_kfold_summary_10runs_2seeds_20260216.json + fixed_kfold_gate_profiles_10runs_2seeds_20260216.json | n_runs 조건은 충족했지만 mean/pooled 조건 미달로 gate fail 유지`
- `2026-02-16 | daac/pathA-protocol-default | random25 vs kfold25 비교/선정 자동화 스크립트 추가 후 기본 split protocol을 kfold25로 결정 | tests 29 passed + split_protocol_selection_20260216.json(kfold25 우선) + fixed_kfold_summary_25runs_5seeds_20260216.json + repeated_random_summary_25runs_300_324_20260216.json | gate fail 유지(임계치 미달), threshold/데이터 규모 재설계 필요`
- `2026-02-16 | daac/pathA-gate-tuning-scale120 | pass/fail 타깃 기반 게이트 임계치 튜닝 수행 후 scale-aware 프로파일(scale120_tuned) 추가 | tests 32 passed + gate_profile_tuning_20260216.json + gate_scale120_tuned_{seed42_51,seed52_61,random25,kfold25}_20260216.json | tuned profile 분리력 확인, active 전환 전 독립 블록 재검증 필요`
- `2026-02-16 | daac/pathA-kfold25-block2 | 독립 fixed-kfold 블록(305~309) 25-run 추가 검증 및 gate/profile 재평가 | fixed_kfold_summary_25runs_5seeds_305_309_20260216.json + fixed_kfold_gate_profiles_25runs_5seeds_305_309_20260216.json + fixed_kfold_subgroup_summary_25runs_5seeds_305_309_20260216.json | 모든 gate fail로 독립 블록 재현성 추가 확인 필요`
- `2026-02-16 | daac/pathA-kfold25-block3 | 독립 fixed-kfold 블록(310~314) 25-run 추가 검증 및 seed-drift 비교 | fixed_kfold_summary_25runs_5seeds_310_314_20260216.json + fixed_kfold_gate_profiles_25runs_5seeds_310_314_20260216.json + seed_drift_fixed_kfold25_300_304_vs_310_314_20260216.json | 초기 양성 블록 대비 방향성 역전 원인 진단 필요`
- `2026-02-16 | daac/pathA-gate-policy-refresh | 확장 블록(최소 3개 kfold25 포함) 기준 게이트 재튜닝 후 보수형 프로파일(scale120_conservative) 추가 및 active 전환 | gate_profile_tuning_with_kfold_block3_20260216.json + gate_profile_tuning_conservative_seed42_only_20260216.json + config update | 운영 게이트는 보수형 유지, 후보 모델 개선 후 재튜닝`
- `2026-02-16 | daac/pathA-kfold75-scaleup | fixed-kfold split-seed 300~314(15 seeds x 5 folds = 75 runs) 확장 검증 실행 및 gate/profile 재평가 | fixed_kfold_summary_75runs_15seeds_300_314_20260216.json + fixed_kfold_gate_profiles_75runs_15seeds_300_314_20260216.json + fixed_kfold_subgroup_summary_75runs_15seeds_300_314_20260216.json + gate_profile_tuning_conservative_with_kfold75_20260216.json | ΔF1 mean=-0.0010, sign 34/34/7로 near-zero; 보수형 gate 재확인`
- `2026-02-16 | daac/pathA-kfold-variance-diagnostics | kfold 블록 변동성 진단 스크립트/테스트 추가 및 25x3+75 통합 리포트 생성 | tests/test_patha_kfold_variance.py + experiments/analyze_patha_kfold_variance.py + kfold_variance_diagnostics_25x3_plus_75_20260216.json | split_seed/test_fold 혼합 변동 유지로 모델-side 개선 우선`

## Golden Rules

Immutable:
- Preserve verdict contract: `authentic`, `manipulated`, `ai_generated`, `uncertain`.
- Keep Watermark legacy removed. Use `FATFORMER` naming consistently. Do not reintroduce `WATERMARK` enums/roles.
- Keep graceful degradation. Missing model/checkpoint/API must not crash the whole pipeline.
- Do not hardcode API keys, secrets, or user-specific absolute paths.

Do:
- Update rules and docs when behavior changes.
- Keep model path and thresholds configurable via `configs/settings.py` and `configs/tool_thresholds.json`.
- Use relative imports inside `src/` modules as documented in `CLAUDE.md`.
- Keep confidence values normalized to `[0.0, 1.0]`.
- Add or update tests for behavior changes.

Don't:
- Do not directly edit vendored external model repositories unless explicitly required (`CAT-Net-main`, `MVSS-Net-master`, `TruFor-main`, `OmniGuard-main`, `Mesorch-main`, `Integrated Submodules/FatFormer`).
- Do not commit raw datasets or new large binary checkpoints to this repo.
- Do not bypass fallback behavior with hard failures.

## Standards & References

Coding conventions:
- Follow existing Python style and type hints.
- Keep docstrings/comments in Korean for project-facing logic, as used in current codebase.
- Reuse existing dataclasses and enums (`ToolResult`, `AgentResponse`, `Verdict`, `AgentRole`).

Git strategy:
- Branch from feature branch for scoped work.
- Preferred commit format: Conventional Commits style.
  - `feat(scope): ...`
  - `fix(scope): ...`
  - `docs(scope): ...`
  - `refactor(scope): ...`
- Run relevant tests or smoke checks before push.

Maintenance policy:
- Use this `AGENTS.md` as the active status and roadmap tracker.
- If runtime behavior, thresholds, architecture, or experiment pipeline changes, update this file first, then synchronize:
  - `README.md`
  - `docs/research/MAIFS_TECHNICAL_THEORY.md`
  - `docs/research/DAAC_RESEARCH_PLAN.md`
- If rule/doc and code diverge, code has priority for immediate correctness, then this file and related docs must be updated in the same work cycle.

## Context Map (Action-Based Routing)

- **[Runtime orchestration and cross-module flow](./src/AGENTS.md)** — `src/maifs.py`, package-level architecture, integration changes.
- **[Forensic tools and model inference](./src/tools/AGENTS.md)** — CAT-Net, Noise, FatFormer, Spatial tool logic and thresholds.
- **[Agent behavior and role logic](./src/agents/AGENTS.md)** — specialist/manager response rules, trust handling, debate interfaces.
- **[Meta learning and DAAC internals](./src/meta/AGENTS.md)** — 43/47-dim features, trainer/evaluator/router behavior.
- **[Experiment runners and result generation](./experiments/AGENTS.md)** — phase configs, run scripts, output folder conventions.
- **[Global config and threshold policy](./configs/AGENTS.md)** — settings paths, backend toggles, trust and threshold sources.
- **[Utility and evaluation scripts](./scripts/AGENTS.md)** — CLI utilities, calibration/evaluation tooling.
- **[Test changes and environment-dependent cases](./tests/AGENTS.md)** — pytest scope, skip policy, regression coverage.
- **[Research and technical documentation updates](./docs/research/AGENTS.md)** — plan/theory docs synchronization policy.
