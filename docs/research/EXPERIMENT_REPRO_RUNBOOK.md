# Experiment Repro Runbook

이 문서는 공개 `MAIFS` 번들에서 어떤 실험이 바로 다시 실행되는지, 어떤 실험은 외부 데이터나 서브모듈이 더 필요한지 정리한 실행 가이드입니다.

## 1. DAAC

### 바로 되는 것

- paper-style cache rerun
  - `python3 experiments/run_paper_final.py experiments/configs/paper_final_dsA.yaml`
  - `python3 experiments/run_paper_final.py experiments/configs/paper_final_dsB.yaml`
  - `python3 experiments/run_paper_final.py experiments/configs/paper_final_dsC.yaml`
- standalone agent / ensemble 비교
  - `python3 experiments/run_agent_eval.py`
- cross-dataset validation
  - `python3 experiments/run_cross_dataset_validation.py`
- 6-set aggregate cache 확인
  - `experiments/results/phase2_patha_case3_multidata/`
- 경량 공개 재현
  - `cd Server_Reproduction/DAAC && python3 run_daac_retrain_lightweight.py`

### 외부 자산이 더 필요한 것

- 원래 base 1500장 main table을 cache 없이 다시 만들기
- raw image에서 Path-A agent output JSONL을 다시 수집하기
- full 4-agent backend를 처음부터 다시 구동하기

### 필요한 외부 자산

- dataset image tree
  - `datasets/CASIA2_subset/`
  - `datasets/GenImage_subset/`
  - `datasets/IMD2020_subset/`
  - `datasets/OpenSDID_subset/`
  - `datasets/AI-GenBench_proxy/`
- backend source / checkpoints
  - `external/CAT-Net-main`
  - `external/MVSS-Net-master`
  - `external/Mesorch-main`
  - `Integrated Submodules/FatFormer`

### 공개본의 알려진 공백

- 원래 `paper_final.yaml`이 가리키던 base cache
  - `experiments/results/phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl`
  - 이 파일은 원본 `MAIFS copy`에도 없어서 공개본에 복구할 수 없었습니다.
- 그래서 공개본 기본 엔트리포인트는 실제로 남아 있는 `dsA` cache를 가리키도록 바꾸었습니다.
- 당시 결과 snapshot 자체는 `experiments/results/paper_final/`에 보존했습니다.

## 2. PAR / ARV

### 바로 되는 것

- 메인 PAR server rerun
  - `cd Server_Reproduction/ARV && python3 run_arv_experiment.py`
- stage-1 weighting sweep
  - `cd Server_Reproduction/ARV && python3 run_icwmv_weighting_sweep.py`
- MobileCLIP generalist 비교
  - `cd Server_Reproduction/ARV && python3 run_arv_generalist.py`
- backbone transfer
  - `cd Server_Reproduction/ARV && python3 run_arv_backbone_transfer.py`
- strong-backbone repeat
  - `cd Server_Reproduction/ARV && python3 run_arv_strong_backbone_repeats.py`
- support analysis
  - `cd Server_Reproduction/ARV && python3 run_arv_support_analyses.py`

### 이 번들에서 같이 제공하는 cache

- `experiments/results/backbone_eval/`
- `experiments/results/specm_eval/`
- `experiments/results/specm_complementary_eval/`
- `experiments/results/hema_icwmv_veto/`
- `experiments/results/repeat_arv/`
- `experiments/results/paper_support/`

### 외부 자산이 더 필요한 것

- raw image에서 backbone/specm JSONL cache를 처음부터 다시 만들기
- 새로운 dataset으로 stage-1/stage-2를 처음부터 다시 학습하기
- 대형 checkpoint export를 새로 생성하기

## 3. Raspberry Pi 5

### 바로 되는 것

- 1단계 ICWMV latency
  - `cd Raspberry_pi5_Experiment/ICWMV && bash run_rpi5_latency_benchmark.sh all /path/to/test_image.jpg`
- ARV fixed-input end-to-end latency
  - `cd Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5 && bash run_arv_e2e_benchmark.sh all`
- ARV active keep/revert probe workflow
  - `cd Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5 && bash run_arv_active_workflow.sh cpu`

### 포함 자산

- fixed benchmark input
  - `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/assets/benchmark_input.png`
- active probe set
  - `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/assets/active_probe_set/`
- stage-2 runtime
  - `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/common/scripts/arv_stage2_runtime.py`
- active discovery / active latency scripts
  - `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/common/scripts/discover_arv_active_cases.py`
  - `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/common/scripts/benchmark_arv_active_cases.py`

## 4. Dataset Provenance

raw image는 제외했지만 아래 provenance 파일은 포함했습니다.

- `datasets/CASIA2_subset/au_list.txt`
- `datasets/CASIA2_subset/tp_list.txt`
- `datasets/OpenSDID_subset/manifest.jsonl`
- `datasets/AI-GenBench_fakepart_subset/manifest.jsonl`
- `datasets/AI-GenBench_proxy/meta.json`

OpenSDI / AI-GenBench subset 재생성은 아래 스크립트를 사용합니다.

- `python3 scripts/prepare_sota_datasets.py`
