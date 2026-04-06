# MAIFS

`MAIFS`는 두 편의 연구를 다시 실행하고 검토하는 데 필요한 코드, 설정, 경량 결과 캐시만 남긴 공개용 연구 번들입니다. raw image dataset, 대형 weight, 외부 서브모듈 전체는 제외했지만, 논문 수치와 Raspberry Pi 5 측정을 추적할 수 있도록 `experiments/`, `src/`, `configs/`, `deploy/`, `Server_Reproduction/`, `Raspberry_pi5_Experiment/`를 복구했습니다.

## 연구 축

- `DAAC`
  - 서버 환경의 4-agent disagreement-aware consensus 연구
  - 최종 논문: `docs/research/papers/DAAC_final_20260318.pdf`
  - 원본 논문 재현 코드: `experiments/`, `src/meta/`, `src/tools/`
  - 경량 공개 재현 번들: `Server_Reproduction/DAAC/`
- `PAR`
  - 경량 2단계 결합/수정 기반 이미지 위변조 탐지 연구
  - 최종 논문: `docs/research/papers/PAR_final_20260406.pdf`
  - 저장소 내부 구현/결과물의 역사적 이름은 `ARV`로 유지
  - 서버 재현 번들: `Server_Reproduction/ARV/`
  - 온디바이스 측정: `Raspberry_pi5_Experiment/`

## 현재 포함된 핵심 자산

- `experiments/`
  - DAAC paper-style cache rerun, cross-dataset validation, PAR/ARV fusion 실험, MobileCLIP 비교, RPi export 관련 스크립트
- `experiments/results/`
  - DAAC `dsA/dsB/dsC/dsD/OpenSDI/aigenproxy` Path-A 캐시
  - PAR/ARV server rerun용 backbone/specm/veto/repeat cache
  - paper-style summary JSON, repeated split summary, 6-set aggregate cache
- `src/`, `configs/`, `deploy/`, `requirements*.txt`
  - 원 논문 스크립트가 직접 참조하는 연구 코어와 실행 환경
- `datasets/`
  - raw image 대신 manifest, split provenance, dataset list만 포함
- `Server_Reproduction/DAAC/`
  - 경량 DAAC 공개 재현 엔트리포인트
- `Server_Reproduction/ARV/`
  - PAR 공개 재현 엔트리포인트
  - alpha-aware rerun과 weighting sweep 포함
- `Raspberry_pi5_Experiment/ICWMV/`
  - 1단계 경량 결합 지연 측정
- `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/`
  - fixed-input 종단간 측정과 active keep/revert probe workflow
- `docs/research/papers/`
  - 최종 PDF, 확장 초안, 그림, 공유 결과 archive
- `docs/research/EXPERIMENT_REPRO_RUNBOOK.md`
  - 어떤 실험이 cache-only로 바로 되는지, 무엇이 외부 자산이 필요한지 정리한 실행 가이드

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional-tools.txt
```

### DAAC cache-based rerun

```bash
python3 experiments/run_paper_final.py experiments/configs/paper_final_dsA.yaml
python3 experiments/run_agent_eval.py
python3 experiments/run_cross_dataset_validation.py
```

### PAR/ARV server rerun

```bash
cd Server_Reproduction/ARV
pip install -r requirements_arv.txt
python3 run_arv_experiment.py
python3 run_icwmv_weighting_sweep.py
python3 run_arv_generalist.py
python3 run_arv_strong_backbone_repeats.py
python3 run_arv_support_analyses.py
```

### Raspberry Pi 5

```bash
cd Raspberry_pi5_Experiment/ICWMV
bash run_rpi5_latency_benchmark.sh all /path/to/test_image.jpg
```

```bash
cd Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5
bash run_arv_e2e_benchmark.sh all
bash run_arv_active_workflow.sh cpu
```

## 중요한 제한 사항

- raw dataset과 대용량 dataset zip은 포함하지 않습니다.
- 대형 학습 weight와 외부 서브모듈 전체는 포함하지 않습니다.
- `scripts/prepare_sota_datasets.py`와 `datasets/` manifest는 포함했지만 실제 이미지 원본은 사용자가 별도로 준비해야 합니다.
- `DAAC`의 원래 main base-set table이 사용한 `phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl` cache는 원본 `MAIFS copy` 아카이브에도 없었습니다.
  - 따라서 `experiments/results/paper_final/`에는 당시 결과 snapshot을 보존하되,
  - 실제 기본 rerun 엔트리포인트는 현재 보존된 `paper_final_dsA.yaml` 등 공개 cache 기준으로 안내합니다.
- 최신 짧은 논문 이름은 `PAR`이지만, 저장소 내부 폴더/스크립트/결과물은 기존 실험명 `ARV`를 유지합니다.
