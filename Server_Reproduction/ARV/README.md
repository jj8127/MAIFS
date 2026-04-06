# Server Reproduction: ARV

이 디렉토리는 ARV의 서버 전용 정확도 재현 자산을 모아 둔 self-contained 번들이다.

## 포함 범위

- `run_arv_experiment.py`
  - 메인 ARV/PAR 재현 실험
  - `alpha=1.0/1.5` 가중 방식 비교까지 지원
- `run_icwmv_weighting_sweep.py`
  - stage-1 weighting sweep
- `run_arv_generalist.py`
  - MobileCLIP 계열 일반화 비교 실험
- `run_arv_backbone_transfer.py`
  - 다른 백본으로의 전이 비교 실험
- `run_arv_strong_backbone_repeats.py`
  - strong backbone 반복 실험
- `run_arv_support_analyses.py`
  - support analysis와 paper helper 통계
- `prepare_server_data.sh`
  - full tree에서 현재 경량 서버 번들을 다시 만드는 보조 스크립트
- `data/experiments/...`
  - MNV2, MobileCLIP, SpecM, veto rerun에 필요한 최소 결과 캐시와 보조 스크립트

## 포함하지 않는 것

- raw image dataset 전체
- 대형 학습 weight
- full submodule checkout

온디바이스 시간 측정은 `Raspberry_pi5_Experiment/` 아래에서 수행한다.

## 빠른 시작

```bash
cd Server_Reproduction/ARV
pip install -r requirements_arv.txt
python3 run_arv_experiment.py
```

## 현재 번들에서 바로 되는 것

- `python3 run_arv_experiment.py`
- `python3 run_icwmv_weighting_sweep.py`
- `python3 run_arv_generalist.py`
- `python3 run_arv_backbone_transfer.py --backbones mnv2_strong mnv2_weak mnv2_nofinetune`
- `python3 run_arv_strong_backbone_repeats.py`
- `python3 run_arv_support_analyses.py`

## 추가 자산이 필요한 것

- raw 이미지에서 prediction cache를 처음부터 다시 만들고 싶다면 실제 dataset과 weight가 더 필요하다.
- 그 경우 root `experiments/`, `datasets/`, `scripts/prepare_sota_datasets.py`, 그리고 외부 backend checkout을 함께 사용해야 한다.
