# Server Reproduction: ARV

이 디렉토리는 ARV의 서버 전용 정확도 재현 자산을 모아 둔 self-contained 번들이다.

## 포함 범위

- `run_arv_experiment.py`
  - 메인 ARV 재현 실험
- `run_arv_generalist.py`
  - MobileCLIP 계열 일반화 비교 실험
- `run_arv_backbone_transfer.py`
  - 다른 백본으로의 전이 비교 실험
- `data/experiments/...`
  - MNV2 기반 재현에 필요한 최소 결과 캐시와 보조 스크립트

## 포함하지 않는 것

- Raspberry Pi 5 시간 측정 자산
- Coral/PCIe 온디바이스 실행 스크립트

온디바이스 시간 측정은 `Raspberry_pi5_Experiment/` 아래에서 수행한다.

## 빠른 시작

```bash
cd Server_Reproduction/ARV
python3 run_arv_experiment.py
```

## 현재 번들에서 바로 되는 것

- `python3 run_arv_experiment.py`
- `python3 run_arv_backbone_transfer.py --backbones mnv2_strong mnv2_weak mnv2_nofinetune`

## 추가 자산이 필요한 것

- `python3 run_arv_generalist.py`
- `python3 run_arv_backbone_transfer.py` 기본 실행

이 경로들은 MobileCLIP JSONL, seed JSONL, 실제 이미지 데이터셋, `specm_v4` 결과 캐시가 추가로 필요하다.
