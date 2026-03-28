# Raspberry Pi 5 Experiment Hub

이 디렉토리는 Raspberry Pi 5 실험 자산의 공식 허브입니다.

- `common/`
  - 1단계 공용 모델과 canonical 추론 스크립트
- `ICWMV/`
  - 1단계 경량 2-모델 결합의 Raspberry Pi 5 지연 측정
- `ARV_EndToEnd_RPi5/`
  - ARV stage-2까지 포함한 종단간 지연 측정
- `docs/`
  - 논문 근거가 되는 상태 문서와 요약 문서

## 디렉토리 원칙

- 환경 디렉토리 이름은 항상 아래 셋만 사용합니다.
  - `rpi5_cpu_only`
  - `rpi5_coral_usb`
  - `rpi5_pcie_hat`
- 공개 실행 진입점은 각 번들의 상위 런처 스크립트입니다.
- 1단계 모델은 `common/models/`에 공용으로 둡니다.
- ARV stage-2 모델은 `ARV_EndToEnd_RPi5/common/models/arv_stage2/`에 둡니다.

## 빠른 시작

### ICWMV 측정

```bash
cd Raspberry_pi5_Experiment/ICWMV
bash run_rpi5_latency_benchmark.sh all /path/to/test_image.jpg
```

### ARV 종단간 측정

```bash
cd Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5
bash run_arv_e2e_benchmark.sh all
```

## 주요 문서

- 상태 문서: [TEMP1_EXPERIMENT_STATUS_20260327.md](docs/TEMP1_EXPERIMENT_STATUS_20260327.md)
- 결과 요약: [RESULT_SUMMARY_20260326.md](docs/RESULT_SUMMARY_20260326.md)
- 서버 정확도 재현: [Server_Reproduction/ARV](../Server_Reproduction/ARV/README.md)
