# ARV End-to-End Raspberry Pi 5

이 디렉토리는 Raspberry Pi 5에서 ARV 전체 파이프라인 종단간 지연을 측정하는 공식 번들입니다.

포함 내용:

- 2단계 ARV 저장 모델
  - `common/models/arv_stage2/*.json`
  - `common/models/arv_stage2/manifest.json`
- CPU / Coral USB / PCIe HAT 실행 스크립트
- 기본 벤치마크 입력 이미지
  - `assets/benchmark_input.png`

1단계 경량 모델은 상위 공용 경로를 사용합니다.

- `../common/models/onnx_quant/`
- `../common/models/tflite_edgetpu/`
- `../common/models/tflite_edgetpu_sweep/`

이 번들의 목표는 다음 두 가지를 함께 측정하는 것입니다.

1. 실제 분기 기준 `real-path` 종단간 지연
2. stage-2를 강제로 실행한 `forced-stage2` 종단간 지연

## 디렉토리 구조

- `common/`
  - ARV stage-2 모델, 벤치마크 스크립트, AI agent 프롬프트
- `rpi5_cpu_only/`
  - Raspberry Pi 5 CPU 전용 측정
- `rpi5_coral_usb/`
  - Raspberry Pi 5 + Coral USB 전용 측정
- `rpi5_pcie_hat/`
  - Raspberry Pi 5 + PCIe 연결 HAT 전용 측정
- `assets/`
  - 기본 입력 이미지

## 가장 빠른 실행

```bash
bash run_arv_e2e_benchmark.sh cpu
```

```bash
bash run_arv_e2e_benchmark.sh all
```

특정 이미지를 쓰고 싶으면 두 번째 인자로 넘깁니다.

```bash
bash run_arv_e2e_benchmark.sh coral /home/pi/test.jpg
```

## 출력 위치

각 환경 디렉토리 아래에 저장됩니다.

- `rpi5_cpu_only/results/`
- `rpi5_cpu_only/logs/`
- `rpi5_coral_usb/results/`
- `rpi5_coral_usb/logs/`
- `rpi5_pcie_hat/results/`
- `rpi5_pcie_hat/logs/`

## 기본 측정 규약

- `paper_v2`
  - `warmup=0`
  - `runs=10`
  - `threads=4`

확장 측정은 아래처럼 사용합니다.

```bash
bash run_arv_e2e_benchmark.sh all "" --protocol extended
```

## Python 환경

### CPU

```bash
cd rpi5_cpu_only
bash setup_env.sh
```

### Coral USB / PCIe HAT

```bash
cd rpi5_coral_usb
bash setup_env.sh
```

또는

```bash
cd rpi5_pcie_hat
bash setup_env.sh
```

주의:

- Coral / PCIe HAT은 `libedgetpu` 시스템 런타임이 필요합니다.
- PCIe HAT은 `lspci`와 `/dev/apex*`에서 장치가 보여야 정상 측정됩니다.

## AI agent용 지시

- `common/prompts/RPI5_ARV_E2E_AGENT_PROMPT.md`
