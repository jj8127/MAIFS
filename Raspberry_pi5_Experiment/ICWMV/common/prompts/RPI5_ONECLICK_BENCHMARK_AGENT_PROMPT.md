# Raspberry Pi 5 One-Click Benchmark Prompt

이 문서는 AI agent가 `Raspberry_pi5_Experiment/ICWMV` 아래의 자동화 스크립트만 이용해 Raspberry Pi 5 지연 실험을 재현하도록 지시하기 위한 프롬프트입니다.

## 목표

- `CPU`, `Coral Edge TPU`, `PCIe HAT` 세 경로를 같은 프로토콜로 측정합니다.
- 기본 프로토콜은 `paper_v2`입니다.
- `paper_v2`는 `threads=4`, 단일 입력 이미지 반복, `warmup 0`, `10회 실측`을 의미합니다.
- 결과를 각 환경 디렉토리의 `results/`와 `logs/`에 자동 저장합니다.
- 측정 실패 시에도 `unmeasured` JSON을 남겨 무엇이 막혔는지 기록합니다.

## 에이전트에게 줄 지시문

아래 지시를 그대로 사용합니다.

---

`Raspberry_pi5_Experiment/ICWMV`만 사용해서 Raspberry Pi 5 지연 실험을 수행해.

반드시 다음 순서로 진행해.

1. `cd Raspberry_pi5_Experiment/ICWMV`
2. CPU 경로를 먼저 측정
3. Coral 경로를 측정
4. PCIe HAT 경로를 측정
5. 세 경로 결과를 각각 JSON과 로그로 저장
6. 측정 실패 시에도 중단하지 말고 `unmeasured` 결과 파일을 남겨
7. 실험이 끝나면 어떤 값이 논문에 바로 반영 가능하고 어떤 값이 아직 미측정인지 구분해서 보고해

CPU, Coral, PCIe HAT을 한 번에 돌릴 때는 아래 명령을 사용해.

```bash
cd Raspberry_pi5_Experiment/ICWMV
bash run_rpi5_latency_benchmark.sh all /path/to/test_image.jpg
```

확장 프로토콜이 필요하면 아래처럼 실행해.

```bash
bash run_rpi5_latency_benchmark.sh all /path/to/test_image.jpg --protocol extended
```

개별 경로만 다시 돌릴 때는 아래 명령을 사용해.

```bash
bash run_rpi5_latency_benchmark.sh cpu /path/to/test_image.jpg
bash run_rpi5_latency_benchmark.sh coral /path/to/test_image.jpg
bash run_rpi5_latency_benchmark.sh pcie-hat /path/to/test_image.jpg
```

추가 조건:

- CPU는 기본적으로 `.venv-rpi5-cpu`가 있으면 그 Python을 쓰고, 없으면 시스템 `python3`를 사용해.
- Coral과 PCIe HAT은 `tflite-runtime`과 `libedgetpu.so.1` 런타임이 준비된 Python 환경을 사용해.
- 결과 파일은 각 환경 디렉토리의 `results/YYYYMMDD_rpi5_*_latency.json` 형식이어야 해.
- 로그 파일은 각 환경 디렉토리의 `logs/YYYYMMDD_rpi5_*.log` 형식이어야 해.
- 논문 수치는 직접 측정한 값만 사용하고, 실패한 경로는 `미측정`으로 남겨.
- 논문 본문은 별도 요청이 있기 전에는 수정하지 마.

최종 보고에는 아래를 포함해.

1. 각 환경의 측정 성공 여부
2. CPU / Coral / PCIe HAT의 총 지연 평균
3. 워밍업 제외 여부와 실측 횟수
4. 측정 실패 시 실패 원인
5. 논문에 바로 넣을 수 있는 값과 아직 넣으면 안 되는 값

---

## 빠른 수동 실행

```bash
cd Raspberry_pi5_Experiment/ICWMV
bash run_rpi5_latency_benchmark.sh all /path/to/test_image.jpg
```
