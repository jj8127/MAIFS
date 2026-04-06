# Active Probe Set

이 디렉토리는 Raspberry Pi 5에서 **ARV가 실제로 개입하는 keep 또는 revert 경로의 종단간 추론시간**을 측정하기 위한 최소 예제 이미지 묶음이다.

구성 원칙:

- OpenSDI subset에서 실제 CPU discovery를 통해 keep/revert가 확인된 이미지들만 골랐다.
- 이미지 수를 최소화하면서도 keep와 revert가 모두 나올 가능성을 높이기 위해 authentic / manipulated 예제를 함께 넣었다.
- 이 디렉토리만으로 `run_arv_active_probe_benchmark.sh`를 바로 실행할 수 있다.

권장 실행:

```bash
cd ARV_EndToEnd_RPi5
bash run_arv_active_probe_benchmark.sh cpu
```

또는

```bash
cd ARV_EndToEnd_RPi5
bash run_arv_active_probe_benchmark.sh all
```

중요:

- CPU 기준으로는 이 이미지들에서 실제 keep/revert 사례가 확인되었다.
- Coral USB와 PCIe HAT에서도 같은 이미지로 먼저 discovery를 수행한 뒤, 실제 active case만 benchmark에 사용하도록 워크플로가 설계되어 있다.
- 따라서 외부 이미지 폴더 없이도 이 번들 하나만으로 ARV-active latency 실험을 시작할 수 있다.
