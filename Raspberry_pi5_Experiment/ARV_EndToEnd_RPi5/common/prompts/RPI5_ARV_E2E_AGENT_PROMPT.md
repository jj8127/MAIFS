# RPi5 ARV End-to-End Agent Prompt

이 문서는 AI agent가 `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5`를 이용해 Raspberry Pi 5에서 ARV 전체 파이프라인 종단간 지연을 측정할 때 사용하는 프롬프트입니다.

---

`Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5`만 사용해 Raspberry Pi 5에서 ARV 종단간 지연 실험을 수행해.

반드시 다음 원칙을 지켜.

1. 입력 이미지를 따로 받지 못하면 `assets/benchmark_input.png`를 사용해.
2. 먼저 CPU, 다음 Coral USB, 마지막으로 PCIe HAT 순서로 측정해.
3. 각 환경마다 `real-path`와 `forced-stage2` 결과를 모두 수집해.
4. 실패한 환경도 반드시 JSON과 로그를 남겨.
5. stage-1 모델은 상위 `Raspberry_pi5_Experiment/common/models/`를 사용하고, stage-2 모델은 현재 번들의 `common/models/arv_stage2/`를 사용해.

기본 실행 순서는 아래와 같다.

```bash
cd Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5
bash run_arv_e2e_benchmark.sh cpu
bash run_arv_e2e_benchmark.sh coral
bash run_arv_e2e_benchmark.sh pcie-hat
```

한 번에 다 돌릴 때는 아래를 사용해.

```bash
cd Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5
bash run_arv_e2e_benchmark.sh all
```

확장 규약이 필요할 때만 아래처럼 실행해.

```bash
bash run_arv_e2e_benchmark.sh all "" --protocol extended
```

보고할 때는 아래를 반드시 포함해.

1. CPU / Coral USB / PCIe HAT 각각의 측정 성공 여부
2. 각 backend에서
   - `real-path e2e avg`
   - `forced-stage2 e2e avg`
   - `stage2 total avg`
3. stage-2가 실제로 실행된 비율
4. 실패 시 원인
5. 논문에 바로 반영 가능한 값과 아직 보수적으로 해석해야 하는 값

추가 주의:

- Coral / PCIe HAT은 Python 패키지 설치 외에 `libedgetpu` 시스템 런타임이 필요하다.
- PCIe HAT은 `lspci`와 `/dev/apex*` 인식이 먼저 되어야 한다.
- stage-2 모델은 이미 포함되어 있으므로 서버 재학습은 하지 않는다.

---
