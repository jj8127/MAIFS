# TEMP(1).zip 실험 실행 정리

작성일: 2026-03-27

이 문서는 `/home/jj/Downloads/TEMP (1).zip` 를 `/home/jj/maifs/TEMP_1_zip_bundle` 로 새로 압축 해제한 뒤, 내부의 `temp/` 와 `temp2/` 에 대해 실제로 수행한 실험 결과를 정리한 것이다.

## 1. 실행 기준

- 압축 파일: `/home/jj/Downloads/TEMP (1).zip`
- 추출 경로: `/home/jj/maifs/TEMP_1_zip_bundle`
- 공통 입력 이미지:
  - `/home/jj/maifs/temp_zip_bundle/temp/benchmark_input.png`
  - 새 압축본에는 벤치마크 입력 이미지가 없어서, 기존에 검증해 둔 동일 목적의 입력 이미지를 사용했다.

사용한 Python:

- CPU 시간 측정 runner: `/home/jj/maifs_env/bin/python`
- Coral/PCIe 시간 측정 runner: `/home/jj/maifs/temp_zip_bundle/.venv-coral39/bin/python`
- ARV 실험 runner: `python3`

선택 이유:

- `python3` 와 `maifs_env` 는 `onnxruntime`, `numpy`, `scikit-learn`, `xgboost` 를 사용할 수 있었다.
- Coral 경로는 새 번들 안에 전용 Python 3.9 venv가 없어서, 기존에 설치되어 있던 `tflite_runtime` 포함 Python 3.9 Coral venv를 runner로 재사용했다.

## 2. 전체 요약

### 성공한 실험

- `temp/run_rpi5_latency_benchmark.sh all ...`
- `temp2/run_rpi5_latency_benchmark.sh all ...`
- `temp2/run_arv_experiment.py`

### 실패한 실험

- `temp2/run_arv_generalist.py`
- `temp2/run_arv_backbone_transfer.py --backbones clip_ft4_strong clip_zeroshot_weak`

### 미완료 실험

- `temp2/run_arv_backbone_transfer.py` 기본 실행

이 마지막 항목은 새 번들 코드 기준으로 기본 backbones가 이미 `mnv2_strong mnv2_weak mnv2_nofinetune` 으로 바뀌어 있어 README 설명과 다르다. 실제로 실행했지만, 이번 세션에서는 결과 파일을 남기지 못한 채 장시간 계산 후 비정상 종료 상태(`defunct`)로 끝나 미완료로 분류했다.

## 3. temp 실험 결과

실행 명령:

```bash
CPU_PYTHON=/home/jj/maifs_env/bin/python \
CORAL_PYTHON=/home/jj/maifs/temp_zip_bundle/.venv-coral39/bin/python \
bash run_rpi5_latency_benchmark.sh all /home/jj/maifs/temp_zip_bundle/temp/benchmark_input.png
```

작업 디렉토리:

```text
/home/jj/maifs/TEMP_1_zip_bundle/temp
```

### 3-1. CPU 시간 측정

- 결과 파일: `temp/results/20260327_rpi5_cpu_latency.json`
- 상태: 성공

핵심 수치 (`paper_v2`, 10회 실측):

- `mnv2_avg = 140.32 ms`
- `specm_avg = 79.99 ms`
- `total_avg = 220.32 ms`
- `total_std = 34.625 ms`

### 3-2. Coral 시간 측정

- 결과 파일: `temp/results/20260327_rpi5_coral_latency.json`
- 상태: 성공

핵심 수치 (`paper_v2`, 10회 실측):

- `mnv2_avg = 29.74 ms`
- `specm_avg = 36.16 ms`
- `total_avg = 65.91 ms`
- `total_std = 2.045 ms`

### 3-3. PCIe HAT 시간 측정

- 결과 파일: `temp/results/20260327_rpi5_pcie_hat_latency.json`
- 상태: 실패가 아니라 `unmeasured`

기록된 직접 사유:

- `PCIe Coral 장치가 탐지되지 않아 측정 불가`

확인된 상태:

- `libedgetpu.so.1`: 있음
- runner env의 `tflite_runtime`: 있음
- `lsusb`: `Google Inc.` 장치 보임
- `/boot/firmware/config.txt` 관련 설정:
  - `dtparam=pciex1=on`
  - `dtparam=pcie_tperst_clk_ms=100`
- `lspci | grep -i apex`: 비어 있음
- `/dev/apex*`: 비어 있음

해석:

- USB Coral 쪽 환경은 보이지만, PCIe HAT 쪽 Apex 장치가 실제로 enumerate되지 않았다.
- 따라서 이 경로는 코드 문제가 아니라 하드웨어/연결/부팅 상태 문제로 측정이 불가했다.

## 4. temp2 실험 결과

## 4-1. temp2 시간 측정

실행 명령:

```bash
CPU_PYTHON=/home/jj/maifs_env/bin/python \
CORAL_PYTHON=/home/jj/maifs/temp_zip_bundle/.venv-coral39/bin/python \
bash run_rpi5_latency_benchmark.sh all /home/jj/maifs/temp_zip_bundle/temp/benchmark_input.png
```

작업 디렉토리:

```text
/home/jj/maifs/TEMP_1_zip_bundle/temp2
```

### CPU

- 결과 파일: `temp2/results/20260327_rpi5_cpu_latency.json`
- 상태: 성공

핵심 수치:

- `mnv2_avg = 128.76 ms`
- `specm_avg = 87.23 ms`
- `total_avg = 215.99 ms`
- `total_std = 49.377 ms`

### Coral

- 결과 파일: `temp2/results/20260327_rpi5_coral_latency.json`
- 상태: 성공

핵심 수치:

- `mnv2_avg = 30.10 ms`
- `specm_avg = 36.69 ms`
- `total_avg = 66.79 ms`
- `total_std = 2.143 ms`

### PCIe HAT

- 결과 파일: `temp2/results/20260327_rpi5_pcie_hat_latency.json`
- 상태: `unmeasured`
- 사유: `temp` 와 동일하게 PCIe Apex 장치 미탐지

## 4-2. ARV 메인 재현 실험

실행 명령:

```bash
python3 run_arv_experiment.py
```

상태: 성공

결과 파일:

- `temp2/data/experiments/results/hema_icwmv_veto/comp_nots_richer_veto_20260327_200437.json`

핵심 결과:

- `avg_f1 = 0.9647`
- `avg_corr = 0.3596`
- `avg_net_gain = 8.0`

데이터셋별:

- `base`: `0.9579`, `broken=8`, `net_gain=21`, `corr=0.3537`
- `dsC`: `0.9811`, `broken=7`, `net_gain=2`, `corr=0.5294`
- `opensdi`: `0.9545`, `broken=4`, `net_gain=5`, `corr=0.3333`
- `aigenproxy`: `0.9655`, `broken=0`, `net_gain=4`, `corr=0.2222`

기준선:

- `ICWMV avg_f1 = 0.9630`
- `scalar_veto avg_f1 = 0.9622`
- `meta_warmstart_veto avg_f1 = 0.9625`

해석:

- 새 압축본의 `temp2` 에서도 ARV 메인 재현은 정상적으로 동작했다.
- 결과는 이전 번들에서 확인했던 값과 동일 계열로 재현되었다.

## 5. 실패한 실험과 실패 이유

## 5-1. generalist ARV 실험 실패

실행 명령:

```bash
python3 run_arv_generalist.py
```

상태: 실패

직접 출력된 메시지:

```text
[run] clip_ft4_strong
입력 자산이 부족해 generalist ARV 실험을 시작할 수 없습니다.
- generalist: clip_ft4_strong
- missing: Missing backbone JSONL: /home/jj/maifs/TEMP_1_zip_bundle/temp2/data/experiments/results/backbone_eval/mobileclip_s2_finetuned_base_20260319_061834.jsonl
- 현재 최소 temp2 번들에는 MobileCLIP JSONL이 포함되어 있지 않습니다.
- 이 스크립트를 실행하려면 mobileclip_s2_finetuned_*.jsonl 과 mobileclip_s2_zeroshot_scored_*.jsonl 을 추가로 넣어야 합니다.
```

상세 원인:

- `run_arv_generalist.py` 는 MobileCLIP 기반 generalist 비교를 기본 전제로 한다.
- 현재 최소 번들에는 `mobilenetv2_dualstream_*.jsonl` 은 들어 있지만 `mobileclip_s2_*.jsonl` 은 없다.
- 따라서 첫 번째 MobileCLIP 백본 로딩 단계에서 즉시 중단된다.

직접 부족한 MobileCLIP JSONL:

- `mobileclip_s2_finetuned_base_20260319_061834.jsonl`
- `mobileclip_s2_finetuned_dsC_20260319_061834.jsonl`
- `mobileclip_s2_finetuned_opensdi_20260319_061834.jsonl`
- `mobileclip_s2_finetuned_aigenproxy_20260319_061834.jsonl`
- `mobileclip_s2_zeroshot_scored_base_20260323_141929.jsonl`
- `mobileclip_s2_zeroshot_scored_dsC_20260323_141929.jsonl`
- `mobileclip_s2_zeroshot_scored_opensdi_20260323_141929.jsonl`
- `mobileclip_s2_zeroshot_scored_aigenproxy_20260323_141929.jsonl`

현재 결론:

- 이 실패는 코드 문법 문제가 아니라 입력 자산 부족 때문이다.

## 5-2. MobileCLIP backbone transfer 실패

실행 명령:

```bash
python3 run_arv_backbone_transfer.py --backbones clip_ft4_strong clip_zeroshot_weak
```

상태: 실패

직접 출력된 메시지:

```text
[Backbone=clip_ft4_strong]
입력 자산이 부족해 backbone transfer를 계속할 수 없습니다.
- backbone: clip_ft4_strong
- missing: [Errno 2] No such file or directory: '/home/jj/maifs/TEMP_1_zip_bundle/temp2/data/experiments/results/backbone_eval/mobileclip_s2_finetuned_base_20260319_061834.jsonl'
- 현재 최소 temp2 번들에서 기본 지원되는 실행은 `--backbones mnv2_strong mnv2_weak mnv2_nofinetune` 입니다.
- MobileCLIP 비교를 하려면 mobileclip JSONL, seed JSONL, datasets 이미지, specm_v4 JSONL을 추가로 넣어야 합니다.
```

상세 원인:

- `clip_ft4_strong` 는 fine-tuned MobileCLIP 캐시를 바로 읽어야 한다.
- 하지만 그 JSONL이 현재 `backbone_eval/` 에 없다.
- `clip_zeroshot_weak` 도 이론상 현장 생성 코드가 있으나, 실제로는 추가 seed JSONL과 원본 이미지 데이터셋이 필요하다.

추가로 필요한 자산:

- `mobileclip_s2_finetuned_*.jsonl` 4개
- `mobileclip_s2_zeroshot_scored_*.jsonl` 4개
- seed `mobileclip_s2_*.jsonl`
- 실제 `datasets/...` 이미지
- `specm_v4_*.jsonl`

현재 결론:

- MobileCLIP backbone transfer 실패 원인도 generalist와 마찬가지로 입력 자산 부족이다.

## 6. 미완료 실험

## 6-1. 기본 run_arv_backbone_transfer.py 실행

실행 명령:

```bash
python3 run_arv_backbone_transfer.py
```

상태: 미완료

관찰된 사실:

- 이번 압축본의 `run_arv_backbone_transfer.py` 기본 인자는 이미
  - `mnv2_strong`
  - `mnv2_weak`
  - `mnv2_nofinetune`
  로 바뀌어 있었다.
- 즉 README의 "기본 실행은 현재 최소 번들만으로는 불가" 라는 설명과 달리, 코드 기본값은 MNV2-only 비교로 수정되어 있었다.

이번 실행에서 발생한 일:

- 프로세스는 30분 이상 CPU를 계속 사용했다.
- 결과 파일 `temp2/data/experiments/results/arv_backbone_transfer/*.json` 은 생성되지 않았다.
- 마지막 확인 시 프로세스는 `defunct` 상태였고, 실험 결과를 회수할 수 없었다.

왜 성공으로 분류하지 않았는가:

- 산출 JSON이 없었다.
- 표준 출력으로 최종 요약도 회수되지 않았다.
- 따라서 "성공"으로 기록할 근거가 부족하다.

왜 일반적인 입력 부족 실패로 분류하지 않았는가:

- 이 경로는 명시적 `FileNotFoundError` 로 즉시 죽은 것이 아니었다.
- 실제로 장시간 계산이 진행됐다.
- 따라서 이 항목은 "실패"보다 "미완료/비정상 종료"로 보는 것이 더 정확하다.

보수적 해석:

- 현재 세션 기준으로는 이 실험이 끝까지 성공했다고 말할 수 없다.
- 다만 코드 기본값과 README 설명이 어긋나 있으므로, 다음 실행 때는 이 항목을 별도 터미널/장시간 세션으로 다시 돌리는 것이 안전하다.

## 7. 추가 메모

### 7-1. temp2 README와 실제 코드의 불일치

`temp2/README.md` 는 다음을 현재 최소 번들에서 바로 안 되는 것으로 적고 있다.

- `python3 run_arv_generalist.py`
- `python3 run_arv_backbone_transfer.py` 기본 실행

하지만 실제 코드에서는:

- `run_arv_generalist.py` 는 여전히 실패한다.
- `run_arv_backbone_transfer.py` 기본 인자는 이미 `mnv2` 3종으로 바뀌어 있다.

즉, README는 일부 구간이 최신 코드 상태를 완전히 반영하지 못하고 있다.

### 7-2. 현재 가장 확실하게 성공한 핵심 실험

이번 압축본 기준으로 가장 안정적으로 재현된 핵심 실험은 다음 두 가지다.

- `temp` / `temp2` 의 Raspberry Pi 5 시간 측정
- `temp2/run_arv_experiment.py` ARV 메인 재현

이 둘은 실제 결과 파일까지 정상 생성되었다.
