# SHIELD RPi5 배포 가이드

> 최종 업데이트: 2026-03-24
> 이 문서는 **현재 저장소에 포함된 배포 진입점** 기준으로 정리되어 있습니다.

---

## 1. 배포 개요

현재 SHIELD의 배포 파이프라인은 다음 조합을 기준으로 합니다.

- **Generalist**: `MNV2`
- **Specialist**: `SpecM-v4`
- **Fusion**: `ICWMV` (역신뢰도 가중 다수결)

배포 경로는 두 가지입니다.

| 경로 | 형식 | 정확도 | RPi5 평균 레이턴시 | 상태 |
|------|------|--------|-------------------|------|
| ONNX CPU | INT8 ONNX Runtime | **0.9535** | **88.3ms** | 현재 주력 |
| Coral USB | Edge TPU TFLite | 0.9308 | **63.4ms** | 속도 우위, 정확도 gap 존재 |

현재 연구 결론상 **정확도 우선이면 ONNX CPU**, **속도/전력 우선 실험이면 Coral** 경로를 권장합니다.

---

## 2. 저장소 기준 배포 파일

### 공통

| 파일 | 설명 |
|------|------|
| `deploy/rpi5_infer.py` | RPi5 통합 추론 진입점 |
| `deploy/requirements_rpi5.txt` | CPU 경로 의존성 |
| `deploy/requirements_rpi5_coral.txt` | Coral 경로 의존성 |
| `deploy/setup_rpi5_coral_env.sh` | Python 3.9 Coral 전용 venv 부트스트랩 |

### 모델 경로

| 경로 | 설명 |
|------|------|
| `weights/onnx_quant/mnv2_int8_dynamic.onnx` | CPU용 MNV2 |
| `weights/onnx_quant/specm_v4_int8_static.onnx` | CPU용 SpecM-v4 |
| `weights/tflite_edgetpu_sweep/mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite` | Coral용 tuned MNV2 |
| `weights/tflite_edgetpu/specm_v4_coral_ft_int8_full_edgetpu.tflite` | Coral용 coral-ft SpecM |

---

## 3. 설치

### 3.1 CPU 경로

```bash
pip install -r deploy/requirements_rpi5.txt
```

### 3.2 Coral 경로

Coral은 `tflite-runtime` 제약 때문에 **Python 3.9**를 별도로 써야 합니다.

```bash
bash deploy/setup_rpi5_coral_env.sh
```

수동 설치가 필요할 경우:

```bash
python3.9 -m venv .venv-coral
source .venv-coral/bin/activate
pip install -r deploy/requirements_rpi5_coral.txt
```

추가로 `libedgetpu1-std` 설치가 필요합니다.

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update && sudo apt install libedgetpu1-std
```

---

## 4. 사용법

### 4.1 ONNX CPU

```bash
python deploy/rpi5_infer.py image.jpg --backend onnx
python deploy/rpi5_infer.py image.jpg --backend onnx --threads 4
python deploy/rpi5_infer.py image.jpg --backend onnx --json
```

### 4.2 Coral USB

```bash
source .venv-coral/bin/activate

python deploy/rpi5_infer.py image.jpg --backend edgetpu
python deploy/rpi5_infer.py image.jpg --backend edgetpu --json
python deploy/rpi5_infer.py image.jpg --backend edgetpu \
  --delegate-path /usr/lib/aarch64-linux-gnu/libedgetpu.so.1
```

### 4.3 자동 선택

```bash
python deploy/rpi5_infer.py image.jpg
```

현재 `auto`는 정확도 보수성을 위해 **ONNX를 우선** 사용합니다.

---

## 5. 출력 형식

### 기본 텍스트

```text
백엔드 : onnx_cpu
판정   : 조작 (manipulated, 72.3%)
scores : auth=0.214  manip=0.723  aigen=0.063
MNV2   : auth=0.301  manip=0.542  aigen=0.157
SpecM  : auth=0.247  manip=0.753
레이턴시: MNV2=53.7ms  SpecM=34.6ms  합계=88.3ms
```

### JSON

```bash
python deploy/rpi5_infer.py image.jpg --backend onnx --json
```

---

## 6. 성능 요약

### 6.1 정확도

| 경로 | avg macro-F1 | 기준 |
|------|-------------|------|
| 서버 기준 ICWMV + v4 | 0.9652 | strong MNV2, 4-DS LOO-CD |
| RPi5 ONNX | **0.9535** | deployment path |
| RPi5 Coral | 0.9308 | tuned MNV2 + coral-ft SpecM |

### 6.2 RPi5 실측 레이턴시

| 단계 | ONNX CPU | Coral Edge TPU | 비고 |
|------|----------|----------------|------|
| MNV2 | 53.7ms | **29.0ms** | Coral 1.85x 가속 |
| SpecM | 34.6ms | 34.4ms | fallback 영향으로 거의 동일 |
| **Total** | **88.3ms** | **63.4ms** | Coral 1.39x 가속 |
| 편차 | ±9.5ms | **±0.8ms** | Coral이 더 안정적 |

### 6.3 콜드스타트

| 경로 | 모델 로드 시간 |
|------|---------------|
| ONNX CPU | **270ms** |
| Coral | **2,661ms** |

즉, Coral은 warm latency는 유리하지만 cold start는 훨씬 느립니다.

---

## 7. 현재 권장 해석

- **실험/논문용 주력 경로**: ONNX CPU
- **속도 실험 / 후속연구용 경로**: Coral USB
- `w_spec`는 Coral의 coral-ft 조합에서 `0.2`가 best였고, ONNX 주력선은 기본 `v4` 설정을 사용합니다.

---

## 8. 한계 및 주의사항

- `SpecM`은 `ai_generated`를 직접 다루지 못하므로, 해당 클래스는 `MNV2`가 전담합니다.
- Coral 경로는 속도는 좋지만 현재 정확도 기준으로는 ONNX보다 불리합니다.
- `tflite-runtime` 제약 때문에 Coral은 Python 3.9 분리 환경이 필요합니다.
- `auto` 백엔드는 의도적으로 ONNX 우선입니다.

---

## 9. 모델 재생성

```bash
# ONNX 양자화 export 및 검증
.venv-qwen/bin/python experiments/run_rpi5_model_export.py

# Coral export + compile
.venv-qwen/bin/python experiments/run_edgetpu_export.py --models mnv2_coral specm_v4_coral

# Coral용 MNV2 PTQ sweep
.venv-qwen/bin/python experiments/run_mnv2_coral_quant_sweep.py
```

