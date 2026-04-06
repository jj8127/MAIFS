# RPi5 On-Device 실험 가이드

> 최종 업데이트: 2026-03-24
> 대상 시스템: `MNV2 + SpecM-v4 + ICWMV`

이 문서는 **RPi5에서 현재 저장소 기준 배포 경로를 재현하거나 재측정할 때** 사용하는 실험 절차입니다.

---

## 1. 준비 파일

RPi5로 최소한 아래 파일을 가져갑니다.

```text
deploy/rpi5_infer.py
deploy/requirements_rpi5.txt
deploy/requirements_rpi5_coral.txt
deploy/setup_rpi5_coral_env.sh

weights/onnx_quant/mnv2_int8_dynamic.onnx
weights/onnx_quant/specm_v4_int8_static.onnx

weights/tflite_edgetpu_sweep/mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite
weights/tflite_edgetpu/specm_v4_coral_ft_int8_full_edgetpu.tflite
```

권장 작업 디렉터리 예시는 `~/maifs/` 입니다.

---

## 2. CPU 경로 재현

### 2.1 설치

```bash
cd ~/maifs
pip install -r deploy/requirements_rpi5.txt
```

### 2.2 단일 이미지 실행

```bash
python deploy/rpi5_infer.py image.jpg --backend onnx
python deploy/rpi5_infer.py image.jpg --backend onnx --threads 4
python deploy/rpi5_infer.py image.jpg --backend onnx --threads 4 --json
```

### 2.3 레이턴시 10회 반복

```bash
for i in $(seq 1 10); do
  python deploy/rpi5_infer.py image.jpg --backend onnx --threads 4 --json 2>/dev/null \
    | python -c "import sys,json; d=json.load(sys.stdin); print(d['latency']['total_ms'])"
done | awk '{s+=$1; n++} END {printf \"평균: %.1fms (n=%d)\\n\", s/n, n}'
```

기준 실측값:

- `88.3ms` total
- `53.7ms` MNV2
- `34.6ms` SpecM

---

## 3. Coral 경로 재현

### 3.1 libedgetpu 설치

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
```

USB 연결 확인:

```bash
lsusb | grep -i "Google\\|Coral"
```

### 3.2 Python 3.9 환경

```bash
cd ~/maifs
bash deploy/setup_rpi5_coral_env.sh
```

### 3.3 단일 이미지 실행

```bash
source .venv-coral/bin/activate

python deploy/rpi5_infer.py image.jpg --backend edgetpu
python deploy/rpi5_infer.py image.jpg --backend edgetpu --json
python deploy/rpi5_infer.py image.jpg --backend edgetpu \
  --delegate-path /usr/lib/aarch64-linux-gnu/libedgetpu.so.1
```

### 3.4 레이턴시 10회 반복

```bash
source .venv-coral/bin/activate

for i in $(seq 1 10); do
  python deploy/rpi5_infer.py image.jpg --backend edgetpu --json 2>/dev/null \
    | python -c "import sys,json; d=json.load(sys.stdin); print(d['latency']['total_ms'])"
done | awk '{s+=$1; n++} END {printf \"평균: %.1fms (n=%d)\\n\", s/n, n}'
```

기준 실측값:

- `63.4ms` total
- `29.0ms` MNV2
- `34.4ms` SpecM

---

## 4. CPU vs Coral 비교

```bash
IMG=image.jpg

echo "=== ONNX CPU ==="
python deploy/rpi5_infer.py "$IMG" --backend onnx --threads 4 --json

echo ""
echo "=== Coral Edge TPU ==="
source .venv-coral/bin/activate
python deploy/rpi5_infer.py "$IMG" --backend edgetpu --json
```

현재 해석:

- **CPU ONNX**: 정확도 주력
- **Coral**: 속도 주력

---

## 5. 기록 양식

### 5.1 레이턴시

| 경로 | MNV2 (ms) | SpecM (ms) | Total (ms) | 비고 |
|------|-----------|------------|------------|------|
| CPU ONNX | | | | |
| Coral TPU | | | | |

### 5.2 콜드스타트

| 경로 | Load Time (ms) |
|------|----------------|
| CPU ONNX | |
| Coral TPU | |

기준값:

- CPU ONNX: `270ms`
- Coral TPU: `2661ms`

---

## 6. 현재 판단 기준

실험 결과를 기록할 때 아래 기준을 같이 메모하는 편이 좋습니다.

1. ONNX 경로가 여전히 `0.9535` 수준을 유지하는가
2. Coral 경로가 `0.9308` 근처인지, 추가 drift가 있는가
3. Coral warm latency가 `63.4ms` 수준으로 재현되는가
4. 장시간 반복 시 Coral 편차가 여전히 `±0.8ms` 수준인지

---

## 7. 트러블슈팅

### Coral이 안 잡힐 때

```bash
lsusb | grep -i "Google\\|Coral"
```

### `tflite_runtime` import 오류

```bash
.venv-coral/bin/python --version
```

Python `3.9.x`가 아니면 Coral 경로가 깨집니다.

### delegate 경로 오류

```bash
find /usr -name 'libedgetpu.so*' 2>/dev/null
```

찾은 경로를 `--delegate-path`에 넣습니다.

