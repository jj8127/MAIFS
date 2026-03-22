# SHIELD RPi5 / Coral 배포 가이드

## 구성

- **모델**:
  - ONNX 경로: MNV2-Dynamic INT8 + SpecM-v4-Dynamic INT8
  - Coral 경로: tuned `mnv2_coral_qsweep_qtpc_cal064_ioint8` + `specm_v4_coral_ft`
    (없으면 기존 `mnv2_coral` / `specm_v4_coral` fallback)
- **추론 스크립트**: `rpi5_infer.py`
- **예상 지연**:
  - ONNX: ~112ms 실측 (RPi5, threads=4)
  - Coral: MNV2 기준 ~5ms급 기대, 실제는 Edge TPU compiler report 확인 필요

## 설치

### 1. ONNX 경로

```bash
# Python 3.11+/3.13+ 가능
pip install -r requirements_rpi5.txt

# 서버에서 복사
#   weights/onnx_quant/mnv2_int8_dynamic.onnx
#   weights/onnx_quant/specm_v4_int8_dynamic.onnx
```

### 2. Coral 경로

```bash
# Python 3.13은 tflite-runtime 미지원 -> Python 3.9 venv 분리
bash setup_rpi5_coral_env.sh

# 서버에서 먼저 변환
#   python experiments/run_edgetpu_export.py --models mnv2_coral specm_v4_coral

# RPi5로 복사
#   weights/tflite/*.tflite
#   weights/tflite_edgetpu/*_edgetpu.tflite
#   weights/tflite_sweep/mnv2_coral_qsweep_qtpc_cal064_ioint8.tflite
#   weights/tflite_edgetpu_sweep/mnv2_coral_qsweep_qtpc_cal064_ioint8_edgetpu.tflite
#   weights/tflite/specm_v4_coral_ft_int8_full.tflite
#   weights/tflite_edgetpu/specm_v4_coral_ft_int8_full_edgetpu.tflite
```

## 사용법

```bash
# 자동 선택 (현재는 정확도 검증된 onnx 우선)
python rpi5_infer.py photo.jpg

# 백엔드 명시
python rpi5_infer.py photo.jpg --backend onnx
python rpi5_infer.py photo.jpg --backend tflite
python rpi5_infer.py photo.jpg --backend edgetpu

# JSON 출력
python rpi5_infer.py photo.jpg --json

# RPi5 멀티코어 활용 (onnx/tflite only)
python rpi5_infer.py photo.jpg --threads 4

# 모델 경로 직접 지정
python rpi5_infer.py photo.jpg \
  --backend edgetpu \
  --mnv2 /home/pi/models/mnv2_coral_int8_full_edgetpu.tflite \
  --specm /home/pi/models/specm_v4_coral_int8_full_edgetpu.tflite
```

## 출력 예시

```
백엔드: edgetpu
판정: 조작 (manipulated, 72.3%)
  auth=0.214  manip=0.723  aigen=0.063
MNV2: auth=0.301  manip=0.542  aigen=0.157
SpecM: auth=0.247  manip=0.753
추론: 138ms  (모델 로드: 680ms)
```

```json
{
  "backend": "edgetpu",
  "verdict": "manipulated",
  "confidence": 0.7231,
  "scores": {"authentic": 0.214, "manipulated": 0.723, "ai_generated": 0.063},
  "mnv2_scores": {"authentic": 0.301, "manipulated": 0.542, "ai_generated": 0.157},
  "specm_scores": {"authentic": 0.247, "manipulated": 0.753},
  "latency_ms": 138.4,
  "load_ms": 682.1
}
```

## 성능

| 경로 | 지연 | 비고 |
|------|------|------|
| ONNX (MNV2+SpecM-v4) | avg 112ms | 2026-03-21 RPi5 실측 |
| TFLite CPU (`*_coral`) | 미측정 | tuned MNV2 + `specm_v4_coral_ft` 우선 사용 |
| Edge TPU (`*_coral`) | 실측 대기 | tuned MNV2 + `specm_v4_coral_ft` compile 완료 |

현재 정확도 재평가(2026-03-21):
- `current_onnx` avg macro-F1: `0.9535`
- `coral_tflite` avg macro-F1: `0.9051` (`-0.0483`)
- `specm_v4_coral` avg manip-F1: `0.5827` vs current `0.8392`
- `mnv2_coral`도 full INT8 TFLite에서 avg macro-F1 `0.9096`까지 하락

2026-03-22 MNV2 PTQ sweep:
- best MNV2 Coral candidate: `mnv2_coral_qsweep_qtpc_cal064_ioint8(.tflite)`
- MNV2 avg macro-F1: `0.9173` vs baseline coral `0.9096` (`+0.0077`)
- tuned MNV2 + existing `specm_v4_coral` pair avg macro-F1: `0.9160`
  vs baseline coral pair `0.9051` (`+0.0108`)
- tuned MNV2 Edge TPU compile: **151/151 ops** mapped

2026-03-22 SpecM Coral fine-tune:
- `specm_v4_coral_ft` TFLite standalone avg manip-F1: `0.8360`
  vs old coral `0.5827` (`+0.2533`)
- tuned MNV2 + `specm_v4_coral_ft` pair:
  - `w_spec=1.0`: avg macro-F1 `0.8891`
  - `w_spec=0.2`: avg macro-F1 `0.9308` (best)
- current ONNX `0.9535` 대비 gap: `-0.0226`

## 조합 로직 (ICWMV)

```
auth  = (MNV2(auth)  + SpecM(auth))  / 2   ← 양쪽 기여
manip = (MNV2(manip) + SpecM(manip)) / 2   ← 양쪽 기여
aigen =  MNV2(aigen)                        ← MNV2만 기여 (SpecM은 AI탐지 불가)
→ renormalize → argmax
```

## 한계

- AI 생성 이미지 탐지는 MNV2만 담당 (SpecG 없이 단독)
- SpecG(AI-gen 전문 모델)는 141MB로 RPi5 지연 예산 초과 → 서버 전용
- Coral 경로는 Python 3.9 `tflite-runtime`이 필요하며, 서버에서 TFLite/Edge TPU 컴파일을 선행해야 함
- `inference_rpi5.py`/`rpi5_infer.py`는 explicit `--backend tflite|edgetpu`일 때
  tuned `mnv2_coral_qsweep_qtpc_cal064_ioint8*` + `specm_v4_coral_ft*`를 먼저 사용함
- 이 조합에서는 `w_spec=0.2`가 자동 기본값이며, CLI `--w-spec`로 override 가능
- `auto`는 정확도 검증이 끝날 때까지 ONNX를 우선 사용함
- 4-DS 재평가 기준 최신 explicit Coral pair best는 avg macro-F1=`0.9308`
- current ONNX `0.9535` 대비 아직 `-0.0226` 남아 있어,
  최종 배포 확정 전에는 실제 Coral USB 실측과 추가 미세조정이 권장됨
