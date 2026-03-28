# Raspberry Pi 5 ARV 실험 결과 정리

기준 디렉토리: `/data/jj812_files/result_log`

## 1. 전체 상태 요약

- 성공적으로 직접 측정된 항목
  - Raspberry Pi 5 CPU 환경의 1단계 추론 지연
  - ARV 2단계 거부 모듈의 추가 오버헤드
- 아직 직접 측정되지 않은 항목
  - Raspberry Pi 5 + Coral Edge TPU의 1단계 지연
  - Raspberry Pi 5 + PCIe 연결 HAT의 1단계 지연
  - Raspberry Pi 5에서의 ARV 전체 파이프라인 end-to-end 실측
- 부분적으로만 확보된 항목
  - ARV 2단계 conflict 케이스는 실제 override 샘플이 아니라 synthetic override 조합으로 측정됨

## 2. 핵심 결과

## 2.1 Raspberry Pi 5 CPU 1단계 실측

출처:
- `20260325_rpi5_cpu_latency.json`
- `20260325_rpi5_cpu.log`

측정 조건:
- 장치: Raspberry Pi 5 Model B Rev 1.0
- Python: 3.13.5
- 스레드 수: 4
- 측정 방식: 5회 warm-up 제외 후 30회 측정
- 입력 이미지: `/home/jj/test.jpg` 단일 이미지 반복

핵심 수치:
- 기본 분류기 평균: `115.313 ms`
- 조작 전문 모델 평균: `69.273 ms`
- 총 지연 평균: `184.59 ms`
- 총 지연 표준편차: `32.948 ms`
- 총 지연 범위: `149.8 ms` ~ `320.5 ms`

해석:
- Raspberry Pi 5 CPU만으로도 1단계 2-모델 파이프라인은 실행된다.
- 다만 기존 초안에서 기대한 80~90 ms 수준보다 훨씬 느리다.
- 현재 수치는 `동적 양자화 MNV2 + 정적 양자화 SpecM-v4` 조합 기준이므로, 논문에는 정확히 이 조합으로 명시해야 한다.
- 이 측정은 지연 실험이며, 온디바이스 정확도 실험으로 해석하면 안 된다.

## 2.2 ARV 2단계 오버헤드

출처:
- `20260325_arv_stage2_overhead.json`
- `20260325_arv_stage2.log`

핵심 수치:
- non-conflict real
  - 특징 생성 평균: `0.133 ms`
  - `predict_proba` 평균: `2.654 ms`
  - 2단계 총 오버헤드 평균: `2.787 ms`
- conflict synthetic override
  - 특징 생성 평균: `0.185 ms`
  - `predict_proba` 평균: `3.463 ms`
  - 2단계 총 오버헤드 평균: `3.648 ms`

주의:
- 실제 로컬 샘플 35장에서는 override가 발생하지 않아, conflict 케이스는 synthetic 조합으로 측정되었다.
- 따라서 `2단계 전체 파이프라인 실측`이라고 쓰면 안 되고, `2단계 모듈 오버헤드의 부분 측정`으로 기술해야 한다.

추정:
- CPU 기준 1단계 평균 `184.59 ms`에 2단계 오버헤드만 단순 합산하면
  - non-conflict 추정 총합: 약 `187.38 ms`
  - conflict 추정 총합: 약 `188.24 ms`
- 이 수치는 직접 측정값이 아니라 단순 합산 추정치다.

## 2.3 ARV 저장 모델 상태

출처:
- `20260325_arv_export_status.json`
- `20260325_arv_export.log`

상태:
- 재생성(regeneration)은 실패
- 실패 원인: `FileNotFoundError`
- 누락 경로:
  - `/home/jj/maifs/experiments/results/hema_icwmv_veto/hema_icwmv_veto_loo_cd_20260323_114321.json`

하지만 다음은 이미 존재함:
- bundled ARV 모델 존재: `true`
- bundled manifest 존재:
  - `/home/jj/maifs/temp/artifacts/arv_models/manifest.json`

해석:
- 결과 재생성 경로는 깨졌지만, 번들에 포함된 저장 모델 자체는 있어서 2단계 오버헤드 측정은 가능했다.
- 즉 `재생성 실패`와 `ARV 저장 모델 부재`는 같은 문제가 아니다.

## 2.4 Coral Edge TPU 측정 상태

출처:
- `20260325_rpi5_coral_latency.json`
- `20260325_rpi5_coral.log`

상태:
- `unmeasured`

원인:
- Edge TPU delegate 로드 실패
- `libedgetpu.so.1`는 존재
- Python 3.9 환경에 `pycoral` 미설치
- `tflite_runtime`는 설치됨
- `lsusb` 결과: `unable to initialize libusb: -99`

해석:
- 현재 Coral 측정 실패 원인은 모델 문제가 아니라 런타임/장치 접근 문제에 가깝다.
- 따라서 논문에는 Coral 수치를 넣으면 안 된다.

## 2.5 PCIe HAT 측정 상태

출처:
- `20260325_rpi5_pcie_hat_latency.json`
- `20260325_rpi5_pcie_hat.log`

상태:
- `unmeasured`

원인:
- `config.txt`에 PCIe 관련 설정 없음
- `lspci`에서 Apex 장치 미탐지
- `/dev/apex*` 미존재

해석:
- PCIe HAT은 아직 장치 자체가 올라오지 않은 상태다.
- 현재는 소프트웨어 최적화 문제가 아니라 하드웨어 인식 단계가 막혀 있다.

## 3. 논문 반영 가능 항목

현재 바로 논문에 반영 가능한 항목:
- Raspberry Pi 5 CPU 기준 1단계 추론 평균 `184.59 ms`
- ARV 2단계 오버헤드가 대략 `2.8~3.6 ms` 수준이라는 점

현재 논문에 직접 수치로 쓰면 안 되는 항목:
- Coral Edge TPU 레이턴시
- PCIe HAT 레이턴시
- Raspberry Pi 5에서의 ARV 전체 구조 end-to-end 레이턴시
- 실제 conflict 샘플 기반 ARV 2단계 전체 오버헤드

## 4. 가장 냉정한 결론

1. 지금 확보된 온디바이스 실측은 사실상 CPU 경로뿐이다.
2. CPU 경로에서 1단계 구조는 돌아가지만, 총 `184.59 ms`로 기대보다 느리다.
3. ARV 2단계 자체는 매우 가볍다. 병목은 2단계가 아니라 1단계 추론이다.
4. Coral/PCIe 결과가 비어 있으므로, 현재 논문 메시지는 `CPU에서의 실행 가능성 + ARV 2단계의 낮은 추가 비용`까지만 정직하게 주장하는 편이 맞다.

## 5. 다음 우선순위

1. Coral Python 3.9 환경에 `pycoral`을 올려 delegate 로드부터 복구
2. PCIe HAT에서 `/dev/apex_0`가 보이도록 하드웨어 인식 복구
3. 실제 override가 발생하는 로컬 샘플 세트를 모아 ARV conflict 실측 보강
4. CPU 경로에서 정적 ONNX 조합으로 다시 측정해 지연 감소 여지 확인
