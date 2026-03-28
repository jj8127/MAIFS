# ARV End-to-End Raspberry Pi 5 + Coral USB

이 디렉토리는 Raspberry Pi 5 + Coral USB 환경에서 ARV 전체 파이프라인 종단간 지연을 측정할 때 사용한다.

## 환경 준비

```bash
cd rpi5_coral_usb
bash setup_env.sh
```

시스템에는 `libedgetpu` 런타임이 먼저 설치되어 있어야 한다.

## 실행

```bash
cd ..
bash run_arv_e2e_benchmark.sh coral
```
