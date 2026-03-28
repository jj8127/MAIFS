# ARV End-to-End Raspberry Pi 5 CPU

이 디렉토리는 Raspberry Pi 5 CPU만 사용해 ARV 전체 파이프라인 종단간 지연을 측정할 때 사용한다.

## 환경 준비

```bash
cd rpi5_cpu_only
bash setup_env.sh
```

## 실행

```bash
cd ..
bash run_arv_e2e_benchmark.sh cpu
```

또는

```bash
cd rpi5_cpu_only
bash run_benchmark.sh
```
