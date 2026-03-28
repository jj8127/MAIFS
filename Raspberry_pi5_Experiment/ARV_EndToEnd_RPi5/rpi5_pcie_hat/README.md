# ARV End-to-End Raspberry Pi 5 + PCIe HAT

이 디렉토리는 Raspberry Pi 5 + PCIe 연결 HAT 환경에서 ARV 전체 파이프라인 종단간 지연을 측정할 때 사용한다.

## 환경 준비

```bash
cd rpi5_pcie_hat
bash setup_env.sh
```

추가 조건:

- `lspci`에서 Apex 장치가 보여야 한다.
- `/dev/apex*`가 존재해야 한다.
- 시스템에 `libedgetpu` 런타임이 설치되어 있어야 한다.

## 실행

```bash
cd ..
bash run_arv_e2e_benchmark.sh pcie-hat
```
