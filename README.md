# MAIFS

공개용 `MAIFS` 저장소는 두 연구 축만 남긴 경량 재현 저장소입니다.

- `DAAC`
  - 서버 환경의 disagreement-aware consensus 재현 번들
- `ARV (v3)`
  - Lightweight Image Forgery Detection via Asymmetric Risk-aware Veto 논문 초안과 재현 자산

## 저장소 구조

- `docs/research/papers/`
  - 현재 공개 기준 논문 문서와 그림 자산
- `Server_Reproduction/DAAC/`
  - DAAC self-contained 재현 번들
- `Server_Reproduction/ARV/`
  - ARV self-contained 재현 번들
- `Raspberry_pi5_Experiment/`
  - Raspberry Pi 5 측정 허브
  - `ICWMV/`: 1단계 경량 결합 지연 측정
  - `ARV_EndToEnd_RPi5/`: ARV 종단간 지연 측정

## 현재 메인 문서

- ARV v3 초안: `docs/research/papers/PAPER_DRAFT_ARV_v3.md`
- ARV 구조도 프롬프트: `docs/research/papers/ARV_ARCHITECTURE_DIAGRAM_PROMPT.md`
- DAAC 연구 계획: `docs/research/DAAC_RESEARCH_PLAN.md`

## 빠른 시작

### ARV 서버 재현

```bash
cd Server_Reproduction/ARV
python3 run_arv_experiment.py
```

### DAAC 서버 재현

```bash
cd Server_Reproduction/DAAC
python3 run_daac_retrain_lightweight.py
```

### Raspberry Pi 5 ICWMV 측정

```bash
cd Raspberry_pi5_Experiment/ICWMV
bash run_rpi5_latency_benchmark.sh all /path/to/test_image.jpg
```

### Raspberry Pi 5 ARV 종단간 측정

```bash
cd Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5
bash run_arv_e2e_benchmark.sh all
```

## 공개 범위

이 저장소는 논문 재현과 연구 공유를 위한 공개본입니다.

- raw dataset은 포함하지 않습니다.
- 대형 학습 weight와 외부 서브모듈은 포함하지 않습니다.
- Raspberry Pi 측정에 필요한 경량 모델과 결과 JSON/JSONL만 포함합니다.
