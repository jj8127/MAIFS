# MAIFS

`MAIFS`는 두 편의 연구를 재현하고 검토하는 데 필요한 자산만 남긴 공개용 경량 저장소입니다. 무거운 원시 데이터, 대형 학습 체크포인트, 외부 서브모듈 전체는 제외하고, 논문 수치와 Raspberry Pi 5 실측을 다시 확인할 수 있는 최소한의 번들만 유지합니다.

## 연구 축

- `DAAC`
  - 서버 환경의 4-agent disagreement-aware consensus 연구
  - 최종 논문: `docs/research/papers/DAAC_final_20260318.pdf`
  - 재현 번들: `Server_Reproduction/DAAC/`
- `PAR`
  - 경량 2단계 결합/수정 기반 이미지 위변조 탐지 연구
  - 최종 논문: `docs/research/papers/PAR_final_20260406.pdf`
  - 저장소 내부 구현/결과물의 역사적 이름은 `ARV`로 유지
  - 재현 번들: `Server_Reproduction/ARV/`
  - 온디바이스 실측: `Raspberry_pi5_Experiment/`

## 현재 남겨 둔 핵심 자산

- `Server_Reproduction/DAAC/`
  - DAAC 재현용 스크립트, 최소 prediction cache, 결과 JSON
- `Server_Reproduction/ARV/`
  - PAR 연구선의 서버 재현 번들
  - 스크립트/결과 파일 이름은 기존 실험명 `ARV`를 사용
- `Raspberry_pi5_Experiment/ICWMV/`
  - 1단계 경량 결합 구조의 Raspberry Pi 5 지연 측정
- `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/`
  - stage-2까지 포함한 종단간 Raspberry Pi 5 측정
- `Raspberry_pi5_Experiment/common/`
  - Pi 측정에서 공용으로 쓰는 경량 모델과 추론 스크립트
- `docs/research/papers/`
  - 최종 PDF, 확장 초안, 그림, 공유용 결과 archive
- `docs/research/RPi5_EXPERIMENT_GUIDE.md`
  - Raspberry Pi 5 재측정 절차
- `envs/`
  - 원 작업 저장소에서 분리한 conda 환경 명세

## 논문과 디렉토리 매핑

- `DAAC_final_20260318.pdf`
  - 핵심 구현/결과: `Server_Reproduction/DAAC/`
- `PAR_final_20260406.pdf`
  - 서버 실험 재현: `Server_Reproduction/ARV/`
  - 온디바이스 실측: `Raspberry_pi5_Experiment/`
- `PAPER_DRAFT_ARV_v3.md`
  - PAR 논문선의 확장 초안
  - 짧은 최종 PDF에 다 담지 못한 표/ablation/설명 보조 문서
- `PAPER_DRAFT_ARV_v3_SHARED_RESULTS.tar.gz`
  - 확장 초안 본문 수치에 대응하는 공유용 결과 묶음

## 빠른 시작

### DAAC 서버 재현

```bash
cd Server_Reproduction/DAAC
pip install -r requirements_daac.txt
python3 run_daac_retrain_lightweight.py
```

### PAR/ARV 서버 재현

```bash
cd Server_Reproduction/ARV
pip install -r requirements_arv.txt
python3 run_arv_experiment.py
```

추가 분석은 아래 엔트리포인트를 사용합니다.

- `python3 run_arv_backbone_transfer.py`
- `python3 run_arv_strong_backbone_repeats.py`
- `python3 run_arv_support_analyses.py`

### Raspberry Pi 5 측정

1단계 경량 결합 지연 측정:

```bash
cd Raspberry_pi5_Experiment/ICWMV
bash run_rpi5_latency_benchmark.sh all /path/to/test_image.jpg
```

종단간 ARV 번들 측정:

```bash
cd Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5
bash run_arv_e2e_benchmark.sh all
```

자세한 환경 구성과 재측정 절차는 `docs/research/RPi5_EXPERIMENT_GUIDE.md`를 참고합니다.

## 공개 범위

이 저장소는 논문 재현과 연구 근거 공유를 위한 공개본입니다.

- raw dataset과 대용량 dataset zip은 포함하지 않습니다.
- 대형 학습 weight와 외부 서브모듈 전체는 포함하지 않습니다.
- 논문 수치를 다시 추적할 수 있도록 최소 prediction cache, 결과 JSON/JSONL, stage-2 저장 모델, benchmark summary만 남깁니다.
- 최신 짧은 논문 이름은 `PAR`이지만, 저장소 내부 폴더/스크립트/결과물은 기존 실험명 `ARV`를 유지합니다.
