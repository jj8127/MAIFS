# MAIFS

`MAIFS`는 두 편의 연구를 다시 읽고, 핵심 결과를 검증하고, 필요한 실험을 다시 돌려볼 수 있게 정리한 공개용 연구 저장소입니다. 이 저장소는 raw image dataset, 대형 weight, 외부 서브모듈 전체를 포함하지는 않지만, 논문 PDF, 대표 figure, 결과 캐시, 재현 스크립트, Raspberry Pi 5 측정 자산까지 한곳에서 확인할 수 있도록 구성했습니다.

## 한눈에 보기

| 연구 | 무엇을 다루는가 | 대표 성과 | 논문 |
| --- | --- | --- | --- |
| `DAAC` | 4-agent disagreement-aware consensus를 이용한 서버 측 이미지 위변조 판별 | 반복 검증에서 `DAAC-GBM 0.8613 ± 0.0148` vs `COBRA 0.2664 ± 0.0086` | [DAAC_final_20260318.pdf](docs/research/papers/DAAC_final_20260318.pdf) |
| `PAR` | 경량 2-model 구조와 ARV veto를 이용한 엣지 친화적 판정 수정 | avg macro-F1 `0.9581 -> 0.9652`, 역교정 `40 -> 18` | [PAR_final_20260406.pdf](docs/research/papers/PAR_final_20260406.pdf) |

## 논문과 대표 자료

- DAAC 최종 논문: [docs/research/papers/DAAC_final_20260318.pdf](docs/research/papers/DAAC_final_20260318.pdf)
- PAR 최종 논문: [docs/research/papers/PAR_final_20260406.pdf](docs/research/papers/PAR_final_20260406.pdf)
- PAR 확장 초안과 표/설명: [docs/research/papers/PAPER_DRAFT_ARV_v3.md](docs/research/papers/PAPER_DRAFT_ARV_v3.md)
- 공유 결과 archive: [docs/research/papers/PAPER_DRAFT_ARV_v3_SHARED_RESULTS.tar.gz](docs/research/papers/PAPER_DRAFT_ARV_v3_SHARED_RESULTS.tar.gz)
- 실험 재현 안내: [docs/research/EXPERIMENT_REPRO_RUNBOOK.md](docs/research/EXPERIMENT_REPRO_RUNBOOK.md)

## 대표 Figure

![PAR / ARV 2-stage pipeline](docs/research/papers/figures/그림1.png)

위 그림은 PAR 연구의 핵심 구조를 보여줍니다. `MobileNetV2` 기반 3-class 기본 분류기와 2-class 보조 모델을 먼저 역신뢰도 가중 결합하고, 그 뒤 `ARV`가 변경을 받아들일지 되돌릴지 선택적으로 결정합니다. 즉, 이 연구의 초점은 "더 많이 바꾸는 것"이 아니라 "유익한 수정은 살리고 해로운 수정은 막는 것"입니다.

## 핵심 성과

### DAAC

- 연구 질문: 개별 모델의 출력보다, 모델들 사이의 `불일치 패턴` 자체가 더 강한 일반화 신호가 될 수 있는가?
- 대표 결과: paper-style 반복 검증에서 `DAAC-GBM`은 avg macro-F1 `0.8613 ± 0.0148`을 기록했고, 비교 기준 `COBRA`는 `0.2664 ± 0.0086`에 머물렀습니다.
- 개별 에이전트 대비 이점: 가장 높은 단일 에이전트인 `frequency`도 `0.4284` 수준이어서, 메타 합의 계층의 기여가 뚜렷합니다.
- 포함된 자산: `dsA / dsB / dsC / dsD / OpenSDI / aigenproxy` 경로의 cache 기반 재현 결과와 paper summary JSON이 포함되어 있습니다.
- 대표 결과 파일:
  - [experiments/results/paper_final/paper_final_20260304_141508.json](experiments/results/paper_final/paper_final_20260304_141508.json)
  - [experiments/results/agent_eval/agent_eval_paper_final_20260305_053951.json](experiments/results/agent_eval/agent_eval_paper_final_20260305_053951.json)

### PAR / ARV

- 연구 질문: 무거운 다중 전문가 합의의 통찰을, 엣지 장치에서도 쓸 수 있는 경량 2-model 구조로 줄이면서도 성능 향상을 얻을 수 있는가?
- 메인 결과: 최종 시스템은 기본 분류기 단독 avg macro-F1 `0.9581`을 `0.9652`로 높였고, 1단계 결합에서 생기던 역교정을 `40건 -> 18건`으로 줄였습니다.
- 데이터셋별 의미: `OpenSDI`에서는 F1이 `0.9424 -> 0.9545`로 올라가, 역교정이 많이 발생하는 환경에서 ARV의 가치가 특히 크게 드러났습니다.
- 추가 검증: strong backbone 두 종류를 합친 반복 실험에서 역교정 총합이 평균 `30.2건`으로 줄어 `63.6%` 감소했습니다.
- 통계적 뒷받침: pooled 비교에서 macro-F1은 `0.9562 -> 0.9643`, paired bootstrap 95% CI는 `[0.0045, 0.0117]`, exact McNemar test는 `p=5.8×10^-5`였습니다.
- 대표 결과 문서:
  - [docs/research/papers/PAPER_DRAFT_ARV_v3.md](docs/research/papers/PAPER_DRAFT_ARV_v3.md)
  - [Raspberry_pi5_Experiment/docs/ARV_E2E_ALL_BACKENDS_EXPERIMENT_SUMMARY_20260329.md](Raspberry_pi5_Experiment/docs/ARV_E2E_ALL_BACKENDS_EXPERIMENT_SUMMARY_20260329.md)

### Raspberry Pi 5 온디바이스 성과

- 1단계 ICWMV 기준 평균 총 지연: `CPU 123.72 ms`, `USB Edge TPU 64.56 ms`, `PCIe HAT 54.17 ms`
- 고정 benchmark input 기준 전체 ARV bundle real-path 종단간 지연: `CPU 276.995 ms`, `USB 66.102 ms`, `PCIe 51.388 ms`
- 저장소에는 fixed-input benchmark뿐 아니라 실제 keep/revert 사례를 찾는 active probe workflow도 포함되어 있습니다.
- 관련 자산:
  - [Raspberry_pi5_Experiment/README.md](Raspberry_pi5_Experiment/README.md)
  - [Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/README.md](Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/README.md)
  - [docs/research/RPi5_EXPERIMENT_GUIDE.md](docs/research/RPi5_EXPERIMENT_GUIDE.md)

## 이 저장소에 포함된 것

- [experiments/](experiments/)
  - DAAC paper-style rerun, cross-dataset validation, PAR/ARV fusion, MobileCLIP 비교, export 및 보조 분석 스크립트
- [experiments/results/](experiments/results/)
  - DAAC paper summary, agent evaluation, cross-dataset cache, PAR/ARV backbone/specm/veto/repeat cache
- [src/](src/), [configs/](configs/), [deploy/](deploy/)
  - 원 논문 스크립트가 직접 참조하는 연구 코어와 설정
- [Server_Reproduction/DAAC/](Server_Reproduction/DAAC/)
  - 공개용 경량 DAAC 재현 번들
- [Server_Reproduction/ARV/](Server_Reproduction/ARV/)
  - 공개용 PAR 재현 번들과 alpha weighting sweep
- [Raspberry_pi5_Experiment/](Raspberry_pi5_Experiment/)
  - ICWMV 및 ARV의 Raspberry Pi 5 측정 자산
- [datasets/](datasets/)
  - raw image 대신 manifest, split provenance, subset list만 포함
- [docs/research/papers/](docs/research/papers/)
  - 최종 PDF, 초안, figure, 공유 결과 archive

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional-tools.txt
```

### DAAC cache-based rerun

```bash
python3 experiments/run_paper_final.py experiments/configs/paper_final_dsA.yaml
python3 experiments/run_agent_eval.py experiments/configs/paper_final.yaml
python3 experiments/run_cross_dataset_validation.py
```

### PAR / ARV server rerun

```bash
cd Server_Reproduction/ARV
pip install -r requirements_arv.txt
python3 run_arv_experiment.py
python3 run_icwmv_weighting_sweep.py
python3 run_arv_generalist.py
python3 run_arv_strong_backbone_repeats.py
python3 run_arv_support_analyses.py
```

### Raspberry Pi 5

```bash
cd Raspberry_pi5_Experiment/ICWMV
bash run_rpi5_latency_benchmark.sh all /path/to/test_image.jpg
```

```bash
cd Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5
bash run_arv_e2e_benchmark.sh all
bash run_arv_active_workflow.sh cpu
```

## 공개본의 경계

- raw dataset과 대용량 zip은 포함하지 않습니다.
- 대형 학습 weight와 외부 서브모듈 전체는 포함하지 않습니다.
- [scripts/prepare_sota_datasets.py](scripts/prepare_sota_datasets.py)와 [datasets/](datasets/) provenance는 포함하지만 실제 이미지 원본은 별도로 준비해야 합니다.
- DAAC의 원래 main base-set table이 사용한 `phase2_patha_scale500_gain_predictor/patha_agent_outputs_20260304_080157.jsonl` cache는 원본 아카이브에도 없었습니다.
- 그래서 저장소에는 당시 summary snapshot은 보존하되, 실제 공개 rerun 엔트리포인트는 현재 남아 있는 cache 기준 설정으로 안내합니다.
- 최신 논문 이름은 `PAR`이지만, 저장소 내부 폴더와 일부 스크립트, 결과물은 기존 실험명 `ARV`를 유지합니다.
