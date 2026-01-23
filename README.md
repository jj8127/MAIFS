# MAIFS - Multi-Agent Image Forensic System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 다중 AI 에이전트 기반 이미지 포렌식 시스템

MAIFS는 4개의 전문가 AI 에이전트가 협력하여 이미지의 진위 여부를 판별하는 이미지 포렌식 시스템입니다.
COBRA 합의 알고리즘과 다중 라운드 토론을 통해 판정 근거를 정리하고 설명 가능한 결과를 제공합니다.

**현재 상태**: 핵심 기능 완료, 94/94 테스트 통과 (LLM 통합은 추후 단계)

## 주요 특징

- **다중 에이전트 분석**: 주파수, 노이즈, 워터마크, 공간 분석 전문가
- **COBRA 합의**: RoT, DRWA, AVGA 알고리즘 지원
- **자동 토론**: 의견 불일치 시 토론 프로토콜로 합의 수렴
- **설명 가능성**: 판정 근거와 증거를 구조화된 결과로 제공

## 빠른 시작

```bash
# 저장소 클론
git clone https://github.com/jj8127/MAIFS.git
cd MAIFS

# 의존성 설치
pip install -r requirements.txt

# CLI 사용
python main.py analyze path/to/image.jpg

# 웹 UI
python main.py server
```

## Python API

```python
from src.maifs import MAIFS

maifs = MAIFS()
result = maifs.analyze("path/to/image.jpg")

print(f"판정: {result.verdict}")
print(f"신뢰도: {result.confidence:.1%}")
```

## 문서

| 문서 | 내용 | 대상 |
|------|------|------|
| [README_root.md](README_root.md) | 프로젝트 가이드 (상태, 테스트, 구조) | 모두 |
| [QUICK_START.md](QUICK_START.md) | 30초 시작, 예제 | 신규 사용자 |
| [SYSTEM_STATUS.md](SYSTEM_STATUS.md) | 시스템 상태, 설정, 제한사항 | 모든 사용자 |
| [CHECKPOINT_VALIDATION_REPORT.md](CHECKPOINT_VALIDATION_REPORT.md) | 체크포인트 검증 보고서 | 개발자 |
| [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) | 변경 사항 정리 | 개발자 |
| [MAIFS_IMPLEMENTATION_PLAN.md](MAIFS_IMPLEMENTATION_PLAN.md) | 구현 계획 및 진행 상황 | 관리자 |
| [docs/CONDA_ENV_SNAPSHOTS.md](docs/CONDA_ENV_SNAPSHOTS.md) | conda 환경 스냅샷 및 smoke test | 개발자 |
| [docs/DATASETS.md](docs/DATASETS.md) | 데이터셋 다운로드/배치/실행/출력 위치 | 개발자/연구자 |
| [docs/README.md](docs/README.md) | 상세 소개 (Features/Architecture) | 모두 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 시스템 아키텍처 | 개발자 |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | API 문서 | 개발자 |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | 기여 가이드 | 기여자 |

## 체크포인트 안내

OmniGuard 체크포인트는 용량이 커서 Git에 포함하지 않습니다.
다운로드 후 `OmniGuard-main/checkpoint/`에 배치하세요. 자세한 경로와 검증 결과는
[CHECKPOINT_VALIDATION_REPORT.md](CHECKPOINT_VALIDATION_REPORT.md)와 [docs/README.md](docs/README.md)를 참고하세요.

## 프로젝트 구조

```
MAIFS/
├── src/
│   ├── tools/       # 분석 도구 (FFT, PRNU, HiNet, ViT)
│   ├── agents/      # 전문가 에이전트
│   ├── consensus/   # COBRA 합의 엔진
│   ├── debate/      # 토론 프로토콜
│   └── maifs.py     # 메인 시스템
├── configs/         # 설정
├── docs/            # 문서
├── tests/           # 테스트
├── main.py          # CLI 진입점
└── app.py           # Web UI
```

## 라이선스

MIT License
