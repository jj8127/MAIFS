# MAIFS - Multi-Agent Image Forensic System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 다중 AI 에이전트 기반 이미지 포렌식 시스템

MAIFS는 4개의 전문가 AI 에이전트가 협력하여 이미지의 진위 여부를 판별하는 이미지 포렌식 시스템입니다.
COBRA 합의 알고리즘과 다중 라운드 토론을 통해 판정 근거를 정리하고 설명 가능한 결과를 제공합니다.

**현재 상태**: v0.6.0 - 핵심 구현 완료, 161개 테스트 통과 (10개 스킵), LLM & 토론 시스템 구현 완료

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

> **전체 문서 목록**: [docs/README.md](docs/README.md)

| 문서 | 설명 |
|------|------|
| [PROGRESS_REPORT.md](PROGRESS_REPORT.md) | 현재 진행 상황 (최신) |
| [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md) | 빠른 시작 가이드 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 시스템 아키텍처 |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | API 레퍼런스 |

## 체크포인트 안내

OmniGuard 체크포인트는 용량이 커서 Git에 포함하지 않습니다.
다운로드 후 `OmniGuard-main/checkpoint/`에 배치하세요. 자세한 경로와 검증 결과는
[docs/reports/CHECKPOINT_VALIDATION_REPORT.md](docs/reports/CHECKPOINT_VALIDATION_REPORT.md)를 참고하세요.

## 프로젝트 구조

```
MAIFS/
├── src/
│   ├── tools/       # 분석 도구 (FFT, PRNU, OmniGuard, ViT)
│   ├── agents/      # 전문가 에이전트
│   ├── consensus/   # COBRA 합의 엔진
│   ├── debate/      # 토론 프로토콜
│   ├── llm/         # LLM 통합 (Claude API)
│   ├── knowledge/   # 도메인 지식 (논문 기반)
│   └── maifs.py     # 메인 시스템
├── docs/
│   ├── design/      # 설계 문서
│   ├── guides/      # 가이드
│   ├── research/    # 연구 자료
│   └── reports/     # 보고서
├── tests/           # 107 테스트
├── examples/        # 예제 코드
└── main.py          # CLI 진입점
```

## 라이선스

MIT License
