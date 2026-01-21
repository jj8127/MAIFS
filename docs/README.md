# MAIFS - Multi-Agent Image Forensic System

<p align="center">
  <strong>다중 에이전트 기반 이미지 포렌식 시스템</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Overview

MAIFS는 다중 AI 에이전트가 협력하여 이미지의 진위 여부를 판별하는 포렌식 시스템입니다.

기존 단일 모델 방식의 한계를 극복하고, 여러 전문가 에이전트가 각자의 분석 결과를 토론하고 합의에 도달하여 더 정확하고 설명 가능한 판정을 내립니다.

### Key Capabilities

| 기능 | 설명 |
|------|------|
| **다중 전문가 분석** | 4개의 전문 포렌식 에이전트가 독립적으로 분석 |
| **COBRA 합의** | 신뢰도 기반 가중 합의 알고리즘 (RoT, DRWA, AVGA) |
| **자동 토론** | 의견 불일치 시 에이전트 간 토론으로 합의 도출 |
| **설명 가능성** | 각 판정에 대한 상세 근거 및 증거 제시 |

---

## Features

### 🔬 4개 전문가 에이전트

1. **Frequency Agent** - FFT 기반 주파수 스펙트럼 분석
   - GAN/Diffusion 모델의 격자 아티팩트 탐지
   - 고주파 영역 이상 패턴 분석

2. **Noise Agent** - SRM/PRNU 기반 노이즈 분석
   - 카메라 센서 고유 노이즈 패턴 탐지
   - AI 생성 이미지의 노이즈 특성 분석

3. **Watermark Agent** - HiNet 기반 워터마크 분석
   - 비가시성 워터마크 탐지 및 추출
   - 이미지 무결성 검증

4. **Spatial Agent** - ViT 기반 공간 분석
   - 픽셀 수준 조작 영역 탐지
   - 조작 마스크 생성

### 🤝 COBRA 합의 알고리즘

- **RoT (Root-of-Trust)**: 신뢰/비신뢰 코호트 분리 집계
- **DRWA (Dynamic Reliability Weighted Aggregation)**: 동적 가중치 조정
- **AVGA (Adaptive Variance-Guided Attention)**: 분산 기반 어텐션

### 💬 다중 에이전트 토론

- MAD-Sherlock 기반 토론 프로토콜
- 동기/비동기/구조화 토론 모드
- 자동 수렴 감지 및 합의 도출

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 1.12+
- CUDA 11.0+ (GPU 사용 시)

### Install

```bash
# 저장소 클론
git clone https://github.com/jj8127/MAIFS.git
cd MAIFS

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 개발 모드 설치
pip install -e .
```

### Model Checkpoints

OmniGuard 체크포인트 다운로드:
- [PKU Disk](https://disk.pku.edu.cn/link/AAB048898581E047DE9519CE140F991B3A)
- [Google Drive](https://drive.google.com/file/d/1khdBDUDIRIhPIKlV0ictcbTdWLh-WFY_/view)

```bash
# 체크포인트 폴더에 배치
mkdir -p OmniGuard-main/checkpoint
cp *.pth OmniGuard-main/checkpoint/
```

---

## Quick Start

### CLI 사용

```bash
# 이미지 분석
python main.py analyze image.jpg

# 토론 비활성화
python main.py analyze image.jpg --no-debate

# 보고서 저장
python main.py analyze image.jpg --output report.json

# 합의 알고리즘 선택
python main.py analyze image.jpg --algorithm avga
```

### Python API

```python
from src.maifs import MAIFS

# MAIFS 인스턴스 생성
maifs = MAIFS(
    enable_debate=True,
    consensus_algorithm="drwa"
)

# 이미지 분석
result = maifs.analyze("path/to/image.jpg")

# 결과 확인
print(f"판정: {result.verdict}")
print(f"신뢰도: {result.confidence:.1%}")
print(result.detailed_report)
```

### Web UI

```bash
# Gradio 서버 실행
python main.py server --port 7860

# 브라우저에서 접속
# http://localhost:7860
```

---

## Architecture

```
MAIFS/
├── src/
│   ├── tools/           # 분석 도구 (FFT, PRNU, HiNet, ViT)
│   ├── agents/          # 전문가 에이전트
│   ├── consensus/       # COBRA 합의 엔진
│   ├── debate/          # 토론 프로토콜
│   └── maifs.py         # 메인 시스템
├── configs/             # 설정 파일
├── docs/                # 문서
└── tests/               # 테스트
```

### System Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         입력 이미지                          │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
    ┌───────────────────────┼───────────────────┐
    ▼                       ▼                   ▼
┌─────────┐           ┌─────────┐         ┌─────────┐
│Frequency│           │  Noise  │         │Watermark│    ← 전문가 분석
│  Agent  │           │  Agent  │         │  Agent  │
└────┬────┘           └────┬────┘         └────┬────┘
     │                     │                   │
     └──────────┬──────────┴───────────────────┘
                ▼
        ┌───────────────┐
        │ COBRA 합의    │    ← 합의 도출
        └───────┬───────┘
                │
        불일치? ├─────────┐
                │         ▼
                │  ┌─────────────┐
                │  │ 토론 챔버   │    ← 의견 조율
                │  └──────┬──────┘
                │         │
                └────┬────┘
                     ▼
            ┌─────────────────┐
            │   최종 판정     │
            └─────────────────┘
```

---

## API Reference

### MAIFS Class

```python
class MAIFS:
    """다중 에이전트 이미지 포렌식 시스템"""

    def __init__(
        self,
        enable_debate: bool = True,
        debate_threshold: float = 0.3,
        consensus_algorithm: str = "drwa",
        device: str = "cuda"
    ):
        """
        Args:
            enable_debate: 토론 기능 활성화
            debate_threshold: 토론 개시 임계값
            consensus_algorithm: 합의 알고리즘 ("rot", "drwa", "avga")
            device: 연산 디바이스
        """

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        include_debate: Optional[bool] = None,
        save_report: Optional[Path] = None
    ) -> MAIFSResult:
        """이미지 분석 실행"""
```

### MAIFSResult Class

```python
@dataclass
class MAIFSResult:
    verdict: Verdict           # 최종 판정
    confidence: float          # 신뢰도 (0.0 ~ 1.0)
    summary: str               # 요약
    detailed_report: str       # 상세 보고서
    agent_responses: Dict      # 에이전트별 응답
    consensus_result: ...      # 합의 결과
    debate_result: ...         # 토론 결과 (있는 경우)
```

### Verdict Enum

```python
class Verdict(Enum):
    AUTHENTIC = "authentic"          # 원본 이미지
    MANIPULATED = "manipulated"      # 조작된 이미지
    AI_GENERATED = "ai_generated"    # AI 생성 이미지
    UNCERTAIN = "uncertain"          # 판단 불가
```

---

## Contributing

프로젝트 기여에 관심을 가져주셔서 감사합니다!

자세한 기여 가이드는 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

### Quick Contribution Guide

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{maifs2025,
  title = {MAIFS: Multi-Agent Image Forensic System},
  author = {MAIFS Contributors},
  year = {2025},
  url = {https://github.com/jj8127/MAIFS}
}
```

---

## Acknowledgments

- OmniGuard: HiNet 및 ViT 모델 기반
- AIFo: 에이전트 기반 포렌식 프레임워크 참조
- COBRA: 합의 알고리즘 논문 기반
- MAD-Sherlock: 다중 에이전트 토론 프로토콜 참조
