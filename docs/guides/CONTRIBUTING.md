# Contributing to MAIFS

MAIFS 프로젝트에 기여해 주셔서 감사합니다! 이 문서는 AI 에이전트와 연구자 모두를 위한 기여 가이드라인입니다.

---

## Table of Contents

1. [시작하기](#시작하기)
2. [문서화 규칙](#문서화-규칙)
3. [코드 스타일 가이드](#코드-스타일-가이드)
4. [아키텍처 가이드](#아키텍처-가이드)
5. [AI 에이전트를 위한 가이드](#ai-에이전트를-위한-가이드)
6. [Pull Request 프로세스](#pull-request-프로세스)
7. [테스트 가이드](#테스트-가이드)

---

## 시작하기

### 개발 환경 설정

```bash
# 저장소 클론
git clone https://github.com/jj8127/MAIFS.git
cd MAIFS

# 개발 의존성 설치
pip install -r requirements.txt
pip install -e ".[dev]"

# Pre-commit 훅 설치
pre-commit install
```

### 프로젝트 구조 이해

```
MAIFS/
├── src/
│   ├── tools/           # 분석 도구 (BaseTool 상속)
│   │   ├── base_tool.py       # 도구 기본 클래스
│   │   ├── frequency_tool.py  # FFT 분석
│   │   ├── noise_tool.py      # 노이즈 분석
│   │   ├── watermark_tool.py  # 워터마크 분석
│   │   └── spatial_tool.py    # 공간 분석
│   │
│   ├── agents/          # 에이전트 (BaseAgent 상속)
│   │   ├── base_agent.py      # 에이전트 기본 클래스
│   │   ├── specialist_agents.py # 전문가 에이전트
│   │   └── manager_agent.py   # 관리자 에이전트
│   │
│   ├── consensus/       # 합의 알고리즘
│   │   └── cobra.py           # COBRA 구현
│   │
│   ├── debate/          # 토론 시스템
│   │   ├── protocols.py       # 토론 프로토콜
│   │   └── debate_chamber.py  # 토론 관리
│   │
│   └── maifs.py         # 메인 시스템
│
├── configs/             # 설정
├── docs/                # 문서
├── tests/               # 테스트
└── examples/            # 예제
```

---

## 문서화 규칙

### 1. Docstring 형식 (Google Style)

모든 모듈, 클래스, 함수에 Google Style docstring을 사용합니다.

```python
def analyze(
    self,
    image: np.ndarray,
    context: Optional[Dict] = None
) -> ToolResult:
    """
    이미지 분석을 수행합니다.

    주파수 도메인에서 FFT를 사용하여 GAN 생성 이미지의
    특징적인 격자 패턴을 탐지합니다.

    Args:
        image: RGB 이미지 배열 (H, W, 3), dtype=uint8 또는 float32.
            float32인 경우 [0, 1] 범위로 정규화되어 있어야 함.
        context: 추가 컨텍스트 정보. 다음 키를 포함할 수 있음:
            - 'metadata': 이미지 메타데이터
            - 'other_results': 다른 에이전트의 분석 결과

    Returns:
        ToolResult 객체:
            - verdict: 판정 결과 (Verdict enum)
            - confidence: 신뢰도 (0.0 ~ 1.0)
            - evidence: 증거 딕셔너리
            - explanation: 설명 문자열

    Raises:
        ValueError: image가 올바른 형식이 아닌 경우
        RuntimeError: 분석 중 치명적 오류 발생 시

    Examples:
        >>> tool = FrequencyAnalysisTool()
        >>> image = np.random.rand(256, 256, 3)
        >>> result = tool.analyze(image)
        >>> print(result.verdict)
        Verdict.AUTHENTIC

    Note:
        - GPU 사용 시 CUDA 메모리 약 500MB 필요
        - 입력 이미지는 최소 64x64 이상이어야 함
    """
```

### 2. 타입 힌트

모든 함수에 타입 힌트를 사용합니다.

```python
from typing import Dict, List, Optional, Union, Tuple, Any

def compute_consensus(
    responses: Dict[str, AgentResponse],
    trust_scores: Dict[str, float],
    algorithm: Optional[str] = None
) -> ConsensusResult:
    ...
```

### 3. 모듈 Docstring

각 모듈 상단에 모듈 설명을 포함합니다.

```python
"""
주파수 분석 도구

FFT 기반 주파수 스펙트럼 분석으로 AI 생성 이미지의 특징적 패턴을 탐지합니다.

주요 기능:
    - GAN 아티팩트 탐지 (격자 패턴)
    - 고주파 영역 이상 분석
    - Radial Energy Distribution 분석

References:
    - "Detecting GAN generated images using FFT" (Paper, 2020)
    - OmniGuard frequency analysis module

Author: Your Name
Created: 2025-01-21
"""
```

### 4. 인라인 주석

복잡한 로직에 대해 인라인 주석을 추가합니다.

```python
# COBRA DRWA 알고리즘: 분산이 낮은 에이전트에 높은 가중치 부여
# 수식: ω_t = w_t + ε * (1 - σ_t / σ_max)
variance_factor = 1 - (variance / (max_variance + 1e-10))
dynamic_weight = base_trust + self.epsilon * variance_factor
```

### 5. TODO/FIXME 주석

```python
# TODO(username): 성능 최적화 필요 - 현재 O(n²)
# FIXME: 엣지 케이스 처리 누락 - 빈 응답 시 오류
# NOTE: 이 함수는 Python 3.10+ 필요
# HACK: 임시 해결책, 다음 버전에서 리팩토링 예정
```

---

## 코드 스타일 가이드

### 1. Python 스타일

- **PEP 8** 준수
- **Black** 포매터 사용 (line-length=88)
- **isort** 임포트 정렬

```python
# 올바른 임포트 순서
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base_tool import BaseTool, ToolResult
from .utils import preprocess_image
```

### 2. 네이밍 컨벤션

| 유형 | 스타일 | 예시 |
|------|--------|------|
| 클래스 | PascalCase | `FrequencyAgent`, `ToolResult` |
| 함수/메서드 | snake_case | `analyze_image`, `compute_fft` |
| 변수 | snake_case | `image_array`, `trust_score` |
| 상수 | UPPER_SNAKE | `MAX_ROUNDS`, `DEFAULT_THRESHOLD` |
| Private | _prefix | `_internal_method`, `_cache` |
| 모듈 | snake_case | `frequency_tool.py` |

### 3. 클래스 구조

```python
class NewAgent(BaseAgent):
    """에이전트 설명"""

    # 클래스 상수
    DEFAULT_THRESHOLD = 0.5

    def __init__(self, config: Optional[Dict] = None):
        """초기화"""
        super().__init__(...)
        self._private_attr = None

    # Properties
    @property
    def public_property(self) -> str:
        return self._private_attr

    # Public methods
    def analyze(self, image: np.ndarray) -> AgentResponse:
        """공개 메서드"""
        pass

    # Private methods
    def _internal_process(self) -> None:
        """내부 메서드"""
        pass

    # Static/Class methods
    @staticmethod
    def utility_function() -> None:
        """유틸리티"""
        pass
```

---

## 아키텍처 가이드

### 새로운 Tool 추가

```python
# src/tools/my_new_tool.py

from .base_tool import BaseTool, ToolResult, Verdict

class MyNewTool(BaseTool):
    """
    새로운 분석 도구

    [도구 설명]
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(
            name="my_new_tool",
            description="도구 설명",
            device=device
        )

    def load_model(self) -> None:
        """모델 로드 로직"""
        if self._is_loaded:
            return
        # 모델 로드
        self._is_loaded = True

    def analyze(self, image: np.ndarray) -> ToolResult:
        """분석 로직"""
        # 구현
        return ToolResult(
            tool_name=self.name,
            verdict=Verdict.AUTHENTIC,
            confidence=0.9,
            evidence={...},
            explanation="..."
        )
```

### 새로운 Agent 추가

```python
# src/agents/my_new_agent.py

from .base_agent import BaseAgent, AgentRole, AgentResponse
from ..tools.my_new_tool import MyNewTool

class MyNewAgent(BaseAgent):
    """
    새로운 전문가 에이전트

    [역할 및 책임 설명]
    """

    def __init__(self, llm_model: Optional[str] = None):
        super().__init__(
            name="새로운 전문가",
            role=AgentRole.SPATIAL,  # 적절한 역할 선택
            description="에이전트 설명",
            llm_model=llm_model
        )
        self._tool = MyNewTool()
        self.register_tool(self._tool)

    def analyze(
        self,
        image: np.ndarray,
        context: Optional[Dict] = None
    ) -> AgentResponse:
        """분석 수행"""
        tool_result = self._tool(image)
        reasoning = self.generate_reasoning([tool_result], context)

        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            verdict=tool_result.verdict,
            confidence=tool_result.confidence,
            reasoning=reasoning,
            evidence=tool_result.evidence,
            tool_results=[tool_result]
        )

    def generate_reasoning(
        self,
        tool_results: List[ToolResult],
        context: Optional[Dict] = None
    ) -> str:
        """추론 생성"""
        # 결과 해석 로직
        return "분석 결과 설명..."
```

### 새로운 합의 알고리즘 추가

```python
# src/consensus/my_algorithm.py

from .cobra import ConsensusAlgorithm, ConsensusResult

class MyConsensusAlgorithm(ConsensusAlgorithm):
    """
    새로운 합의 알고리즘

    수식: [수학적 정의]
    """

    def __init__(self, param1: float = 0.5):
        self.param1 = param1

    def aggregate(
        self,
        responses: Dict[str, AgentResponse],
        trust_scores: Dict[str, float]
    ) -> ConsensusResult:
        """합의 집계"""
        # 구현
        return ConsensusResult(
            final_verdict=...,
            confidence=...,
            algorithm_used="my_algorithm",
            ...
        )
```

---

## AI 에이전트를 위한 가이드

### 🤖 AI Agent Integration Protocol

AI 에이전트(Claude, GPT 등)가 MAIFS와 상호작용할 때 따라야 할 프로토콜입니다.

#### 1. 컨텍스트 이해

```
MAIFS 시스템에서 작업할 때:
1. 먼저 관련 파일들을 읽어 현재 구조 파악
2. base_tool.py, base_agent.py의 인터페이스 확인
3. 기존 구현 패턴을 따라 일관성 유지
```

#### 2. 코드 생성 규칙

```python
# AI 에이전트가 코드 생성 시 반드시 포함해야 할 요소:

# 1. 완전한 타입 힌트
def my_function(param: str, optional: Optional[int] = None) -> Dict[str, Any]:
    ...

# 2. Google Style docstring
def my_function(...):
    """
    함수 설명.

    Args:
        param: 파라미터 설명
        optional: 선택적 파라미터 설명

    Returns:
        반환값 설명
    """

# 3. 에러 처리
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise
```

#### 3. 커밋 메시지 형식

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 포맷팅
- `refactor`: 리팩토링
- `test`: 테스트
- `chore`: 기타

예시:
```
feat(agents): add semantic analysis agent

- Implement SemanticAgent with VLM integration
- Add OCR-based text consistency check
- Support multi-language detection

Closes #123
```

#### 4. PR 템플릿

```markdown
## Summary
[변경 사항 요약]

## Changes
- [ ] 변경 1
- [ ] 변경 2

## Testing
- [ ] 단위 테스트 추가
- [ ] 통합 테스트 통과

## Documentation
- [ ] docstring 업데이트
- [ ] README 업데이트 (필요시)
```

---

## Pull Request 프로세스

### 1. 브랜치 전략

```
main           # 프로덕션 브랜치
└── develop    # 개발 브랜치
    ├── feature/xxx    # 기능 개발
    ├── fix/xxx        # 버그 수정
    └── docs/xxx       # 문서 작업
```

### 2. PR 체크리스트

- [ ] 코드가 스타일 가이드를 따름
- [ ] 모든 테스트 통과
- [ ] 새 기능에 대한 테스트 추가
- [ ] 문서 업데이트
- [ ] CHANGELOG 업데이트

### 3. 리뷰 프로세스

1. PR 생성 → 자동 CI 실행
2. 코드 리뷰 요청
3. 리뷰 피드백 반영
4. 승인 후 머지

---

## 테스트 가이드

### 테스트 구조

```
tests/
├── unit/
│   ├── test_tools.py
│   ├── test_agents.py
│   └── test_consensus.py
├── integration/
│   └── test_maifs.py
└── fixtures/
    └── sample_images/
```

### 테스트 작성

```python
# tests/unit/test_tools.py

import pytest
import numpy as np
from src.tools.frequency_tool import FrequencyAnalysisTool
from src.tools.base_tool import Verdict


class TestFrequencyTool:
    """FrequencyAnalysisTool 테스트"""

    @pytest.fixture
    def tool(self):
        """Tool 인스턴스"""
        return FrequencyAnalysisTool()

    @pytest.fixture
    def sample_image(self):
        """샘플 이미지"""
        return np.random.rand(256, 256, 3).astype(np.float32)

    def test_analyze_returns_tool_result(self, tool, sample_image):
        """analyze가 ToolResult를 반환하는지 확인"""
        result = tool.analyze(sample_image)

        assert result.tool_name == "frequency_analyzer"
        assert isinstance(result.verdict, Verdict)
        assert 0.0 <= result.confidence <= 1.0

    def test_analyze_with_invalid_input(self, tool):
        """잘못된 입력 처리 확인"""
        with pytest.raises(ValueError):
            tool.analyze(None)
```

### 테스트 실행

```bash
# 전체 테스트
pytest

# 특정 파일
pytest tests/unit/test_tools.py

# 커버리지 포함
pytest --cov=src --cov-report=html

# 마커로 필터링
pytest -m "not slow"
```

---

## 질문이 있으신가요?

- GitHub Issues에 질문 남기기
- 이메일: your-email@example.com

감사합니다! 🙏
