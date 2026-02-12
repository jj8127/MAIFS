# MAIFS 연구 로드맵

**작성일**: 2026-01-23
**현재 상태**: Phase 4 완료, Phase 5 시작 준비

---

## 📊 현재 진행 상황

### 완료된 단계
```
Phase 1: 핵심 기능 구현 ............ ████████████████████ 100% ✅
Phase 2: 테스트 및 검증 ........... ████████████████████ 100% ✅
Phase 3: OmniGuard 통합 ........... ████████████████████ 100% ✅
Phase 4: LLM 통합 ................. ████████████████████ 100% ✅
```

### Phase 4 완료 내용 (2026-01-23)
- ✅ Claude API 클라이언트 (`src/llm/claude_client.py`)
- ✅ Manager Agent LLM 통합 (`analyze_with_llm()`, `generate_human_report()`)
- ✅ Fallback 모드 (API 없이도 동작)
- ✅ 테스트 코드 (`tests/test_llm_integration.py` - 11개 테스트)
- ✅ 데모 스크립트 (`examples/llm_demo.py`)

### 가용 데이터셋
| 데이터셋 | 유형 | 위치 | 용도 |
|---------|------|------|------|
| HiNet | 스테가노그래피 | `HiNet-main/image/` | 워터마크 탐지 |
| GenImage | AI 생성 이미지 | `datasets/GenImage/` | AI 생성 탐지 |
| IMD2020 | 이미지 조작 | `datasets/IMD2020_subset/` | 조작 탐지 |
| TruFor | 테스트 세트 | `TruFor-main/test_docker/images/` | 검증 |

---

## 🎯 다음 연구 단계

### Phase 4: LLM 통합 (우선순위 1)

#### 4.1 Manager Agent 구현
```
목표: Claude API를 활용한 지능형 Manager Agent 구현
예상 난이도: 중간
```

**구현 항목**:
1. Claude API 연동
   - API 키 설정
   - 요청/응답 핸들링
   - 에러 처리

2. Manager Agent 설계
   - 전문가 에이전트 조율
   - 분석 결과 해석
   - 자연어 보고서 생성

3. 프롬프트 엔지니어링
   - 이미지 포렌식 전문 프롬프트
   - 근거 기반 추론 유도
   - 일관된 출력 형식

#### 4.2 구현 예시
```python
# src/agents/manager_agent.py

import anthropic
from typing import Dict, List
from ..tools.base_tool import ToolResult

class ManagerAgent:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """당신은 이미지 포렌식 전문가입니다.
        4명의 전문가 에이전트로부터 분석 결과를 받아 종합적인 판단을 내립니다.

        전문가 에이전트:
        1. Frequency Agent: FFT 기반 주파수 패턴 분석
        2. Noise Agent: PRNU 노이즈 분석
        3. Watermark Agent: 워터마크 탐지
        4. Spatial Agent: 공간 조작 탐지

        분석 결과를 검토하고 다음을 제공하세요:
        - 최종 판정 (AUTHENTIC, AI_GENERATED, MANIPULATED, UNCERTAIN)
        - 신뢰도 (0.0-1.0)
        - 상세 근거
        - 권장 추가 조사 사항
        """

    def analyze(self, agent_responses: Dict[str, ToolResult]) -> str:
        # 에이전트 응답을 프롬프트로 변환
        prompt = self._format_responses(agent_responses)

        # Claude API 호출
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text
```

---

### Phase 5: 성능 평가 (우선순위 2)

#### 5.1 데이터셋별 평가 계획

**GenImage 평가**:
```
목표: AI 생성 이미지 탐지 정확도 측정
메트릭: Accuracy, Precision, Recall, F1-Score, AUC-ROC
```

**IMD2020 평가**:
```
목표: 이미지 조작 탐지 정확도 측정
메트릭: Pixel-level F1, Image-level Accuracy
```

**HiNet 평가**:
```
목표: 스테가노그래피/워터마크 탐지 정확도
메트릭: Detection Rate, False Positive Rate
```

#### 5.2 평가 스크립트 구조
```python
# experiments/evaluate_genimage.py

from pathlib import Path
from src.maifs import MAIFS
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_on_genimage(data_dir: Path, maifs: MAIFS):
    results = []
    labels = []

    # Real 이미지
    for img_path in (data_dir / "real").glob("*.png"):
        result = maifs.analyze(img_path)
        results.append(result.verdict.value == "AUTHENTIC")
        labels.append(True)

    # AI 생성 이미지
    for img_path in (data_dir / "fake").glob("*.png"):
        result = maifs.analyze(img_path)
        results.append(result.verdict.value == "AI_GENERATED")
        labels.append(False)

    return {
        "accuracy": accuracy_score(labels, results),
        "f1": f1_score(labels, results),
    }
```

---

### Phase 6: 비교 벤치마크 (우선순위 3)

#### 6.1 비교 대상 방법론

| 방법론 | 논문 | 특징 |
|--------|------|------|
| CNNDetection | Wang et al., 2020 | CNN 기반 탐지 |
| Spec | Dzanic et al., 2020 | 스펙트럼 분석 |
| GramNet | Liu et al., 2020 | Gram 행렬 분석 |
| UnivFD | Ojha et al., 2023 | CLIP 기반 범용 탐지 |
| NPR | Tan et al., 2024 | 노이즈 패턴 분석 |

#### 6.2 벤치마크 프레임워크
```python
# experiments/benchmark.py

class BenchmarkFramework:
    def __init__(self):
        self.methods = {
            "MAIFS": MAIFSDetector(),
            "CNNDetection": CNNDetector(),
            "UnivFD": UnivFDDetector(),
            # ...
        }

    def run_benchmark(self, dataset_name: str, data_dir: Path):
        results = {}
        for name, detector in self.methods.items():
            results[name] = self.evaluate(detector, data_dir)
        return results
```

---

### Phase 7: 논문 준비 (우선순위 4)

#### 7.1 논문 구조 (예상)

```
1. Introduction
   - 문제 정의: AI 생성 이미지 탐지의 필요성
   - 기존 방법의 한계: 단일 분석기의 취약점
   - 제안 방법: 다중 에이전트 합의 기반 접근

2. Related Work
   - AI 생성 이미지 탐지
   - 이미지 조작 탐지
   - 다중 에이전트 시스템

3. Proposed Method
   - 시스템 아키텍처
   - 전문가 에이전트 설계
   - COBRA 합의 알고리즘
   - 토론 프로토콜

4. Experiments
   - 데이터셋 (GenImage, IMD2020, etc.)
   - 평가 메트릭
   - 비교 실험
   - Ablation Study

5. Results
   - 정량적 결과
   - 정성적 분석
   - 실패 사례 분석

6. Conclusion
   - 기여점 요약
   - 한계점 및 향후 연구
```

#### 7.2 예상 기여점

1. **다중 에이전트 합의 기반 이미지 포렌식**
   - 단일 탐지기 대비 강건성 향상
   - 다양한 조작 유형에 대한 범용성

2. **COBRA 합의 알고리즘**
   - 신뢰도 기반 동적 가중치
   - 불확실성 처리

3. **토론 기반 추론**
   - 에이전트 간 의견 불일치 해결
   - 해석 가능한 판단 과정

---

## 📋 즉시 실행 가능한 작업

### 1. LLM 통합 시작
```bash
# Claude API 테스트
python -c "
import anthropic
client = anthropic.Anthropic(api_key='YOUR_API_KEY')
response = client.messages.create(
    model='claude-sonnet-4-20250514',
    max_tokens=100,
    messages=[{'role': 'user', 'content': 'Hello'}]
)
print(response.content[0].text)
"
```

### 2. GenImage 데이터 확인
```bash
# GenImage 예제 확인
ls -la datasets/GenImage/Examples/
```

### 3. 실제 이미지 테스트
```bash
# HiNet steg 이미지로 테스트
python -c "
from src.maifs import MAIFS
maifs = MAIFS()
result = maifs.analyze('HiNet-main/image/steg/1')
print(result.to_json())
"
```

---

## 🔬 실험 설계

### 실험 1: AI 생성 이미지 탐지
```
데이터: GenImage 데이터셋
비교 대상: CNNDetection, UnivFD, NPR
메트릭: Accuracy, F1, AUC-ROC
```

### 실험 2: 이미지 조작 탐지
```
데이터: IMD2020, TruFor 테스트셋
비교 대상: ManTraNet, MVSS-Net, TruFor
메트릭: Pixel-F1, Image Accuracy
```

### 실험 3: 합의 알고리즘 비교
```
설정: RoT vs DRWA vs AVGA
데이터: 혼합 테스트셋
메트릭: Accuracy, 수렴 속도
```

### 실험 4: Ablation Study
```
설정: 에이전트 조합 변화
- 4 에이전트 (전체)
- 3 에이전트 (하나씩 제거)
- 2 에이전트 (조합)
메트릭: 정확도 변화
```

---

## 📅 예상 일정

| 단계 | 작업 | 예상 기간 |
|------|------|----------|
| Phase 4 | LLM 통합 | 1-2주 |
| Phase 5 | 성능 평가 | 1-2주 |
| Phase 6 | 벤치마크 | 1주 |
| Phase 7 | 논문 작성 | 2-4주 |

---

## 💡 추가 고려사항

### 연구 방향 선택지

**Option A: AI 생성 이미지 탐지 중심**
- GenImage 데이터셋 집중
- Diffusion 모델 생성 이미지 탐지
- 범용 탐지기와 비교

**Option B: 이미지 조작 탐지 중심**
- IMD2020, TruFor 데이터셋
- 픽셀 수준 조작 영역 탐지
- 조작 유형 분류

**Option C: 통합 접근 (권장)**
- AI 생성 + 조작 탐지 모두
- 다양한 시나리오 대응
- 범용 이미지 포렌식 시스템

---

## 📚 참고 자료

### 주요 논문
1. COBRA (2024) - 합의 알고리즘
2. AIFo (2024) - 에이전트 기반 포렌식
3. MAD-Sherlock (2024) - 다중 에이전트 토론
4. OmniGuard - 조작 위치 탐지 (ViT 기반)
5. HiNet (2021) - 스테가노그래피

### 데이터셋 논문
1. GenImage (2023) - AI 생성 이미지
2. IMD2020 (2020) - 이미지 조작 탐지

---

**다음 단계**: LLM 통합 또는 성능 평가 중 선택하여 진행

어떤 단계부터 시작하시겠습니까?

---

## 참고 프레임워크 비교

| 기능 | MAIFS (현재) | MAD-Sherlock | AIFo | Hybrid-Forensic |
|-----|------------|--------------|------|-----------------|
| 4-Branch 분석 | ✅ | ❌ | ✅ | ✅ |
| LLM 기반 추론 | ✅ 구현 완료 | ✅ | ❌ | ❌ |
| 토론 프로토콜 | ✅ 구현 완료 | ✅ | ✅ | ❌ |
| COBRA 전략 | ✅ | ❌ | ❌ | ✅ |
| Knowledge Base | ✅ | ❌ | ❌ | ❌ |
| 종료 메커니즘 | ✅ 5가지 | ✅ | ❌ | ❌ |

**MAIFS 강점**: LLM + Knowledge Base + 종료 메커니즘  
**보완 가능 영역**: Dempster-Shafer 불확실성 모델링, 재귀적 토론

## 연구 발전 옵션 (고급)

| 옵션 | 개념 | 복잡도 | 우선순위 |
|------|------|--------|---------|
| **Dempster-Shafer 통합** | 확률 대신 믿음+불확실성으로 판정 | 중 | 중 |
| **게임이론 최적화** | 토론을 Nash Equilibrium 게임으로 모델링 | 중 | 중 |
| **Devil's Advocate** | 다수 의견에 의도적 반박 에이전트 도입 | 중 | 중 |
| **재귀적 토론** | 이미지를 영역별로 분할하여 계층적 토론 | 높음 | 낮음 |
