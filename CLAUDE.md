# MAIFS Project Guide

> 이 문서는 AI agent가 세션 시작 시 참조하는 프로젝트 가이드입니다.
> 모든 작업 전에 이 문서를 읽고 프로젝트 맥락을 파악하세요.

## 1. 프로젝트 개요

**MAIFS (Multi-Agent Image Forensic System)** 는 4개의 전문가 AI 에이전트가 협력하여 이미지의 진위(실제/조작/AI 생성)를 판별하는 시스템입니다.

- **언어**: Python 3.9+
- **프레임워크**: PyTorch, Gradio (UI)
- **라이선스**: MIT
- **테스트**: pytest (171 collected, 158 passed, 10 skipped, 3 환경 의존 실패)
- **가상환경**: `.venv-qwen/` (실행: `.venv-qwen/bin/python`)

## 2. 아키텍처 (4계층)

```
[Gradio UI]  app.py
     │
[Orchestration]  src/maifs.py → src/agents/manager_agent.py
     │
[Consensus/Debate]  src/consensus/cobra.py, src/debate/
     │
[Agent Layer]  src/agents/specialist_agents.py
     │
[Tool Layer]  src/tools/*.py → External Models
```

### 2.1 4개 전문가 에이전트

| Agent | Tool | 분석 방식 | 역할 |
|-------|------|----------|------|
| FrequencyAgent | FrequencyAnalysisTool | FFT 스펙트럼 분석 | GAN 업샘플링 아티팩트 탐지 |
| NoiseAgent | NoiseAnalysisTool | PRNU/SRM 센서 노이즈 | 카메라 진위 검증 |
| **FatFormerAgent** | **FatFormerTool** | CLIP ViT-L/14 + DWT | AI 생성 이미지 탐지 |
| SpatialAgent | SpatialAnalysisTool | ViT 기반 조작 탐지 | 부분 조작 영역 검출 |

> **중요**: WatermarkTool/WatermarkAgent는 FatFormerTool/FatFormerAgent로 완전히 교체되었음 (2026-02-12). 코드에 watermark 참조가 남아있으면 안 됨.

### 2.2 합의 메커니즘

- **COBRA**: RoT(Root of Trust), DRWA(Dynamic Reweighting), AVGA(Attention-based) 3개 알고리즘
- **Trust scores**: `frequency: 0.85, noise: 0.80, fatformer: 0.85, spatial: 0.85`
- **DebateChamber**: 에이전트 간 불일치 시 토론 프로토콜 (Sync/Async/Structured)

### 2.3 LLM 통합

- **SubAgentLLM**: Claude API 기반, 각 에이전트의 Tool 결과를 자연어로 해석
- **QwenClient**: vLLM 서버 기반 Qwen 32B, batch inference + guided JSON
- **QwenMAIFSAdapter**: Qwen ↔ MAIFS 브릿지
- **ClaudeClient**: Manager Agent용 최종 종합 분석

## 3. 디렉토리 구조

```
MAIFS/
├── CLAUDE.md                    ← 이 파일 (프로젝트 가이드)
├── app.py                       ← Gradio UI 진입점
├── main.py                      ← CLI 진입점
├── configs/
│   └── settings.py              ← 전역 설정 (경로, trust scores, 모델 설정)
├── src/
│   ├── maifs.py                 ← 메인 오케스트레이터
│   ├── tools/
│   │   ├── base_tool.py         ← BaseTool, Verdict, ToolResult 정의
│   │   ├── frequency_tool.py    ← FFT 분석
│   │   ├── noise_tool.py        ← PRNU/SRM 분석
│   │   ├── fatformer_tool.py    ← FatFormer CLIP+DWT 분석
│   │   └── spatial_tool.py      ← ViT 조작 탐지
│   ├── agents/
│   │   ├── base_agent.py        ← BaseAgent, AgentRole, AgentResponse
│   │   ├── specialist_agents.py ← 4개 전문가 에이전트 구현
│   │   └── manager_agent.py     ← ManagerAgent (최종 종합)
│   ├── consensus/
│   │   └── cobra.py             ← COBRA 합의 알고리즘
│   ├── debate/
│   │   ├── debate_chamber.py    ← 토론 관리
│   │   ├── debate_protocol.py   ← 프로토콜 기본 클래스
│   │   └── protocols.py         ← Sync/Async/Structured 구현
│   ├── llm/
│   │   ├── subagent_llm.py      ← AgentDomain, SubAgentLLM (Claude 기반)
│   │   ├── claude_client.py     ← Claude API 클라이언트
│   │   ├── qwen_client.py       ← AgentRole, QwenClient (vLLM)
│   │   └── qwen_maifs_adapter.py ← Qwen ↔ MAIFS 어댑터
│   └── knowledge/
│       ├── __init__.py           ← KnowledgeBase 클래스
│       ├── frequency_domain_knowledge.md
│       ├── noise_domain_knowledge.md
│       ├── fatformer_domain_knowledge.md
│       └── spatial_domain_knowledge.md
├── Integrated Submodules/
│   └── FatFormer/               ← FatFormer 모델 코드 (git clone)
│       ├── models/              ← CLIP + Forgery-Aware Adapter
│       ├── utils/               ← 데이터셋, 유틸리티
│       ├── pretrained/
│       │   └── ViT-L-14.pt     ← CLIP pretrained (890MB, 배치 완료)
│       └── checkpoint/
│           └── fatformer.pth    ← FatFormer fine-tuned (사용자가 배치 예정)
├── tests/                       ← pytest 테스트
├── docs/                        ← 문서
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── design/                  ← 설계 문서
│   ├── guides/                  ← 가이드
│   ├── research/                ← 연구 문서
│   │   └── DAAC_RESEARCH_PLAN.md ← DAAC 연구 계획 (진행 중)
│   └── reports/                 ← 리포트
├── experiments/                 ← DAAC Phase 1 실험 스크립트 + 결과
│   ├── configs/phase1.yaml     ← 실험 설정
│   ├── run_phase1.py           ← Phase 1 전체 파이프라인
│   └── results/phase1/         ← 실험 결과 (JSON, 리포트)
└── src/meta/                    ← DAAC 메타 분류기 모듈
    ├── simulator.py             ← Gaussian copula 에이전트 시뮬레이터
    ├── features.py              ← 43-dim 메타 특징 추출기
    ├── baselines.py             ← Majority Vote, COBRA 래퍼
    ├── trainer.py               ← LogReg, GBM, MLP 학습/추론
    ├── evaluate.py              ← 평가 지표 + 통계 검정
    └── ablation.py              ← A1~A6 ablation runner
```

## 4. 핵심 데이터 구조

### 4.1 Verdict (판정)
```python
class Verdict(Enum):
    AUTHENTIC = "authentic"         # 원본
    MANIPULATED = "manipulated"     # 부분 조작
    AI_GENERATED = "ai_generated"   # AI 생성
    UNCERTAIN = "uncertain"         # 판단 불가
```

### 4.2 ToolResult
```python
@dataclass
class ToolResult:
    verdict: Verdict
    confidence: float          # 0.0 ~ 1.0
    evidence: Dict[str, Any]   # 도구별 상세 증거
    explanation: str           # 자연어 설명
    processing_time: float     # 초
```

### 4.3 AgentResponse
```python
@dataclass
class AgentResponse:
    verdict: Verdict
    confidence: float
    reasoning: str
    evidence: Dict[str, Any]
    tool_result: ToolResult
    llm_interpretation: Optional[str]
```

## 5. 코딩 규칙

### 5.1 일반 원칙
- **한국어 docstring/주석** 사용 (코드 자체는 영어)
- **type hints** 필수
- **dataclass** 패턴 활용 (ToolResult, AgentResponse 등)
- **Enum** 으로 상태/유형 정의 (Verdict, AgentRole, AgentDomain)

### 5.2 새 Tool 추가 시
```python
class NewTool(BaseTool):
    def __init__(self):
        super().__init__(name="new_tool_name", description="...")

    def load_model(self) -> bool:
        # 모델 로드, 실패 시 self.model = None
        ...

    def analyze(self, image: np.ndarray) -> ToolResult:
        if self.model is None:
            return self._fallback_analysis(image)
        # 실제 분석
        ...

    def _fallback_analysis(self, image: np.ndarray) -> ToolResult:
        return ToolResult(
            verdict=Verdict.UNCERTAIN,
            confidence=0.3,
            evidence={...},
            explanation="모델 미로드 상태",
            processing_time=0.0
        )
```

### 5.3 새 Agent 추가 시
```python
class NewAgent(BaseAgent):
    def __init__(self, use_llm: bool = False):
        tool = NewTool()
        super().__init__(
            role=AgentRole.NEW_ROLE,
            tool=tool,
            domain=AgentDomain.NEW_DOMAIN,
            use_llm=use_llm
        )
```

### 5.4 Graceful Degradation 필수
- 모델 체크포인트 부재 시 → `Verdict.UNCERTAIN, confidence=0.3` 반환
- LLM API 미연결 시 → 규칙 기반 폴백 (`_fallback_interpret`)
- 외부 의존성 실패 시 → `try-except`로 감싸서 시스템 전체 중단 방지

### 5.5 Import 규칙
```python
# 상대 임포트 사용 (src/ 내부)
from ..tools.base_tool import BaseTool, Verdict, ToolResult
from ..agents.base_agent import BaseAgent, AgentRole

# 절대 임포트 (configs, 외부)
from configs.settings import config
```

### 5.6 테스트 규칙
- 테스트 파일: `tests/test_*.py`
- 실행: `.venv-qwen/bin/python -m pytest tests/ -v --tb=short`
- 새 Tool/Agent 추가 시 해당 테스트 클래스 필수 작성
- fallback 모드 테스트 포함 (체크포인트 없이도 통과해야 함)
- 실제 이미지 필요 테스트는 `@pytest.mark.skipif`로 조건부 실행

## 6. 문서화 규칙

### 6.1 문서 위치
| 유형 | 위치 |
|------|------|
| 프로젝트 가이드 (이 문서) | `CLAUDE.md` |
| 시스템 아키텍처 | `docs/ARCHITECTURE.md` |
| API 레퍼런스 | `docs/API_REFERENCE.md` |
| 설계 결정 | `docs/design/*.md` |
| 연구/실험 계획 | `docs/research/*.md` |
| 사용 가이드 | `docs/guides/*.md` |
| 도메인 지식 | `src/knowledge/*.md` |

### 6.2 문서 작성 시 원칙
- 한국어 작성 (기술 용어는 영어 병기)
- 코드 예시 포함
- 변경 이력 상단에 날짜 기록
- 다른 문서와의 링크 명시

### 6.3 변경 사항 기록
중요한 아키텍처 변경은 이 문서의 Section 8 (변경 이력)에 반드시 기록할 것.

## 7. 현재 진행 상황

### 7.1 완료된 작업
- [x] Phase 1~4: 핵심 시스템 (Tool, Agent, COBRA, Debate)
- [x] Phase 5: Qwen vLLM 통합
- [x] WatermarkTool → FatFormerTool 교체 (2026-02-12)
  - FatFormer (CVPR 2024): CLIP ViT-L/14 + Forgery-Aware Adapter
  - 모델 코드: `Integrated Submodules/FatFormer/`
  - Pretrained weights: `pretrained/ViT-L-14.pt` (890MB, 배치 완료)
  - Fine-tuned checkpoint: `checkpoint/fatformer.pth` (사용자 배치 예정)

### 7.2 완료: DAAC Phase 1 — Disagreement Pattern Analysis (Path B)
**Disagreement-Aware Adaptive Consensus** — 메타 분류기 기반 합의 개선

핵심 가설: "에이전트 간 불일치 패턴 자체가 조작 유형을 암시하는 탐지 신호"

상세 계획: `docs/research/DAAC_RESEARCH_PLAN.md`

#### Phase 1 구현 완료 (2026-02-12)

```
src/meta/
├── __init__.py          # 패키지 exports
├── simulator.py         # Gaussian copula 기반 합성 에이전트 출력 생성
├── features.py          # 43-dim 메타 특징 추출 + ablation configs
├── baselines.py         # Majority Vote, COBRA 베이스라인
├── trainer.py           # LogReg, GradientBoosting, MLP 메타 분류기
├── evaluate.py          # Macro-F1, AUROC, ECE, McNemar, Bootstrap CI
└── ablation.py          # A1~A6 ablation runner

experiments/
├── configs/phase1.yaml  # 실험 설정
├── run_phase1.py        # Phase 1 전체 파이프라인
└── results/phase1/      # 실험 결과 (JSON + 리포트)
```

#### Phase 1 실험 결과 (시뮬레이션, Path B)

| 방법 | Macro-F1 | AUROC |
|------|----------|-------|
| Majority Vote | 0.9467 | — |
| COBRA (기존) | 0.9787 | 0.9937 |
| **GBM (43-dim)** | **0.9949** | **1.0000** |
| MLP (43-dim) | 0.9908 | 0.9996 |

Go/No-Go: **3개 조건 모두 PASS → Phase 2 착수 가능**
- C1: A5(Full) > COBRA, F1 +0.016, McNemar p=0.0004 ✓
- C2: 교차 데이터셋 F1 drop 0.51%p < 5%p ✓
- C3: A3(disagreement only) F1=0.641 > random+0.05=0.383 ✓

주요 발견:
- A4 (verdict+confidence, 20-dim) ≈ A5 (full, 43-dim) → 시뮬레이션 한계 가능성
- 특징 중요도 Top: `fatformer_verdict_ai_generated`, `spatial_verdict_manipulated`
- 실데이터(Path A)에서 disagreement 특징 기여도 재검증 필요

실행: `.venv-qwen/bin/python experiments/run_phase1.py`

### 7.3 진행 예정: DAAC Phase 2 — Adaptive Routing
- Meta-Router Network: 이미지 특성 → 에이전트 가중치 동적 생성
- Phase 1 Go 판정 완료 → 착수 가능

### 7.4 미래 계획
- Phase 2: Adaptive Routing (Meta-Router Network)
- Phase 1-A: 실데이터 검증 (FatFormer/Spatial 체크포인트 확보 시)
- Phase 3: 벤치마크 + 논문 작성

## 8. 변경 이력

| 날짜 | 변경 내용 | 영향 범위 |
|------|----------|----------|
| 2026-02-12 | DAAC Phase 1 구현 + 실험 완료 (Path B) | src/meta/, experiments/, CLAUDE.md |
| 2026-02-12 | 코드베이스 클린업 (watermark 잔여 참조 제거, exif_tool 삭제) | scripts/, examples/, tests/, docs/ |
| 2026-02-12 | WatermarkTool → FatFormerTool 전면 교체 | tools, agents, llm, knowledge, configs, tests, app.py |
| 2026-02-12 | DAAC 연구 계획 수립 | docs/research/DAAC_RESEARCH_PLAN.md |
| 2026-02-12 | settings.py path.exists() PermissionError 수정 | configs/settings.py |
| 2026-01-24 | Qwen vLLM 통합 완료 (v0.6.0) | src/llm/qwen_*.py, tests/ |

## 9. 자주 발생하는 문제

### 9.1 "File has not been read yet" 오류
Edit 도구 사용 전에 반드시 Read로 파일을 읽어야 합니다.

### 9.2 PermissionError: `/root/Desktop/...`
`configs/settings.py`의 경로 탐색에서 발생. `try-except (PermissionError, OSError)` 로 이미 수정됨.

### 9.3 pytest 실행 방법
```bash
/home/dsu/Desktop/MAIFS/.venv-qwen/bin/python -m pytest tests/ -v --tb=short
```
- `python` 또는 `python3` 직접 호출 불가 (시스템 python에 pytest 없음)
- 반드시 `.venv-qwen/bin/python` 사용

### 9.4 3개 테스트 항상 실패
`test_checkpoint_loading.py`의 `TestCheckpointAvailability` 3개 테스트는 OmniGuard 체크포인트 디렉토리가 없어서 실패. 환경 의존 이슈이며 코드 결함이 아님.

### 9.5 Enum 불일치 주의
`AgentRole` 과 `AgentDomain` 은 별도 파일에 정의되어 있음:
- `AgentRole`: `src/agents/base_agent.py` (FREQUENCY, NOISE, **FATFORMER**, SPATIAL)
- `AgentDomain`: `src/llm/subagent_llm.py` (FREQUENCY, NOISE, **FATFORMER**, SPATIAL)
- `AgentRole` (Qwen용): `src/llm/qwen_client.py` (FREQUENCY, NOISE, **FATFORMER**, SPATIAL, MANAGER)

모두 `WATERMARK`가 아닌 `FATFORMER`여야 함.

## 10. 핵심 참조 파일 (빠른 탐색용)

| 목적 | 파일 |
|------|------|
| 시스템 진입점 | `src/maifs.py` |
| Tool 기본 클래스 | `src/tools/base_tool.py` |
| Agent 기본 클래스 | `src/agents/base_agent.py` |
| 전문가 에이전트 구현 | `src/agents/specialist_agents.py` |
| 합의 알고리즘 | `src/consensus/cobra.py` |
| 설정 | `configs/settings.py` |
| FatFormer Tool | `src/tools/fatformer_tool.py` |
| DAAC 연구 계획 | `docs/research/DAAC_RESEARCH_PLAN.md` |
| DAAC 메타 분류기 | `src/meta/` |
| Phase 1 실험 | `experiments/run_phase1.py` |
| Phase 1 설정 | `experiments/configs/phase1.yaml` |
| Phase 1 결과 | `experiments/results/phase1/` |
| 테스트 | `tests/test_*.py` |
