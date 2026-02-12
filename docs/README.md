# MAIFS 문서 인덱스

**Multi-Agent Image Forensic System — 전체 문서 목록**

> 프로젝트 개요: [../README.md](../README.md) | 진행 현황: [../PROGRESS_REPORT.md](../PROGRESS_REPORT.md)

---

## 문서 구조

```
docs/
├── README.md                    ← 이 파일 (전체 인덱스)
├── ARCHITECTURE.md              ← 시스템 아키텍처
├── API_REFERENCE.md             ← API 레퍼런스
│
├── design/                      ← 설계 결정사항
│   ├── DEBATE_SYSTEM_DESIGN.md  ← 토론 시스템 전체 (종료조건·예외처리 포함)
│   ├── AGENT_REASONING_IMPROVEMENT.md
│   └── GPU_ALLOCATION.md
│
├── guides/                      ← 사용 가이드
│   ├── QUICK_START.md
│   ├── RESEARCH_ROADMAP.md      ← 연구 방향 + 프레임워크 비교
│   └── CONTRIBUTING.md
│
├── research/                    ← 연구 자료
│   ├── TOOLS_AND_PAPERS.md      ← 툴별 관련 논문
│   ├── TOOL_THEORETICAL_FOUNDATIONS.md  ← 이론 기반 + 전문화 전략
│   ├── RESEARCH_REPORT_Frequency_Tool_Improvement.md
│   └── DATASETS.md
│
└── reports/                     ← 상태/검증 보고서
    ├── FALSE_POSITIVE_FIX_SUMMARY.md
    └── CONDA_ENV_SNAPSHOTS.md
```

---

## 역할별 시작점

| 역할 | 추천 순서 |
|------|-----------|
| **신규 개발자** | README → ARCHITECTURE → API_REFERENCE → guides/QUICK_START |
| **연구자** | PROGRESS_REPORT → guides/RESEARCH_ROADMAP → research/ |
| **설계 검토** | ARCHITECTURE → design/DEBATE_SYSTEM_DESIGN → design/GPU_ALLOCATION |
| **기여자** | guides/QUICK_START → guides/CONTRIBUTING |

---

## 핵심 문서

### 아키텍처 & API
- [ARCHITECTURE.md](ARCHITECTURE.md) — 컴포넌트 구조, 데이터 흐름
- [API_REFERENCE.md](API_REFERENCE.md) — 클래스/함수 레퍼런스

### 설계 (design/)
- [DEBATE_SYSTEM_DESIGN.md](design/DEBATE_SYSTEM_DESIGN.md) — 토론 프로토콜 + 종료 조건(5가지) + 예외처리·최적화 전략
- [AGENT_REASONING_IMPROVEMENT.md](design/AGENT_REASONING_IMPROVEMENT.md) — 에이전트 추론 개선 방안
- [GPU_ALLOCATION.md](design/GPU_ALLOCATION.md) — GPU 2+2 할당 전략

### 가이드 (guides/)
- [QUICK_START.md](guides/QUICK_START.md) — 설치 및 빠른 시작
- [QWEN_INFERENCE_SETUP.md](guides/QWEN_INFERENCE_SETUP.md) — Qwen 모델 다운로드 + vLLM 추론 셋업
- [RESEARCH_ROADMAP.md](guides/RESEARCH_ROADMAP.md) — 연구 방향 + 참고 프레임워크 비교
- [CONTRIBUTING.md](guides/CONTRIBUTING.md) — 기여 가이드

### 연구 (research/)
- [TOOLS_AND_PAPERS.md](research/TOOLS_AND_PAPERS.md) — 툴별 관련 논문 목록
- [TOOL_THEORETICAL_FOUNDATIONS.md](research/TOOL_THEORETICAL_FOUNDATIONS.md) — 각 툴의 이론적 기반 + 전문화 전략
- [RESEARCH_REPORT_Frequency_Tool_Improvement.md](research/RESEARCH_REPORT_Frequency_Tool_Improvement.md) — 주파수 툴 개선 연구
- [DATASETS.md](research/DATASETS.md) — 데이터셋 목록

### 보고서 (reports/)
- [FALSE_POSITIVE_FIX_SUMMARY.md](reports/FALSE_POSITIVE_FIX_SUMMARY.md) — 오탐 수정 내역 및 알고리즘 개선
- [CONDA_ENV_SNAPSHOTS.md](reports/CONDA_ENV_SNAPSHOTS.md) — 환경 스냅샷

### 도메인 지식 (LLM용)
- [../src/knowledge/frequency_domain_knowledge.md](../src/knowledge/frequency_domain_knowledge.md)
- [../src/knowledge/noise_domain_knowledge.md](../src/knowledge/noise_domain_knowledge.md)
- [../src/knowledge/watermark_domain_knowledge.md](../src/knowledge/watermark_domain_knowledge.md)
- [../src/knowledge/spatial_domain_knowledge.md](../src/knowledge/spatial_domain_knowledge.md)
