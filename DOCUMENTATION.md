# MAIFS 문서 가이드
**Documentation Index for Multi-Agent Image Forensic System**

📅 **최종 업데이트**: 2026-01-23

---

## 📂 문서 구조

```
MAIFS/
├── README.md                       # 프로젝트 소개 (시작점)
├── PROGRESS_REPORT.md              # 현재 진행 상황 (최신)
├── DOCUMENTATION.md                # 이 파일 (문서 가이드)
│
├── docs/
│   ├── design/                     # 설계 문서
│   │   ├── DEBATE_SYSTEM_DESIGN.md
│   │   ├── DEBATE_TERMINATION_GUIDE.md
│   │   ├── DEBATE_EDGE_CASES_AND_OPTIMIZATIONS.md
│   │   └── AGENT_REASONING_IMPROVEMENT.md
│   │
│   ├── reports/                    # 보고서
│   │   ├── CHECKPOINT_VALIDATION_REPORT.md
│   │   ├── SYSTEM_STATUS.md
│   │   └── CHANGES_SUMMARY.md
│   │
│   ├── guides/                     # 가이드
│   │   ├── QUICK_START.md
│   │   └── RESEARCH_ROADMAP.md
│   │
│   ├── ARCHITECTURE.md             # 아키텍처
│   └── API_REFERENCE.md            # API 문서
│
└── src/knowledge/                  # 도메인 지식
    ├── frequency_domain_knowledge.md
    ├── noise_domain_knowledge.md
    ├── watermark_domain_knowledge.md
    └── spatial_domain_knowledge.md
```

---

## 🚀 빠른 시작

| 목적 | 문서 |
|------|------|
| 프로젝트 소개 | [README.md](README.md) |
| 현재 진행 상황 | [PROGRESS_REPORT.md](PROGRESS_REPORT.md) |
| 빠른 시작 | [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md) |

---

## 📖 주제별 문서

### 1. 프로젝트 개요

| 문서 | 설명 | 대상 |
|-----|------|------|
| [README.md](README.md) | 프로젝트 소개, 설치, 사용법 | 모든 사용자 |
| [PROGRESS_REPORT.md](PROGRESS_REPORT.md) | 현재 진행 상황, 완료/예정 작업 | 개발자, 연구자 |
| [docs/guides/RESEARCH_ROADMAP.md](docs/guides/RESEARCH_ROADMAP.md) | 연구 방향, 4가지 옵션 | 연구자 |

### 2. 아키텍처 & 설계

| 문서 | 설명 | 대상 |
|-----|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 시스템 아키텍처, 구성 요소 | 개발자 |
| [docs/design/AGENT_REASONING_IMPROVEMENT.md](docs/design/AGENT_REASONING_IMPROVEMENT.md) | Sub-agent 추론 능력 개선 방안 | 개발자 |
| [docs/design/DEBATE_SYSTEM_DESIGN.md](docs/design/DEBATE_SYSTEM_DESIGN.md) | 토론 시스템 전체 설계 | 개발자 |
| [docs/design/DEBATE_TERMINATION_GUIDE.md](docs/design/DEBATE_TERMINATION_GUIDE.md) | 토론 종료 메커니즘 (5가지) | 개발자 |
| [docs/design/DEBATE_EDGE_CASES_AND_OPTIMIZATIONS.md](docs/design/DEBATE_EDGE_CASES_AND_OPTIMIZATIONS.md) | 예외 상황 & 최적화 (12가지) | 개발자 |

### 3. API & 사용법

| 문서 | 설명 | 대상 |
|-----|------|------|
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | API 레퍼런스 | 개발자 |
| [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md) | 빠른 시작 가이드 | 사용자 |

### 4. 보고서 & 검증

| 문서 | 설명 | 대상 |
|-----|------|------|
| [docs/reports/CHECKPOINT_VALIDATION_REPORT.md](docs/reports/CHECKPOINT_VALIDATION_REPORT.md) | OmniGuard 체크포인트 검증 | 개발자 |
| [docs/reports/SYSTEM_STATUS.md](docs/reports/SYSTEM_STATUS.md) | 시스템 상태 점검 | 개발자 |
| [docs/reports/CHANGES_SUMMARY.md](docs/reports/CHANGES_SUMMARY.md) | 변경 사항 요약 | 개발자 |

### 5. 도메인 지식 (논문 기반)

| 문서 | 설명 | 대상 |
|-----|------|------|
| [src/knowledge/frequency_domain_knowledge.md](src/knowledge/frequency_domain_knowledge.md) | 주파수 분석 지식 (FFT, GAN 패턴) | 연구자, LLM |
| [src/knowledge/noise_domain_knowledge.md](src/knowledge/noise_domain_knowledge.md) | 노이즈 분석 지식 (PRNU, 센서 지문) | 연구자, LLM |
| [src/knowledge/watermark_domain_knowledge.md](src/knowledge/watermark_domain_knowledge.md) | 워터마크 탐지 지식 (OmniGuard) | 연구자, LLM |
| [src/knowledge/spatial_domain_knowledge.md](src/knowledge/spatial_domain_knowledge.md) | 공간 분석 지식 (ViT, 조작 탐지) | 연구자, LLM |

---

## 🎯 역할별 추천 문서

### 👨‍💻 개발자 (신규)

```
1단계: README.md                           # 프로젝트 이해
2단계: docs/ARCHITECTURE.md                # 구조 파악
3단계: docs/API_REFERENCE.md               # API 학습
4단계: docs/guides/QUICK_START.md          # 실습
```

### 🔬 연구자

```
1단계: PROGRESS_REPORT.md                  # 현재 상태
2단계: docs/guides/RESEARCH_ROADMAP.md     # 연구 방향
3단계: src/knowledge/*.md                  # 도메인 지식
4단계: docs/design/DEBATE_SYSTEM_DESIGN.md # 핵심 기여
```

### 🏗️ 아키텍트

```
1단계: docs/ARCHITECTURE.md                # 전체 구조
2단계: docs/design/DEBATE_SYSTEM_DESIGN.md # 토론 시스템
3단계: docs/design/DEBATE_EDGE_CASES_AND_OPTIMIZATIONS.md # 최적화
```

### 🧪 테스터

```
1단계: docs/guides/QUICK_START.md          # 사용법
2단계: docs/reports/SYSTEM_STATUS.md       # 현재 상태
3단계: docs/API_REFERENCE.md               # API 테스트
```

---

## 📑 문서 유형별 분류

### 📘 설계 문서 (Design)
토론 시스템, 에이전트 구조 등 **설계 결정사항**을 설명합니다.

- [토론 시스템 설계](docs/design/DEBATE_SYSTEM_DESIGN.md)
- [토론 종료 가이드](docs/design/DEBATE_TERMINATION_GUIDE.md)
- [예외 & 최적화](docs/design/DEBATE_EDGE_CASES_AND_OPTIMIZATIONS.md)
- [추론 능력 개선](docs/design/AGENT_REASONING_IMPROVEMENT.md)

### 📗 가이드 (Guides)
**사용 방법**과 **연구 방향**을 안내합니다.

- [빠른 시작](docs/guides/QUICK_START.md)
- [연구 로드맵](docs/guides/RESEARCH_ROADMAP.md)

### 📙 보고서 (Reports)
**검증 결과**와 **상태 점검**을 기록합니다.

- [체크포인트 검증](docs/reports/CHECKPOINT_VALIDATION_REPORT.md)
- [시스템 상태](docs/reports/SYSTEM_STATUS.md)
- [변경 사항](docs/reports/CHANGES_SUMMARY.md)

### 📕 도메인 지식 (Knowledge)
**과학적 근거**와 **논문 기반 지식**을 제공합니다.

- [주파수 분석 지식](src/knowledge/frequency_domain_knowledge.md)
- [노이즈 분석 지식](src/knowledge/noise_domain_knowledge.md)
- [워터마크 탐지 지식](src/knowledge/watermark_domain_knowledge.md)
- [공간 분석 지식](src/knowledge/spatial_domain_knowledge.md)

---

## 🔍 주제별 검색

### 토론 시스템 관련
- 전체 설계: [DEBATE_SYSTEM_DESIGN.md](docs/design/DEBATE_SYSTEM_DESIGN.md)
- 종료 조건: [DEBATE_TERMINATION_GUIDE.md](docs/design/DEBATE_TERMINATION_GUIDE.md)
- 예외 처리: [DEBATE_EDGE_CASES_AND_OPTIMIZATIONS.md](docs/design/DEBATE_EDGE_CASES_AND_OPTIMIZATIONS.md)

### LLM 통합 관련
- 문제 분석: [AGENT_REASONING_IMPROVEMENT.md](docs/design/AGENT_REASONING_IMPROVEMENT.md)
- 도메인 지식: [src/knowledge/](src/knowledge/)
- API 레퍼런스: [API_REFERENCE.md](docs/API_REFERENCE.md)

### 성능 & 최적화
- 최적화 방안: [DEBATE_EDGE_CASES_AND_OPTIMIZATIONS.md](docs/design/DEBATE_EDGE_CASES_AND_OPTIMIZATIONS.md)
- 진행 보고서: [PROGRESS_REPORT.md](PROGRESS_REPORT.md)

### 데이터셋 & 평가
- 연구 로드맵: [RESEARCH_ROADMAP.md](docs/guides/RESEARCH_ROADMAP.md)
- 진행 보고서: [PROGRESS_REPORT.md](PROGRESS_REPORT.md)

---

## 📝 문서 작성 규칙

### 1. 파일 명명 규칙
```
루트: 대문자_언더스코어.md (README.md, PROGRESS_REPORT.md)
하위: 대문자_언더스코어.md (동일)
```

### 2. 문서 헤더
모든 문서는 다음 정보를 포함합니다:
```markdown
# 제목
**부제목**

📅 **작성일**: YYYY-MM-DD
📊 **버전**: vX.Y.Z (선택적)

---
```

### 3. 내부 링크
```markdown
[문서 이름](상대경로/파일명.md)
[섹션](#앵커-이름)
```

### 4. 이모지 사용
```
📋 목차, 리스트
📁 파일, 디렉토리
🚀 시작, 빠른 시작
✅ 완료, 성공
🔄 진행 중
⏳ 대기
🚧 이슈, 문제
📊 통계, 수치
🔬 연구
🏗️ 아키텍처
```

---

## 🔄 문서 업데이트 정책

### 자동 업데이트
- [PROGRESS_REPORT.md](PROGRESS_REPORT.md): 주요 마일스톤 달성 시

### 수동 업데이트
- [README.md](README.md): 사용법 변경 시
- [API_REFERENCE.md](docs/API_REFERENCE.md): API 변경 시
- 설계 문서: 설계 변경 시

---

## 📞 도움말

### 문서를 찾을 수 없나요?
1. 이 파일([DOCUMENTATION.md](DOCUMENTATION.md))의 검색 기능 사용 (Ctrl+F)
2. [PROGRESS_REPORT.md](PROGRESS_REPORT.md)에서 "관련 문서" 섹션 확인
3. `docs/` 폴더 내 카테고리별 탐색

### 문서 제안
- 새로운 문서 필요: Issue 생성
- 문서 개선: Pull Request

---

**다음 문서**: [README.md](README.md) 또는 [PROGRESS_REPORT.md](PROGRESS_REPORT.md)
