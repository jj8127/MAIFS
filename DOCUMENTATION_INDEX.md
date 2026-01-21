# MAIFS 프로젝트 문서 인덱스

**생성 날짜**: 2026-01-21
**총 문서 수**: 12개 (핵심 문서 기준)

---

## 📚 문서 목록

### 1. 시작/요약

#### 📄 `README.md`
- **용도**: 프로젝트 메인 진입점
- **대상**: 모든 사용자
- **내용**:
  - 프로젝트 개요
  - 빠른 시작
  - 문서 네비게이션
  - 체크포인트 안내

#### 📄 `README_root.md`
- **용도**: 프로젝트 가이드 (상세 상태/구성)
- **대상**: 모든 사용자
- **내용**:
  - 핵심 기능 요약
  - 테스트 결과
  - 설정 및 구조
  - 다음 단계

#### 📄 `QUICK_START.md`
- **용도**: 빠른 시작 가이드
- **대상**: 신규 사용자, 개발자
- **내용**:
  - 30초 시작
  - 기본 사용 예제
  - 합의 알고리즘/토론 옵션
  - 테스트 실행 방법

#### 📄 `SYSTEM_STATUS.md`
- **용도**: 시스템 상태 및 사용 가이드
- **대상**: 모든 사용자
- **내용**:
  - 현재 상태
  - 테스트 결과
  - 설정 및 경로
  - 제한사항/트러블슈팅

---

### 2. 상태/변경/검증

#### 📄 `CHECKPOINT_VALIDATION_REPORT.md`
- **용도**: OmniGuard 체크포인트 검증 보고서
- **대상**: 개발자, 기술자
- **내용**:
  - 체크포인트 검증 결과
  - 모델 로드 상태
  - 테스트 스위트 결과

#### 📄 `CHANGES_SUMMARY.md`
- **용도**: 변경 사항 정리
- **대상**: 개발자
- **내용**:
  - 수정/추가 파일 목록
  - 코드 변경 요약
  - 영향 범위

#### 📄 `MAIFS_IMPLEMENTATION_PLAN.md`
- **용도**: 구현 계획 및 진행 현황
- **대상**: 프로젝트 관리자, 개발자
- **내용**:
  - 완료 작업
  - 우선순위별 작업 목록
  - 다음 단계

---

### 3. 상세 기술 문서 (docs/)

#### 📄 `docs/README.md`
- **용도**: 상세 소개 및 설치/사용 가이드
- **대상**: 모든 사용자

#### 📄 `docs/ARCHITECTURE.md`
- **용도**: 시스템 아키텍처 설명
- **대상**: 개발자

#### 📄 `docs/API_REFERENCE.md`
- **용도**: API 상세 문서
- **대상**: 개발자

#### 📄 `docs/CONTRIBUTING.md`
- **용도**: 기여 가이드
- **대상**: 기여자

#### 📄 `docs/DOCUMENTS_REVIEW_TABLE.md`
- **용도**: 연구 문서 리뷰 테이블
- **대상**: 연구자

---

## 📊 문서 매트릭스

| 문서 | 타입 | 대상 | 분량 | 우선순위 |
|------|------|------|------|----------|
| README.md | 개요 | 모두 | 중간 | 🔴 1 |
| README_root.md | 가이드 | 모두 | 길음 | 🔴 1 |
| QUICK_START.md | 가이드 | 신규 | 중간 | 🔴 1 |
| SYSTEM_STATUS.md | 참고 | 모두 | 길음 | 🟠 2 |
| CHECKPOINT_VALIDATION_REPORT.md | 기술 | 개발 | 길음 | 🟠 2 |
| CHANGES_SUMMARY.md | 기술 | 개발 | 중간 | 🟠 2 |
| MAIFS_IMPLEMENTATION_PLAN.md | 계획 | 관리 | 길음 | 🟡 3 |
| docs/README.md | 소개 | 모두 | 길음 | 🟠 2 |
| docs/ARCHITECTURE.md | 기술 | 개발 | 길음 | 🟠 2 |
| docs/API_REFERENCE.md | 기술 | 개발 | 길음 | 🟠 2 |
| docs/CONTRIBUTING.md | 가이드 | 기여 | 길음 | 🟡 3 |
| docs/DOCUMENTS_REVIEW_TABLE.md | 리서치 | 연구 | 중간 | 🟡 3 |

---

## 🎯 문서 읽기 순서

### 신규 사용자
1. `README.md` (프로젝트 개요)
2. `QUICK_START.md` (빠른 시작)
3. `README_root.md` (상세 가이드)
4. `SYSTEM_STATUS.md` (필요 시)

### 개발자
1. `README.md`
2. `QUICK_START.md`
3. `CHECKPOINT_VALIDATION_REPORT.md`
4. `CHANGES_SUMMARY.md`
5. `SYSTEM_STATUS.md`
6. `docs/ARCHITECTURE.md`
7. `docs/API_REFERENCE.md`

### 프로젝트 관리자
1. `README.md`
2. `MAIFS_IMPLEMENTATION_PLAN.md`
3. `CHECKPOINT_VALIDATION_REPORT.md`
4. `CHANGES_SUMMARY.md`

---

## 📍 문서 위치

```
MAIFS/
├── README.md
├── README_root.md
├── QUICK_START.md
├── SYSTEM_STATUS.md
├── CHECKPOINT_VALIDATION_REPORT.md
├── CHANGES_SUMMARY.md
├── MAIFS_IMPLEMENTATION_PLAN.md
├── DOCUMENTATION_INDEX.md
├── docs/
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── CONTRIBUTING.md
│   ├── DOCUMENTS_REVIEW_TABLE.md
│   └── DOCUMENTS_REVIEW_TABLE.csv
└── documents/  # 수집된 논문/자료 원문
```

---

## 🔍 문서별 주요 내용

### README.md
- 프로젝트 소개
- 빠른 시작
- 문서 네비게이션
- 체크포인트 안내

### QUICK_START.md
- 30초 시작
- 기본 사용 예제
- 합의 알고리즘 선택
- 토론 옵션
- 테스트 실행

### SYSTEM_STATUS.md
- 현재 상태 (테이블)
- 테스트 결과
- 기능 검증
- 설정 정보
- 제한사항/트러블슈팅

### CHECKPOINT_VALIDATION_REPORT.md
- 체크포인트 검증 결과
- 테스트 통과 현황
- 모델 로드 상태 분석

### CHANGES_SUMMARY.md
- 수정/추가 파일 목록
- 변경 통계
- 영향 범위
