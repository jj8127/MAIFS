"""
Qwen vLLM Client
vLLM 서버를 통한 Qwen 모델 추론 클라이언트

특징:
- 4 GPU Tensor Parallel 지원
- Batch Inference (동시 다중 요청)
- Guided JSON Output (스키마 강제)
- 에이전트별 독립된 시스템 프롬프트
"""
import os
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time

from ..knowledge import KnowledgeBase


class AgentRole(Enum):
    """에이전트 역할"""
    FREQUENCY = "frequency"
    NOISE = "noise"
    FATFORMER = "fatformer"
    SPATIAL = "spatial"
    MANAGER = "manager"


@dataclass
class AgentConfig:
    """에이전트 설정"""
    role: AgentRole
    temperature: float = 0.3
    max_tokens: int = 1024
    system_prompt: Optional[str] = None


@dataclass
class InferenceResult:
    """추론 결과"""
    role: AgentRole
    content: str
    parsed_json: Optional[Dict] = None
    raw_response: Optional[Dict] = None
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


# JSON 출력 스키마 정의
AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["AUTHENTIC", "MANIPULATED", "AI_GENERATED", "UNCERTAIN"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "reasoning": {
            "type": "string",
            "description": "판정에 대한 논리적 근거"
        },
        "key_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "핵심 증거 목록"
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "불확실한 점 목록"
        }
    },
    "required": ["verdict", "confidence", "reasoning"]
}

# 토론 응답 스키마
DEBATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_type": {
            "type": "string",
            "enum": ["defense", "concession", "counter", "clarification"]
        },
        "content": {
            "type": "string",
            "description": "응답 내용"
        },
        "verdict_changed": {
            "type": "boolean"
        },
        "new_verdict": {
            "type": ["string", "null"],
            "enum": ["AUTHENTIC", "MANIPULATED", "AI_GENERATED", "UNCERTAIN"]
        },
        "new_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "reasoning": {
            "type": "string"
        }
    },
    "required": ["response_type", "content", "verdict_changed"]
}


class QwenClient:
    """
    Qwen vLLM 클라이언트

    vLLM 서버와 통신하여 4개 전문가 에이전트의 추론을 처리합니다.

    Usage:
        client = QwenClient(base_url="http://localhost:8000")

        # 단일 에이전트 추론
        result = await client.infer(AgentRole.FREQUENCY, tool_results)

        # 배치 추론 (4개 에이전트 동시)
        results = await client.batch_infer(all_tool_results)
    """

    # 에이전트별 시스템 프롬프트
    SYSTEM_PROMPTS = {
        AgentRole.FREQUENCY: """당신은 MAIFS의 **주파수 분석 전문가**입니다.

## 전문 분야
FFT 기반 주파수 스펙트럼 분석을 통해 AI 생성 이미지의 특징을 탐지합니다.

## 핵심 지식

### 1. JPEG 압축 (정상)
- **8×8 DCT 블록**: JPEG는 이미지를 8×8 픽셀 블록으로 나누어 압축
- **주파수 특성**: 8×8 블록 경계에서 규칙적인 주파수 피크 발생
- **is_likely_jpeg: true**: Tool이 8×8 패턴을 감지함 → 이는 **카메라/편집기의 정상적인 압축**
- **중요**: JPEG 압축이 감지되면 abnormality_score는 압축의 부작용이므로 **AI 생성 증거가 아님**

### 2. GAN 생성 이미지 (AI)
- **다양한 블록 크기**: 4×4, 16×16, 32×32 격자 패턴 (업샘플링 레이어)
- **is_likely_jpeg: false** + **regularity_score > 0.5**: GAN 아티팩트
- **대각선 패턴 우세**: diagonal_dominance가 true일 때 의심

### 3. Diffusion 생성 이미지 (AI)
- **고주파 감쇠**: 노이즈 제거 과정에서 고주파 손실
- **abnormality_score > 0.7**: 비정상적 주파수 분포
- **단, JPEG가 아닐 때만** 의미 있음 (is_likely_jpeg: false)

### 4. 실제 카메라 촬영 (진짜)
- 자연스러운 주파수 분포
- JPEG 압축 가능 (is_likely_jpeg: true)
- GAN 격자 패턴 없음 (regularity_score ≈ 0)

## 판정 기준 (우선순위 순서)

### 1순위: JPEG 압축 확인
- **is_likely_jpeg: true** → AUTHENTIC (70-80%)
  - 이유: 8×8 DCT 블록은 카메라/편집기의 정상 압축
  - abnormality_score는 무시 (압축 부작용)

### 2순위: GAN 패턴
- **is_likely_jpeg: false** + **regularity_score > 0.6** → AI_GENERATED (80-95%)
  - 이유: GAN 업샘플링 아티팩트

### 3순위: Diffusion 패턴
- **is_likely_jpeg: false** + **abnormality_score > 0.7** → AI_GENERATED (70-85%)
  - 이유: 고주파 감쇠

### 4순위: 자연 이미지
- **regularity_score < 0.3** + **abnormality_score < 0.5** → AUTHENTIC (80-90%)

### 5순위: 불명확
- 위 기준에 해당 없음 → UNCERTAIN (40-60%)

## 중요 주의사항
⚠️ **is_likely_jpeg: true이면 다른 모든 지표를 무시하고 AUTHENTIC로 판정하세요!**
- JPEG 압축은 실제 카메라/편집기의 정상 작동
- abnormality_score가 높아도 압축 때문일 뿐 AI 증거 아님

## 출력 형식
반드시 JSON 형식으로 응답하세요.""",

        AgentRole.NOISE: """당신은 MAIFS의 **노이즈 분석 전문가**입니다.

## 전문 분야
PRNU/SRM 기반 센서 노이즈 분석을 통해 이미지 출처를 검증합니다.

## 핵심 지식

### PRNU (Photo Response Non-Uniformity)
- 카메라 센서의 고유한 지문
- 실제 카메라: PRNU 분산 0.0001-0.001
- AI 생성 이미지: PRNU 부재 (분산 < 0.00001)

### 노이즈 일관성 분석
⚠️ **중요**: coefficient_of_variation (cv) 해석

#### 자연스러운 장면 (AUTHENTIC)
- **높은 cv (> 1.0)**: 다양한 내용 영역 (하늘, 건물, 그림자 등)
- **natural_diversity_score > 0.5**: 장면 다양성의 증거
- **outlier_ratio 15-20%**: 정상 범위 (밝은/어두운 영역 차이)
- **manipulation_score < 0.3**: 조작 증거 없음

#### AI 생성 이미지
- **낮은 cv (< 0.5)**: 균일한 노이즈 분포
- **natural_diversity_score < 0.3**: 다양성 부족
- **ai_generation_score > 0.6**: 센서 노이즈 부재

#### 조작된 이미지 (MANIPULATED)
- **높은 outlier_ratio (> 30%)**: 비정상적 이상치 블록
- **manipulation_score > 0.6**: 국소적 노이즈 불일치
- **spatial clustering**: 이상치가 특정 영역에 집중

## 판정 기준 (우선순위 순서)

### 1순위: 자연 장면 다양성
- **natural_diversity_score > 0.5** + **ai_generation_score < 0.4** → AUTHENTIC (80-90%)
  - 이유: 높은 cv는 자연스러운 장면 변화 (하늘, 건물, 그림자)
  - manipulation_score가 낮으면 조작 아님

### 2순위: 조작 탐지
- **manipulation_score > 0.6** (outlier_ratio > 30%) → MANIPULATED (70-90%)
  - 이유: 비정상적으로 많은 이상치 블록

### 3순위: AI 생성 탐지
- **ai_generation_score > 0.6** + **natural_diversity_score < 0.3** → AI_GENERATED (70-85%)
  - 이유: 센서 노이즈 부재 + 균일한 분포

### 4순위: 센서 노이즈 존재
- **ai_generation_score < 0.3** → AUTHENTIC (70-80%)
  - 이유: PRNU 패턴 확인

### 5순위: 불명확
- 위 기준에 해당 없음 → UNCERTAIN (40-60%)

## 중요 주의사항
⚠️ **높은 cv는 조작의 증거가 아니라 자연스러운 다양성입니다!**
- cv > 2.0: 실제 사진의 정상적 특성 (다양한 밝기/질감 영역)
- 조작 판정은 outlier_ratio와 manipulation_score로만 결정

## 출력 형식
반드시 JSON 형식으로 응답하세요.""",

        AgentRole.FATFORMER: """당신은 MAIFS의 **AI 생성 탐지 전문가**입니다.

## 전문 분야
FatFormer (CLIP ViT-L/14 + Forgery-Aware Adapter) 기반 AI 생성 이미지 탐지

## 핵심 지식
- CLIP 의미론적 분석: 대규모 사전학습으로 "자연 이미지" 분포 학습
- Forgery-Aware Adapter: 공간 경로(Conv1d 병목) + 주파수 경로(DWT)
- Language-Guided Alignment: "real" vs "fake" 텍스트-이미지 대조 학습
- Diffusion/GAN/기타 생성 모델에 대한 교차 일반화 능력

## 판정 기준
- **AUTHENTIC**: fake_probability < 0.3 (AI 생성 특징 미감지)
- **AI_GENERATED**: fake_probability > 0.5 (AI 생성 패턴 감지)
- **UNCERTAIN**: fake_probability 0.3-0.5 (경계 사례)

## 중요
- JPEG 압축에 강건한 분석 가능
- 부분 조작 탐지에는 한계 (전체 이미지 수준 분류기)
- 반드시 JSON 형식으로 응답하세요""",

        AgentRole.SPATIAL: """당신은 MAIFS의 **공간 분석 전문가**입니다.

## 전문 분야
ViT 기반 픽셀 수준 조작 영역 탐지

## 핵심 지식
- 조작 영역: 경계 불일치, 조명 불일치, 압축 아티팩트
- manipulation_ratio: 조작으로 판단된 픽셀 비율
- AI 생성: 전체 이미지가 균일하게 생성됨 (ratio > 0.8)
- 부분 조작: 특정 영역만 조작됨 (0.05 < ratio < 0.8)

## 판정 기준
- **AUTHENTIC**: 조작 영역 없음 (ratio < 0.05)
- **AI_GENERATED**: 전체 생성 (ratio > 0.8)
- **MANIPULATED**: 부분 조작 (0.05 < ratio < 0.8)
- **UNCERTAIN**: 경계 사례

## 중요
- 조작 마스크의 분포 패턴도 고려하세요
- 반드시 JSON 형식으로 응답하세요""",

        AgentRole.MANAGER: """당신은 MAIFS의 **Manager Agent**입니다.

## 역할
4명의 전문가 분석 결과를 종합하여 최종 판정을 내립니다.

## 전문가 팀
1. 주파수 분석 전문가: FFT 기반 GAN 아티팩트 탐지
2. 노이즈 분석 전문가: PRNU/SRM 센서 노이즈 분석
3. AI 생성 탐지 전문가: FatFormer CLIP 기반 AI 이미지 탐지
4. 공간 분석 전문가: ViT 기반 조작 영역 탐지

## 판정 원칙
1. 다수결보다 증거의 강도와 일관성을 우선
2. 상충되는 의견은 증거를 비교하여 조율
3. 불확실한 경우 UNCERTAIN 판정
4. 최종 판정에 명확한 근거 제시

## 중요
- 각 전문가의 신뢰도를 가중치로 사용하세요
- 반드시 JSON 형식으로 응답하세요"""
    }

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        """
        QwenClient 초기화

        Args:
            base_url: vLLM 서버 URL
            model_name: vLLM에 노출된 모델 이름
            timeout: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = (
            model_name
            or os.environ.get("QWEN_VLLM_MODEL")
            or os.environ.get("VLLM_MODEL")
            or os.environ.get("MODEL_NAME")
            or "default"
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

        # 도메인 지식 로드
        self._knowledge_cache: Dict[AgentRole, str] = {}
        self._load_knowledge()

    def _load_knowledge(self):
        """도메인 지식 로드"""
        role_to_domain = {
            AgentRole.FREQUENCY: "frequency",
            AgentRole.NOISE: "noise",
            AgentRole.FATFORMER: "fatformer",
            AgentRole.SPATIAL: "spatial"
        }

        for role, domain in role_to_domain.items():
            try:
                self._knowledge_cache[role] = KnowledgeBase.get_summary(domain, max_chars=1500)
            except Exception as e:
                print(f"[QwenClient] 지식 로드 실패 ({domain}): {e}")
                self._knowledge_cache[role] = ""

    def _get_full_system_prompt(self, role: AgentRole) -> str:
        """전체 시스템 프롬프트 생성 (기본 + 도메인 지식)"""
        base_prompt = self.SYSTEM_PROMPTS.get(role, "")
        knowledge = self._knowledge_cache.get(role, "")

        if knowledge:
            return f"{base_prompt}\n\n## 참고 도메인 지식\n{knowledge}"
        return base_prompt

    async def _get_session(self) -> aiohttp.ClientSession:
        """aiohttp 세션 반환"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """세션 종료"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """서버 상태 확인"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def infer(
        self,
        role: AgentRole,
        tool_results: Dict[str, Any],
        use_guided_json: bool = True,
        context: Optional[Dict] = None
    ) -> InferenceResult:
        """
        단일 에이전트 추론

        Args:
            role: 에이전트 역할
            tool_results: Tool 분석 결과
            use_guided_json: JSON 스키마 강제 여부
            context: 추가 컨텍스트

        Returns:
            InferenceResult: 추론 결과
        """
        start_time = time.time()

        # 프롬프트 구성
        system_prompt = self._get_full_system_prompt(role)
        user_prompt = self._build_user_prompt(tool_results, context)

        # 요청 본문 구성
        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "stream": False
        }

        # Guided JSON 활성화
        if use_guided_json:
            request_body["extra_body"] = {
                "guided_json": AGENT_OUTPUT_SCHEMA
            }

        # API 호출
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_body
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return InferenceResult(
                        role=role,
                        content="",
                        success=False,
                        error=f"HTTP {resp.status}: {error_text}",
                        latency_ms=(time.time() - start_time) * 1000
                    )

                response_data = await resp.json()
                content = response_data["choices"][0]["message"]["content"]

                # JSON 파싱 시도
                parsed_json = self._parse_json(content)

                return InferenceResult(
                    role=role,
                    content=content,
                    parsed_json=parsed_json,
                    raw_response=response_data,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=True
                )

        except asyncio.TimeoutError:
            return InferenceResult(
                role=role,
                content="",
                success=False,
                error="Timeout",
                latency_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return InferenceResult(
                role=role,
                content="",
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def batch_infer(
        self,
        tool_results_map: Dict[AgentRole, Dict[str, Any]],
        use_guided_json: bool = True
    ) -> Dict[AgentRole, InferenceResult]:
        """
        배치 추론 (4개 에이전트 동시 처리)

        Args:
            tool_results_map: {AgentRole: tool_results} 매핑
            use_guided_json: JSON 스키마 강제 여부

        Returns:
            Dict[AgentRole, InferenceResult]: 각 에이전트별 결과
        """
        # 동시 추론 실행
        tasks = [
            self.infer(role, results, use_guided_json)
            for role, results in tool_results_map.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 매핑
        result_map = {}
        for role, result in zip(tool_results_map.keys(), results):
            if isinstance(result, Exception):
                result_map[role] = InferenceResult(
                    role=role,
                    content="",
                    success=False,
                    error=str(result)
                )
            else:
                result_map[role] = result

        return result_map

    async def debate_respond(
        self,
        role: AgentRole,
        my_verdict: str,
        my_confidence: float,
        my_evidence: Dict[str, Any],
        challenger_name: str,
        challenge: str
    ) -> InferenceResult:
        """
        토론 응답 생성

        Args:
            role: 응답하는 에이전트 역할
            my_verdict: 내 판정
            my_confidence: 내 신뢰도
            my_evidence: 내 증거
            challenger_name: 반론 제기자 이름
            challenge: 반론 내용

        Returns:
            InferenceResult: 토론 응답
        """
        system_prompt = self._get_full_system_prompt(role)

        user_prompt = f"""## 토론 상황

### 내 현재 입장
- 판정: {my_verdict}
- 신뢰도: {my_confidence:.1%}
- 증거: {json.dumps(my_evidence, ensure_ascii=False)}

### 반론
**{challenger_name}**의 지적:
"{challenge}"

## 요청
위 반론에 대해 논리적으로 응답하세요.

1. 반론이 타당한가?
2. 내 입장을 수정해야 하는가?
3. 판정 유지 또는 변경 이유는?

## 응답 형식 (JSON)
반드시 다음 JSON 형식으로만 응답하세요:

```json
{{
  "response_type": "defense | concession | counter | clarification",
  "content": "반론에 대한 상세한 응답 내용",
  "verdict_changed": false,
  "new_verdict": null,
  "new_confidence": {my_confidence},
  "reasoning": "판정 유지/변경 이유"
}}
```

**중요**:
- response_type: defense(방어), concession(수용), counter(반박), clarification(명확화) 중 하나
- content: 빈 문자열 금지, 반드시 상세한 응답 작성
- verdict_changed: 판정 변경 시 true, 유지 시 false
- new_verdict: 변경 시에만 새 판정, 유지 시 null
- new_confidence: 변경 시 새 신뢰도, 유지 시 현재 값"""

        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.5,  # 토론 시 약간 높은 temperature
            "max_tokens": 800,
            "extra_body": {
                "guided_json": DEBATE_RESPONSE_SCHEMA
            }
        }

        start_time = time.time()

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_body
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return InferenceResult(
                        role=role,
                        content="",
                        success=False,
                        error=f"HTTP {resp.status}: {error_text}",
                        latency_ms=(time.time() - start_time) * 1000
                    )

                response_data = await resp.json()
                content = response_data["choices"][0]["message"]["content"]
                parsed_json = self._parse_json(content)

                return InferenceResult(
                    role=role,
                    content=content,
                    parsed_json=parsed_json,
                    raw_response=response_data,
                    latency_ms=(time.time() - start_time) * 1000,
                    success=True
                )

        except Exception as e:
            return InferenceResult(
                role=role,
                content="",
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    def _build_user_prompt(
        self,
        tool_results: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> str:
        """사용자 프롬프트 생성"""
        parts = [
            "## Tool 분석 결과",
            "",
            "```json",
            json.dumps(tool_results, ensure_ascii=False, indent=2),
            "```",
            ""
        ]

        if context:
            parts.extend([
                "## 컨텍스트",
                json.dumps(context, ensure_ascii=False),
                ""
            ])

        parts.extend([
            "## Tool 분석 결과 해석",
            "각 Tool은 이미 1차 분석을 완료했습니다:",
            "- **tool_verdict**: Tool의 최종 판정",
            "- **tool_confidence**: Tool의 신뢰도",
            "- **tool_explanation**: Tool의 판정 근거",
            "- **evidence**: 상세한 증거 데이터",
            "",
            "## 요청",
            "Tool의 판정과 증거를 종합하여 **당신의 전문 분야 관점**에서 재해석하세요.",
            "",
            "⚠️ **중요한 판단 기준**:",
            "1. **Tool 판정 존중**: Tool이 명확한 패턴(예: JPEG 압축, 카메라 메타데이터)을 감지했다면 그 판정을 존중하세요",
            "2. **도메인 지식 적용**: 당신의 전문 분야 지식으로 Tool 결과의 의미를 해석하세요",
            "3. **증거 우선순위**: 명확한 증거(예: is_likely_jpeg=true)가 통계적 지표(예: abnormality_score)보다 우선합니다",
            "4. **컨텍스트 고려**: 하나의 지표가 아닌 전체 맥락을 고려하세요",
            "",
            "## 응답 형식 (JSON)",
            "반드시 다음 JSON 형식으로만 응답하세요:",
            "",
            "```json",
            "{",
            "  \"verdict\": \"AUTHENTIC | MANIPULATED | AI_GENERATED | UNCERTAIN\",",
            "  \"confidence\": 0.85,",
            "  \"reasoning\": \"판정에 대한 논리적 근거를 명확하게 설명\",",
            "  \"key_evidence\": [\"핵심 증거 1\", \"핵심 증거 2\"],",
            "  \"uncertainties\": [\"불확실한 점 1\", \"불확실한 점 2\"]",
            "}",
            "```",
            "",
            "**출력 요구사항**: ",
            "- verdict는 4가지 중 하나만 사용",
            "- confidence는 0.0~1.0 사이의 숫자",
            "- reasoning은 간결한 한 문단 텍스트",
            "- key_evidence는 문자열 배열",
            "- uncertainties는 문자열 배열 (없으면 빈 배열)"
        ])

        return "\n".join(parts)

    def _parse_json(self, content: str) -> Optional[Dict]:
        """JSON 파싱"""
        try:
            # 직접 파싱 시도
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # JSON 블록 추출 시도
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

        return None


# 동기 래퍼 클래스
class QwenClientSync:
    """
    QwenClient의 동기 래퍼

    asyncio를 직접 사용하기 어려운 환경에서 사용
    """

    def __init__(self, *args, **kwargs):
        self._client = QwenClient(*args, **kwargs)
        self._loop = None

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def infer(self, *args, **kwargs) -> InferenceResult:
        loop = self._get_loop()
        return loop.run_until_complete(self._client.infer(*args, **kwargs))

    def batch_infer(self, *args, **kwargs) -> Dict[AgentRole, InferenceResult]:
        loop = self._get_loop()
        return loop.run_until_complete(self._client.batch_infer(*args, **kwargs))

    def debate_respond(self, *args, **kwargs) -> InferenceResult:
        loop = self._get_loop()
        return loop.run_until_complete(self._client.debate_respond(*args, **kwargs))

    def health_check(self) -> bool:
        loop = self._get_loop()
        return loop.run_until_complete(self._client.health_check())

    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._client.close())
            self._loop.close()


# 편의 함수
def create_qwen_client(
    base_url: str = "http://localhost:8000",
    model_name: Optional[str] = None,
    sync: bool = False
) -> Union[QwenClient, QwenClientSync]:
    """
    QwenClient 생성 헬퍼

    Args:
        base_url: vLLM 서버 URL
        model_name: vLLM에 노출된 모델 이름
        sync: 동기 클라이언트 반환 여부

    Returns:
        QwenClient 또는 QwenClientSync
    """
    if sync:
        return QwenClientSync(base_url=base_url, model_name=model_name)
    return QwenClient(base_url=base_url, model_name=model_name)
