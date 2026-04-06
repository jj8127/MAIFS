"""
Sub-Agent LLM
각 전문가 에이전트를 위한 LLM 추론 엔진

Tool 결과를 도메인 지식을 바탕으로 해석하고,
토론에서 논리적인 의견을 생성합니다.
"""
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum

from ..knowledge import KnowledgeBase


class AgentDomain(Enum):
    """에이전트 도메인"""
    FREQUENCY = "frequency"
    NOISE = "noise"
    FATFORMER = "fatformer"
    SPATIAL = "spatial"


@dataclass
class ReasoningResult:
    """추론 결과"""
    interpretation: str          # Tool 결과 해석
    reasoning: str               # 논리적 추론
    verdict_rationale: str       # 판정 근거
    confidence_rationale: str    # 신뢰도 근거
    key_findings: List[str] = field(default_factory=list)  # 핵심 발견
    uncertainties: List[str] = field(default_factory=list)  # 불확실성


@dataclass
class DebateResponse:
    """토론 응답"""
    response_type: str           # "defense", "concession", "counter", "clarification"
    content: str                 # 응답 내용
    verdict_changed: bool        # 의견 변경 여부
    new_verdict: Optional[str] = None  # 변경된 의견 (있을 경우)
    new_confidence: Optional[float] = None  # 변경된 신뢰도
    reasoning: str = ""          # 변경/유지 이유


class SubAgentLLM:
    """
    Sub-Agent LLM 추론 엔진

    각 전문가 에이전트가 Tool 결과를 이해하고
    토론에 참여할 수 있도록 LLM 기반 추론을 제공합니다.

    Usage:
        llm = SubAgentLLM(AgentDomain.FREQUENCY)
        reasoning = llm.interpret_results(tool_results)
        response = llm.respond_to_challenge(challenge, my_verdict, my_evidence)
    """

    # 도메인별 역할 설명
    DOMAIN_ROLES = {
        AgentDomain.FREQUENCY: {
            "name": "주파수 분석 전문가",
            "expertise": "FFT 기반 주파수 스펙트럼 분석",
            "focus": "GAN/Diffusion 모델의 격자 아티팩트, 주기적 패턴, 고주파 특성"
        },
        AgentDomain.NOISE: {
            "name": "노이즈 분석 전문가",
            "expertise": "PRNU/SRM 기반 센서 노이즈 분석",
            "focus": "카메라 센서 지문, 노이즈 일관성, AI 생성 노이즈 패턴"
        },
        AgentDomain.FATFORMER: {
            "name": "AI 생성 탐지 전문가",
            "expertise": "FatFormer (CLIP ViT-L/14 + Forgery-Aware Adapter) 기반 AI 생성 이미지 탐지",
            "focus": "AI 생성 이미지 분류, Diffusion 모델 탐지, JPEG 강건 탐지, 교차 생성기 일반화"
        },
        AgentDomain.SPATIAL: {
            "name": "공간 분석 전문가",
            "expertise": "ViT 기반 픽셀 수준 조작 탐지",
            "focus": "조작 영역 탐지, 조명 일관성, 경계 분석"
        }
    }

    def __init__(
        self,
        domain: AgentDomain,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1500,
        use_full_knowledge: bool = False
    ):
        """
        SubAgentLLM 초기화

        Args:
            domain: 에이전트 도메인 (FREQUENCY, NOISE, FATFORMER, SPATIAL)
            api_key: Anthropic API 키
            model: 사용할 Claude 모델
            max_tokens: 최대 응답 토큰
            use_full_knowledge: 전체 지식 사용 여부 (False면 요약본 사용)
        """
        self.domain = domain
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.use_full_knowledge = use_full_knowledge

        self._client = None
        self._available = False
        self._knowledge = None

        self._initialize()

    def _initialize(self):
        """초기화: API 클라이언트 및 도메인 지식 로드"""
        # 도메인 지식 로드
        try:
            if self.use_full_knowledge:
                self._knowledge = KnowledgeBase.load(self.domain.value)
            else:
                self._knowledge = KnowledgeBase.get_summary(self.domain.value)
        except Exception as e:
            print(f"[SubAgentLLM:{self.domain.value}] 지식 로드 실패: {e}")
            self._knowledge = ""

        # API 클라이언트 초기화
        if not self.api_key:
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._available = True
        except ImportError:
            pass
        except Exception as e:
            print(f"[SubAgentLLM:{self.domain.value}] API 초기화 실패: {e}")

    @property
    def is_available(self) -> bool:
        """API 사용 가능 여부"""
        return self._available and self._client is not None

    def _get_system_prompt(self) -> str:
        """도메인별 시스템 프롬프트 생성"""
        role_info = self.DOMAIN_ROLES[self.domain]

        return f"""당신은 MAIFS의 {role_info['name']}입니다.

## 역할
{role_info['expertise']} 전문가로서, 이미지 포렌식 분석에서 {role_info['focus']}을 담당합니다.

## 도메인 지식
다음은 당신의 전문 분야에 대한 과학적 지식입니다:

{self._knowledge}

## 판정 기준
- **AUTHENTIC**: 원본 이미지 - 조작/생성 흔적 없음
- **MANIPULATED**: 부분 조작 - 특정 영역에서 불일치 발견
- **AI_GENERATED**: AI 생성 - GAN/Diffusion 특성 감지
- **UNCERTAIN**: 판단 불가 - 증거 불충분

## 분석 원칙
1. 과학적 근거에 기반하여 판단하세요
2. 도메인 지식의 임계값과 패턴을 참고하세요
3. 불확실한 부분은 명확히 표시하세요
4. 다른 전문가와 의견이 다를 경우 논리적 근거를 제시하세요
5. 새로운 증거에 열린 자세를 유지하세요

## 중요
- 수치 데이터를 도메인 지식에 따라 해석하세요
- 단순히 수치를 나열하지 말고, 그 의미를 설명하세요
- 판정에 대한 확신 수준을 정직하게 보고하세요
"""

    def interpret_results(
        self,
        tool_results: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> ReasoningResult:
        """
        Tool 결과 해석 및 추론 생성

        Args:
            tool_results: Tool이 반환한 분석 결과
            context: 추가 컨텍스트 (다른 에이전트 정보 등)

        Returns:
            ReasoningResult: 해석 및 추론 결과
        """
        if not self.is_available:
            return self._fallback_interpret(tool_results)

        prompt = self._build_interpretation_prompt(tool_results, context)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_interpretation_response(response.content[0].text)
        except Exception as e:
            print(f"[SubAgentLLM:{self.domain.value}] API 호출 실패: {e}")
            return self._fallback_interpret(tool_results)

    def _build_interpretation_prompt(
        self,
        tool_results: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> str:
        """해석 요청 프롬프트 생성"""
        prompt_parts = [
            "## Tool 분석 결과",
            "",
            "다음은 분석 도구가 반환한 결과입니다:",
            "",
            "```json",
            json.dumps(tool_results, ensure_ascii=False, indent=2),
            "```",
            ""
        ]

        if context:
            prompt_parts.extend([
                "## 컨텍스트 정보",
                "",
                json.dumps(context, ensure_ascii=False, indent=2),
                ""
            ])

        prompt_parts.extend([
            "## 요청",
            "",
            "위 Tool 결과를 도메인 지식에 기반하여 해석하고, 다음 형식으로 응답하세요:",
            "",
            "```json",
            "{",
            '  "interpretation": "Tool 결과 수치의 의미 해석",',
            '  "reasoning": "판정으로 이어지는 논리적 추론 과정",',
            '  "verdict_rationale": "최종 판정(AUTHENTIC/MANIPULATED/AI_GENERATED/UNCERTAIN)과 그 이유",',
            '  "confidence_rationale": "신뢰도(0.0-1.0) 수준과 그 근거",',
            '  "key_findings": ["핵심 발견 1", "핵심 발견 2"],',
            '  "uncertainties": ["불확실한 점 1", "불확실한 점 2"]',
            "}",
            "```"
        ])

        return "\n".join(prompt_parts)

    def _parse_interpretation_response(self, response_text: str) -> ReasoningResult:
        """LLM 응답 파싱"""
        try:
            # JSON 블록 추출
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                return ReasoningResult(
                    interpretation=data.get("interpretation", ""),
                    reasoning=data.get("reasoning", ""),
                    verdict_rationale=data.get("verdict_rationale", ""),
                    confidence_rationale=data.get("confidence_rationale", ""),
                    key_findings=data.get("key_findings", []),
                    uncertainties=data.get("uncertainties", [])
                )
        except json.JSONDecodeError:
            pass

        # JSON 파싱 실패 시 텍스트 그대로 반환
        return ReasoningResult(
            interpretation=response_text,
            reasoning="",
            verdict_rationale="",
            confidence_rationale="",
            key_findings=[],
            uncertainties=[]
        )

    def _fallback_interpret(self, tool_results: Dict[str, Any]) -> ReasoningResult:
        """API 불가 시 규칙 기반 해석"""
        # 도메인별 기본 해석 로직
        key_findings = []
        uncertainties = []

        if self.domain == AgentDomain.FREQUENCY:
            if tool_results.get("ai_generation_score", 0) > 0.6:
                key_findings.append("AI 생성 점수가 높음")
            if tool_results.get("grid_analysis", {}).get("is_grid_pattern"):
                key_findings.append("격자 패턴 탐지됨")

        elif self.domain == AgentDomain.NOISE:
            ai_score = tool_results.get("ai_detection", {}).get("ai_generation_score", 0)
            if ai_score > 0.6:
                key_findings.append("센서 노이즈 부재로 AI 생성 의심")

        elif self.domain == AgentDomain.FATFORMER:
            fake_prob = tool_results.get("fake_probability", 0)
            if fake_prob > 0.7:
                key_findings.append(f"FatFormer AI 생성 확률 높음 ({fake_prob:.0%})")
            elif fake_prob > 0.5:
                key_findings.append(f"FatFormer AI 생성 가능성 ({fake_prob:.0%})")
            elif fake_prob > 0.3:
                uncertainties.append(f"FatFormer 판단 경계 ({fake_prob:.0%})")
            else:
                key_findings.append("FatFormer: AI 생성 특징 미감지")

        elif self.domain == AgentDomain.SPATIAL:
            ratio = tool_results.get("manipulation_ratio", 0)
            if ratio > 0.8:
                key_findings.append(f"조작 영역 {ratio:.0%} - AI 생성 의심")
            elif ratio > 0.05:
                key_findings.append(f"부분 조작 영역 탐지 ({ratio:.0%})")

        return ReasoningResult(
            interpretation="[규칙 기반 해석] Tool 결과를 기본 임계값으로 해석",
            reasoning="LLM API 미사용 - 도메인 지식 기반 규칙 적용",
            verdict_rationale="규칙 기반 판정",
            confidence_rationale="규칙 기반 신뢰도",
            key_findings=key_findings,
            uncertainties=uncertainties
        )

    def respond_to_challenge(
        self,
        challenger_name: str,
        challenge: str,
        my_verdict: str,
        my_confidence: float,
        my_evidence: Dict[str, Any],
        my_reasoning: str
    ) -> DebateResponse:
        """
        다른 에이전트의 반론에 대응

        Args:
            challenger_name: 반론 제기 에이전트 이름
            challenge: 반론 내용
            my_verdict: 내 판정
            my_confidence: 내 신뢰도
            my_evidence: 내 증거
            my_reasoning: 내 추론

        Returns:
            DebateResponse: 토론 응답
        """
        if not self.is_available:
            return self._fallback_respond(
                challenger_name, challenge, my_verdict, my_confidence
            )

        prompt = self._build_debate_prompt(
            challenger_name, challenge, my_verdict, my_confidence, my_evidence, my_reasoning
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_debate_response(response.content[0].text)
        except Exception as e:
            print(f"[SubAgentLLM:{self.domain.value}] 토론 응답 실패: {e}")
            return self._fallback_respond(
                challenger_name, challenge, my_verdict, my_confidence
            )

    def _build_debate_prompt(
        self,
        challenger_name: str,
        challenge: str,
        my_verdict: str,
        my_confidence: float,
        my_evidence: Dict[str, Any],
        my_reasoning: str
    ) -> str:
        """토론 응답 프롬프트 생성"""
        return f"""## 토론 상황

### 내 현재 입장
- **판정**: {my_verdict}
- **신뢰도**: {my_confidence:.1%}
- **추론**: {my_reasoning}
- **증거**: {json.dumps(my_evidence, ensure_ascii=False, indent=2)}

### 반론
**{challenger_name}**의 지적:
\"{challenge}\"

## 요청

위 반론에 대해 논리적으로 응답하세요. 다음을 고려하세요:
1. 반론이 타당한가? - 내 증거로 반박 가능한가?
2. 새로운 관점이 있는가? - 내가 놓친 부분이 있는가?
3. 의견을 수정해야 하는가? - 증거를 재평가할 필요가 있는가?

다음 JSON 형식으로 응답하세요:

```json
{{
  "response_type": "defense|concession|counter|clarification",
  "content": "응답 내용 (반박, 인정, 반론, 또는 설명)",
  "verdict_changed": false,
  "new_verdict": null,
  "new_confidence": null,
  "reasoning": "판정 유지 또는 변경 이유"
}}
```

응답 유형 설명:
- **defense**: 내 입장 방어 - 증거로 반박
- **concession**: 부분/전체 인정 - 상대 논점 수용
- **counter**: 반론 - 상대 논점에 대한 반대 논거 제시
- **clarification**: 설명 - 오해 해소 또는 추가 설명
"""

    def _parse_debate_response(self, response_text: str) -> DebateResponse:
        """토론 응답 파싱"""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                return DebateResponse(
                    response_type=data.get("response_type", "defense"),
                    content=data.get("content", ""),
                    verdict_changed=data.get("verdict_changed", False),
                    new_verdict=data.get("new_verdict"),
                    new_confidence=data.get("new_confidence"),
                    reasoning=data.get("reasoning", "")
                )
        except json.JSONDecodeError:
            pass

        # 파싱 실패 시 방어 응답
        return DebateResponse(
            response_type="defense",
            content=response_text,
            verdict_changed=False,
            reasoning="JSON 파싱 실패로 원문 반환"
        )

    def _fallback_respond(
        self,
        challenger_name: str,
        challenge: str,
        my_verdict: str,
        my_confidence: float
    ) -> DebateResponse:
        """API 불가 시 규칙 기반 응답"""
        return DebateResponse(
            response_type="defense",
            content=(
                f"{challenger_name}의 지적을 검토했습니다. "
                f"내 분석 결과 {my_verdict} 판정은 "
                f"신뢰도 {my_confidence:.1%}의 증거에 기반합니다. "
                f"현재로서는 판정을 유지합니다."
            ),
            verdict_changed=False,
            reasoning="[규칙 기반] API 미사용으로 기본 방어 응답"
        )

    def generate_challenge(
        self,
        target_verdict: str,
        target_confidence: float,
        target_evidence: Dict[str, Any],
        my_verdict: str,
        my_evidence: Dict[str, Any]
    ) -> str:
        """
        다른 에이전트에 대한 반론 생성

        Args:
            target_verdict: 대상 에이전트의 판정
            target_confidence: 대상 에이전트의 신뢰도
            target_evidence: 대상 에이전트의 증거
            my_verdict: 내 판정
            my_evidence: 내 증거

        Returns:
            str: 반론 내용
        """
        if not self.is_available:
            return self._fallback_challenge(
                target_verdict, my_verdict, my_evidence
            )

        prompt = f"""## 반론 생성

### 대상 에이전트 입장
- **판정**: {target_verdict}
- **신뢰도**: {target_confidence:.1%}
- **증거**: {json.dumps(target_evidence, ensure_ascii=False, indent=2)}

### 내 입장
- **판정**: {my_verdict}
- **증거**: {json.dumps(my_evidence, ensure_ascii=False, indent=2)}

## 요청

내 전문 분야의 관점에서 대상 에이전트의 판정에 대해 건설적인 반론을 제기하세요.

다음 원칙을 따르세요:
1. 과학적 근거에 기반한 질문
2. 모호하거나 불확실한 부분 지적
3. 대안적 해석 가능성 제시
4. 존중하는 어조 유지

간결하게 2-3문장으로 반론하세요.
"""

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=500,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"[SubAgentLLM:{self.domain.value}] 반론 생성 실패: {e}")
            return self._fallback_challenge(
                target_verdict, my_verdict, my_evidence
            )

    def _fallback_challenge(
        self,
        target_verdict: str,
        my_verdict: str,
        my_evidence: Dict[str, Any]
    ) -> str:
        """API 불가 시 규칙 기반 반론"""
        if target_verdict == my_verdict:
            return f"판정이 일치합니다 ({my_verdict}). 추가 검토가 필요한 부분이 있을까요?"

        role_info = self.DOMAIN_ROLES[self.domain]
        return (
            f"내 {role_info['focus']} 분석 결과는 {my_verdict}입니다. "
            f"{target_verdict} 판정의 근거를 더 자세히 설명해주시겠습니까?"
        )


# 편의 함수
def create_subagent_llm(domain: str, **kwargs) -> SubAgentLLM:
    """SubAgentLLM 생성 헬퍼"""
    domain_enum = AgentDomain(domain)
    return SubAgentLLM(domain_enum, **kwargs)
