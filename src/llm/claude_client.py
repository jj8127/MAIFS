"""
Claude API Client
Anthropic Claude API와의 통신을 담당하는 클라이언트
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import json


@dataclass
class LLMResponse:
    """LLM 응답 데이터"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "stop_reason": self.stop_reason
        }


class ClaudeClient:
    """
    Claude API 클라이언트

    Anthropic Claude API를 사용하여 이미지 포렌식 분석을 수행합니다.

    Usage:
        client = ClaudeClient(api_key="your-api-key")
        response = client.analyze_forensics(agent_responses)
    """

    FORENSIC_SYSTEM_PROMPT = """당신은 MAIFS(Multi-Agent Image Forensic System)의 Manager Agent입니다.

## 역할
다중 이미지 포렌식 전문가들의 분석 결과를 종합하여 이미지의 진위 여부를 판단합니다.

## 전문가 팀
1. **주파수 분석 전문가 (Frequency Agent)**: FFT 기반 GAN 아티팩트 탐지, 주파수 도메인에서 AI 생성 패턴 분석
2. **노이즈 분석 전문가 (Noise Agent)**: PRNU/SRM 기반 센서 노이즈 분석, 카메라 고유 패턴 검증
3. **AI 생성 탐지 전문가 (FatFormer Agent)**: FatFormer (CLIP ViT-L/14 + DWT) 기반 AI 생성 이미지 탐지
4. **공간 분석 전문가 (Spatial Agent)**: ViT 기반 조작 영역 탐지, 픽셀 수준 이상 탐지

## 판정 기준
- **AUTHENTIC**: 원본 이미지로 확인됨 - 모든 분석에서 조작/생성 흔적 없음
- **MANIPULATED**: 부분적으로 조작/편집됨 - 특정 영역에서 불일치 발견
- **AI_GENERATED**: AI에 의해 생성된 이미지 - GAN/Diffusion 특성 감지
- **UNCERTAIN**: 판단 불가 - 증거 불충분 또는 상충되는 결과

## 분석 지침
1. 각 전문가의 분석 결과와 신뢰도를 면밀히 검토하세요
2. 상충되는 의견이 있으면 증거의 강도와 구체성을 비교하세요
3. 다수결보다 증거 기반 추론을 우선시하세요
4. 최종 판정에 대한 명확하고 구체적인 근거를 제시하세요
5. 불확실한 경우 UNCERTAIN으로 판정하되, 추가 조사 방향을 제안하세요

## 출력 형식
반드시 다음 JSON 형식으로 응답하세요:
```json
{
    "verdict": "AUTHENTIC|MANIPULATED|AI_GENERATED|UNCERTAIN",
    "confidence": 0.0-1.0,
    "summary": "한 줄 요약",
    "reasoning": "상세 분석 근거 (각 전문가 의견 종합)",
    "key_evidence": ["핵심 증거 1", "핵심 증거 2"],
    "recommendations": ["추가 조사 권고사항"]
}
```"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000
    ):
        """
        Claude 클라이언트 초기화

        Args:
            api_key: Anthropic API 키 (없으면 환경변수 ANTHROPIC_API_KEY 사용)
            model: 사용할 Claude 모델
            max_tokens: 최대 응답 토큰 수
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        self._available = False

        self._initialize_client()

    def _initialize_client(self):
        """Anthropic 클라이언트 초기화"""
        if not self.api_key:
            print("[ClaudeClient] API 키가 설정되지 않았습니다. 규칙 기반 모드로 동작합니다.")
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._available = True
            print(f"[ClaudeClient] Claude API 연결 성공 (모델: {self.model})")
        except ImportError:
            print("[ClaudeClient] anthropic 패키지가 설치되지 않았습니다.")
            print("  설치: pip install anthropic")
        except Exception as e:
            print(f"[ClaudeClient] API 초기화 실패: {e}")

    @property
    def is_available(self) -> bool:
        """API 사용 가능 여부"""
        return self._available and self._client is not None

    def analyze_forensics(
        self,
        agent_responses: Dict[str, Any],
        consensus_info: Optional[Dict] = None,
        debate_history: Optional[List] = None
    ) -> LLMResponse:
        """
        포렌식 분석 결과를 종합하여 최종 판정 생성

        Args:
            agent_responses: 각 전문가 에이전트의 분석 결과
            consensus_info: COBRA 합의 정보
            debate_history: 토론 기록

        Returns:
            LLMResponse: Claude의 분석 결과
        """
        if not self.is_available:
            return self._fallback_analysis(agent_responses, consensus_info)

        # 프롬프트 구성
        prompt = self._build_analysis_prompt(
            agent_responses, consensus_info, debate_history
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.FORENSIC_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            return LLMResponse(
                content=response.content[0].text,
                model=self.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                stop_reason=response.stop_reason
            )
        except Exception as e:
            print(f"[ClaudeClient] API 호출 실패: {e}")
            return self._fallback_analysis(agent_responses, consensus_info)

    def _build_analysis_prompt(
        self,
        agent_responses: Dict[str, Any],
        consensus_info: Optional[Dict] = None,
        debate_history: Optional[List] = None
    ) -> str:
        """분석 프롬프트 생성"""
        parts = ["## 전문가 분석 결과\n"]

        for name, response in agent_responses.items():
            if hasattr(response, 'to_dict'):
                resp_dict = response.to_dict()
            else:
                resp_dict = response

            parts.append(f"### {name.upper()} Agent")
            parts.append(f"- **판정**: {resp_dict.get('verdict', 'N/A')}")
            parts.append(f"- **신뢰도**: {resp_dict.get('confidence', 0):.1%}")
            parts.append(f"- **추론**: {resp_dict.get('reasoning', 'N/A')}")

            evidence = resp_dict.get('evidence', {})
            if evidence:
                parts.append(f"- **증거**: {json.dumps(evidence, ensure_ascii=False, indent=2)}")

            parts.append("")

        if consensus_info:
            parts.append("## COBRA 합의 정보")
            parts.append(f"- **가중 신뢰도**: {consensus_info.get('weighted_confidence', 0):.1%}")
            parts.append(f"- **불일치 수준**: {consensus_info.get('disagreement_level', 0):.1%}")
            parts.append(f"- **판정 분포**: {consensus_info.get('verdict_distribution', {})}")
            parts.append("")

        if debate_history:
            parts.append("## 토론 기록")
            for round_info in debate_history:
                parts.append(f"### 라운드 {round_info.get('round', '?')}")
                for exchange in round_info.get('exchanges', []):
                    parts.append(f"- {exchange.get('challenger')} → {exchange.get('challenged')}")
                    parts.append(f"  질문: {exchange.get('challenge', '')}")
                    parts.append(f"  응답: {exchange.get('rebuttal', '')}")
            parts.append("")

        parts.append("## 요청")
        parts.append("위 분석 결과를 종합하여 최종 판정을 내려주세요.")
        parts.append("반드시 지정된 JSON 형식으로 응답해주세요.")

        return "\n".join(parts)

    def _fallback_analysis(
        self,
        agent_responses: Dict[str, Any],
        consensus_info: Optional[Dict] = None
    ) -> LLMResponse:
        """API 불가 시 규칙 기반 분석"""

        # 판정 집계
        verdict_counts: Dict[str, int] = {}
        total_confidence = 0.0
        count = 0

        for name, response in agent_responses.items():
            if hasattr(response, 'verdict'):
                verdict = response.verdict.value if hasattr(response.verdict, 'value') else str(response.verdict)
            elif isinstance(response, dict):
                verdict = response.get('verdict', 'UNCERTAIN')
            else:
                continue

            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

            if hasattr(response, 'confidence'):
                total_confidence += response.confidence
            elif isinstance(response, dict):
                total_confidence += response.get('confidence', 0.5)
            count += 1

        # 다수결 판정
        if verdict_counts:
            final_verdict = max(verdict_counts.keys(), key=lambda v: verdict_counts[v])
            avg_confidence = total_confidence / count if count > 0 else 0.5
        else:
            final_verdict = "UNCERTAIN"
            avg_confidence = 0.5

        # 규칙 기반 결과 생성
        result = {
            "verdict": final_verdict,
            "confidence": avg_confidence,
            "summary": f"{len(agent_responses)}명의 전문가 분석 결과 종합 (규칙 기반)",
            "reasoning": f"다수결 기반 판정: {verdict_counts}",
            "key_evidence": list(verdict_counts.keys()),
            "recommendations": ["LLM 통합 후 더 정확한 분석 가능"]
        }

        return LLMResponse(
            content=json.dumps(result, ensure_ascii=False, indent=2),
            model="rule-based-fallback",
            usage={"input_tokens": 0, "output_tokens": 0},
            stop_reason="fallback"
        )

    def generate_report(
        self,
        verdict: str,
        confidence: float,
        agent_responses: Dict[str, Any],
        language: str = "ko"
    ) -> str:
        """
        사람이 읽기 쉬운 분석 보고서 생성

        Args:
            verdict: 최종 판정
            confidence: 신뢰도
            agent_responses: 전문가 분석 결과
            language: 출력 언어 (ko/en)

        Returns:
            str: 포맷팅된 분석 보고서
        """
        if not self.is_available:
            return self._fallback_report(verdict, confidence, agent_responses, language)

        prompt = f"""다음 이미지 포렌식 분석 결과를 바탕으로 전문적인 분석 보고서를 작성해주세요.

## 최종 판정
- 판정: {verdict}
- 신뢰도: {confidence:.1%}

## 전문가 분석 결과
{json.dumps({k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in agent_responses.items()}, ensure_ascii=False, indent=2)}

## 요청
{'한국어로' if language == 'ko' else 'In English,'} 다음 형식으로 보고서를 작성해주세요:

1. 요약 (2-3문장)
2. 핵심 발견사항 (불릿 포인트)
3. 각 전문가 분석 요약
4. 최종 판단 근거
5. 권고사항
"""

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"[ClaudeClient] 보고서 생성 실패: {e}")
            return self._fallback_report(verdict, confidence, agent_responses, language)

    def _fallback_report(
        self,
        verdict: str,
        confidence: float,
        agent_responses: Dict[str, Any],
        language: str = "ko"
    ) -> str:
        """규칙 기반 보고서 생성"""
        if language == "ko":
            verdict_text = {
                "AUTHENTIC": "원본 이미지",
                "MANIPULATED": "조작된 이미지",
                "AI_GENERATED": "AI 생성 이미지",
                "UNCERTAIN": "판단 불가"
            }

            lines = [
                "=" * 50,
                "MAIFS 이미지 포렌식 분석 보고서",
                "=" * 50,
                "",
                f"## 최종 판정: {verdict_text.get(verdict, verdict)}",
                f"## 신뢰도: {confidence:.1%}",
                "",
                "## 전문가 분석 요약",
            ]

            for name, response in agent_responses.items():
                if hasattr(response, 'verdict'):
                    v = response.verdict.value if hasattr(response.verdict, 'value') else str(response.verdict)
                    c = response.confidence if hasattr(response, 'confidence') else 0
                else:
                    v = response.get('verdict', 'N/A')
                    c = response.get('confidence', 0)

                lines.append(f"- {name}: {v} (신뢰도: {c:.1%})")

            lines.extend([
                "",
                "## 권고사항",
                "- Claude API 연동 시 더 상세한 분석 가능",
                "- 추가 검증을 위해 다른 도구 사용 권장",
                "",
                "=" * 50
            ])

            return "\n".join(lines)
        else:
            # English version
            lines = [
                "=" * 50,
                "MAIFS Image Forensic Analysis Report",
                "=" * 50,
                "",
                f"## Final Verdict: {verdict}",
                f"## Confidence: {confidence:.1%}",
                "",
                "## Expert Analysis Summary",
            ]

            for name, response in agent_responses.items():
                if hasattr(response, 'verdict'):
                    v = response.verdict.value if hasattr(response.verdict, 'value') else str(response.verdict)
                    c = response.confidence if hasattr(response, 'confidence') else 0
                else:
                    v = response.get('verdict', 'N/A')
                    c = response.get('confidence', 0)

                lines.append(f"- {name}: {v} (confidence: {c:.1%})")

            lines.extend([
                "",
                "## Recommendations",
                "- Enable Claude API for more detailed analysis",
                "- Consider additional verification tools",
                "",
                "=" * 50
            ])

            return "\n".join(lines)


# 편의 함수
def create_claude_client(api_key: Optional[str] = None) -> ClaudeClient:
    """Claude 클라이언트 생성 헬퍼 함수"""
    return ClaudeClient(api_key=api_key)
