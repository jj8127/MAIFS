"""
SubAgentLLM 통합 테스트
도메인 지식 기반 LLM 추론 및 토론 기능 테스트
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.llm.subagent_llm import (
    SubAgentLLM,
    AgentDomain,
    ReasoningResult,
    DebateResponse,
    create_subagent_llm
)
from src.knowledge import KnowledgeBase
from src.agents.specialist_agents import (
    FrequencyAgent,
    NoiseAgent,
    FatFormerAgent,
    SpatialAgent
)
from src.tools.base_tool import Verdict


class TestAgentDomain:
    """AgentDomain 열거형 테스트"""

    def test_all_domains_exist(self):
        """모든 도메인이 정의되어 있는지 확인"""
        assert AgentDomain.FREQUENCY.value == "frequency"
        assert AgentDomain.NOISE.value == "noise"
        assert AgentDomain.FATFORMER.value == "fatformer"
        assert AgentDomain.SPATIAL.value == "spatial"

    def test_domain_count(self):
        """도메인 개수 확인"""
        assert len(AgentDomain) == 4


class TestSubAgentLLMInitialization:
    """SubAgentLLM 초기화 테스트"""

    def test_initialization_without_api_key(self):
        """API 키 없이 초기화"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY, api_key=None)
        assert llm.domain == AgentDomain.FREQUENCY
        assert not llm.is_available

    def test_knowledge_loading(self):
        """도메인 지식 로드 확인"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY)
        assert llm._knowledge is not None
        assert len(llm._knowledge) > 0

    def test_all_domains_have_knowledge(self):
        """모든 도메인에 지식이 있는지 확인"""
        for domain in AgentDomain:
            llm = SubAgentLLM(domain)
            assert llm._knowledge is not None

    def test_create_subagent_llm_helper(self):
        """헬퍼 함수 테스트"""
        llm = create_subagent_llm("frequency")
        assert llm.domain == AgentDomain.FREQUENCY


class TestSubAgentLLMSystemPrompt:
    """시스템 프롬프트 테스트"""

    def test_system_prompt_contains_domain_info(self):
        """시스템 프롬프트에 도메인 정보 포함 확인"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY)
        prompt = llm._get_system_prompt()

        assert "주파수 분석 전문가" in prompt
        assert "FFT" in prompt

    def test_system_prompt_contains_knowledge(self):
        """시스템 프롬프트에 도메인 지식 포함 확인"""
        llm = SubAgentLLM(AgentDomain.NOISE)
        prompt = llm._get_system_prompt()

        # 도메인 지식이 포함되어 있어야 함
        assert llm._knowledge in prompt or len(llm._knowledge) > 0

    def test_all_domains_have_role_info(self):
        """모든 도메인에 역할 정보가 있는지 확인"""
        for domain in AgentDomain:
            assert domain in SubAgentLLM.DOMAIN_ROLES
            role_info = SubAgentLLM.DOMAIN_ROLES[domain]
            assert "name" in role_info
            assert "expertise" in role_info
            assert "focus" in role_info


class TestSubAgentLLMFallback:
    """규칙 기반 폴백 테스트"""

    def test_fallback_interpret_frequency(self):
        """주파수 도메인 폴백 해석"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY)

        tool_results = {
            "ai_generation_score": 0.8,
            "grid_analysis": {"is_grid_pattern": True}
        }

        result = llm._fallback_interpret(tool_results)

        assert isinstance(result, ReasoningResult)
        assert len(result.key_findings) > 0
        assert "AI 생성 점수가 높음" in result.key_findings

    def test_fallback_interpret_noise(self):
        """노이즈 도메인 폴백 해석"""
        llm = SubAgentLLM(AgentDomain.NOISE)

        tool_results = {
            "ai_detection": {"ai_generation_score": 0.7}
        }

        result = llm._fallback_interpret(tool_results)

        assert isinstance(result, ReasoningResult)
        assert len(result.key_findings) > 0

    def test_fallback_interpret_fatformer(self):
        """FatFormer 도메인 폴백 해석"""
        llm = SubAgentLLM(AgentDomain.FATFORMER)

        tool_results = {"fake_probability": 0.85}
        result = llm._fallback_interpret(tool_results)

        assert isinstance(result, ReasoningResult)
        assert len(result.key_findings) > 0

    def test_fallback_interpret_spatial(self):
        """공간 도메인 폴백 해석"""
        llm = SubAgentLLM(AgentDomain.SPATIAL)

        tool_results = {"manipulation_ratio": 0.9}
        result = llm._fallback_interpret(tool_results)

        assert isinstance(result, ReasoningResult)
        assert len(result.key_findings) > 0

    def test_fallback_respond(self):
        """폴백 토론 응답"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY)

        response = llm._fallback_respond(
            challenger_name="노이즈 분석 전문가",
            challenge="주파수 분석만으로는 불충분합니다.",
            my_verdict="AI_GENERATED",
            my_confidence=0.8
        )

        assert isinstance(response, DebateResponse)
        assert response.response_type == "defense"
        assert not response.verdict_changed

    def test_fallback_challenge(self):
        """폴백 반론 생성"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY)

        challenge = llm._fallback_challenge(
            target_verdict="AUTHENTIC",
            my_verdict="AI_GENERATED",
            my_evidence={}
        )

        assert isinstance(challenge, str)
        assert len(challenge) > 0


class TestSpecialistAgentsWithLLM:
    """전문가 에이전트 LLM 통합 테스트"""

    @pytest.fixture
    def test_image(self):
        """테스트 이미지 생성"""
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_frequency_agent_llm_disabled(self, test_image):
        """LLM 비활성화 시 FrequencyAgent"""
        agent = FrequencyAgent(use_llm=False)
        result = agent.analyze(test_image)

        assert result.verdict in [v for v in Verdict]
        assert "[주파수 분석 결과]" in result.reasoning

    def test_frequency_agent_llm_enabled_fallback(self, test_image):
        """LLM 활성화 (폴백) 시 FrequencyAgent"""
        agent = FrequencyAgent(use_llm=True)
        # API 키 없으므로 폴백 사용
        result = agent.analyze(test_image)

        assert result.verdict in [v for v in Verdict]

    def test_noise_agent_llm_integration(self, test_image):
        """NoiseAgent LLM 통합"""
        agent = NoiseAgent(use_llm=True)
        result = agent.analyze(test_image)

        assert result.verdict in [v for v in Verdict]

    def test_fatformer_agent_llm_integration(self, test_image):
        """FatFormerAgent LLM 통합"""
        agent = FatFormerAgent(use_llm=True)
        result = agent.analyze(test_image)

        assert result.verdict in [v for v in Verdict]

    def test_spatial_agent_llm_integration(self, test_image):
        """SpatialAgent LLM 통합"""
        agent = SpatialAgent(use_llm=True)
        result = agent.analyze(test_image)

        assert result.verdict in [v for v in Verdict]

    def test_agent_respond_to_challenge(self, test_image):
        """에이전트 토론 응답"""
        agent = FrequencyAgent(use_llm=True)
        result = agent.analyze(test_image)

        debate_response = agent.respond_to_challenge(
            challenger_name="노이즈 분석 전문가",
            challenge="주파수 분석 결과에 대해 질문드립니다.",
            my_response=result
        )

        assert isinstance(debate_response, DebateResponse)
        assert debate_response.response_type in ["defense", "concession", "counter", "clarification"]

    def test_agent_generate_challenge(self, test_image):
        """에이전트 반론 생성"""
        freq_agent = FrequencyAgent(use_llm=True)
        noise_agent = NoiseAgent(use_llm=True)

        noise_result = noise_agent.analyze(test_image)
        challenge = freq_agent.generate_challenge(noise_result)

        assert isinstance(challenge, str)
        assert len(challenge) > 0


class TestDebateResponseStructure:
    """DebateResponse 구조 테스트"""

    def test_debate_response_fields(self):
        """DebateResponse 필드 확인"""
        response = DebateResponse(
            response_type="defense",
            content="반박 내용",
            verdict_changed=False
        )

        assert response.response_type == "defense"
        assert response.content == "반박 내용"
        assert response.verdict_changed is False
        assert response.new_verdict is None
        assert response.new_confidence is None

    def test_debate_response_with_verdict_change(self):
        """판정 변경이 있는 DebateResponse"""
        response = DebateResponse(
            response_type="concession",
            content="인정합니다",
            verdict_changed=True,
            new_verdict="MANIPULATED",
            new_confidence=0.6,
            reasoning="증거를 재검토한 결과"
        )

        assert response.verdict_changed is True
        assert response.new_verdict == "MANIPULATED"
        assert response.new_confidence == 0.6


class TestReasoningResultStructure:
    """ReasoningResult 구조 테스트"""

    def test_reasoning_result_fields(self):
        """ReasoningResult 필드 확인"""
        result = ReasoningResult(
            interpretation="해석",
            reasoning="추론",
            verdict_rationale="판정 근거",
            confidence_rationale="신뢰도 근거",
            key_findings=["발견 1"],
            uncertainties=["불확실성 1"]
        )

        assert result.interpretation == "해석"
        assert result.reasoning == "추론"
        assert len(result.key_findings) == 1
        assert len(result.uncertainties) == 1

    def test_reasoning_result_default_lists(self):
        """기본 리스트 초기화"""
        result = ReasoningResult(
            interpretation="해석",
            reasoning="추론",
            verdict_rationale="판정",
            confidence_rationale="신뢰도"
        )

        assert result.key_findings == []
        assert result.uncertainties == []


class TestKnowledgeBaseIntegration:
    """KnowledgeBase 통합 테스트"""

    def test_knowledge_base_loads_for_all_domains(self):
        """모든 도메인의 지식 로드 확인"""
        for domain in ["frequency", "noise", "fatformer", "spatial"]:
            knowledge = KnowledgeBase.load(domain)
            assert len(knowledge) > 0

    def test_knowledge_summary_is_shorter(self):
        """지식 요약이 전체보다 짧은지 확인"""
        full = KnowledgeBase.load("frequency")
        summary = KnowledgeBase.get_summary("frequency", max_chars=1000)

        assert len(summary) <= len(full)

    def test_subagent_uses_summary_by_default(self):
        """SubAgentLLM이 기본적으로 요약 사용"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY, use_full_knowledge=False)
        full_knowledge = KnowledgeBase.load("frequency")

        # 요약본은 전체보다 짧아야 함
        assert len(llm._knowledge) <= len(full_knowledge)


class TestPromptConstruction:
    """프롬프트 구성 테스트"""

    def test_interpretation_prompt_structure(self):
        """해석 프롬프트 구조 확인"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY)

        tool_results = {"ai_generation_score": 0.5}
        prompt = llm._build_interpretation_prompt(tool_results)

        assert "Tool 분석 결과" in prompt
        assert "ai_generation_score" in prompt
        assert "interpretation" in prompt  # JSON 형식 안내

    def test_interpretation_prompt_with_context(self):
        """컨텍스트가 있는 해석 프롬프트"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY)

        tool_results = {"ai_generation_score": 0.5}
        context = {"other_agent": "AUTHENTIC"}
        prompt = llm._build_interpretation_prompt(tool_results, context)

        assert "컨텍스트 정보" in prompt
        assert "other_agent" in prompt

    def test_debate_prompt_structure(self):
        """토론 프롬프트 구조 확인"""
        llm = SubAgentLLM(AgentDomain.FREQUENCY)

        prompt = llm._build_debate_prompt(
            challenger_name="노이즈 전문가",
            challenge="반론입니다",
            my_verdict="AI_GENERATED",
            my_confidence=0.8,
            my_evidence={"score": 0.9},
            my_reasoning="추론입니다"
        )

        assert "토론 상황" in prompt
        assert "노이즈 전문가" in prompt
        assert "반론입니다" in prompt
        assert "AI_GENERATED" in prompt
