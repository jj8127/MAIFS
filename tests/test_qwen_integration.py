"""
Qwen vLLM 통합 테스트
QwenClient 및 QwenMAIFSAdapter 테스트
"""
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from src.llm.qwen_client import (
    QwenClient,
    QwenClientSync,
    AgentRole,
    InferenceResult,
    AGENT_OUTPUT_SCHEMA,
    DEBATE_RESPONSE_SCHEMA
)
from src.llm.qwen_maifs_adapter import (
    QwenMAIFSAdapter,
    QwenAnalysisResult,
    create_qwen_adapter
)
from src.tools.base_tool import Verdict


class TestAgentRole:
    """AgentRole 열거형 테스트"""

    def test_all_roles_exist(self):
        assert AgentRole.FREQUENCY.value == "frequency"
        assert AgentRole.NOISE.value == "noise"
        assert AgentRole.FATFORMER.value == "fatformer"
        assert AgentRole.SPATIAL.value == "spatial"
        assert AgentRole.MANAGER.value == "manager"

    def test_role_count(self):
        assert len(AgentRole) == 5


class TestQwenClientInitialization:
    """QwenClient 초기화 테스트"""

    def test_default_initialization(self):
        client = QwenClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 60.0
        assert client.max_retries == 3

    def test_custom_initialization(self):
        client = QwenClient(
            base_url="http://custom:9000",
            timeout=30.0,
            max_retries=5
        )
        assert client.base_url == "http://custom:9000"
        assert client.timeout == 30.0
        assert client.max_retries == 5

    def test_knowledge_loading(self):
        client = QwenClient()
        # 모든 역할에 대해 지식이 로드되어야 함
        assert AgentRole.FREQUENCY in client._knowledge_cache
        assert AgentRole.NOISE in client._knowledge_cache
        assert AgentRole.FATFORMER in client._knowledge_cache
        assert AgentRole.SPATIAL in client._knowledge_cache


class TestSystemPrompts:
    """시스템 프롬프트 테스트"""

    def test_all_roles_have_prompts(self):
        for role in [AgentRole.FREQUENCY, AgentRole.NOISE,
                     AgentRole.FATFORMER, AgentRole.SPATIAL, AgentRole.MANAGER]:
            assert role in QwenClient.SYSTEM_PROMPTS
            assert len(QwenClient.SYSTEM_PROMPTS[role]) > 100

    def test_frequency_prompt_content(self):
        prompt = QwenClient.SYSTEM_PROMPTS[AgentRole.FREQUENCY]
        assert "주파수" in prompt or "FFT" in prompt
        assert "GAN" in prompt

    def test_noise_prompt_content(self):
        prompt = QwenClient.SYSTEM_PROMPTS[AgentRole.NOISE]
        assert "PRNU" in prompt or "노이즈" in prompt

    def test_fatformer_prompt_content(self):
        prompt = QwenClient.SYSTEM_PROMPTS[AgentRole.FATFORMER]
        assert "FatFormer" in prompt or "AI 생성" in prompt

    def test_spatial_prompt_content(self):
        prompt = QwenClient.SYSTEM_PROMPTS[AgentRole.SPATIAL]
        assert "공간" in prompt or "ViT" in prompt

    def test_full_system_prompt_includes_knowledge(self):
        client = QwenClient()
        full_prompt = client._get_full_system_prompt(AgentRole.FREQUENCY)
        base_prompt = QwenClient.SYSTEM_PROMPTS[AgentRole.FREQUENCY]

        # 전체 프롬프트가 기본 프롬프트보다 길어야 함 (지식 포함)
        assert len(full_prompt) >= len(base_prompt)


class TestJSONSchemas:
    """JSON 스키마 테스트"""

    def test_agent_output_schema_structure(self):
        assert "type" in AGENT_OUTPUT_SCHEMA
        assert AGENT_OUTPUT_SCHEMA["type"] == "object"
        assert "properties" in AGENT_OUTPUT_SCHEMA
        assert "verdict" in AGENT_OUTPUT_SCHEMA["properties"]
        assert "confidence" in AGENT_OUTPUT_SCHEMA["properties"]
        assert "reasoning" in AGENT_OUTPUT_SCHEMA["properties"]

    def test_verdict_enum_values(self):
        verdict_schema = AGENT_OUTPUT_SCHEMA["properties"]["verdict"]
        assert "enum" in verdict_schema
        assert "AUTHENTIC" in verdict_schema["enum"]
        assert "MANIPULATED" in verdict_schema["enum"]
        assert "AI_GENERATED" in verdict_schema["enum"]
        assert "UNCERTAIN" in verdict_schema["enum"]

    def test_debate_response_schema_structure(self):
        assert "type" in DEBATE_RESPONSE_SCHEMA
        assert "response_type" in DEBATE_RESPONSE_SCHEMA["properties"]
        assert "verdict_changed" in DEBATE_RESPONSE_SCHEMA["properties"]


class TestInferenceResult:
    """InferenceResult 테스트"""

    def test_successful_result(self):
        result = InferenceResult(
            role=AgentRole.FREQUENCY,
            content='{"verdict": "AI_GENERATED", "confidence": 0.85}',
            parsed_json={"verdict": "AI_GENERATED", "confidence": 0.85},
            latency_ms=150.0,
            success=True
        )

        assert result.success
        assert result.role == AgentRole.FREQUENCY
        assert result.parsed_json["verdict"] == "AI_GENERATED"

    def test_failed_result(self):
        result = InferenceResult(
            role=AgentRole.NOISE,
            content="",
            success=False,
            error="Connection timeout"
        )

        assert not result.success
        assert result.error == "Connection timeout"


class TestQwenClientPromptBuilding:
    """프롬프트 생성 테스트"""

    def test_build_user_prompt(self):
        client = QwenClient()
        tool_results = {"ai_score": 0.8, "grid_pattern": True}

        prompt = client._build_user_prompt(tool_results)

        assert "Tool 분석 결과" in prompt
        assert "ai_score" in prompt
        assert "JSON" in prompt

    def test_build_user_prompt_with_context(self):
        client = QwenClient()
        tool_results = {"score": 0.5}
        context = {"other_agent": "AUTHENTIC"}

        prompt = client._build_user_prompt(tool_results, context)

        assert "컨텍스트" in prompt
        assert "other_agent" in prompt


class TestQwenClientJSONParsing:
    """JSON 파싱 테스트"""

    def test_parse_valid_json(self):
        client = QwenClient()
        content = '{"verdict": "AUTHENTIC", "confidence": 0.9}'

        result = client._parse_json(content)

        assert result is not None
        assert result["verdict"] == "AUTHENTIC"

    def test_parse_json_with_surrounding_text(self):
        client = QwenClient()
        content = 'Here is my analysis:\n{"verdict": "MANIPULATED", "confidence": 0.7}\nEnd.'

        result = client._parse_json(content)

        assert result is not None
        assert result["verdict"] == "MANIPULATED"

    def test_parse_invalid_json(self):
        client = QwenClient()
        content = "This is not JSON at all"

        result = client._parse_json(content)

        assert result is None


class TestQwenAnalysisResult:
    """QwenAnalysisResult 테스트"""

    def test_result_creation(self):
        result = QwenAnalysisResult(
            verdict=Verdict.AI_GENERATED,
            confidence=0.85,
            reasoning="High frequency artifacts detected",
            key_evidence=["Grid pattern", "Spectral anomaly"],
            uncertainties=["Noise level unclear"]
        )

        assert result.verdict == Verdict.AI_GENERATED
        assert result.confidence == 0.85
        assert len(result.key_evidence) == 2

    def test_to_dict(self):
        result = QwenAnalysisResult(
            verdict=Verdict.AUTHENTIC,
            confidence=0.9,
            reasoning="Natural noise pattern"
        )

        d = result.to_dict()

        assert d["verdict"] == "AUTHENTIC"
        assert d["confidence"] == 0.9
        assert "reasoning" in d


class TestQwenMAIFSAdapter:
    """QwenMAIFSAdapter 테스트"""

    def test_initialization(self):
        adapter = QwenMAIFSAdapter(
            base_url="http://localhost:8000",
            enable_debate=True,
            max_debate_rounds=3
        )

        assert adapter.enable_debate
        assert adapter.max_debate_rounds == 3

    def test_name_to_role_mapping(self):
        adapter = QwenMAIFSAdapter()

        assert adapter._name_to_role("frequency") == AgentRole.FREQUENCY
        assert adapter._name_to_role("noise") == AgentRole.NOISE
        assert adapter._name_to_role("fatformer") == AgentRole.FATFORMER
        assert adapter._name_to_role("spatial") == AgentRole.SPATIAL
        assert adapter._name_to_role("unknown") is None

    def test_convert_successful_result(self):
        adapter = QwenMAIFSAdapter()

        inference_result = InferenceResult(
            role=AgentRole.FREQUENCY,
            content='{"verdict": "AI_GENERATED", "confidence": 0.8, "reasoning": "test"}',
            parsed_json={
                "verdict": "AI_GENERATED",
                "confidence": 0.8,
                "reasoning": "test",
                "key_evidence": ["evidence1"]
            },
            success=True
        )

        result = adapter._convert_result(inference_result)

        assert result.verdict == Verdict.AI_GENERATED
        assert result.confidence == 0.8
        assert result.reasoning == "test"

    def test_convert_failed_result(self):
        adapter = QwenMAIFSAdapter()

        inference_result = InferenceResult(
            role=AgentRole.NOISE,
            content="",
            success=False,
            error="Timeout"
        )

        result = adapter._convert_result(inference_result)

        assert result.verdict == Verdict.UNCERTAIN
        assert result.confidence == 0.5
        assert "Timeout" in result.reasoning

    def test_group_by_verdict(self):
        adapter = QwenMAIFSAdapter()

        results = {
            "frequency": QwenAnalysisResult(Verdict.AI_GENERATED, 0.8, ""),
            "noise": QwenAnalysisResult(Verdict.AI_GENERATED, 0.7, ""),
            "fatformer": QwenAnalysisResult(Verdict.AUTHENTIC, 0.6, ""),
            "spatial": QwenAnalysisResult(Verdict.AI_GENERATED, 0.9, "")
        }

        groups = adapter._group_by_verdict(results)

        assert len(groups[Verdict.AI_GENERATED]) == 3
        assert len(groups[Verdict.AUTHENTIC]) == 1


class TestCreateHelpers:
    """헬퍼 함수 테스트"""

    def test_create_qwen_adapter_async(self):
        adapter = create_qwen_adapter(sync=False)
        assert isinstance(adapter, QwenMAIFSAdapter)

    def test_create_qwen_adapter_sync(self):
        adapter = create_qwen_adapter(sync=True)
        from src.llm.qwen_maifs_adapter import QwenMAIFSAdapterSync
        assert isinstance(adapter, QwenMAIFSAdapterSync)


class TestQwenClientSyncWrapper:
    """동기 래퍼 테스트"""

    def test_sync_client_initialization(self):
        client = QwenClientSync(base_url="http://localhost:8000")
        assert client._client is not None
        client.close()


class TestDebateGeneration:
    """토론 생성 테스트"""

    def test_generate_challenge(self):
        adapter = QwenMAIFSAdapter()

        challenger = QwenAnalysisResult(
            verdict=Verdict.AI_GENERATED,
            confidence=0.85,
            reasoning="High frequency artifacts detected in the spectrum"
        )
        target = QwenAnalysisResult(
            verdict=Verdict.AUTHENTIC,
            confidence=0.7,
            reasoning="Natural noise pattern"
        )

        challenge = adapter._generate_challenge(
            "frequency", challenger, "noise", target
        )

        assert "frequency" in challenge
        assert "AI_GENERATED" in challenge
        assert "AUTHENTIC" in challenge
