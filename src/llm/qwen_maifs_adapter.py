"""
Qwen MAIFS Adapter
QwenClient를 MAIFS 시스템과 통합하는 어댑터

기존 MAIFS 파이프라인에 Qwen vLLM 기반 추론을 연결합니다.
"""
import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .qwen_client import (
    QwenClient,
    QwenClientSync,
    AgentRole,
    InferenceResult
)
from ..tools.base_tool import Verdict


@dataclass
class QwenAnalysisResult:
    """Qwen 분석 결과"""
    verdict: Verdict
    confidence: float
    reasoning: str
    key_evidence: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    raw_result: Optional[InferenceResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value.upper(),  # 대문자로 반환 (일관성)
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_evidence": self.key_evidence,
            "uncertainties": self.uncertainties
        }


class QwenMAIFSAdapter:
    """
    Qwen MAIFS 어댑터

    Tool 결과를 받아 Qwen LLM으로 해석하고,
    MAIFS 호환 형식으로 반환합니다.

    Usage:
        adapter = QwenMAIFSAdapter(base_url="http://localhost:8000")

        # Tool 결과들을 Qwen으로 분석
        results = await adapter.analyze_with_qwen(tool_results_map)

        # 토론 수행
        debate_result = await adapter.conduct_debate(results)
    """

    # AgentRole과 기존 MAIFS 에이전트 이름 매핑
    ROLE_NAME_MAP = {
        AgentRole.FREQUENCY: "frequency",
        AgentRole.NOISE: "noise",
        AgentRole.FATFORMER: "fatformer",
        AgentRole.SPATIAL: "spatial"
    }

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: Optional[str] = None,
        enable_debate: bool = True,
        max_debate_rounds: int = 3,
        consensus_threshold: float = 0.7
    ):
        """
        어댑터 초기화

        Args:
            base_url: vLLM 서버 URL
            model_name: vLLM에 노출된 모델 이름
            enable_debate: 토론 활성화 여부
            max_debate_rounds: 최대 토론 라운드
            consensus_threshold: 합의 임계값
        """
        self.client = QwenClient(base_url=base_url, model_name=model_name)
        self.enable_debate = enable_debate
        self.max_debate_rounds = max_debate_rounds
        self.consensus_threshold = consensus_threshold

    async def analyze_with_qwen(
        self,
        tool_results_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, QwenAnalysisResult]:
        """
        Tool 결과를 Qwen으로 분석

        Args:
            tool_results_map: {"frequency": {...}, "noise": {...}, ...}

        Returns:
            Dict[str, QwenAnalysisResult]: 에이전트별 분석 결과
        """
        # 역할 매핑
        role_results_map = {}
        for name, results in tool_results_map.items():
            role = self._name_to_role(name)
            if role:
                role_results_map[role] = results

        # 배치 추론
        inference_results = await self.client.batch_infer(role_results_map)

        # 결과 변환
        analysis_results = {}
        for role, result in inference_results.items():
            name = self.ROLE_NAME_MAP.get(role, role.value)
            qwen_result = self._convert_result(result)

            # Fallback 모드 정보를 uncertainties에 추가
            tool_results = tool_results_map.get(name, {})
            if tool_results.get("fallback_mode"):
                fallback_msg = (
                    f"{name.capitalize()} tool is running in fallback mode "
                    f"(model unavailable). Results may be less accurate."
                )
                if fallback_msg not in qwen_result.uncertainties:
                    qwen_result.uncertainties.insert(0, fallback_msg)

            analysis_results[name] = qwen_result

        return analysis_results

    async def analyze_single(
        self,
        agent_name: str,
        tool_results: Dict[str, Any]
    ) -> QwenAnalysisResult:
        """
        단일 에이전트 분석

        Args:
            agent_name: 에이전트 이름 (frequency, noise, fatformer, spatial)
            tool_results: Tool 분석 결과

        Returns:
            QwenAnalysisResult: 분석 결과
        """
        role = self._name_to_role(agent_name)
        if not role:
            raise ValueError(f"Unknown agent name: {agent_name}")

        result = await self.client.infer(role, tool_results)
        return self._convert_result(result)

    async def conduct_debate(
        self,
        analysis_results: Dict[str, QwenAnalysisResult]
    ) -> Dict[str, Any]:
        """
        토론 수행

        Args:
            analysis_results: 에이전트별 분석 결과

        Returns:
            Dict: 토론 결과 (변경된 판정, 합의 여부 등)
        """
        if not self.enable_debate:
            return {"debate_conducted": False}

        # 불일치 확인
        verdicts = {name: r.verdict for name, r in analysis_results.items()}
        unique_verdicts = set(verdicts.values())

        if len(unique_verdicts) <= 1:
            return {
                "debate_conducted": False,
                "reason": "unanimous_agreement",
                "verdict": list(unique_verdicts)[0].value if unique_verdicts else "UNCERTAIN"
            }

        # 토론 수행
        debate_history = []
        current_results = dict(analysis_results)

        for round_num in range(self.max_debate_rounds):
            round_exchanges = []

            # 다수 의견과 소수 의견 식별
            verdict_groups = self._group_by_verdict(current_results)
            majority_verdict, majority_agents = max(
                verdict_groups.items(),
                key=lambda x: len(x[1])
            )

            # 소수 의견 에이전트들이 다수에게 반론 제기
            for verdict, agents in verdict_groups.items():
                if verdict == majority_verdict:
                    continue

                for challenger_name in agents:
                    challenger = current_results[challenger_name]

                    # 다수 의견 대표에게 반론
                    for target_name in majority_agents[:1]:  # 대표 1명에게만
                        target = current_results[target_name]

                        # 반론 생성 및 응답
                        challenge = self._generate_challenge(
                            challenger_name, challenger, target_name, target
                        )

                        response = await self.client.debate_respond(
                            role=self._name_to_role(target_name),
                            my_verdict=target.verdict.value,
                            my_confidence=target.confidence,
                            my_evidence={"reasoning": target.reasoning},
                            challenger_name=challenger_name,
                            challenge=challenge
                        )

                        # 응답 처리
                        if response.success and response.parsed_json:
                            parsed = response.parsed_json
                            verdict_changed = parsed.get("verdict_changed", False)

                            if verdict_changed:
                                new_verdict_str = parsed.get("new_verdict")
                                new_confidence = parsed.get("new_confidence", target.confidence)

                                if new_verdict_str:
                                    try:
                                        # 대문자를 소문자로 변환 (Verdict enum은 소문자)
                                        new_verdict = Verdict(str(new_verdict_str).lower())
                                        current_results[target_name] = QwenAnalysisResult(
                                            verdict=new_verdict,
                                            confidence=new_confidence,
                                            reasoning=parsed.get("reasoning", target.reasoning),
                                            key_evidence=target.key_evidence,
                                            uncertainties=target.uncertainties
                                        )
                                    except (ValueError, AttributeError):
                                        # Invalid verdict, keep original
                                        pass

                            round_exchanges.append({
                                "challenger": challenger_name,
                                "target": target_name,
                                "challenge": challenge,
                                "response": parsed.get("content", ""),
                                "verdict_changed": verdict_changed
                            })

            debate_history.append({
                "round": round_num + 1,
                "exchanges": round_exchanges
            })

            # 합의 확인
            current_verdicts = {name: r.verdict for name, r in current_results.items()}
            if len(set(current_verdicts.values())) == 1:
                break

        # 최종 결과
        final_verdicts = {name: r.verdict for name, r in current_results.items()}
        consensus = len(set(final_verdicts.values())) == 1

        return {
            "debate_conducted": True,
            "rounds": len(debate_history),
            "consensus_reached": consensus,
            "final_verdicts": {k: v.value for k, v in final_verdicts.items()},
            "history": debate_history,
            "updated_results": current_results
        }

    def _name_to_role(self, name: str) -> Optional[AgentRole]:
        """에이전트 이름을 역할로 변환"""
        name_lower = name.lower()
        for role, role_name in self.ROLE_NAME_MAP.items():
            if role_name in name_lower or role.value in name_lower:
                return role
        return None

    def _convert_result(self, result: InferenceResult) -> QwenAnalysisResult:
        """InferenceResult를 QwenAnalysisResult로 변환"""
        if not result.success or not result.parsed_json:
            return QwenAnalysisResult(
                verdict=Verdict.UNCERTAIN,
                confidence=0.5,
                reasoning=result.error or "분석 실패",
                raw_result=result
            )

        parsed = result.parsed_json
        if not isinstance(parsed, dict):
            return QwenAnalysisResult(
                verdict=Verdict.UNCERTAIN,
                confidence=0.5,
                reasoning="응답 형식이 올바르지 않습니다.",
                raw_result=result
            )

        # verdict 변환 (다양한 필드명 지원)
        verdict_str = (
            parsed.get("verdict") or
            parsed.get("analysis_result") or
            parsed.get("prediction") or
            "UNCERTAIN"
        )
        try:
            # LLM은 대문자로 반환, Verdict enum은 소문자
            verdict = Verdict(str(verdict_str).lower())
        except (ValueError, AttributeError):
            verdict = Verdict.UNCERTAIN

        # confidence 변환 (숫자 또는 텍스트 레벨)
        confidence = parsed.get("confidence")
        if confidence is None:
            # confidence_level 텍스트를 숫자로 변환
            conf_level = str(parsed.get("confidence_level", "")).lower()
            confidence_map = {
                "high": 0.85,
                "medium": 0.65,
                "low": 0.4,
                "uncertain": 0.5
            }
            confidence = confidence_map.get(conf_level, 0.5)

        # confidence가 문자열인 경우 처리
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 0.5

        # 범위 제한
        confidence = max(0.0, min(1.0, float(confidence)))

        # reasoning 변환 (객체면 요약)
        reasoning = parsed.get("reasoning", "")
        if not isinstance(reasoning, str):
            # 객체인 경우 핵심 내용 추출
            if isinstance(reasoning, dict):
                parts = []
                for key, value in reasoning.items():
                    if isinstance(value, dict):
                        conclusion = value.get("conclusion") or value.get("description", "")
                        if conclusion:
                            parts.append(f"{key}: {conclusion}")
                    else:
                        parts.append(f"{key}: {value}")
                reasoning = "; ".join(parts) if parts else json.dumps(reasoning, ensure_ascii=False)
            else:
                reasoning = json.dumps(reasoning, ensure_ascii=False)

        # key_evidence 변환
        key_evidence = parsed.get("key_evidence", [])
        if isinstance(key_evidence, str):
            key_evidence = [key_evidence]
        elif key_evidence is None:
            key_evidence = []
        elif not isinstance(key_evidence, list):
            key_evidence = [str(key_evidence)]

        # uncertainties 변환
        uncertainties = parsed.get("uncertainties", [])
        if isinstance(uncertainties, str):
            uncertainties = [uncertainties]
        elif uncertainties is None:
            uncertainties = []
        elif not isinstance(uncertainties, list):
            uncertainties = [str(uncertainties)]

        return QwenAnalysisResult(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            key_evidence=key_evidence,
            uncertainties=uncertainties,
            raw_result=result
        )

    def _group_by_verdict(
        self,
        results: Dict[str, QwenAnalysisResult]
    ) -> Dict[Verdict, List[str]]:
        """판정별로 에이전트 그룹화"""
        groups: Dict[Verdict, List[str]] = {}
        for name, result in results.items():
            if result.verdict not in groups:
                groups[result.verdict] = []
            groups[result.verdict].append(name)
        return groups

    def _generate_challenge(
        self,
        challenger_name: str,
        challenger: QwenAnalysisResult,
        target_name: str,
        target: QwenAnalysisResult
    ) -> str:
        """반론 생성"""
        # Verdict 값을 대문자로 표시 (가독성)
        target_verdict = target.verdict.value.upper()
        challenger_verdict = challenger.verdict.value.upper()

        return (
            f"저는 {challenger_name}입니다. "
            f"당신의 {target_verdict} 판정에 대해 질문이 있습니다. "
            f"제 분석 결과는 {challenger_verdict}입니다. "
            f"근거: {challenger.reasoning[:200]}... "
            f"당신의 판정 근거를 더 자세히 설명해주시겠습니까?"
        )

    async def close(self):
        """리소스 정리"""
        await self.client.close()


# 동기 버전
class QwenMAIFSAdapterSync:
    """QwenMAIFSAdapter의 동기 래퍼"""

    def __init__(self, *args, **kwargs):
        self._adapter = QwenMAIFSAdapter(*args, **kwargs)
        self._loop = None

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def analyze_with_qwen(self, *args, **kwargs):
        loop = self._get_loop()
        return loop.run_until_complete(
            self._adapter.analyze_with_qwen(*args, **kwargs)
        )

    def analyze_single(self, *args, **kwargs):
        loop = self._get_loop()
        return loop.run_until_complete(
            self._adapter.analyze_single(*args, **kwargs)
        )

    def conduct_debate(self, *args, **kwargs):
        loop = self._get_loop()
        return loop.run_until_complete(
            self._adapter.conduct_debate(*args, **kwargs)
        )

    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._adapter.close())
            self._loop.close()


def create_qwen_adapter(
    base_url: str = "http://localhost:8000",
    sync: bool = False,
    **kwargs
):
    """
    Qwen MAIFS 어댑터 생성

    Args:
        base_url: vLLM 서버 URL
        sync: 동기 어댑터 반환 여부
        **kwargs: 추가 설정

    Returns:
        QwenMAIFSAdapter 또는 QwenMAIFSAdapterSync
    """
    if sync:
        return QwenMAIFSAdapterSync(base_url=base_url, **kwargs)
    return QwenMAIFSAdapter(base_url=base_url, **kwargs)
