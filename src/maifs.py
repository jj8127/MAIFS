"""
MAIFS - Multi-Agent Image Forensic System

메인 시스템 클래스: 전체 파이프라인 통합 및 실행
"""
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import time
import json

from .agents.manager_agent import ManagerAgent, ForensicReport
from .agents.specialist_agents import (
    FrequencyAgent,
    NoiseAgent,
    WatermarkAgent,
    SpatialAgent
)
from .agents.base_agent import AgentResponse
from .consensus.cobra import COBRAConsensus, ConsensusResult
from .debate.debate_chamber import DebateChamber, DebateResult
from .tools.base_tool import Verdict


@dataclass
class MAIFSResult:
    """MAIFS 분석 최종 결과"""
    # 최종 판정
    verdict: Verdict
    confidence: float

    # 상세 정보
    summary: str
    detailed_report: str

    # 구성 요소 결과
    agent_responses: Dict[str, AgentResponse] = field(default_factory=dict)
    consensus_result: Optional[ConsensusResult] = None
    debate_result: Optional[DebateResult] = None

    # 메타데이터
    image_info: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON 직렬화용 딕셔너리 변환"""
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "summary": self.summary,
            "detailed_report": self.detailed_report,
            "agent_responses": {
                k: v.to_dict() for k, v in self.agent_responses.items()
            },
            "consensus": self.consensus_result.to_dict() if self.consensus_result else None,
            "debate": self.debate_result.to_dict() if self.debate_result else None,
            "image_info": self.image_info,
            "processing_time": self.processing_time
        }

    def to_json(self, indent: int = 2) -> str:
        """JSON 문자열 변환"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def get_verdict_explanation(self) -> str:
        """판정 설명"""
        explanations = {
            Verdict.AUTHENTIC: "원본 이미지로 확인되었습니다. 조작이나 AI 생성 흔적이 발견되지 않았습니다.",
            Verdict.MANIPULATED: "이미지의 일부가 조작된 것으로 탐지되었습니다. 합성, 편집, 또는 수정된 영역이 존재합니다.",
            Verdict.AI_GENERATED: "AI에 의해 생성된 이미지로 판단됩니다. GAN, Diffusion 모델 등의 특징이 탐지되었습니다.",
            Verdict.UNCERTAIN: "판정을 내리기 어렵습니다. 추가 분석이나 전문가 검토가 필요합니다."
        }
        return explanations.get(self.verdict, "알 수 없는 판정")


class MAIFS:
    """
    Multi-Agent Image Forensic System

    다중 에이전트 기반 이미지 포렌식 시스템

    Features:
    - 4개 전문가 에이전트 (주파수, 노이즈, 워터마크, 공간)
    - COBRA 기반 합의 알고리즘
    - 불일치 시 자동 토론 진행
    - 종합적인 분석 보고서 생성

    Usage:
        >>> maifs = MAIFS()
        >>> result = maifs.analyze("path/to/image.jpg")
        >>> print(result.verdict, result.confidence)
    """

    VERSION = "0.1.0"

    def __init__(
        self,
        enable_debate: bool = True,
        debate_threshold: float = 0.3,
        consensus_algorithm: str = "drwa",
        device: str = "cuda"
    ):
        """
        Args:
            enable_debate: 토론 기능 활성화 여부
            debate_threshold: 토론 개시 불일치 임계값
            consensus_algorithm: 합의 알고리즘 ("rot", "drwa", "avga")
            device: 연산 디바이스 ("cuda" or "cpu")
        """
        self.enable_debate = enable_debate
        self.debate_threshold = debate_threshold
        self.device = device
        self.consensus_algorithm = consensus_algorithm

        # 전문가 에이전트 초기화
        self.agents: Dict[str, Any] = {
            "frequency": FrequencyAgent(),
            "noise": NoiseAgent(),
            "watermark": WatermarkAgent(),
            "spatial": SpatialAgent(),
        }

        # 에이전트별 신뢰도
        self.trust_scores: Dict[str, float] = {
            "frequency": 0.85,
            "noise": 0.80,
            "watermark": 0.90,
            "spatial": 0.85,
        }

        # 합의 엔진
        self.consensus_engine = COBRAConsensus(
            default_algorithm=consensus_algorithm
        )

        # 토론 챔버
        self.debate_chamber = DebateChamber(
            consensus_engine=self.consensus_engine,
            disagreement_threshold=debate_threshold
        )

        # Manager Agent
        self.manager = ManagerAgent()

    def analyze(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        include_debate: Optional[bool] = None,
        save_report: Optional[Path] = None
    ) -> MAIFSResult:
        """
        이미지 분석 실행

        Args:
            image: 분석할 이미지 (경로, numpy 배열, PIL 이미지)
            include_debate: 토론 포함 여부 (None이면 자동 결정)
            save_report: 보고서 저장 경로

        Returns:
            MAIFSResult: 분석 결과
        """
        start_time = time.time()

        # 이미지 로드 및 전처리
        img_array, img_info = self._load_image(image)

        # Step 1: 모든 전문가 에이전트 분석
        print("[MAIFS] 전문가 분석 시작...")
        agent_responses = self._collect_agent_analyses(img_array)

        # Step 2: 초기 합의 계산
        print("[MAIFS] 합의 계산 중...")
        consensus_result = self.consensus_engine.aggregate(
            agent_responses, self.trust_scores,
            algorithm=self.consensus_algorithm
        )

        # Step 3: 토론 필요성 판단 및 진행
        debate_result = None
        should_debate = include_debate if include_debate is not None else self.enable_debate

        if should_debate and self.debate_chamber.should_debate(agent_responses):
            print("[MAIFS] 의견 불일치 감지, 토론 시작...")
            debate_result = self.debate_chamber.conduct_debate(
                agent_responses,
                trust_scores=self.trust_scores
            )
            # 토론 후 합의로 업데이트
            if debate_result.consensus_result:
                consensus_result = debate_result.consensus_result

        # Step 4: 최종 결과 생성
        processing_time = time.time() - start_time

        result = MAIFSResult(
            verdict=consensus_result.final_verdict,
            confidence=consensus_result.confidence,
            summary=self._generate_summary(consensus_result, debate_result),
            detailed_report=self._generate_detailed_report(
                agent_responses, consensus_result, debate_result
            ),
            agent_responses=agent_responses,
            consensus_result=consensus_result,
            debate_result=debate_result,
            image_info=img_info,
            processing_time=processing_time
        )

        # 보고서 저장
        if save_report:
            self._save_report(result, save_report)

        print(f"[MAIFS] 분석 완료: {result.verdict.value} ({result.confidence:.1%})")
        return result

    def _load_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> tuple:
        """이미지 로드"""
        img_info = {}

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise ValueError(f"이미지 파일을 찾을 수 없습니다: {path}")

            img_info["path"] = str(path)
            img_info["filename"] = path.name

            pil_img = Image.open(path).convert("RGB")
            img_array = np.array(pil_img)

        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            img_array = np.array(pil_img)

        elif isinstance(image, np.ndarray):
            img_array = image
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)

        else:
            raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")

        img_info["shape"] = img_array.shape
        img_info["dtype"] = str(img_array.dtype)

        return img_array, img_info

    def _collect_agent_analyses(
        self,
        image: np.ndarray
    ) -> Dict[str, AgentResponse]:
        """모든 에이전트의 분석 수집"""
        responses = {}

        for name, agent in self.agents.items():
            try:
                response = agent.analyze(image)
                responses[name] = response
                print(f"  - {name}: {response.verdict.value} ({response.confidence:.1%})")
            except Exception as e:
                print(f"  - {name}: 오류 발생 - {e}")

        return responses

    def _generate_summary(
        self,
        consensus: ConsensusResult,
        debate: Optional[DebateResult]
    ) -> str:
        """요약 생성"""
        verdict_text = {
            Verdict.AUTHENTIC: "원본 이미지",
            Verdict.MANIPULATED: "조작된 이미지",
            Verdict.AI_GENERATED: "AI 생성 이미지",
            Verdict.UNCERTAIN: "판단 불가"
        }

        lines = [
            f"[최종 판정] {verdict_text.get(consensus.final_verdict, '알 수 없음')}",
            f"신뢰도: {consensus.confidence:.1%}",
            f"합의 알고리즘: {consensus.algorithm_used}",
        ]

        if debate:
            lines.append(f"토론 라운드: {debate.total_rounds}")
            if debate.convergence_achieved:
                lines.append(f"수렴: {debate.convergence_reason}")

        return "\n".join(lines)

    def _generate_detailed_report(
        self,
        responses: Dict[str, AgentResponse],
        consensus: ConsensusResult,
        debate: Optional[DebateResult]
    ) -> str:
        """상세 보고서 생성"""
        lines = [
            "=" * 60,
            "MAIFS 이미지 포렌식 분석 보고서",
            "=" * 60,
            "",
            "## 1. 전문가별 분석 결과",
            ""
        ]

        for name, response in responses.items():
            lines.extend([
                f"### {response.agent_name}",
                response.reasoning,
                ""
            ])

        lines.extend([
            "## 2. 합의 분석",
            f"- 알고리즘: {consensus.algorithm_used}",
            f"- 판정 분포: {consensus.verdict_scores}",
            f"- 에이전트 가중치: {consensus.agent_weights}",
            f"- 불일치 수준: {consensus.disagreement_level:.2%}",
            ""
        ])

        if debate:
            lines.extend([
                "## 3. 토론 기록",
                debate.get_summary(),
                ""
            ])

        lines.extend([
            "## 4. 최종 판정",
            f"- 판정: {consensus.final_verdict.value}",
            f"- 신뢰도: {consensus.confidence:.1%}",
            "",
            "=" * 60
        ])

        return "\n".join(lines)

    def _save_report(self, result: MAIFSResult, path: Path) -> None:
        """보고서 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(result.to_json())
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(result.detailed_report)

        print(f"[MAIFS] 보고서 저장: {path}")


# 편의 함수
def analyze_image(
    image_path: Union[str, Path],
    **kwargs
) -> MAIFSResult:
    """
    이미지 분석 편의 함수

    Args:
        image_path: 이미지 경로
        **kwargs: MAIFS 생성자 인자

    Returns:
        MAIFSResult: 분석 결과

    Example:
        >>> result = analyze_image("test.jpg")
        >>> print(result.verdict)
    """
    maifs = MAIFS(**kwargs)
    return maifs.analyze(image_path)
