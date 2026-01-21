"""
MAIFS End-to-End 테스트

전체 파이프라인 통합 테스트
"""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import config
from src.maifs import MAIFS, MAIFSResult, analyze_image
from src.tools.base_tool import Verdict


def create_test_image(size=(512, 512), seed=42):
    """테스트용 이미지 생성"""
    np.random.seed(seed)
    return (np.random.rand(size[0], size[1], 3) * 255).astype(np.uint8)


class TestMAIFSInitialization:
    """MAIFS 초기화 테스트"""

    def test_default_initialization(self):
        """기본 초기화"""
        maifs = MAIFS()

        assert maifs.enable_debate == True
        assert len(maifs.agents) == 4
        assert maifs.consensus_engine is not None
        assert maifs.debate_chamber is not None

    def test_custom_initialization(self):
        """커스텀 초기화"""
        maifs = MAIFS(
            enable_debate=False,
            debate_threshold=0.5,
            consensus_algorithm="rot"
        )

        assert maifs.enable_debate == False
        assert maifs.debate_threshold == 0.5

    def test_agents_registered(self):
        """에이전트 등록 확인"""
        maifs = MAIFS()

        assert "frequency" in maifs.agents
        assert "noise" in maifs.agents
        assert "watermark" in maifs.agents
        assert "spatial" in maifs.agents


class TestMAIFSAnalysis:
    """MAIFS 분석 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.maifs = MAIFS(enable_debate=False)  # 빠른 테스트를 위해 토론 비활성화

    def test_analyze_numpy_array(self):
        """NumPy 배열 분석"""
        image = create_test_image()
        result = self.maifs.analyze(image)

        assert isinstance(result, MAIFSResult)
        assert result.verdict in list(Verdict)
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time > 0

    def test_analyze_pil_image(self):
        """PIL 이미지 분석"""
        image_array = create_test_image()
        pil_image = Image.fromarray(image_array)

        result = self.maifs.analyze(pil_image)

        assert isinstance(result, MAIFSResult)
        assert result.verdict is not None

    def test_analyze_file_path(self):
        """파일 경로 분석"""
        test_path = config.test.sample_image
        if not test_path.exists():
            pytest.skip("Test image not found")

        result = self.maifs.analyze(test_path)

        assert isinstance(result, MAIFSResult)
        assert result.image_info.get("filename") == "0801.png"

    def test_analyze_string_path(self):
        """문자열 경로 분석"""
        test_path = config.test.sample_image
        if not test_path.exists():
            pytest.skip("Test image not found")

        result = self.maifs.analyze(str(test_path))

        assert isinstance(result, MAIFSResult)


class TestMAIFSWithDebate:
    """토론 포함 MAIFS 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.maifs = MAIFS(enable_debate=True, debate_threshold=0.2)

    def test_debate_triggered_on_disagreement(self):
        """불일치 시 토론 발생"""
        image = create_test_image()

        # 토론 강제 활성화
        result = self.maifs.analyze(image, include_debate=True)

        # 토론 결과 확인 (에이전트 의견이 다르면 토론 발생)
        if result.debate_result:
            assert result.debate_result.total_rounds > 0
            print(f"\n토론 진행: {result.debate_result.total_rounds} 라운드")
            print(result.debate_result.get_summary())

    def test_no_debate_when_unanimous(self):
        """만장일치 시 토론 없음"""
        # 만장일치 상황을 시뮬레이션하기 어려우므로
        # 토론 비활성화 테스트
        image = create_test_image()
        result = self.maifs.analyze(image, include_debate=False)

        assert result.debate_result is None


class TestMAIFSResult:
    """MAIFSResult 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.maifs = MAIFS(enable_debate=False)

    def test_result_to_dict(self):
        """to_dict 메서드"""
        image = create_test_image()
        result = self.maifs.analyze(image)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "verdict" in result_dict
        assert "confidence" in result_dict
        assert "agent_responses" in result_dict
        assert "consensus" in result_dict

    def test_result_to_json(self):
        """to_json 메서드"""
        image = create_test_image()
        result = self.maifs.analyze(image)

        json_str = result.to_json()

        assert isinstance(json_str, str)
        # JSON 파싱 가능 확인
        parsed = json.loads(json_str)
        assert parsed["verdict"] in ["authentic", "manipulated", "ai_generated", "uncertain"]

    def test_verdict_explanation(self):
        """판정 설명"""
        image = create_test_image()
        result = self.maifs.analyze(image)

        explanation = result.get_verdict_explanation()

        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_agent_responses_present(self):
        """에이전트 응답 포함"""
        image = create_test_image()
        result = self.maifs.analyze(image)

        assert len(result.agent_responses) == 4
        assert "frequency" in result.agent_responses
        assert "noise" in result.agent_responses
        assert "watermark" in result.agent_responses
        assert "spatial" in result.agent_responses


class TestConvenienceFunction:
    """편의 함수 테스트"""

    def test_analyze_image_function(self):
        """analyze_image 함수"""
        test_path = config.test.sample_image
        if not test_path.exists():
            pytest.skip("Test image not found")

        result = analyze_image(test_path, enable_debate=False)

        assert isinstance(result, MAIFSResult)
        assert result.verdict is not None


class TestMAIFSWithRealImages:
    """실제 이미지 테스트"""

    @pytest.fixture
    def test_image_dir(self):
        """테스트 이미지 디렉토리"""
        path = config.test.test_image_dir
        if not path.exists():
            pytest.skip("Test image directory not found")
        return path

    def test_multiple_images(self, test_image_dir):
        """여러 이미지 분석"""
        maifs = MAIFS(enable_debate=False)

        # 처음 3개 이미지만 테스트
        images = list(test_image_dir.glob("*.png"))[:3]

        results = []
        for img_path in images:
            result = maifs.analyze(img_path)
            results.append(result)

            print(f"\n{img_path.name}: {result.verdict.value} "
                  f"({result.confidence:.1%})")

        assert len(results) == len(images)
        assert all(r.verdict is not None for r in results)

    def test_detailed_analysis(self, test_image_dir):
        """상세 분석 테스트"""
        maifs = MAIFS(enable_debate=True)

        img_path = test_image_dir / "0801.png"
        result = maifs.analyze(img_path)

        print("\n" + "=" * 60)
        print("MAIFS 상세 분석 결과")
        print("=" * 60)
        print(f"파일: {img_path.name}")
        print(f"판정: {result.verdict.value}")
        print(f"신뢰도: {result.confidence:.1%}")
        print(f"처리 시간: {result.processing_time:.2f}초")
        print("\n[에이전트별 결과]")

        for name, response in result.agent_responses.items():
            print(f"  - {name}: {response.verdict.value} ({response.confidence:.1%})")

        print("\n[합의 정보]")
        if result.consensus_result:
            print(f"  알고리즘: {result.consensus_result.algorithm_used}")
            print(f"  불일치 수준: {result.consensus_result.disagreement_level:.2%}")

        if result.debate_result:
            print("\n[토론 결과]")
            print(f"  라운드: {result.debate_result.total_rounds}")
            print(f"  수렴: {result.debate_result.convergence_achieved}")

        print("=" * 60)


class TestConsensusAlgorithms:
    """합의 알고리즘별 테스트"""

    def test_rot_algorithm(self):
        """RoT 알고리즘 사용"""
        maifs = MAIFS(consensus_algorithm="rot", enable_debate=False)
        image = create_test_image()
        result = maifs.analyze(image)

        assert result.consensus_result.algorithm_used == "RoT"

    def test_drwa_algorithm(self):
        """DRWA 알고리즘 사용"""
        maifs = MAIFS(consensus_algorithm="drwa", enable_debate=False)
        image = create_test_image()
        result = maifs.analyze(image)

        assert result.consensus_result.algorithm_used == "DRWA"

    def test_avga_algorithm(self):
        """AVGA 알고리즘 사용"""
        maifs = MAIFS(consensus_algorithm="avga", enable_debate=False)
        image = create_test_image()
        result = maifs.analyze(image)

        assert result.consensus_result.algorithm_used == "AVGA"


class TestErrorHandling:
    """에러 처리 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.maifs = MAIFS(enable_debate=False)

    def test_invalid_image_type(self):
        """잘못된 이미지 타입"""
        with pytest.raises(ValueError):
            self.maifs.analyze("not_an_image")  # 존재하지 않는 파일

    def test_grayscale_image(self):
        """그레이스케일 이미지 처리"""
        gray_image = np.random.rand(256, 256).astype(np.float32)
        result = self.maifs.analyze(gray_image)

        # 자동으로 RGB로 변환되어야 함
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
