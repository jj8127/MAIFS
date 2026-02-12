"""
MAIFS Tools 테스트

각 분석 도구의 기능을 검증합니다.
"""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.settings import config
from src.tools.base_tool import Verdict, ToolResult
from src.tools.frequency_tool import FrequencyAnalysisTool
from src.tools.noise_tool import NoiseAnalysisTool
from src.tools.fatformer_tool import FatFormerTool
from src.tools.spatial_tool import SpatialAnalysisTool


# 테스트용 이미지 생성
def create_dummy_image(size=(512, 512), seed=42):
    """테스트용 더미 이미지 생성"""
    np.random.seed(seed)
    return (np.random.rand(size[0], size[1], 3) * 255).astype(np.uint8)


def create_gradient_image(size=(512, 512)):
    """그라디언트 이미지 생성 (자연스러운 패턴)"""
    h, w = size
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    r = (xx * 255).astype(np.uint8)
    g = (yy * 255).astype(np.uint8)
    b = ((xx + yy) / 2 * 255).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


def create_checkerboard_image(size=(512, 512), block_size=32):
    """체커보드 이미지 생성 (격자 패턴 - AI 생성 이미지 시뮬레이션)"""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i:i+block_size, j:j+block_size] = 255

    return img


class TestFrequencyTool:
    """주파수 분석 도구 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.tool = FrequencyAnalysisTool()

    def test_tool_initialization(self):
        """도구 초기화 테스트"""
        assert self.tool.name == "frequency_analyzer"
        assert self.tool._is_loaded == True  # FFT는 외부 모델 불필요

    def test_analyze_random_image(self):
        """랜덤 이미지 분석"""
        image = create_dummy_image()
        result = self.tool.analyze(image)

        assert isinstance(result, ToolResult)
        assert result.tool_name == "frequency_analyzer"
        assert result.verdict in [Verdict.AUTHENTIC, Verdict.AI_GENERATED, Verdict.UNCERTAIN]
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time > 0

    def test_analyze_gradient_image(self):
        """그라디언트 이미지 분석 (자연스러운 패턴)"""
        image = create_gradient_image()
        result = self.tool.analyze(image)

        assert result.verdict in [Verdict.AUTHENTIC, Verdict.UNCERTAIN]
        assert "grid_analysis" in result.evidence
        assert "high_frequency_analysis" in result.evidence

    def test_analyze_checkerboard_image(self):
        """체커보드 이미지 분석 (격자 패턴)"""
        image = create_checkerboard_image()
        result = self.tool.analyze(image)

        # 규칙적인 격자 패턴은 높은 피크를 보임
        grid_analysis = result.evidence.get("grid_analysis", {})
        assert "horizontal_peaks" in grid_analysis
        assert "vertical_peaks" in grid_analysis

    def test_evidence_structure(self):
        """증거 구조 검증"""
        image = create_dummy_image()
        result = self.tool.analyze(image)

        # 필수 증거 필드 확인
        assert "ai_generation_score" in result.evidence
        assert "grid_analysis" in result.evidence
        assert "high_frequency_analysis" in result.evidence

        # ai_generation_score 범위 검증
        ai_score = result.evidence["ai_generation_score"]
        assert 0.0 <= ai_score <= 1.0

    def test_different_image_sizes(self):
        """다양한 이미지 크기 처리"""
        sizes = [(256, 256), (512, 512), (1024, 1024), (640, 480)]

        for size in sizes:
            image = create_dummy_image(size=size)
            result = self.tool.analyze(image)

            assert result.verdict is not None
            assert result.confidence >= 0.0


class TestNoiseTool:
    """노이즈 분석 도구 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.tool = NoiseAnalysisTool()

    def test_tool_initialization(self):
        """도구 초기화 테스트"""
        assert self.tool.name == "noise_analyzer"

    def test_analyze_random_image(self):
        """랜덤 이미지 분석"""
        image = create_dummy_image()
        result = self.tool.analyze(image)

        assert isinstance(result, ToolResult)
        assert result.tool_name == "noise_analyzer"
        assert result.verdict in [Verdict.AUTHENTIC, Verdict.AI_GENERATED,
                                  Verdict.MANIPULATED, Verdict.UNCERTAIN]
        assert 0.0 <= result.confidence <= 1.0

    def test_prnu_statistics(self):
        """PRNU 통계 검증"""
        image = create_dummy_image()
        result = self.tool.analyze(image)

        assert "prnu_stats" in result.evidence
        prnu_stats = result.evidence["prnu_stats"]

        assert "mean" in prnu_stats
        assert "std" in prnu_stats
        assert "variance" in prnu_stats
        assert "kurtosis" in prnu_stats

    def test_consistency_analysis(self):
        """일관성 분석 검증"""
        image = create_dummy_image()
        result = self.tool.analyze(image)

        assert "consistency_analysis" in result.evidence
        consistency = result.evidence["consistency_analysis"]

        assert "consistency_score" in consistency
        assert 0.0 <= consistency["consistency_score"] <= 1.0

    def test_ai_detection(self):
        """AI 탐지 점수 검증"""
        image = create_dummy_image()
        result = self.tool.analyze(image)

        assert "ai_detection" in result.evidence
        ai_detection = result.evidence["ai_detection"]

        assert "ai_generation_score" in ai_detection
        assert 0.0 <= ai_detection["ai_generation_score"] <= 1.0


class TestFatFormerTool:
    """FatFormer AI 생성 탐지 도구 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.tool = FatFormerTool()

    def test_tool_initialization(self):
        """도구 초기화 테스트"""
        assert self.tool.name == "fatformer_analyzer"
        assert self.tool.input_resolution == 224

    def test_fallback_mode(self):
        """Fallback 모드 테스트 (모델 없을 때)"""
        # 모델 로드 시도
        self.tool.load_model()

        image = create_dummy_image()
        result = self.tool.analyze(image)

        assert isinstance(result, ToolResult)
        assert result.tool_name == "fatformer_analyzer"

        # 모델이 없으면 fallback 모드
        if self.tool._model is None:
            assert result.evidence.get("fallback_mode") == True
            assert result.verdict == Verdict.UNCERTAIN

    def test_analyze_returns_valid_result(self):
        """분석 결과 유효성 검증"""
        image = create_dummy_image()
        result = self.tool.analyze(image)

        assert result.verdict in [Verdict.AUTHENTIC, Verdict.UNCERTAIN,
                                  Verdict.MANIPULATED, Verdict.AI_GENERATED]
        assert result.confidence >= 0.0
        assert result.processing_time >= 0.0


class TestSpatialTool:
    """공간 분석 도구 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.tool = SpatialAnalysisTool()

    def test_tool_initialization(self):
        """도구 초기화 테스트"""
        assert self.tool.name == "spatial_analyzer"
        assert self.tool.input_size == 1024

    def test_fallback_mode(self):
        """Fallback 모드 테스트"""
        self.tool.load_model()

        image = create_dummy_image()
        result = self.tool.analyze(image)

        assert isinstance(result, ToolResult)

        # 모델이 없으면 fallback 모드
        if self.tool._model is None:
            assert result.evidence.get("fallback_mode") == True

    def test_analyze_returns_valid_result(self):
        """분석 결과 유효성 검증"""
        image = create_dummy_image()
        result = self.tool.analyze(image)

        assert result.verdict in [Verdict.AUTHENTIC, Verdict.UNCERTAIN,
                                  Verdict.MANIPULATED, Verdict.AI_GENERATED]
        assert result.confidence >= 0.0


class TestToolConsistency:
    """모든 도구의 일관성 테스트"""

    def setup_method(self):
        """테스트 전 설정"""
        self.tools = [
            FrequencyAnalysisTool(),
            NoiseAnalysisTool(),
            FatFormerTool(),
            SpatialAnalysisTool(),
        ]

    def test_all_tools_return_valid_result(self):
        """모든 도구가 유효한 결과 반환"""
        image = create_dummy_image()

        for tool in self.tools:
            result = tool(image)  # __call__ 테스트

            assert isinstance(result, ToolResult)
            assert result.tool_name == tool.name
            assert result.verdict in list(Verdict)
            assert 0.0 <= result.confidence <= 1.0

    def test_all_tools_to_dict(self):
        """모든 도구의 to_dict 메서드 테스트"""
        image = create_dummy_image()

        for tool in self.tools:
            result = tool(image)
            result_dict = result.to_dict()

            assert isinstance(result_dict, dict)
            assert "tool_name" in result_dict
            assert "verdict" in result_dict
            assert "confidence" in result_dict
            assert "evidence" in result_dict


# 실제 이미지 테스트 (선택적)
class TestWithRealImage:
    """실제 이미지로 테스트 (이미지 파일이 있을 때만)"""

    @pytest.fixture
    def real_image_path(self):
        """실제 이미지 경로"""
        path = config.test.sample_image
        if not path.exists():
            pytest.skip("Test image not found")
        return path

    def test_frequency_tool_real_image(self, real_image_path):
        """주파수 도구 - 실제 이미지"""
        image = np.array(Image.open(real_image_path).convert("RGB"))
        tool = FrequencyAnalysisTool()
        result = tool(image)

        assert result.verdict is not None
        print(f"\n[FrequencyTool] Verdict: {result.verdict.value}, "
              f"Confidence: {result.confidence:.2%}")

    def test_noise_tool_real_image(self, real_image_path):
        """노이즈 도구 - 실제 이미지"""
        image = np.array(Image.open(real_image_path).convert("RGB"))
        tool = NoiseAnalysisTool()
        result = tool(image)

        assert result.verdict is not None
        print(f"\n[NoiseTool] Verdict: {result.verdict.value}, "
              f"Confidence: {result.confidence:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
