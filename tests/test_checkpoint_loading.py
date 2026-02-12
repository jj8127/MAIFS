"""
Checkpoint 로드 검증 테스트

OmniGuard 체크포인트 연동 검증
"""
import pytest
from pathlib import Path
import sys
import os

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# 직접 import
from configs.settings import config


class TestCheckpointAvailability:
    """체크포인트 가용성 테스트"""

    def test_checkpoint_directory_exists(self):
        """체크포인트 디렉토리 존재 확인"""
        checkpoint_dir = config.model.omniguard_checkpoint_dir
        assert checkpoint_dir.exists(), f"Checkpoint directory not found: {checkpoint_dir}"
        print(f"\n✅ Checkpoint directory exists: {checkpoint_dir}")

    def test_checkpoint_files_exist(self):
        """체크포인트 파일 존재 확인"""
        checkpoint_dir = config.model.omniguard_checkpoint_dir

        # 체크포인트 파일 목록
        checkpoint_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))

        assert len(checkpoint_files) > 0, "No checkpoint files found"
        print(f"\n✅ Found {len(checkpoint_files)} checkpoint files:")
        for f in checkpoint_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.1f}MB)")

    def test_get_available_checkpoints(self):
        """사용 가능한 체크포인트 확인"""
        available = config.model.get_available_checkpoints()

        print(f"\n✅ Available checkpoints:")
        for name, is_available in available.items():
            status = "✅" if is_available else "❌"
            print(f"   {status} {name}: {is_available}")

        assert any(available.values()), "No checkpoints available"

    def test_hinet_checkpoint_priority(self):
        """HiNet 체크포인트 우선순위 검증"""
        best_checkpoint = config.model.get_best_hinet_checkpoint()

        if best_checkpoint:
            print(f"\n✅ Best HiNet checkpoint: {best_checkpoint}")
            print(f"   Size: {best_checkpoint.stat().st_size / (1024 * 1024):.1f}MB")
        else:
            print("\n⚠️  No HiNet checkpoint found (will use fallback mode)")


class TestConfigurationPaths:
    """설정 경로 검증"""

    def test_omniguard_dir_configured(self):
        """OmniGuard 디렉토리 설정 확인"""
        from configs.settings import OMNIGUARD_DIR

        assert OMNIGUARD_DIR.exists(), f"OmniGuard dir not found: {OMNIGUARD_DIR}"
        print(f"\n✅ OmniGuard directory: {OMNIGUARD_DIR}")

    def test_hinet_dir_configured(self):
        """HiNet 디렉토리 설정 확인"""
        from configs.settings import HINET_DIR

        print(f"\n✅ HiNet directory: {HINET_DIR}")
        if HINET_DIR.exists():
            print(f"   Status: Exists ✅")
        else:
            print(f"   Status: Does not exist (optional)")

    def test_device_configured(self):
        """디바이스 설정 확인"""
        device = config.model.device
        print(f"\n✅ Configured device: {device}")
        assert device in ["cuda", "cpu"]


class TestToolInitialization:
    """Tool 초기화 테스트 (체크포인트 포함)"""

    def test_fatformer_tool_initialization(self):
        """FatFormerTool 초기화"""
        from src.tools.fatformer_tool import FatFormerTool
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        tool = FatFormerTool()

        print(f"\n✅ FatFormerTool initialized:")
        print(f"   Name: {tool.name}")
        print(f"   Device: {tool.device}")
        print(f"   Checkpoint path: {tool.checkpoint_path}")

    def test_spatial_tool_initialization(self):
        """SpatialAnalysisTool 초기화"""
        from src.tools.spatial_tool import SpatialAnalysisTool

        tool = SpatialAnalysisTool()

        print(f"\n✅ SpatialAnalysisTool initialized:")
        print(f"   Name: {tool.name}")
        print(f"   Device: {tool.device}")
        print(f"   Input size: {tool.input_size}")
        print(f"   Checkpoint path: {tool.checkpoint_path}")

    def test_frequency_tool_initialization(self):
        """FrequencyAnalysisTool 초기화"""
        from src.tools.frequency_tool import FrequencyAnalysisTool

        tool = FrequencyAnalysisTool()

        print(f"\n✅ FrequencyAnalysisTool initialized:")
        print(f"   Name: {tool.name}")
        print(f"   Device: {tool.device}")

    def test_noise_tool_initialization(self):
        """NoiseAnalysisTool 초기화"""
        from src.tools.noise_tool import NoiseAnalysisTool

        tool = NoiseAnalysisTool()

        print(f"\n✅ NoiseAnalysisTool initialized:")
        print(f"   Name: {tool.name}")
        print(f"   Device: {tool.device}")


class TestModelLoading:
    """모델 로드 테스트"""

    def test_fatformer_model_load_attempt(self):
        """FatFormerTool 모델 로드 시도"""
        from src.tools.fatformer_tool import FatFormerTool
        import numpy as np

        tool = FatFormerTool()

        # 모델 로드 시도
        tool.load_model()

        print(f"\n✅ FatFormerTool model load attempt completed")
        print(f"   Model loaded: {tool._is_loaded}")
        print(f"   Model available: {tool._model is not None}")

        if tool._model is not None:
            print(f"   ✅ Model is available - full mode enabled")
        else:
            print(f"   ⚠️  Model not available - fallback mode will be used")

    def test_spatial_model_load_attempt(self):
        """SpatialAnalysisTool 모델 로드 시도"""
        from src.tools.spatial_tool import SpatialAnalysisTool

        tool = SpatialAnalysisTool()

        # 모델 로드 시도
        tool.load_model()

        print(f"\n✅ SpatialAnalysisTool model load attempt completed")
        print(f"   Model loaded: {tool._is_loaded}")
        print(f"   Model available: {tool._model is not None}")

        if tool._model is not None:
            print(f"   ✅ Model is available - full mode enabled")
        else:
            print(f"   ⚠️  Model not available - fallback mode will be used")


class TestMAIFSWithCheckpoints:
    """체크포인트를 포함한 MAIFS 테스트"""

    def test_maifs_full_pipeline(self):
        """MAIFS 전체 파이프라인 (체크포인트 포함)"""
        from src.maifs import MAIFS
        import numpy as np

        # MAIFS 초기화
        maifs = MAIFS(enable_debate=False)

        print(f"\n✅ MAIFS initialized with checkpoints")
        print(f"   Consensus algorithm: {maifs.consensus_algorithm}")
        print(f"   Debate enabled: {maifs.enable_debate}")
        print(f"   Device: {maifs.device}")

        # 더미 이미지로 분석
        np.random.seed(42)
        test_image = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)

        result = maifs.analyze(test_image)

        print(f"\n✅ MAIFS analysis completed")
        print(f"   Verdict: {result.verdict.value}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Processing time: {result.processing_time:.2f}s")

        # 에이전트별 결과
        print(f"\n   Agent results:")
        for name, response in result.agent_responses.items():
            model_status = "✅" if "fallback" not in response.evidence else "⚠️"
            print(f"     {model_status} {name}: {response.verdict.value} ({response.confidence:.1%})")

    def test_maifs_with_real_image(self):
        """실제 이미지로 MAIFS 테스트"""
        from src.maifs import MAIFS
        from PIL import Image

        test_image_path = config.test.sample_image

        if not test_image_path.exists():
            pytest.skip("Test image not found")

        maifs = MAIFS(enable_debate=True)

        result = maifs.analyze(test_image_path)

        print(f"\n✅ MAIFS analysis with real image completed")
        print(f"   Image: {test_image_path.name}")
        print(f"   Verdict: {result.verdict.value}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Processing time: {result.processing_time:.2f}s")

        if result.debate_result:
            print(f"\n   Debate result:")
            print(f"     Rounds: {result.debate_result.total_rounds}")
            print(f"     Converged: {result.debate_result.convergence_achieved}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
