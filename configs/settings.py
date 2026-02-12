"""
MAIFS (Multi-Agent Image Forensic System) 설정 파일
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os


# 기본 경로 설정
BASE_DIR = Path(__file__).parent.parent


def get_env_path(var_name: str) -> Optional[Path]:
    """환경 변수에서 경로를 읽어옵니다."""
    value = os.environ.get(var_name)
    if not value:
        return None
    return Path(value).expanduser()


def get_omniguard_path() -> Path:
    """OmniGuard 경로 자동 감지"""
    env_path = get_env_path("MAIFS_OMNIGUARD_DIR")
    if env_path:
        return env_path
    possible_paths = [
        # Linux (현재 MAIFS 내부)
        BASE_DIR / "OmniGuard-main",
        # Linux (별도 Tri-Shield 디렉토리)
        BASE_DIR.parent / "Tri-Shield" / "OmniGuard-main",
        Path("/root/Desktop/Tri-Shield/OmniGuard-main"),
        # Windows
        Path("e:/Downloads/OmniGuard-main/OmniGuard-main"),
        Path("E:/Downloads/OmniGuard-main/OmniGuard-main"),
    ]
    for path in possible_paths:
        try:
            if path.exists():
                return path
        except (PermissionError, OSError):
            continue
    return BASE_DIR / "OmniGuard-main"


def get_hinet_path() -> Path:
    """HiNet 체크포인트 경로 자동 감지"""
    env_path = get_env_path("MAIFS_HINET_DIR")
    if env_path:
        return env_path
    possible_paths = [
        BASE_DIR / "HiNet_root" / "HiNet" / "logging" / "finetuned_log",
        BASE_DIR.parent / "HiNet_root" / "HiNet" / "logging" / "finetuned_log",
        Path("/root/Desktop/HiNet_root/HiNet/logging/finetuned_log"),
        Path("/root/Desktop/HiNet_root/HiNet/logging/qat_qbackend_safe_20250812_221928_ep100_calib128"),
        Path("e:/HiNet_root/HiNet/logging/finetuned_log"),
    ]
    for path in possible_paths:
        try:
            if path.exists():
                return path
        except (PermissionError, OSError):
            continue
    return BASE_DIR / "HiNet_root" / "HiNet" / "logging" / "finetuned_log"


def get_test_image_dir() -> Path:
    """테스트 이미지 디렉토리 자동 감지"""
    env_path = get_env_path("MAIFS_TEST_IMAGE_DIR")
    if env_path:
        return env_path
    return BASE_DIR / "data" / "images"


def get_sample_image() -> Path:
    """샘플 이미지 자동 감지"""
    env_path = get_env_path("MAIFS_SAMPLE_IMAGE")
    if env_path:
        return env_path
    return get_test_image_dir() / "sample.png"


OMNIGUARD_DIR = get_omniguard_path()
HINET_DIR = get_hinet_path()


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    # OmniGuard 체크포인트 디렉토리
    omniguard_checkpoint_dir: Path = field(default_factory=lambda: OMNIGUARD_DIR / "checkpoint")

    # HiNet 워터마크 모델 (OmniGuard 버전) - 실제 사용 가능한 체크포인트
    hinet_checkpoint: Path = field(default_factory=lambda: OMNIGUARD_DIR / "checkpoint" / "checkpoint-175.pth")

    # HiNet 로컬 체크포인트 (직접 학습한 버전)
    hinet_local_checkpoint: Path = field(default_factory=lambda: HINET_DIR / "last.pth")

    # ViT 이미지 조작 탐지 모델 (대체 모델)
    vit_checkpoint: Path = field(default_factory=lambda: OMNIGUARD_DIR / "checkpoint" / "model_checkpoint_01500.pt")
    vit_input_size: int = 1024
    vit_patch_size: int = 16
    vit_embed_dim: int = 768

    # UNet 탐지 모델 (대체 모델)
    unet_checkpoint: Path = field(default_factory=lambda: OMNIGUARD_DIR / "checkpoint" / "model_checkpoint_00540.pt")

    # FatFormer 체크포인트
    fatformer_checkpoint: Path = field(default_factory=lambda: BASE_DIR / "Integrated Submodules" / "FatFormer" / "checkpoint" / "fatformer.pth")

    # 디바이스 설정
    device: str = field(default_factory=lambda: "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

    def get_available_checkpoints(self) -> Dict[str, bool]:
        """사용 가능한 체크포인트 확인"""
        checkpoint_dir = self.omniguard_checkpoint_dir
        # 실제 다운로드된 체크포인트 파일 확인
        available_files = []
        if checkpoint_dir.exists():
            available_files = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))

        return {
            "omniguard_models": len(available_files) > 0,
            "hinet_checkpoint": self.hinet_checkpoint.exists(),
            "hinet_local": self.hinet_local_checkpoint.exists(),
            "all_checkpoint_files": len(available_files),
        }

    def get_best_hinet_checkpoint(self) -> Optional[Path]:
        """사용 가능한 최적의 HiNet 체크포인트 반환"""
        # 우선순위: hinet_checkpoint > hinet_local_checkpoint > 다른 모든 체크포인트
        if self.hinet_checkpoint.exists():
            return self.hinet_checkpoint
        if self.hinet_local_checkpoint.exists():
            return self.hinet_local_checkpoint

        # 폴백: 다른 모든 checkpoint 파일 중 하나
        checkpoint_dir = self.omniguard_checkpoint_dir
        if checkpoint_dir.exists():
            all_checkpoints = list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt"))
            if all_checkpoints:
                return sorted(all_checkpoints, key=lambda x: x.stat().st_size, reverse=True)[0]

        return None


@dataclass
class AgentConfig:
    """에이전트 관련 설정"""
    # Manager Agent LLM
    manager_model: str = "gpt-4o"  # or "claude-3-5-sonnet"
    manager_temperature: float = 0.3

    # Specialist Agent LLM (로컬)
    specialist_model: str = "llama-3.2-korean"
    specialist_model_path: Path = field(default_factory=lambda: Path("/root/models/Llama-3.2-Korean"))

    # 에이전트 타임아웃 (초)
    agent_timeout: int = 60

    # 최대 토론 라운드
    max_debate_rounds: int = 3

    # LLM 사용 여부 (False면 규칙 기반 추론)
    use_llm: bool = False


@dataclass
class COBRAConfig:
    """COBRA 합의 알고리즘 설정"""
    # 신뢰 임계값 (RoT에서 사용)
    trust_threshold: float = 0.7

    # 초기 신뢰도 (각 에이전트)
    initial_trust: Dict[str, float] = field(default_factory=lambda: {
        "frequency": 0.85,
        "noise": 0.80,
        "fatformer": 0.85,
        "spatial": 0.85,
        "semantic": 0.75
    })

    # 합의 알고리즘 선택: "rot", "drwa", "avga", "auto"
    consensus_algorithm: str = "drwa"

    # DRWA epsilon (동적 가중치 조정)
    drwa_epsilon: float = 0.1

    # AVGA 설정
    avga_temperature: float = 1.0
    avga_shift_rate: float = 0.2

    # RoT alpha (비신뢰 코호트 가중치 감쇠)
    rot_alpha: float = 0.3


@dataclass
class DebateConfig:
    """토론 프로토콜 설정"""
    # 토론 활성화 여부
    enable_debate: bool = True

    # 토론 활성화 임계값 (불일치 정도)
    disagreement_threshold: float = 0.3

    # 최대 토론 라운드
    max_rounds: int = 3

    # 수렴 임계값 (신뢰도 변화)
    convergence_threshold: float = 0.1

    # 토론 모드: "synchronous", "asynchronous", "structured"
    debate_mode: str = "asynchronous"

    # Judge 모델 (선택적, None이면 COBRA 합의 사용)
    judge_model: Optional[str] = None


@dataclass
class TestConfig:
    """테스트 관련 설정"""
    # 테스트 이미지 디렉토리
    test_image_dir: Path = field(default_factory=get_test_image_dir)

    # 결과 저장 디렉토리
    test_output_dir: Path = field(default_factory=lambda: BASE_DIR / "test_results")

    # 샘플 이미지 (기본)
    sample_image: Path = field(default_factory=get_sample_image)


@dataclass
class MAIFSConfig:
    """전체 MAIFS 시스템 설정"""
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    cobra: COBRAConfig = field(default_factory=COBRAConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    test: TestConfig = field(default_factory=TestConfig)

    # 로깅 설정
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")

    # 결과 저장
    output_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs")

    def __post_init__(self):
        """디렉토리 생성"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test.test_output_dir.mkdir(parents=True, exist_ok=True)

    def print_info(self):
        """설정 정보 출력"""
        print("=" * 50)
        print("MAIFS Configuration")
        print("=" * 50)
        print(f"OmniGuard Path: {OMNIGUARD_DIR}")
        print(f"HiNet Path: {HINET_DIR}")
        print(f"Device: {self.model.device}")
        print(f"Checkpoints: {self.model.get_available_checkpoints()}")
        print(f"Consensus Algorithm: {self.cobra.consensus_algorithm}")
        print(f"Debate Mode: {self.debate.debate_mode}")
        print("=" * 50)


# 기본 설정 인스턴스
config = MAIFSConfig()


if __name__ == "__main__":
    config.print_info()
