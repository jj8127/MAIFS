"""
공간 분석 도구
OmniGuard의 ViT 모델을 래핑하여 이미지 조작 영역 탐지 수행
"""
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

from .base_tool import BaseTool, ToolResult, Verdict

# 설정에서 경로 로드
try:
    from ..configs.settings import config, OMNIGUARD_DIR
    OMNIGUARD_PATH = OMNIGUARD_DIR
except ImportError:
    OMNIGUARD_PATH = Path(__file__).resolve().parents[2] / "OmniGuard-main"

# OmniGuard 모듈 경로 추가
if OMNIGUARD_PATH.exists():
    sys.path.insert(0, str(OMNIGUARD_PATH))


class SpatialAnalysisTool(BaseTool):
    """
    공간 분석 도구 (Image Manipulation Localization)

    Window Attention ViT + Feature Pyramid Network를 사용하여:
    - 픽셀 수준 조작 영역 탐지
    - 경계 불일치 분석
    - 텍스처 이상 탐지
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        input_size: int = None,
        device: str = None
    ):
        # 설정에서 로드
        if device is None:
            try:
                device = config.model.device
            except:
                device = "cpu"

        if input_size is None:
            try:
                input_size = config.model.vit_input_size
            except:
                input_size = 1024

        super().__init__(
            name="spatial_analyzer",
            description="ViT 기반 이미지 조작 영역 탐지. "
                       "픽셀 수준에서 조작된 영역을 찾아 마스크로 반환합니다.",
            device=device
        )

        # 체크포인트 경로 설정
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        else:
            try:
                self.checkpoint_path = config.model.omniguard_checkpoint_dir
            except:
                self.checkpoint_path = OMNIGUARD_PATH / "checkpoint"

        self.input_size = input_size

    def load_model(self) -> None:
        """ViT IML 모델 로드"""
        if self._is_loaded:
            return

        try:
            from iml_vit_model import iml_vit_model

            self._model = iml_vit_model(
                input_size=self.input_size,
                patch_size=16,
                embed_dim=768,
                vit_pretrain_path=None  # 사전학습 가중치는 체크포인트에 포함
            )
            self._model = self._model.to(self.device)

            # 체크포인트 로드
            checkpoint_file = self.checkpoint_path / "iml_vit.pth"
            if checkpoint_file.exists():
                state_dict = torch.load(checkpoint_file, map_location=self.device)
                self._model.load_state_dict(state_dict, strict=False)
                print(f"[SpatialTool] 체크포인트 로드: {checkpoint_file}")

            self._model.eval()
            self._is_loaded = True
            print("[SpatialTool] ViT IML 모델 로드 완료")

        except ImportError as e:
            print(f"[SpatialTool] OmniGuard 모듈 임포트 실패: {e}")
            print("[SpatialTool] Fallback 모드로 전환")
            self._model = None
            self._is_loaded = True

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """이미지 전처리"""
        original_size = (image.shape[0], image.shape[1])

        # numpy to PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        # 리사이즈
        pil_image = pil_image.resize((self.input_size, self.input_size), Image.BILINEAR)

        # numpy로 변환 및 정규화
        img_array = np.array(pil_image).astype(np.float32) / 255.0

        # 정규화 (ImageNet 표준)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

        # 배치 차원 추가
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.to(self.device), original_size

    def _postprocess_mask(
        self,
        mask: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """마스크 후처리"""
        # (1, 1, H, W) -> (H, W)
        mask = mask.squeeze().cpu().numpy()

        # 원본 크기로 리사이즈
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((original_size[1], original_size[0]), Image.BILINEAR)

        return np.array(mask_pil).astype(np.float32) / 255.0

    def _analyze_mask(self, mask: np.ndarray) -> dict:
        """마스크 분석"""
        # 이진화
        binary_mask = (mask > 0.5).astype(np.uint8)

        # 조작 영역 비율
        manipulation_ratio = np.mean(binary_mask)

        # 연결된 영역 수 (간단한 분석)
        # TODO: cv2.connectedComponents 사용 가능

        return {
            "manipulation_ratio": float(manipulation_ratio),
            "max_intensity": float(np.max(mask)),
            "mean_intensity": float(np.mean(mask)),
            "threshold_used": 0.5
        }

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        공간 분석 실행

        Args:
            image: RGB 이미지 (H, W, 3)

        Returns:
            ToolResult: 조작 영역 분석 결과
        """
        start_time = time.time()

        if self._model is None:
            return self._fallback_analysis(image, start_time)

        try:
            # 전처리
            img_tensor, original_size = self._preprocess(image)

            # Container 이미지 (원본과 동일하게 사용)
            container_tensor = img_tensor.clone()

            # 마스크 예측
            with torch.no_grad():
                mask_pred = self._model(img_tensor, container_tensor)

            # 후처리
            mask = self._postprocess_mask(mask_pred, original_size)

            # 마스크 분석
            mask_analysis = self._analyze_mask(mask)

            # 판정
            manipulation_ratio = mask_analysis["manipulation_ratio"]

            if manipulation_ratio < 0.05:
                verdict = Verdict.AUTHENTIC
                confidence = 1.0 - manipulation_ratio
                explanation = (
                    f"이미지에서 유의미한 조작 영역이 탐지되지 않았습니다. "
                    f"조작 비율: {manipulation_ratio:.2%}"
                )
            elif manipulation_ratio > 0.8:
                verdict = Verdict.AI_GENERATED
                confidence = manipulation_ratio
                explanation = (
                    f"이미지 대부분이 조작/생성된 것으로 보입니다. "
                    f"조작 비율: {manipulation_ratio:.2%}. AI 생성 이미지일 가능성이 높습니다."
                )
            else:
                verdict = Verdict.MANIPULATED
                confidence = manipulation_ratio
                explanation = (
                    f"이미지의 일부 영역이 조작된 것으로 탐지되었습니다. "
                    f"조작 비율: {manipulation_ratio:.2%}. 마스크를 통해 조작 영역을 확인하세요."
                )

            processing_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence=mask_analysis,
                explanation=explanation,
                manipulation_mask=mask,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ToolResult(
                tool_name=self.name,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                evidence={"error": str(e)},
                explanation=f"공간 분석 중 오류 발생: {str(e)}",
                processing_time=processing_time
            )

    def _fallback_analysis(self, image: np.ndarray, start_time: float) -> ToolResult:
        """모델 없이 기본 분석 수행 (Edge detection 기반)"""
        try:
            # 간단한 에지 기반 분석
            gray = np.mean(image, axis=2)

            # Sobel 근사 (간단한 구현)
            dx = np.abs(np.diff(gray, axis=1))
            dy = np.abs(np.diff(gray, axis=0))

            # 에지 강도
            edge_mean = (np.mean(dx) + np.mean(dy)) / 2

            processing_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                verdict=Verdict.UNCERTAIN,
                confidence=0.3,
                evidence={
                    "fallback_mode": True,
                    "edge_mean": float(edge_mean),
                    "image_shape": image.shape
                },
                explanation="ViT 모델을 사용할 수 없어 기본 에지 분석만 수행되었습니다.",
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ToolResult(
                tool_name=self.name,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                evidence={"error": str(e), "fallback_mode": True},
                explanation=f"Fallback 분석 중 오류: {str(e)}",
                processing_time=processing_time
            )


# LangChain Tool 호환 래퍼
def create_langchain_tool():
    """LangChain용 Tool 생성"""
    try:
        from langchain.tools import Tool as LCTool

        tool_instance = SpatialAnalysisTool()

        def run_analysis(image_path: str) -> str:
            """이미지 파일 경로로 공간 분석 실행"""
            image = np.array(Image.open(image_path).convert("RGB"))
            result = tool_instance(image)
            return str(result.to_dict())

        return LCTool(
            name="spatial_analyzer",
            description=tool_instance.description,
            func=run_analysis
        )

    except ImportError:
        return None
