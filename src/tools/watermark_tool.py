"""
워터마크 분석 도구
OmniGuard의 HiNet 모델을 래핑하여 워터마크 탐지 및 추출 수행
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


class WatermarkTool(BaseTool):
    """
    워터마크 분석 도구

    HiNet 기반 역변환 신경망을 사용하여:
    - 비가시성 워터마크 탐지
    - 워터마크 정보 추출 (100비트)
    - 워터마크 위변조 여부 확인
    """

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = None):
        # 설정에서 디바이스 로드
        if device is None:
            try:
                device = config.model.device
            except:
                device = "cpu"

        super().__init__(
            name="watermark_analyzer",
            description="HiNet 기반 비가시성 워터마크 탐지 및 추출. "
                       "이미지에 숨겨진 워터마크 정보를 복원하고 무결성을 검증합니다.",
            device=device
        )

        # 체크포인트 경로 설정 (우선순위: 인자 > OmniGuard > 로컬 HiNet)
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        else:
            try:
                self.checkpoint_path = config.model.omniguard_checkpoint_dir
            except:
                self.checkpoint_path = OMNIGUARD_PATH / "checkpoint"

        self.watermark_bits = 100  # 워터마크 비트 수
        self.image_size = (256, 256)  # HiNet 입력 크기

    def load_model(self) -> None:
        """HiNet 모델 로드"""
        if self._is_loaded:
            return

        try:
            from hinet import Hinet

            self._model = Hinet()
            self._model = self._model.to(self.device)

            # 체크포인트 로드 (있는 경우)
            checkpoint_file = self.checkpoint_path / "hinet.pth"
            if checkpoint_file.exists():
                state_dict = torch.load(checkpoint_file, map_location=self.device)
                self._model.load_state_dict(state_dict)
                print(f"[WatermarkTool] 체크포인트 로드: {checkpoint_file}")

            self._model.eval()
            self._is_loaded = True
            print("[WatermarkTool] HiNet 모델 로드 완료")

        except ImportError as e:
            print(f"[WatermarkTool] OmniGuard 모듈 임포트 실패: {e}")
            print("[WatermarkTool] Fallback 모드로 전환")
            self._model = None
            self._is_loaded = True  # fallback 모드로 표시

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리"""
        # numpy to PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        # 리사이즈
        pil_image = pil_image.resize(self.image_size, Image.BILINEAR)

        # numpy로 변환 및 정규화
        img_array = np.array(pil_image).astype(np.float32) / 255.0

        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        # 배치 차원 추가
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.to(self.device)

    def _decode_watermark(self, tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        워터마크 디코딩

        Returns:
            (추출된 비트 배열, BER - Bit Error Rate 추정치)
        """
        # HiNet의 출력에서 워터마크 채널 추출
        # 실제 구현에서는 모델 출력의 특정 채널이 워터마크 정보를 담음
        watermark_channel = tensor[:, 3:, :, :]  # 알파 채널 이후

        # 평균 풀링으로 비트 추출
        pooled = F.adaptive_avg_pool2d(watermark_channel, (10, 10))
        bits = (pooled > 0.5).float().flatten().cpu().numpy()

        # BER 추정 (신호 대 잡음비 기반)
        signal_strength = torch.abs(pooled - 0.5).mean().item()
        estimated_ber = max(0, 0.5 - signal_strength)

        return bits[:self.watermark_bits], estimated_ber

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        워터마크 분석 실행

        Args:
            image: RGB 이미지 (H, W, 3)

        Returns:
            ToolResult: 워터마크 분석 결과
        """
        start_time = time.time()

        if self._model is None:
            # Fallback: 기본 분석 (모델 없이)
            return self._fallback_analysis(image, start_time)

        try:
            # 전처리
            img_tensor = self._preprocess(image)

            # 워터마크 추출 (reverse mode)
            with torch.no_grad():
                output = self._model(img_tensor, rev=True)

            # 워터마크 디코딩
            extracted_bits, ber = self._decode_watermark(output)

            # 워터마크 존재 여부 판단
            has_watermark = ber < 0.3  # BER 30% 미만이면 워터마크 존재

            # 신뢰도 계산
            confidence = 1.0 - ber if has_watermark else ber

            # 판정
            if has_watermark:
                verdict = Verdict.AUTHENTIC  # 유효한 워터마크 = 원본
                explanation = (
                    f"유효한 워터마크가 탐지되었습니다. "
                    f"BER: {ber:.2%}. 이미지의 무결성이 확인됩니다."
                )
            else:
                verdict = Verdict.UNCERTAIN
                explanation = (
                    f"워터마크가 탐지되지 않거나 손상되었습니다. "
                    f"BER: {ber:.2%}. 추가 분석이 필요합니다."
                )

            processing_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "has_watermark": has_watermark,
                    "bit_error_rate": ber,
                    "extracted_bits_sample": extracted_bits[:10].tolist(),
                    "signal_detected": ber < 0.5
                },
                explanation=explanation,
                raw_output=output.cpu().numpy() if output is not None else None,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ToolResult(
                tool_name=self.name,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                evidence={"error": str(e)},
                explanation=f"워터마크 분석 중 오류 발생: {str(e)}",
                processing_time=processing_time
            )

    def _fallback_analysis(self, image: np.ndarray, start_time: float) -> ToolResult:
        """모델 없이 기본 분석 수행"""
        processing_time = time.time() - start_time

        return ToolResult(
            tool_name=self.name,
            verdict=Verdict.UNCERTAIN,
            confidence=0.3,
            evidence={
                "fallback_mode": True,
                "image_shape": image.shape,
                "model_available": False
            },
            explanation="HiNet 모델을 사용할 수 없어 기본 분석만 수행되었습니다. "
                       "체크포인트 파일을 확인해주세요.",
            processing_time=processing_time
        )


# LangChain Tool 호환 래퍼
def create_langchain_tool():
    """LangChain용 Tool 생성"""
    try:
        from langchain.tools import Tool as LCTool
        from langchain.pydantic_v1 import BaseModel, Field

        class WatermarkInput(BaseModel):
            image_path: str = Field(description="분석할 이미지 파일 경로")

        tool_instance = WatermarkTool()

        def run_analysis(image_path: str) -> str:
            """이미지 파일 경로로 워터마크 분석 실행"""
            image = np.array(Image.open(image_path).convert("RGB"))
            result = tool_instance(image)
            return str(result.to_dict())

        return LCTool(
            name="watermark_analyzer",
            description=tool_instance.description,
            func=run_analysis,
            args_schema=WatermarkInput
        )

    except ImportError:
        return None
