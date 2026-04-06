"""
FatFormer 기반 AI 생성 이미지 탐지 도구
CLIP ViT-L/14 + Forgery-Aware Adapter를 사용하여 AI 생성 이미지를 분류
"""
import sys
import time
import importlib
import json
from pathlib import Path
from typing import Optional
import numpy as np
from argparse import Namespace

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from .base_tool import BaseTool, ToolResult, Verdict

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# 설정에서 경로 로드
try:
    from configs.settings import config
except ImportError:
    config = None

# FatFormer 모듈 경로
FATFORMER_ROOT = Path(__file__).resolve().parents[2] / "Integrated Submodules" / "FatFormer"


class FatFormerTool(BaseTool):
    """
    FatFormer 기반 AI 생성 이미지 탐지 도구

    CLIP ViT-L/14 백본에 Forgery-Aware Adapter (공간 + DWT 주파수 경로)를 결합하여:
    - AI 생성 이미지 vs 실제 이미지 이진 분류
    - Diffusion 모델 생성 이미지 탐지
    - JPEG 압축에 강건한 탐지
    - 다양한 생성기(GAN, Diffusion, etc.)에 대한 교차 일반화
    """

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = None):
        # 설정에서 디바이스 로드
        if device is None:
            try:
                device = config.model.device
            except Exception:
                device = "cpu"

        super().__init__(
            name="fatformer_analyzer",
            description="FatFormer (CLIP ViT-L/14 + Forgery-Aware Adapter) 기반 AI 생성 이미지 탐지. "
                       "의미론적 특징과 주파수 특징을 결합하여 AI 생성 여부를 판별합니다.",
            device=device
        )

        # 체크포인트 경로 설정
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            try:
                self.checkpoint_path = config.model.fatformer_checkpoint
            except Exception:
                self.checkpoint_path = FATFORMER_ROOT / "checkpoint" / "fatformer.pth"

        # 모델 설정
        self.input_resolution = 224
        self.img_resolution = 256
        self._thresholds = self._load_thresholds()
        fat_cfg = self._thresholds.get("fatformer", {})
        self.ai_threshold = float(fat_cfg.get("ai_threshold", 0.5))
        self.auth_threshold = float(fat_cfg.get("auth_threshold", self.ai_threshold))

        # 전처리 파이프라인 (CLIP 표준)
        self._transform = Compose([
            Resize(self.input_resolution, interpolation=BICUBIC),
            CenterCrop(self.input_resolution),
            ToTensor(),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

    def _load_thresholds(self) -> dict:
        """Load calibrated thresholds from configs/tool_thresholds.json if present."""
        threshold_path = Path(__file__).resolve().parents[2] / "configs" / "tool_thresholds.json"
        if threshold_path.exists():
            try:
                with threshold_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _get_model_args(self) -> Namespace:
        """FatFormer 모델 생성에 필요한 인자"""
        return Namespace(
            backbone='CLIP:ViT-L/14',
            num_classes=2,
            num_vit_adapter=8,
            num_context_embedding=8,
            init_context_embedding="",
            hidden_dim=768,
            clip_vision_width=1024,
            frequency_encoder_layer=2,
            decoder_layer=4,
            num_heads=12,
        )

    def load_model(self) -> None:
        """FatFormer 모델 로드"""
        if self._is_loaded:
            return

        # FatFormer 모듈 경로 추가
        if FATFORMER_ROOT.exists():
            fatformer_str = str(FATFORMER_ROOT)
            if fatformer_str not in sys.path:
                sys.path.insert(0, fatformer_str)

        # CLIP pretrained 가중치 확인
        pretrained_path = FATFORMER_ROOT / "pretrained" / "ViT-L-14.pt"
        if not pretrained_path.exists():
            print(f"[FatFormerTool] CLIP pretrained 가중치 미존재: {pretrained_path}")
            print("[FatFormerTool] Fallback 모드로 전환")
            self._model = None
            self._is_loaded = True
            return

        try:
            # 다른 서브모듈(MVSS 등)의 `models`와 이름 충돌 방지
            loaded_models = sys.modules.get("models")
            if loaded_models is not None:
                loaded_file = getattr(loaded_models, "__file__", "") or ""
                if str(FATFORMER_ROOT) not in loaded_file:
                    sys.modules.pop("models", None)

            build_model = importlib.import_module("models").build_model

            args = self._get_model_args()
            self._model = build_model(args)
            self._model = self._model.to(self.device)

            # FatFormer 학습된 체크포인트 로드
            if self.checkpoint_path.exists():
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False
                )
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                self._model.load_state_dict(state_dict, strict=False)
                print(f"[FatFormerTool] FatFormer 체크포인트 로드: {self.checkpoint_path}")
            else:
                print(f"[FatFormerTool] FatFormer 체크포인트 미존재: {self.checkpoint_path}")
                print("[FatFormerTool] CLIP pretrained 가중치만 사용 (성능 저하 가능)")

            self._model.eval()
            self._is_loaded = True
            print("[FatFormerTool] FatFormer 모델 로드 완료")

        except ImportError as e:
            print(f"[FatFormerTool] FatFormer 모듈 임포트 실패: {e}")
            print("[FatFormerTool] Fallback 모드로 전환")
            self._model = None
            self._is_loaded = True

        except Exception as e:
            print(f"[FatFormerTool] 모델 로드 실패: {e}")
            print("[FatFormerTool] Fallback 모드로 전환")
            self._model = None
            self._is_loaded = True

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리 (CLIP 표준)"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image).convert("RGB")

        img_tensor = self._transform(pil_image)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.to(self.device)

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        AI 생성 이미지 탐지 실행

        Args:
            image: RGB 이미지 (H, W, 3)

        Returns:
            ToolResult: AI 생성 탐지 결과
        """
        start_time = time.time()

        if not self._is_loaded:
            self.load_model()

        if self._model is None:
            return self._fallback_analysis(image, start_time)

        try:
            # 전처리
            img_tensor = self._preprocess(image)

            # 추론
            with torch.no_grad():
                logits = self._model(img_tensor)  # [1, 2]

            # 확률 계산
            probs = F.softmax(logits, dim=1)
            real_prob = probs[0, 0].item()
            fake_prob = probs[0, 1].item()

            # 판정
            if fake_prob >= self.ai_threshold:
                verdict = Verdict.AI_GENERATED
                confidence = fake_prob
                explanation = (
                    f"FatFormer가 AI 생성 이미지로 판별했습니다. "
                    f"AI 생성 확률: {fake_prob:.2%}. "
                    f"CLIP 의미론적 특징과 DWT 주파수 특징 모두에서 "
                    f"AI 생성 패턴이 감지되었습니다."
                )
            elif fake_prob > self.auth_threshold:
                verdict = Verdict.UNCERTAIN
                confidence = 0.5
                explanation = (
                    f"AI 생성 여부를 확실히 판단하기 어렵습니다. "
                    f"AI 생성 확률: {fake_prob:.2%}. "
                    f"다른 분석 도구의 결과를 참고하세요."
                )
            else:
                verdict = Verdict.AUTHENTIC
                confidence = real_prob
                explanation = (
                    f"FatFormer가 실제 이미지로 판별했습니다. "
                    f"실제 이미지 확률: {real_prob:.2%}. "
                    f"AI 생성 특징이 감지되지 않았습니다."
                )

            processing_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "fake_probability": fake_prob,
                    "real_probability": real_prob,
                    "model_backbone": "CLIP:ViT-L/14",
                    "detection_method": "fatformer",
                    "adapter_type": "forgery_aware_adapter",
                    "frequency_pathway": "DWT",
                },
                explanation=explanation,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ToolResult(
                tool_name=self.name,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                evidence={"error": str(e)},
                explanation=f"FatFormer 분석 중 오류 발생: {str(e)}",
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
            explanation="FatFormer 모델을 사용할 수 없어 기본 분석만 수행되었습니다. "
                       "CLIP pretrained 가중치와 FatFormer 체크포인트를 확인해주세요.",
            processing_time=processing_time
        )
