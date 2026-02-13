"""
공간 분석 도구
OmniGuard의 ViT 모델을 래핑하여 이미지 조작 영역 탐지 수행
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

from .base_tool import BaseTool, ToolResult, Verdict

# 설정에서 경로 로드
try:
    from configs.settings import config, OMNIGUARD_DIR
    OMNIGUARD_PATH = OMNIGUARD_DIR
except ImportError:
    config = None
    OMNIGUARD_PATH = Path(__file__).resolve().parents[2] / "OmniGuard-main"

TRUFOR_ROOT = Path(__file__).resolve().parents[2] / "TruFor-main" / "TruFor_train_test"

# OmniGuard 모듈 경로 추가
if OMNIGUARD_PATH.exists():
    sys.path.insert(0, str(OMNIGUARD_PATH))
if TRUFOR_ROOT.exists():
    sys.path.insert(0, str(TRUFOR_ROOT))


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
        device: str = None,
        backend: Optional[str] = None,
        use_mvss_mask: Optional[bool] = None,
        mvss_weight: float = 0.4
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

        self.backend = (backend or os.environ.get("MAIFS_SPATIAL_BACKEND", "omniguard")).lower()

        # 체크포인트 경로 설정
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            try:
                self.checkpoint_path = config.model.omniguard_checkpoint_dir
            except:
                self.checkpoint_path = OMNIGUARD_PATH / "checkpoint"

        self.input_size = input_size
        self._thresholds = self._load_thresholds()
        spatial_cfg = self._thresholds.get("spatial", {})
        self.mask_threshold = float(spatial_cfg.get("mask_threshold", 0.4))
        self.authentic_ratio_threshold = float(spatial_cfg.get("authentic_ratio_threshold", 0.05))
        self.ai_ratio_threshold = float(spatial_cfg.get("ai_ratio_threshold", 0.8))
        self.mvss_score_low = float(spatial_cfg.get("mvss_score_low", 0.2))
        self.mvss_score_high = float(spatial_cfg.get("mvss_score_high", 0.8))
        env_use_mvss = os.environ.get("MAIFS_SPATIAL_USE_MVSS")
        if use_mvss_mask is not None:
            self.use_mvss_mask = bool(use_mvss_mask)
        elif env_use_mvss is not None:
            self.use_mvss_mask = env_use_mvss == "1"
        else:
            # MVSS 체크포인트가 있으면 기본으로 마스크 융합 활성화
            mvss_ckpt = Path(
                os.environ.get(
                    "MAIFS_MVSS_CHECKPOINT",
                    Path(__file__).resolve().parents[2] / "MVSS-Net-master" / "ckpt" / "mvssnet_casia.pt"
                )
            )
            self.use_mvss_mask = mvss_ckpt.exists()
        env_weight = os.environ.get("MAIFS_SPATIAL_MVSS_WEIGHT")
        default_mvss_weight = float(spatial_cfg.get("mvss_weight", mvss_weight))
        self.mvss_weight = float(env_weight) if env_weight is not None else default_mvss_weight
        self._mvss_tool = None
        self._trufor_config = None

        trufor_ckpt = os.environ.get("MAIFS_TRUFOR_CHECKPOINT")
        if trufor_ckpt:
            self._trufor_checkpoint = Path(trufor_ckpt)
        else:
            self._trufor_checkpoint = TRUFOR_ROOT / "pretrained_models" / "trufor.pth.tar"

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

    def load_model(self) -> None:
        """ViT IML 모델 로드"""
        if self._is_loaded:
            return

        if self.backend == "trufor":
            self._load_trufor_model()
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

            # 체크포인트 경로 결정 (파일 > 기본 iml_vit.pth > 설정값)
            checkpoint_file = None
            if isinstance(self.checkpoint_path, Path) and self.checkpoint_path.is_file():
                checkpoint_file = self.checkpoint_path
            else:
                candidate = self.checkpoint_path / "iml_vit.pth"
                if candidate.exists():
                    checkpoint_file = candidate
                else:
                    try:
                        checkpoint_file = config.model.vit_checkpoint
                    except Exception:
                        checkpoint_file = None

            if checkpoint_file and checkpoint_file.exists():
                state_dict = torch.load(checkpoint_file, map_location=self.device)
                if isinstance(state_dict, dict):
                    for key in ("state_dict", "model", "net"):
                        if key in state_dict and isinstance(state_dict[key], dict):
                            state_dict = state_dict[key]
                            break
                self._model.load_state_dict(state_dict, strict=False)
                print(f"[SpatialTool] 체크포인트 로드: {checkpoint_file}")
            else:
                print("[SpatialTool] 체크포인트를 찾지 못했습니다. Fallback 모드로 전환합니다.")
                self._model = None
                self._is_loaded = True
                return

            self._model.eval()
            self._is_loaded = True
            print("[SpatialTool] ViT IML 모델 로드 완료")

        except ImportError as e:
            print(f"[SpatialTool] OmniGuard 모듈 임포트 실패: {e}")
            print("[SpatialTool] Fallback 모드로 전환")
            self._model = None
            self._is_loaded = True

    def _load_trufor_model(self) -> None:
        """TruFor 모델 로드"""
        if self._is_loaded:
            return
        if not TRUFOR_ROOT.exists():
            print("[SpatialTool] TruFor 디렉토리를 찾지 못했습니다. Fallback 모드로 전환합니다.")
            self._model = None
            self._is_loaded = True
            return
        if not self._trufor_checkpoint.exists():
            print(f"[SpatialTool] TruFor 체크포인트 미존재: {self._trufor_checkpoint}")
            self._model = None
            self._is_loaded = True
            return

        try:
            from lib.config import config as trufor_config
            from lib.utils import get_model

            trufor_config.defrost()
            trufor_config.merge_from_file(str(TRUFOR_ROOT / "lib" / "config" / "trufor_ph3.yaml"))
            trufor_config.merge_from_list(["TEST.MODEL_FILE", str(self._trufor_checkpoint)])
            trufor_config.freeze()

            try:
                checkpoint = torch.load(
                    self._trufor_checkpoint,
                    map_location=self.device,
                    weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(self._trufor_checkpoint, map_location=self.device)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

            model = get_model(trufor_config)
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device)
            model.eval()

            self._model = model
            self._trufor_config = trufor_config
            self._is_loaded = True
            print(f"[SpatialTool] TruFor 모델 로드 완료: {self._trufor_checkpoint}")
        except Exception as e:
            print(f"[SpatialTool] TruFor 모델 로드 실패: {e}")
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
        binary_mask = (mask > self.mask_threshold).astype(np.uint8)

        # 조작 영역 비율
        manipulation_ratio = np.mean(binary_mask)

        # 연결된 영역 수 (간단한 분석)
        # TODO: cv2.connectedComponents 사용 가능

        return {
            "manipulation_ratio": float(manipulation_ratio),
            "max_intensity": float(np.max(mask)),
            "mean_intensity": float(np.mean(mask)),
            "threshold_used": self.mask_threshold
        }

    def _get_mvss_mask(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """MVSS 마스크를 가져와 공간 마스크와 결합"""
        if not self.use_mvss_mask:
            return None

        try:
            if self._mvss_tool is None:
                from .noise_tool import NoiseAnalysisTool
                self._mvss_tool = NoiseAnalysisTool(device=self.device, backend="mvss")

            result = self._mvss_tool(image)
            if result.manipulation_mask is None:
                return None
            score = float(result.evidence.get("mvss_score", 0.0))
            return result.manipulation_mask, score
        except Exception:
            return None

    def _merge_masks(
        self,
        spatial_mask: np.ndarray,
        mvss_mask: np.ndarray,
        mvss_score: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """공간 마스크와 MVSS 마스크 결합"""
        if mvss_mask.shape != spatial_mask.shape:
            mvss_pil = Image.fromarray((mvss_mask * 255).astype(np.uint8))
            mvss_pil = mvss_pil.resize((spatial_mask.shape[1], spatial_mask.shape[0]), Image.BILINEAR)
            mvss_mask = np.array(mvss_pil).astype(np.float32) / 255.0

        base_weight = min(max(self.mvss_weight, 0.0), 1.0)

        # MVSS 점수가 낮으면 융합 가중치를 자동으로 낮춰 과검출을 완화
        if mvss_score is None:
            effective_weight = base_weight
        else:
            low = min(self.mvss_score_low, self.mvss_score_high)
            high = max(self.mvss_score_low, self.mvss_score_high)
            if high <= low + 1e-8:
                confidence_scale = 1.0 if mvss_score >= high else 0.0
            else:
                confidence_scale = (float(mvss_score) - low) / (high - low)
                confidence_scale = min(max(confidence_scale, 0.0), 1.0)
            effective_weight = base_weight * confidence_scale

        merged = (1.0 - effective_weight) * spatial_mask + effective_weight * mvss_mask
        return np.clip(merged, 0.0, 1.0), float(effective_weight)

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        공간 분석 실행

        Args:
            image: RGB 이미지 (H, W, 3)

        Returns:
            ToolResult: 조작 영역 분석 결과
        """
        start_time = time.time()

        if not self._is_loaded:
            self.load_model()

        if self._model is None:
            return self._fallback_analysis(image, start_time)

        try:
            if self.backend == "trufor":
                return self._analyze_trufor(image, start_time)

            # 전처리
            img_tensor, original_size = self._preprocess(image)

            # Container 이미지 (원본과 동일하게 사용)
            container_tensor = img_tensor.clone()

            # 마스크 예측
            with torch.no_grad():
                mask_pred = self._model(img_tensor, container_tensor)

            # 후처리
            mask = self._postprocess_mask(mask_pred, original_size)

            mvss_info = self._get_mvss_mask(image)
            if mvss_info is not None:
                mvss_mask, mvss_score = mvss_info
                mask, effective_mvss_weight = self._merge_masks(mask, mvss_mask, mvss_score)

            # 마스크 분석
            mask_analysis = self._analyze_mask(mask)
            if mvss_info is not None:
                mask_analysis["mvss_used"] = True
                mask_analysis["mvss_weight"] = float(self.mvss_weight)
                mask_analysis["mvss_effective_weight"] = float(effective_mvss_weight)
                mask_analysis["mvss_score"] = float(mvss_score)
                mask_analysis["mvss_mask_mean"] = float(np.mean(mvss_mask))
                mask_analysis["mvss_mask_max"] = float(np.max(mvss_mask))

            # 판정
            manipulation_ratio = mask_analysis["manipulation_ratio"]

            if manipulation_ratio < self.authentic_ratio_threshold:
                verdict = Verdict.AUTHENTIC
                confidence = 1.0 - manipulation_ratio
                explanation = (
                    f"이미지에서 유의미한 조작 영역이 탐지되지 않았습니다. "
                    f"조작 비율: {manipulation_ratio:.2%}"
                )
            elif manipulation_ratio > self.ai_ratio_threshold:
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

    def _analyze_trufor(self, image: np.ndarray, start_time: float) -> ToolResult:
        """TruFor 기반 공간 조작 탐지"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        rgb = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        rgb = rgb.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred, conf, det, _ = self._model(rgb, save_np=False)

        conf_map = None
        if conf is not None:
            conf = torch.squeeze(conf, 0)
            conf = torch.sigmoid(conf)[0]
            conf_map = conf.cpu().numpy()

        det_score = None
        if det is not None:
            det_score = float(torch.sigmoid(det).item())

        pred = torch.squeeze(pred, 0)
        pred = F.softmax(pred, dim=0)[1]
        pred = pred.cpu().numpy()

        if pred.shape != image.shape[:2]:
            pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
            pred_pil = pred_pil.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
            pred = np.array(pred_pil).astype(np.float32) / 255.0

        mask_analysis = self._analyze_mask(pred)
        if det_score is not None:
            mask_analysis["trufor_score"] = det_score
        if conf_map is not None:
            mask_analysis["trufor_conf_mean"] = float(np.mean(conf_map))
            mask_analysis["trufor_conf_min"] = float(np.min(conf_map))
            mask_analysis["trufor_conf_max"] = float(np.max(conf_map))

        manipulation_ratio = mask_analysis["manipulation_ratio"]
        if manipulation_ratio < self.authentic_ratio_threshold:
            verdict = Verdict.AUTHENTIC
            confidence = 1.0 - manipulation_ratio
            explanation = (
                f"TruFor가 유의미한 조작 영역을 탐지하지 않았습니다. "
                f"조작 비율: {manipulation_ratio:.2%}"
            )
        elif manipulation_ratio > self.ai_ratio_threshold:
            verdict = Verdict.AI_GENERATED
            confidence = manipulation_ratio
            explanation = (
                f"TruFor가 이미지 대부분의 조작 가능성을 탐지했습니다. "
                f"조작 비율: {manipulation_ratio:.2%}"
            )
        else:
            verdict = Verdict.MANIPULATED
            confidence = manipulation_ratio
            explanation = (
                f"TruFor가 이미지 일부 조작 영역을 탐지했습니다. "
                f"조작 비율: {manipulation_ratio:.2%}"
            )

        processing_time = time.time() - start_time
        return ToolResult(
            tool_name=self.name,
            verdict=verdict,
            confidence=confidence,
            evidence=mask_analysis,
            explanation=explanation,
            manipulation_mask=pred,
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
