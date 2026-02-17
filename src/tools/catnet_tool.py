"""
CAT-Net 기반 압축 아티팩트 분석 도구
Compression Artifact Tracing Network(CAT-Net)로 조작 영역을 탐지합니다.
"""
import argparse
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base_tool import BaseTool, ToolResult, Verdict
from .frequency_tool import FrequencyAnalysisTool

try:
    from configs.settings import config, CATNET_DIR
except ImportError:
    config = None
    CATNET_DIR = Path(__file__).resolve().parents[2] / "CAT-Net-main"


class CATNetAnalysisTool(BaseTool):
    """CAT-Net 압축 아티팩트 분석 도구"""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            try:
                device = config.model.device
            except Exception:
                device = "cpu"

        super().__init__(
            name="catnet_analyzer",
            description=(
                "CAT-Net 기반 압축 아티팩트 분석. JPEG 압축 흔적 및 이중 압축 기반 "
                "조작 영역을 픽셀 단위로 탐지합니다."
            ),
            device=device,
        )

        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            env_ckpt = os.environ.get("MAIFS_CATNET_CHECKPOINT")
            if env_ckpt:
                self.checkpoint_path = Path(env_ckpt)
            else:
                try:
                    self.checkpoint_path = config.model.catnet_checkpoint
                except Exception:
                    self.checkpoint_path = CATNET_DIR / "output" / "splicing_dataset" / "CAT_full" / "CAT_full_v2.pth.tar"

        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            env_cfg = os.environ.get("MAIFS_CATNET_CONFIG")
            if env_cfg:
                self.config_path = Path(env_cfg)
            else:
                try:
                    self.config_path = config.model.catnet_config
                except Exception:
                    self.config_path = CATNET_DIR / "experiments" / "CAT_full.yaml"

        self.mask_threshold = 0.4
        self.authentic_ratio_threshold = 0.03
        self.manipulated_ratio_threshold = 0.10
        self._thresholds = self._load_thresholds()
        comp_cfg = self._thresholds.get("compression", {})
        self.mask_threshold = float(comp_cfg.get("mask_threshold", self.mask_threshold))
        self.authentic_ratio_threshold = float(
            comp_cfg.get("authentic_ratio_threshold", self.authentic_ratio_threshold)
        )
        self.manipulated_ratio_threshold = float(
            comp_cfg.get("manipulated_ratio_threshold", self.manipulated_ratio_threshold)
        )

        self._fallback_tool = FrequencyAnalysisTool(device=device)
        self._load_error: Optional[str] = None

    def _load_thresholds(self) -> Dict[str, Any]:
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

    def _prepare_catnet_modules(self) -> bool:
        if not CATNET_DIR.exists():
            self._load_error = f"CAT-Net 디렉토리 미존재: {CATNET_DIR}"
            return False

        catnet_str = str(CATNET_DIR)
        if catnet_str not in sys.path:
            sys.path.insert(0, catnet_str)

        # 다른 서브모듈의 lib 패키지와 충돌 방지
        for module_name in [
            "lib",
            "lib.config",
            "lib.models",
            "lib.models.network_CAT",
            "Splicing",
            "Splicing.data",
            "Splicing.data.dataset_arbitrary",
            "project_config",
        ]:
            loaded = sys.modules.get(module_name)
            if loaded is None:
                continue
            loaded_file = str(getattr(loaded, "__file__", "") or "")
            if loaded_file and catnet_str not in loaded_file:
                sys.modules.pop(module_name, None)
        return True

    def load_model(self) -> None:
        if self._is_loaded:
            return

        if not self._prepare_catnet_modules():
            print(f"[CATNetTool] {self._load_error}. Frequency fallback 사용")
            self._model = None
            self._is_loaded = True
            return

        if not self.config_path.exists():
            self._load_error = f"CAT-Net config 미존재: {self.config_path}"
            print(f"[CATNetTool] {self._load_error}. Frequency fallback 사용")
            self._model = None
            self._is_loaded = True
            return

        if not self.checkpoint_path.exists():
            self._load_error = f"CAT-Net checkpoint 미존재: {self.checkpoint_path}"
            print(f"[CATNetTool] {self._load_error}. Frequency fallback 사용")
            self._model = None
            self._is_loaded = True
            return

        try:
            # Legacy CAT-Net code uses removed numpy aliases on modern numpy.
            if not hasattr(np, "float"):
                np.float = float  # type: ignore[attr-defined]

            cat_models = importlib.import_module("lib.models")
            cat_config_module = importlib.import_module("lib.config")
            cat_config = cat_config_module.config
            update_config = cat_config_module.update_config

            pretrained_rgb = CATNET_DIR / "pretrained_models" / "hrnetv2_w48_imagenet_pretrained.pth"
            pretrained_dct = CATNET_DIR / "pretrained_models" / "DCT_djpeg.pth.tar"
            args = argparse.Namespace(
                cfg=str(self.config_path),
                opts=[
                    "TEST.MODEL_FILE", str(self.checkpoint_path),
                    "TEST.FLIP_TEST", "False",
                    "TEST.NUM_SAMPLES", "0",
                    "MODEL.PRETRAINED_RGB", str(pretrained_rgb),
                    "MODEL.PRETRAINED_DCT", str(pretrained_dct),
                ],
            )
            update_config(cat_config, args)

            model = eval("cat_models." + cat_config.MODEL.NAME + ".get_seg_model")(cat_config)
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()

            self._model = model
            self._is_loaded = True
            self._load_error = None
            print(f"[CATNetTool] 모델 로드 완료: {self.checkpoint_path}")
        except Exception as e:
            self._load_error = str(e)
            print(f"[CATNetTool] 모델 로드 실패: {e}. Frequency fallback 사용")
            self._model = None
            self._is_loaded = True

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        with tempfile.TemporaryDirectory(prefix="maifs_catnet_") as tmpdir:
            image_path = Path(tmpdir) / "input.jpg"
            Image.fromarray(image).convert("RGB").save(image_path, quality=100, subsampling=0)

            arbitrary_dataset_module = importlib.import_module("Splicing.data.dataset_arbitrary")
            arbitrary_cls = getattr(arbitrary_dataset_module, "arbitrary")
            dataset = arbitrary_cls(
                crop_size=None,
                grid_crop=True,
                blocks=("RGB", "DCTvol", "qtable"),
                DCT_channels=1,
                tamp_list=str(image_path),
                read_from_jpeg=True,
            )
            tensor, _, qtable = dataset.get_tamp(0)

        return tensor.unsqueeze(0).to(self.device), qtable.unsqueeze(0).to(self.device)

    def _postprocess_mask(self, pred: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        mask = F.softmax(pred.squeeze(0), dim=0)[1].detach().cpu().numpy()
        if mask.shape != original_size:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((original_size[1], original_size[0]), Image.BILINEAR)
            mask = np.array(mask_pil).astype(np.float32) / 255.0
        return np.clip(mask, 0.0, 1.0)

    def _analyze_mask(self, mask: np.ndarray) -> Dict[str, float]:
        binary = (mask >= self.mask_threshold).astype(np.uint8)
        return {
            "manipulation_ratio": float(np.mean(binary)),
            "mean_intensity": float(np.mean(mask)),
            "max_intensity": float(np.max(mask)),
            "mask_threshold": float(self.mask_threshold),
        }

    def analyze(self, image: np.ndarray) -> ToolResult:
        start_time = time.time()
        if not self._is_loaded:
            self.load_model()

        if self._model is None:
            fallback = self._fallback_tool(image)
            # AGENTS.md 정책: graceful degradation 유지 + 조용한 성능 저하 차단
            # - fallback_raw_* : 사후 분석/진단용으로 원본 FFT 판정 보존
            # - verdict/confidence : UNCERTAIN으로 cap → COBRA에서 가중치 자동 감소
            fallback.evidence = {
                **fallback.evidence,
                "backend": "frequency_fallback",
                "catnet_available": False,
                "catnet_error": self._load_error or "",
                "fallback_raw_verdict": fallback.verdict.value,
                "fallback_raw_confidence": float(fallback.confidence),
                "fallback_mode": True,
            }
            fallback.verdict = Verdict.UNCERTAIN
            fallback.confidence = min(float(fallback.confidence), 0.35)
            fallback.explanation = (
                "CAT-Net을 사용할 수 없어 Frequency fallback으로 분석했습니다 "
                "(verdict: UNCERTAIN cap, confidence ≤ 0.35). "
                "원본 FFT 판정은 evidence.fallback_raw_verdict를 참조하세요. "
                + fallback.explanation
            )
            fallback.tool_name = self.name
            return fallback

        try:
            original_size = (image.shape[0], image.shape[1])
            img_tensor, qtable = self._preprocess(image)

            with torch.no_grad():
                pred = self._model(img_tensor, qtable)
            mask = self._postprocess_mask(pred, original_size)
            mask_analysis = self._analyze_mask(mask)
            manipulation_ratio = mask_analysis["manipulation_ratio"]

            if manipulation_ratio < self.authentic_ratio_threshold:
                verdict = Verdict.AUTHENTIC
                confidence = float(1.0 - manipulation_ratio)
                explanation = (
                    f"CAT-Net 기준 압축 조작 흔적이 낮습니다. "
                    f"조작 비율: {manipulation_ratio:.2%}"
                )
            elif manipulation_ratio >= self.manipulated_ratio_threshold:
                verdict = Verdict.MANIPULATED
                confidence = float(manipulation_ratio)
                explanation = (
                    f"CAT-Net이 압축 아티팩트 기반 조작 영역을 탐지했습니다. "
                    f"조작 비율: {manipulation_ratio:.2%}"
                )
            else:
                verdict = Verdict.UNCERTAIN
                confidence = float(manipulation_ratio)
                explanation = (
                    f"CAT-Net 탐지 신호가 경계 구간입니다. "
                    f"조작 비율: {manipulation_ratio:.2%}"
                )

            processing_time = time.time() - start_time
            evidence = {
                "backend": "catnet",
                "catnet_available": True,
                "compression_artifact_score": manipulation_ratio,
                **mask_analysis,
            }
            return ToolResult(
                tool_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence=evidence,
                explanation=explanation,
                manipulation_mask=mask,
                processing_time=processing_time,
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return ToolResult(
                tool_name=self.name,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                evidence={
                    "backend": "catnet",
                    "catnet_available": True,
                    "error": str(e),
                },
                explanation=f"CAT-Net 분석 중 오류 발생: {e}",
                processing_time=processing_time,
            )


def create_langchain_tool():
    """LangChain Tool 호환 래퍼"""
    from langchain.tools import Tool

    tool_instance = CATNetAnalysisTool()

    def _analyze_image(image_path: str) -> str:
        image = np.array(Image.open(image_path).convert("RGB"))
        result = tool_instance(image)
        return result.to_dict()

    return Tool(
        name="catnet_analyzer",
        description=(
            "CAT-Net 기반 압축 아티팩트 분석 도구입니다. JPEG 조작 흔적과 "
            "이중 압축 기반 포렌식 신호를 탐지합니다."
        ),
        func=_analyze_image,
    )
