"""
노이즈 분석 도구
SRM(Spatial Rich Model) 필터와 PRNU(Photo Response Non-Uniformity) 분석
"""
import time
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage, signal
from PIL import Image
import torch

from .base_tool import BaseTool, ToolResult, Verdict

_MAIFS_ROOT = Path(__file__).resolve().parents[2]
MVSS_ROOT = (
    _MAIFS_ROOT / "external" / "MVSS-Net-master"
    if (_MAIFS_ROOT / "external" / "MVSS-Net-master").exists()
    else _MAIFS_ROOT / "MVSS-Net-master"
)


class NoiseAnalysisTool(BaseTool):
    """
    노이즈 분석 도구 (SRM + PRNU)

    센서 노이즈 패턴 분석을 통해:
    - 카메라 고유 PRNU 패턴 탐지
    - AI 생성 이미지의 노이즈 특성 분석
    - 이미지 조작 영역의 노이즈 불일치 탐지
    """

    def __init__(self, device: str = "cpu", backend: Optional[str] = None):
        super().__init__(
            name="noise_analyzer",
            description="SRM 필터 및 PRNU 기반 노이즈 패턴 분석. "
                       "카메라 센서 노이즈와 AI 생성 노이즈의 차이를 탐지합니다.",
            device=device
        )
        self._srm_filters = None
        self._is_loaded = False
        self._backend = (backend or os.environ.get("MAIFS_NOISE_BACKEND", "prnu")).lower()
        self._mvss_model = None
        self._mvss_checkpoint = Path(
            os.environ.get("MAIFS_MVSS_CHECKPOINT", MVSS_ROOT / "ckpt" / "mvssnet_casia.pt")
        )
        self._thresholds = self._load_thresholds()

        noise_cfg = self._thresholds.get("noise", {})
        self.manipulation_threshold = float(noise_cfg.get("manipulation_threshold", 0.6))
        self.ai_threshold = float(noise_cfg.get("ai_threshold", 0.6))
        self.ai_diversity_threshold = float(noise_cfg.get("ai_diversity_threshold", 0.3))
        self.authentic_diversity_threshold = float(noise_cfg.get("authentic_diversity_threshold", 0.5))
        self.authentic_ai_threshold = float(noise_cfg.get("authentic_ai_threshold", 0.4))
        self.mvss_threshold = float(noise_cfg.get("mvss_threshold", 0.5))
        self.mvss_auth_threshold = float(noise_cfg.get("mvss_auth_threshold", self.mvss_threshold))
        self.mvss_uncertain_margin = float(noise_cfg.get("mvss_uncertain_margin", 0.0))

    def _load_thresholds(self) -> Dict[str, Any]:
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
        """SRM 필터 초기화"""
        if self._is_loaded:
            return

        if self._backend == "mvss":
            self._load_mvss_model()
            return

        # SRM 필터 정의 (30개 중 대표적인 필터들)
        self._srm_filters = self._initialize_srm_filters()
        self._is_loaded = True
        print("[NoiseTool] SRM 필터 초기화 완료")

    def _load_mvss_model(self) -> None:
        """MVSS-Net 모델 로드"""
        if self._is_loaded:
            return
        if not MVSS_ROOT.exists():
            print("[NoiseTool] MVSS-Net 디렉토리를 찾지 못했습니다. PRNU 모드로 전환합니다.")
            self._backend = "prnu"
            self._is_loaded = False
            self.load_model()
            return
        if not self._mvss_checkpoint.exists():
            print(f"[NoiseTool] MVSS 체크포인트 미존재: {self._mvss_checkpoint}. PRNU 모드로 전환합니다.")
            self._backend = "prnu"
            self._is_loaded = False
            self.load_model()
            return

        try:
            sys.path.insert(0, str(MVSS_ROOT))
            from models.mvssnet import get_mvss

            def _load_state(weights: Path):
                state = torch.load(str(weights), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    return state["state_dict"]
                if isinstance(state, dict) and "model" in state:
                    return state["model"]
                return state

            model = get_mvss(pretrained_base=False, sobel=True, n_input=3, constrain=True)
            model.load_state_dict(_load_state(self._mvss_checkpoint), strict=True)
            model = model.to(self.device)
            model.eval()

            self._mvss_model = model
            self._is_loaded = True
            print(f"[NoiseTool] MVSS-Net 모델 로드 완료: {self._mvss_checkpoint}")
        except Exception as e:
            print(f"[NoiseTool] MVSS 모델 로드 실패: {e}. PRNU 모드로 전환합니다.")
            self._backend = "prnu"
            self._is_loaded = False
            self.load_model()

    def _mvss_inference(self, image: np.ndarray, model: torch.nn.Module) -> Tuple[np.ndarray, float]:
        """Run MVSS-Net without albumentations dependency."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("MVSS 입력은 HxWx3 RGB 이미지여야 합니다.")

        img = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=True)

        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            _, seg = model(tensor)
            seg = torch.sigmoid(seg)

            if torch.isnan(seg).any() or torch.isinf(seg).any():
                max_score = 0.0
            else:
                max_score = float(seg.max().item())

            seg = seg.squeeze(0)
            if seg.dim() == 3:
                seg = seg[0]
            seg = seg.clamp(0.0, 1.0) * 255.0
            mask = seg.to(torch.uint8).cpu().numpy()

        return mask, max_score

    def _initialize_srm_filters(self) -> list:
        """
        Spatial Rich Model 필터 초기화

        Fridrich의 SRM에서 가장 효과적인 필터들 선택
        """
        filters = []

        # 1st order edge filters
        edge_h = np.array([[-1, 1]])
        edge_v = np.array([[-1], [1]])
        filters.extend([edge_h, edge_v])

        # 2nd order filters (Laplacian variants)
        laplacian_1 = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]) / 4.0

        laplacian_2 = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]) / 8.0

        filters.extend([laplacian_1, laplacian_2])

        # 3rd order SQUARE filters
        square_3x3 = np.array([
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1]
        ]) / 4.0

        square_5x5 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ]) / 12.0

        filters.extend([square_3x3, square_5x5])

        # High-pass residual filter
        hp_filter = np.array([
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, -4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0]
        ]) / 4.0

        filters.append(hp_filter)

        return filters

    def _extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        노이즈 잔차 추출

        디노이징 필터 적용 후 원본과의 차이로 노이즈 추출
        """
        # 가우시안 디노이징
        denoised = ndimage.gaussian_filter(image, sigma=1.5)

        # 잔차 = 원본 - 디노이즈
        residual = image.astype(np.float32) - denoised.astype(np.float32)

        return residual

    def _apply_srm_filters(self, image: np.ndarray) -> np.ndarray:
        """
        SRM 필터 적용

        Returns:
            필터 응답의 합성 특징 맵
        """
        if len(image.shape) == 3:
            # 그레이스케일 변환
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # 정규화
        if gray.max() > 1.0:
            gray = gray / 255.0

        # 모든 필터 응답 수집
        responses = []
        for f in self._srm_filters:
            response = ndimage.convolve(gray, f, mode='reflect')
            responses.append(np.abs(response))

        # 평균 응답
        combined = np.mean(responses, axis=0)

        return combined

    def _compute_ela_map(self, image: np.ndarray, quality: int = 95) -> np.ndarray:
        """Error Level Analysis map (JPEG recompression difference)."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        original = Image.fromarray(image)
        buffer = BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")
        buffer.close()

        diff = np.abs(np.array(original, dtype=np.float32) - np.array(recompressed, dtype=np.float32))
        ela_gray = np.mean(diff, axis=2) / 255.0
        return ela_gray

    def _compute_prnu_block_correlations(
        self,
        residual: np.ndarray,
        block_size: int
    ) -> Dict[str, float]:
        """
        PRNU 블록 상관 분석

        각 블록 잔차와 전체 평균 블록 패턴의 상관을 측정하여
        지역적 PRNU 불일치(조작)를 탐지합니다.
        """
        h, w = residual.shape
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size
        if n_blocks_h == 0 or n_blocks_w == 0:
            return {
                "prnu_corr_mean": 0.0,
                "prnu_corr_std": 0.0,
                "prnu_corr_low_ratio": 0.0
            }

        blocks = []
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = residual[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ].astype(np.float32)
                block = block - np.mean(block)
                blocks.append(block.reshape(-1))

        if not blocks:
            return {
                "prnu_corr_mean": 0.0,
                "prnu_corr_std": 0.0,
                "prnu_corr_low_ratio": 0.0
            }

        blocks = np.stack(blocks, axis=0)
        mean_vec = np.mean(blocks, axis=0)
        mean_vec = mean_vec - np.mean(mean_vec)
        mean_norm = np.linalg.norm(mean_vec) + 1e-8

        norms = np.linalg.norm(blocks, axis=1) + 1e-8
        corrs = (blocks @ mean_vec) / (norms * mean_norm)

        q1 = np.percentile(corrs, 25)
        q3 = np.percentile(corrs, 75)
        iqr = q3 - q1
        low_thresh = q1 - 1.5 * iqr
        low_ratio = float(np.mean(corrs < low_thresh)) if iqr > 0 else 0.0

        return {
            "prnu_corr_mean": float(np.mean(corrs)),
            "prnu_corr_std": float(np.std(corrs)),
            "prnu_corr_low_ratio": low_ratio
        }

    def _estimate_prnu(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        PRNU (Photo Response Non-Uniformity) 추정

        Returns:
            (PRNU 패턴 추정치, 통계 정보)
        """
        # 노이즈 잔차 추출
        if len(image.shape) == 3:
            residuals = []
            for c in range(3):
                residual = self._extract_noise_residual(image[:, :, c])
                residuals.append(residual)
            prnu_estimate = np.mean(residuals, axis=0)
        else:
            prnu_estimate = self._extract_noise_residual(image)

        # 통계 계산
        stats = {
            "mean": float(np.mean(prnu_estimate)),
            "std": float(np.std(prnu_estimate)),
            "variance": float(np.var(prnu_estimate)),
            "skewness": float(self._compute_skewness(prnu_estimate)),
            "kurtosis": float(self._compute_kurtosis(prnu_estimate))
        }

        return prnu_estimate, stats

    def _compute_skewness(self, data: np.ndarray) -> float:
        """비대칭도 계산"""
        n = data.size
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.sum(((data - mean) / std) ** 3) / n

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """첨도 계산"""
        n = data.size
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.sum(((data - mean) / std) ** 4) / n - 3

    def _analyze_noise_consistency(
        self,
        image: np.ndarray,
        block_size: int = 64,
        srm_response: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        블록별 노이즈 일관성 분석 (개선됨)

        자연 이미지: 높은 cv (다양한 내용) → 정상
        조작된 이미지: 통계적 이상치 블록 존재 → 의심
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        h, w = gray.shape
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size
        if n_blocks_h == 0 or n_blocks_w == 0:
            return {
                "block_variance_mean": 0.0,
                "block_variance_std": 0.0,
                "coefficient_of_variation": 0.0,
                "outlier_ratio": 0.0,
                "manipulation_score": 0.0,
                "natural_diversity_score": 0.0,
                "num_blocks": 0,
                "consistency_score": 1.0
            }

        # 전체 잔차 (블록별 재계산 대신 사용)
        residual_full = self._extract_noise_residual(gray)

        # 블록별 노이즈 분산 계산
        ela_map = self._compute_ela_map(image)
        ela_block_energies = []
        block_variances = []
        block_energies = []
        block_srm_energies = []
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                residual = residual_full[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ]
                block_variances.append(np.var(residual))
                block_energies.append(np.mean(np.abs(residual)))
                ela_block = ela_map[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ]
                ela_block_energies.append(np.mean(ela_block))
                if srm_response is not None:
                    srm_block = srm_response[
                        i*block_size:(i+1)*block_size,
                        j*block_size:(j+1)*block_size
                    ]
                    block_srm_energies.append(np.mean(srm_block))

        block_variances = np.array(block_variances)
        block_energies = np.array(block_energies)
        ela_block_energies = np.array(ela_block_energies)
        block_srm_energies = np.array(block_srm_energies) if block_srm_energies else None

        # 변동계수 계산
        if np.mean(block_variances) > 0:
            cv = np.std(block_variances) / np.mean(block_variances)
        else:
            cv = 0.0

        if np.mean(block_energies) > 0:
            energy_cv = np.std(block_energies) / np.mean(block_energies)
        else:
            energy_cv = 0.0
        if np.mean(ela_block_energies) > 0:
            ela_energy_cv = np.std(ela_block_energies) / np.mean(ela_block_energies)
        else:
            ela_energy_cv = 0.0

        # 이상치 탐지 (조작된 영역 찾기)
        # IQR 방법: Q3 + 1.5*IQR 이상인 블록 = 이상치
        def outlier_ratio(values: np.ndarray) -> float:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr
            outliers = np.sum((values > upper) | (values < lower))
            return outliers / len(values) if len(values) else 0.0

        outlier_ratio_var = outlier_ratio(block_variances)
        outlier_ratio_energy = outlier_ratio(block_energies)
        outlier_ratio_ela = outlier_ratio(ela_block_energies)
        outlier_ratio_srm = outlier_ratio(block_srm_energies) if block_srm_energies is not None else 0.0
        prnu_stats = self._compute_prnu_block_correlations(residual_full, block_size)
        outlier_ratio_prnu = float(prnu_stats["prnu_corr_low_ratio"])

        outlier_ratio_max = max(
            outlier_ratio_var,
            outlier_ratio_energy,
            outlier_ratio_ela,
            outlier_ratio_srm,
            outlier_ratio_prnu
        )

        # 조작 점수: 이상치 비율 기반
        # 자연 이미지는 15-20% 정도의 이상치 가능 (하늘, 그림자, 디테일 등)
        # 20% 이상이면 의심, 30% 이상이면 확실
        if outlier_ratio_max > 0.30:
            manipulation_score = 1.0
        elif outlier_ratio_max > 0.20:
            manipulation_score = (outlier_ratio_max - 0.20) / 0.10  # 20-30% 구간에서 선형
        else:
            manipulation_score = 0.0

        # 자연 다양성 점수: 높은 cv는 자연 이미지의 증거
        # cv > 0.5: 자연스러운 장면 변화 (하늘, 나무, 건물 등)
        # cv < 0.3: 균일한 장면 또는 AI 생성
        diversity_signal = max(cv, energy_cv)
        natural_diversity_score = min(diversity_signal / 1.0, 1.0) if diversity_signal > 0.5 else 0.0

        return {
            "block_variance_mean": float(np.mean(block_variances)),
            "block_variance_std": float(np.std(block_variances)),
            "coefficient_of_variation": float(cv),
            "energy_cv": float(energy_cv),
            "ela_energy_cv": float(ela_energy_cv),
            "outlier_ratio": float(outlier_ratio_max),
            "outlier_ratio_variance": float(outlier_ratio_var),
            "outlier_ratio_energy": float(outlier_ratio_energy),
            "outlier_ratio_ela": float(outlier_ratio_ela),
            "outlier_ratio_srm": float(outlier_ratio_srm),
            "prnu_corr_mean": float(prnu_stats["prnu_corr_mean"]),
            "prnu_corr_std": float(prnu_stats["prnu_corr_std"]),
            "prnu_corr_low_ratio": float(prnu_stats["prnu_corr_low_ratio"]),
            "manipulation_score": float(manipulation_score),
            "natural_diversity_score": float(natural_diversity_score),
            "num_blocks": len(block_variances),
            # 하위 호환성을 위해 유지 (deprecated)
            "consistency_score": float(1.0 - manipulation_score)
        }

    def _detect_ai_noise_pattern(
        self,
        prnu_stats: Dict[str, float],
        srm_response: np.ndarray
    ) -> Dict[str, Any]:
        """
        AI 생성 이미지의 노이즈 패턴 탐지

        AI 생성 이미지는:
        - 센서 노이즈가 없음 (낮은 PRNU 분산)
        - 규칙적인 노이즈 패턴 (SRM 응답 균일)
        """
        # PRNU 분석
        prnu_variance = prnu_stats["variance"]
        prnu_kurtosis = prnu_stats["kurtosis"]

        # SRM 응답 분석
        srm_mean = np.mean(srm_response)
        srm_std = np.std(srm_response)

        # AI 생성 이미지 점수
        # 1. 낮은 PRNU 분산 = 센서 노이즈 없음
        prnu_score = 1.0 - min(prnu_variance * 1000, 1.0)

        # 2. 높은 SRM 균일성 = AI 특유의 규칙적 패턴
        if srm_mean > 0:
            srm_uniformity = 1.0 - min(srm_std / srm_mean, 1.0)
        else:
            srm_uniformity = 1.0

        # 3. 비정상적인 첨도 (자연 이미지는 ~0, AI는 다름)
        kurtosis_score = min(abs(prnu_kurtosis) / 3.0, 1.0)

        # 종합 점수
        ai_score = (0.4 * prnu_score + 0.3 * srm_uniformity + 0.3 * kurtosis_score)

        return {
            "prnu_score": float(prnu_score),
            "srm_uniformity": float(srm_uniformity),
            "kurtosis_score": float(kurtosis_score),
            "ai_generation_score": float(ai_score)
        }

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        노이즈 분석 실행

        Args:
            image: RGB 이미지 (H, W, 3)

        Returns:
            ToolResult: 노이즈 분석 결과
        """
        start_time = time.time()

        if not self._is_loaded:
            self.load_model()

        if self._backend == "mvss" and self._mvss_model is not None:
            return self._analyze_mvss(image, start_time)

        try:
            # 정규화
            if image.max() > 1.0:
                image_norm = image / 255.0
            else:
                image_norm = image

            # PRNU 추정
            prnu_pattern, prnu_stats = self._estimate_prnu(image)

            # SRM 필터 적용
            srm_response = self._apply_srm_filters(image_norm)

            # 노이즈 일관성 분석 (SRM 에너지 포함)
            consistency = self._analyze_noise_consistency(image, srm_response=srm_response)

            # AI 생성 패턴 탐지
            ai_detection = self._detect_ai_noise_pattern(prnu_stats, srm_response)

            # 종합 점수
            ai_score = ai_detection["ai_generation_score"]
            manipulation_score = consistency["manipulation_score"]
            natural_diversity = consistency["natural_diversity_score"]
            cv = consistency["coefficient_of_variation"]

            # 판정 (개선된 로직)
            if manipulation_score >= self.manipulation_threshold:
                # 명확한 이상치 블록 존재 → 조작 의심
                verdict = Verdict.MANIPULATED
                confidence = manipulation_score
                explanation = (
                    f"통계적 이상치 블록이 탐지되었습니다 ({consistency['outlier_ratio']:.1%}). "
                    f"이미지의 일부 영역이 조작되었을 가능성이 있습니다."
                )
            elif ai_score >= self.ai_threshold and natural_diversity < self.ai_diversity_threshold:
                # AI 노이즈 패턴 + 낮은 다양성 → AI 생성
                verdict = Verdict.AI_GENERATED
                confidence = ai_score
                explanation = (
                    f"AI 생성 이미지의 노이즈 패턴이 탐지되었습니다. "
                    f"센서 노이즈 부재 및 균일한 노이즈 특성이 확인됩니다. "
                    f"AI 점수: {ai_score:.2%}"
                )
            elif natural_diversity >= self.authentic_diversity_threshold and ai_score <= self.authentic_ai_threshold:
                # 높은 다양성 + 낮은 AI 점수 → 자연 이미지
                verdict = Verdict.AUTHENTIC
                confidence = 0.8
                explanation = (
                    f"자연스러운 장면 다양성이 확인되었습니다 (cv={cv:.2f}). "
                    f"실제 카메라로 촬영된 이미지로 판단됩니다. "
                    f"다양한 내용 영역(하늘, 지면, 물체 등)의 노이즈 특성이 정상적으로 변화합니다."
                )
            elif ai_score <= self.authentic_ai_threshold:
                verdict = Verdict.AUTHENTIC
                confidence = 1.0 - ai_score
                explanation = (
                    f"자연스러운 센서 노이즈 패턴이 확인되었습니다. "
                    f"실제 카메라로 촬영된 이미지로 판단됩니다. "
                    f"AI 점수: {ai_score:.2%}"
                )
            else:
                verdict = Verdict.UNCERTAIN
                confidence = 0.5
                explanation = (
                    f"노이즈 분석 결과가 명확하지 않습니다. "
                    f"AI 점수: {ai_score:.2%}, 다양성: {natural_diversity:.2%}"
                )

            processing_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "prnu_stats": prnu_stats,
                    "consistency_analysis": consistency,
                    "ai_detection": ai_detection,
                    "srm_response_mean": float(np.mean(srm_response)),
                    "srm_response_std": float(np.std(srm_response))
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
                explanation=f"노이즈 분석 중 오류 발생: {str(e)}",
                processing_time=processing_time
            )

    def _analyze_mvss(self, image: np.ndarray, start_time: float) -> ToolResult:
        """MVSS-Net 기반 조작 탐지"""
        try:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pred_mask, max_score = self._mvss_inference(image, self._mvss_model)
            mask = pred_mask.astype(np.float32) / 255.0
            manipulation_ratio = float(np.mean(mask >= 0.5))

            score = float(max_score)
            low_threshold = self.mvss_auth_threshold
            high_threshold = self.mvss_threshold
            if self.mvss_uncertain_margin > 0.0:
                low_threshold = max(0.0, high_threshold - self.mvss_uncertain_margin)

            if score >= high_threshold:
                verdict = Verdict.MANIPULATED
                confidence = score
                explanation = (
                    f"MVSS-Net이 조작 가능성을 높게 탐지했습니다. "
                    f"max_score: {score:.2%}, 조작 비율: {manipulation_ratio:.2%}"
                )
            elif score <= low_threshold:
                verdict = Verdict.AUTHENTIC
                confidence = 1.0 - score
                explanation = (
                    f"MVSS-Net이 조작 신호를 낮게 탐지했습니다. "
                    f"max_score: {score:.2%}"
                )
            else:
                verdict = Verdict.UNCERTAIN
                confidence = 0.5
                explanation = (
                    f"MVSS-Net 결과가 불확실 구간에 있습니다. "
                    f"max_score: {score:.2%}"
                )

            processing_time = time.time() - start_time
            return ToolResult(
                tool_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "backend": "mvss",
                    "mvss_score": score,
                    "manipulation_ratio": manipulation_ratio,
                    "mask_mean": float(np.mean(mask)),
                    "mask_max": float(np.max(mask)),
                },
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
                evidence={"error": str(e), "backend": "mvss"},
                explanation=f"MVSS 분석 중 오류 발생: {str(e)}",
                processing_time=processing_time
            )


# LangChain Tool 호환 래퍼
def create_langchain_tool():
    """LangChain용 Tool 생성"""
    try:
        from langchain.tools import Tool as LCTool

        tool_instance = NoiseAnalysisTool()

        def run_analysis(image_path: str) -> str:
            """이미지 파일 경로로 노이즈 분석 실행"""
            image = np.array(Image.open(image_path).convert("RGB"))
            result = tool_instance(image)
            return str(result.to_dict())

        return LCTool(
            name="noise_analyzer",
            description=tool_instance.description,
            func=run_analysis
        )

    except ImportError:
        return None
