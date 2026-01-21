"""
노이즈 분석 도구
SRM(Spatial Rich Model) 필터와 PRNU(Photo Response Non-Uniformity) 분석
"""
import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage, signal
from PIL import Image

from .base_tool import BaseTool, ToolResult, Verdict


class NoiseAnalysisTool(BaseTool):
    """
    노이즈 분석 도구 (SRM + PRNU)

    센서 노이즈 패턴 분석을 통해:
    - 카메라 고유 PRNU 패턴 탐지
    - AI 생성 이미지의 노이즈 특성 분석
    - 이미지 조작 영역의 노이즈 불일치 탐지
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(
            name="noise_analyzer",
            description="SRM 필터 및 PRNU 기반 노이즈 패턴 분석. "
                       "카메라 센서 노이즈와 AI 생성 노이즈의 차이를 탐지합니다.",
            device=device
        )
        self._srm_filters = None
        self._is_loaded = False

    def load_model(self) -> None:
        """SRM 필터 초기화"""
        if self._is_loaded:
            return

        # SRM 필터 정의 (30개 중 대표적인 필터들)
        self._srm_filters = self._initialize_srm_filters()
        self._is_loaded = True
        print("[NoiseTool] SRM 필터 초기화 완료")

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
        block_size: int = 64
    ) -> Dict[str, Any]:
        """
        블록별 노이즈 일관성 분석

        조작된 영역은 주변과 다른 노이즈 특성을 가짐
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        h, w = gray.shape
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size

        # 블록별 노이즈 분산 계산
        block_variances = []
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = gray[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ]
                residual = self._extract_noise_residual(block)
                block_variances.append(np.var(residual))

        block_variances = np.array(block_variances)

        # 일관성 점수 (분산의 변동계수)
        if np.mean(block_variances) > 0:
            cv = np.std(block_variances) / np.mean(block_variances)
        else:
            cv = 0.0

        # 일관성 점수: 낮을수록 일관적 (자연 이미지)
        consistency_score = 1.0 - min(cv, 1.0)

        return {
            "block_variance_mean": float(np.mean(block_variances)),
            "block_variance_std": float(np.std(block_variances)),
            "coefficient_of_variation": float(cv),
            "consistency_score": float(consistency_score),
            "num_blocks": len(block_variances)
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

            # 노이즈 일관성 분석
            consistency = self._analyze_noise_consistency(image)

            # AI 생성 패턴 탐지
            ai_detection = self._detect_ai_noise_pattern(prnu_stats, srm_response)

            # 종합 점수
            ai_score = ai_detection["ai_generation_score"]
            consistency_score = consistency["consistency_score"]

            # 판정
            if ai_score > 0.6 and consistency_score > 0.7:
                verdict = Verdict.AI_GENERATED
                confidence = ai_score
                explanation = (
                    f"AI 생성 이미지의 노이즈 패턴이 탐지되었습니다. "
                    f"센서 노이즈 부재 및 규칙적인 노이즈 특성이 확인됩니다. "
                    f"AI 점수: {ai_score:.2%}"
                )
            elif consistency_score < 0.4:
                verdict = Verdict.MANIPULATED
                confidence = 1.0 - consistency_score
                explanation = (
                    f"노이즈 패턴의 불일치가 탐지되었습니다. "
                    f"이미지의 일부 영역이 조작되었을 가능성이 있습니다. "
                    f"일관성 점수: {consistency_score:.2%}"
                )
            elif ai_score < 0.3:
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
                    f"AI 점수: {ai_score:.2%}, 일관성: {consistency_score:.2%}"
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
