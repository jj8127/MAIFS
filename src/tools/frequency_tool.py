"""
주파수 분석 도구
FFT 기반 주파수 스펙트럼 분석으로 AI 생성 이미지의 특징적 패턴 탐지
"""
import time
from typing import Tuple, Dict, Any
import numpy as np
from scipy import ndimage
from PIL import Image

from .base_tool import BaseTool, ToolResult, Verdict


class FrequencyAnalysisTool(BaseTool):
    """
    주파수 분석 도구 (FFT Analysis)

    고속 푸리에 변환을 사용하여:
    - GAN/Diffusion 생성 이미지의 격자 패턴(grid artifact) 탐지
    - 고주파 스펙트럼 이상 징후 분석
    - Radial Energy Distribution 분석
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(
            name="frequency_analyzer",
            description="FFT 기반 주파수 스펙트럼 분석. "
                       "AI 생성 이미지의 특징적인 주파수 패턴(격자 아티팩트)을 탐지합니다.",
            device=device
        )
        self._is_loaded = True  # 외부 모델 불필요

    def load_model(self) -> None:
        """모델 로드 (FFT는 외부 모델 불필요)"""
        self._is_loaded = True

    def _compute_fft_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        FFT 스펙트럼 계산

        Args:
            image: 그레이스케일 이미지 (H, W)

        Returns:
            로그 스케일 FFT 크기 스펙트럼
        """
        # 2D FFT
        f_transform = np.fft.fft2(image)

        # 중심으로 이동
        f_shift = np.fft.fftshift(f_transform)

        # 크기 스펙트럼 (로그 스케일)
        magnitude = np.abs(f_shift)
        magnitude = np.log1p(magnitude)

        return magnitude

    def _compute_radial_energy(
        self,
        spectrum: np.ndarray,
        num_bins: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        방사형 에너지 분포 계산

        Args:
            spectrum: FFT 크기 스펙트럼
            num_bins: 방사형 빈 개수

        Returns:
            (반경 배열, 에너지 배열)
        """
        h, w = spectrum.shape
        center_y, center_x = h // 2, w // 2

        # 각 픽셀의 중심으로부터 거리 계산
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

        # 최대 반경
        max_radius = min(center_x, center_y)

        # 빈별 에너지 계산
        radii = np.linspace(0, max_radius, num_bins + 1)
        energy = np.zeros(num_bins)

        for i in range(num_bins):
            mask = (distances >= radii[i]) & (distances < radii[i + 1])
            if np.sum(mask) > 0:
                energy[i] = np.mean(spectrum[mask])

        return radii[:-1], energy

    def _detect_grid_artifacts(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """
        격자 아티팩트 탐지

        GAN 이미지는 특정 주파수에서 규칙적인 피크를 보임
        """
        h, w = spectrum.shape
        center_y, center_x = h // 2, w // 2

        # 수평/수직 방향 프로파일
        horizontal_profile = spectrum[center_y, :]
        vertical_profile = spectrum[:, center_x]

        # 대각선 방향 프로파일
        diag_size = min(h, w)
        diagonal_profile = np.array([
            spectrum[center_y - i, center_x + i]
            for i in range(-diag_size // 2 + 1, diag_size // 2)
            if 0 <= center_y - i < h and 0 <= center_x + i < w
        ])

        # 피크 탐지 (간단한 방법)
        def count_peaks(profile: np.ndarray, threshold: float = 0.1) -> int:
            """국소 최대값 개수 계산"""
            if len(profile) < 3:
                return 0
            # 정규화
            profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)
            peaks = 0
            for i in range(1, len(profile) - 1):
                if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                    if profile[i] > threshold:
                        peaks += 1
            return peaks

        h_peaks = count_peaks(horizontal_profile)
        v_peaks = count_peaks(vertical_profile)
        d_peaks = count_peaks(diagonal_profile)

        # 비정상적인 피크 패턴 점수 (GAN은 규칙적인 피크)
        regularity_score = (h_peaks + v_peaks + d_peaks) / 30.0  # 정규화

        return {
            "horizontal_peaks": h_peaks,
            "vertical_peaks": v_peaks,
            "diagonal_peaks": d_peaks,
            "regularity_score": min(regularity_score, 1.0),
            "is_grid_pattern": regularity_score > 0.5
        }

    def _analyze_high_frequency(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """
        고주파 영역 분석

        자연 이미지는 1/f 법칙을 따르지만 AI 생성 이미지는 다른 패턴을 보임
        """
        h, w = spectrum.shape
        center_y, center_x = h // 2, w // 2

        # 저주파/고주파 영역 정의
        low_freq_radius = min(h, w) // 8
        high_freq_radius = min(h, w) // 3

        # 마스크 생성
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

        low_freq_mask = distances <= low_freq_radius
        high_freq_mask = distances >= high_freq_radius

        # 에너지 계산
        low_freq_energy = np.mean(spectrum[low_freq_mask])
        high_freq_energy = np.mean(spectrum[high_freq_mask])

        # 고주파/저주파 비율
        hf_lf_ratio = high_freq_energy / (low_freq_energy + 1e-8)

        # 자연 이미지의 전형적인 비율과 비교
        # 자연 이미지: 0.1 ~ 0.3, AI 생성: 0.3 ~ 0.6
        abnormality_score = (hf_lf_ratio - 0.2) / 0.3
        abnormality_score = np.clip(abnormality_score, 0, 1)

        return {
            "low_freq_energy": float(low_freq_energy),
            "high_freq_energy": float(high_freq_energy),
            "hf_lf_ratio": float(hf_lf_ratio),
            "abnormality_score": float(abnormality_score)
        }

    def analyze(self, image: np.ndarray) -> ToolResult:
        """
        주파수 분석 실행

        Args:
            image: RGB 이미지 (H, W, 3)

        Returns:
            ToolResult: 주파수 분석 결과
        """
        start_time = time.time()

        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image

            # 정규화
            if gray.max() > 1.0:
                gray = gray / 255.0

            # FFT 스펙트럼 계산
            spectrum = self._compute_fft_spectrum(gray)

            # 방사형 에너지 분포
            radii, radial_energy = self._compute_radial_energy(spectrum)

            # 격자 아티팩트 탐지
            grid_analysis = self._detect_grid_artifacts(spectrum)

            # 고주파 분석
            hf_analysis = self._analyze_high_frequency(spectrum)

            # 종합 점수 계산
            # 격자 패턴 점수와 고주파 이상 점수의 가중 평균
            ai_score = (
                0.6 * grid_analysis["regularity_score"] +
                0.4 * hf_analysis["abnormality_score"]
            )

            # 판정
            if ai_score < 0.3:
                verdict = Verdict.AUTHENTIC
                confidence = 1.0 - ai_score
                explanation = (
                    f"주파수 스펙트럼이 자연 이미지 패턴과 일치합니다. "
                    f"AI 생성 점수: {ai_score:.2%}"
                )
            elif ai_score > 0.6:
                verdict = Verdict.AI_GENERATED
                confidence = ai_score
                explanation = (
                    f"GAN/Diffusion 모델 특유의 주파수 패턴이 탐지되었습니다. "
                    f"AI 생성 점수: {ai_score:.2%}. "
                    f"격자 아티팩트: {'존재' if grid_analysis['is_grid_pattern'] else '없음'}"
                )
            else:
                verdict = Verdict.UNCERTAIN
                confidence = 0.5
                explanation = (
                    f"주파수 분석 결과가 명확하지 않습니다. "
                    f"AI 생성 점수: {ai_score:.2%}. 추가 분석이 필요합니다."
                )

            processing_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "ai_generation_score": float(ai_score),
                    "grid_analysis": grid_analysis,
                    "high_frequency_analysis": hf_analysis,
                    "radial_energy_sample": radial_energy[:10].tolist()
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
                explanation=f"주파수 분석 중 오류 발생: {str(e)}",
                processing_time=processing_time
            )


# LangChain Tool 호환 래퍼
def create_langchain_tool():
    """LangChain용 Tool 생성"""
    try:
        from langchain.tools import Tool as LCTool

        tool_instance = FrequencyAnalysisTool()

        def run_analysis(image_path: str) -> str:
            """이미지 파일 경로로 주파수 분석 실행"""
            image = np.array(Image.open(image_path).convert("RGB"))
            result = tool_instance(image)
            return str(result.to_dict())

        return LCTool(
            name="frequency_analyzer",
            description=tool_instance.description,
            func=run_analysis
        )

    except ImportError:
        return None
