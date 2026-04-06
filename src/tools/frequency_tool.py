"""
주파수 분석 도구
FFT 기반 주파수 스펙트럼 분석으로 AI 생성 이미지의 특징적 패턴 탐지
"""
import time
import json
from pathlib import Path
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
        self._thresholds = self._load_thresholds()

        freq_cfg = self._thresholds.get("frequency", {})
        self.ai_threshold = float(freq_cfg.get("ai_threshold", 0.6))
        self.auth_threshold = float(freq_cfg.get("auth_threshold", 0.3))
        self.jpeg_penalty = float(freq_cfg.get("jpeg_penalty", 0.0))
        self.uncertain_margin = float(freq_cfg.get("uncertain_margin", 0.0))

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
        격자 아티팩트 탐지 (JPEG vs GAN 구분)

        JPEG 압축: 8×8 DCT blocks → 정확히 1/8 간격의 수평/수직 피크
        GAN 아티팩트: 다양한 블록 크기 → 불규칙한 대각선 패턴
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

        # JPEG 8×8 블록 패턴 탐지 (개선된 방법)
        def detect_jpeg_8x8_pattern(profile: np.ndarray, fft_size: int) -> bool:
            """
            8×8 JPEG DCT 블록의 특정 주파수 피크 탐지

            접근: 모든 피크를 세는 대신, 8×8 블록의 **예상 위치**에 피크가 있는지 확인
            """
            if len(profile) < fft_size // 8:
                return False

            # 정규화
            profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)

            # 8×8 블록의 FFT 피크 예상 위치
            # fft_size = 4032 → 8px 주기 → 피크 at indices: fft_size/8, 2*fft_size/8, ...
            center = len(profile) // 2
            block_freq = fft_size / 8.0

            # 예상 피크 위치 (중심에서 양쪽으로)
            expected_positions = []
            for k in range(1, 8):  # 최대 7개 하모닉 확인
                pos_right = int(center + k * block_freq)
                pos_left = int(center - k * block_freq)
                if 0 < pos_right < len(profile):
                    expected_positions.append(pos_right)
                if 0 < pos_left < len(profile):
                    expected_positions.append(pos_left)

            # 예상 위치에서 피크 강도 확인 (±5% 오차 허용)
            tolerance = int(block_freq * 0.05) + 1
            jpeg_peaks_found = 0

            for expected_pos in expected_positions:
                # 예상 위치 근처에서 최대값 찾기
                start = max(0, expected_pos - tolerance)
                end = min(len(profile), expected_pos + tolerance + 1)
                local_region = profile[start:end]

                if len(local_region) > 0:
                    max_val = np.max(local_region)
                    # 임계값 이상이면 JPEG 피크로 간주 (정규화된 값 기준)
                    # 0.15는 배경 노이즈보다 유의미하게 높은 값
                    if max_val > 0.15:
                        jpeg_peaks_found += 1

            # 50% 이상의 예상 위치에서 피크 발견 시 JPEG로 판정
            detection_rate = jpeg_peaks_found / len(expected_positions) if expected_positions else 0
            return detection_rate > 0.5

        # JPEG 패턴 확인
        jpeg_h = detect_jpeg_8x8_pattern(horizontal_profile, w)
        jpeg_v = detect_jpeg_8x8_pattern(vertical_profile, h)
        is_likely_jpeg = jpeg_h or jpeg_v

        # 통계용으로 전체 피크 수도 계산 (기존 방식)
        def count_all_peaks(profile: np.ndarray, threshold: float = 0.15) -> int:
            """모든 국소 최대값 계산 (통계용)"""
            if len(profile) < 3:
                return 0
            profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)
            peaks = 0
            for i in range(1, len(profile) - 1):
                if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                    if profile[i] > threshold:
                        peaks += 1
            return peaks

        h_peaks = count_all_peaks(horizontal_profile)
        v_peaks = count_all_peaks(vertical_profile)
        d_peaks = count_all_peaks(diagonal_profile)

        # GAN 패턴: 대각선 우세, 불규칙한 간격
        diagonal_dominance = d_peaks > max(h_peaks, v_peaks) * 0.5 if max(h_peaks, v_peaks) > 0 else False

        # GAN 점수 계산 (JPEG 아티팩트 제외)
        if is_likely_jpeg:
            # JPEG 압축 아티팩트는 GAN 증거가 아님
            gan_score = 0.0
            is_grid_pattern = False
        else:
            # 피크 수를 해상도로 정규화 (고해상도 보정)
            normalized_peaks = (h_peaks + v_peaks + d_peaks) / max(h, w) * 100
            gan_score = min(normalized_peaks / 3.0, 1.0)

            # 대각선 우세 시 GAN 가능성 증가
            if diagonal_dominance:
                gan_score = min(gan_score * 1.5, 1.0)

            is_grid_pattern = gan_score > 0.5

        return {
            "horizontal_peaks": h_peaks,
            "vertical_peaks": v_peaks,
            "diagonal_peaks": d_peaks,
            "is_likely_jpeg": is_likely_jpeg,
            "diagonal_dominance": diagonal_dominance,
            "regularity_score": gan_score,
            "is_grid_pattern": is_grid_pattern
        }

    def _detect_gan_checkerboard(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """
        GAN Checkerboard 아티팩트 탐지

        Transpose convolution이 만드는 특징적 패턴:
        - 2×2, 4×4 스트라이드 → N/2, N/4 위치에 피크
        - JPEG 8×8과 다른 위치
        """
        h, w = spectrum.shape
        center_y, center_x = h // 2, w // 2

        # GAN upsampling 주파수 (JPEG와 구분)
        # stride=2 → 주파수 w/2, h/2
        # stride=4 → 주파수 w/4, h/4
        gan_frequencies_h = [h // 2, h // 4, h // 3]
        gan_frequencies_w = [w // 2, w // 4, w // 3]

        # 수평 방향 체크
        horizontal_profile = spectrum[center_y, :]
        horizontal_profile = (horizontal_profile - horizontal_profile.min()) / (horizontal_profile.max() - horizontal_profile.min() + 1e-8)

        h_gan_peaks = 0
        for freq in gan_frequencies_w:
            # 중심에서 freq 떨어진 위치 확인 (양방향)
            for offset in [freq, -freq]:
                pos = center_x + offset
                if 0 < pos < len(horizontal_profile) - 1:
                    # 국소 최대값 여부 확인
                    if horizontal_profile[pos] > horizontal_profile[pos-1] and \
                       horizontal_profile[pos] > horizontal_profile[pos+1] and \
                       horizontal_profile[pos] > 0.2:
                        h_gan_peaks += 1

        # 수직 방향 체크
        vertical_profile = spectrum[:, center_x]
        vertical_profile = (vertical_profile - vertical_profile.min()) / (vertical_profile.max() - vertical_profile.min() + 1e-8)

        v_gan_peaks = 0
        for freq in gan_frequencies_h:
            for offset in [freq, -freq]:
                pos = center_y + offset
                if 0 < pos < len(vertical_profile) - 1:
                    if vertical_profile[pos] > vertical_profile[pos-1] and \
                       vertical_profile[pos] > vertical_profile[pos+1] and \
                       vertical_profile[pos] > 0.2:
                        v_gan_peaks += 1

        # 점수 계산 (0-6 피크 정규화)
        checkerboard_score = (h_gan_peaks + v_gan_peaks) / 6.0

        return {
            "h_gan_peaks": h_gan_peaks,
            "v_gan_peaks": v_gan_peaks,
            "checkerboard_score": float(min(checkerboard_score, 1.0))
        }

    def _analyze_power_spectrum_slope(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """
        Power Spectrum Slope 분석 (1/f^α)

        자연 이미지: α ≈ 2.0 (1/f^2)
        GAN 이미지: α ≈ 1.5-1.8 (덜 급격한 감쇠)
        """
        h, w = spectrum.shape
        center_y, center_x = h // 2, w // 2

        # 방사형 평균 계산
        max_radius = min(h, w) // 2
        radii = []
        energies = []

        for r in range(10, max_radius, 5):  # 중심은 너무 강해서 제외
            # 원형 마스크
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
            mask = (distances >= r - 2.5) & (distances < r + 2.5)

            if np.sum(mask) > 0:
                radii.append(r)
                energies.append(np.mean(spectrum[mask]))

        if len(radii) < 10:
            return {
                "slope": 0.0,
                "slope_score": 0.0
            }

        # Log-log 공간에서 선형 회귀
        # 주의: spectrum은 이미 log1p 스케일이므로 energies에 다시 log 적용하지 않음
        log_radii = np.log(radii)
        log_energies = np.array(energies)  # 이미 log 공간

        # 선형 회귀로 기울기 추정
        coeffs = np.polyfit(log_radii, log_energies, 1)
        slope = -coeffs[0]  # 음수로 변환 (감쇠율)

        # 실제 데이터 기반 임계값 (BigGAN 데이터셋)
        # AI 이미지: 0.9-1.1, 자연 이미지: 0.9-1.6 (일부 overlap)
        # slope가 낮을수록 GAN 가능성
        if slope < 1.2:
            slope_score = 1.0  # 강한 AI 신호
        elif slope < 1.5:
            slope_score = (1.5 - slope) / 0.3  # 1.2-1.5 구간 선형
        else:
            slope_score = 0.0  # 자연 이미지

        slope_score = float(np.clip(slope_score, 0, 1))

        return {
            "slope": float(slope),
            "slope_score": slope_score
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

            # GAN 특화 패턴 탐지
            checkerboard_analysis = self._detect_gan_checkerboard(spectrum)
            slope_analysis = self._analyze_power_spectrum_slope(spectrum)

            # 고주파 분석
            hf_analysis = self._analyze_high_frequency(spectrum)

            # 종합 점수 계산 (에러 분석 기반 최적화)
            # - Slope: 가장 효과적 (FP 0.974 vs FN 0.839) - 최고 가중치
            # - Checkerboard: 중간 효과 (FP 0.722 vs FN 0.133)
            # - Grid: 보조 지표 (변별력 낮음)
            # - HF: 최소화 (FP에 과반응 0.989)
            ai_score = (
                0.30 * grid_analysis["regularity_score"] +      # 기존 격자
                0.30 * checkerboard_analysis["checkerboard_score"] +  # GAN checkerboard
                0.35 * slope_analysis["slope_score"] +          # Slope (주요 지표)
                0.05 * hf_analysis["abnormality_score"]         # 고주파 이상 (최소)
            )

            # JPEG 감지 시 점수 감점 (JPEG는 증거가 아닌 컨텍스트)
            is_likely_jpeg = grid_analysis.get("is_likely_jpeg", False)
            adjusted_ai_score = ai_score - self.jpeg_penalty if is_likely_jpeg else ai_score
            adjusted_ai_score = float(np.clip(adjusted_ai_score, 0.0, 1.0))

            # 판정 (임계값 기반, 불확실 구간 최소화 가능)
            low_threshold = self.auth_threshold
            high_threshold = self.ai_threshold
            if self.uncertain_margin > 0.0:
                low_threshold = max(0.0, high_threshold - self.uncertain_margin)

            if adjusted_ai_score >= high_threshold:
                verdict = Verdict.AI_GENERATED
                confidence = adjusted_ai_score
                explanation = (
                    f"주파수 패턴이 AI 생성 특성과 일치합니다. "
                    f"AI 점수: {adjusted_ai_score:.2%}. "
                    f"격자 아티팩트: {'존재' if grid_analysis['is_grid_pattern'] else '없음'}"
                )
            elif adjusted_ai_score <= low_threshold:
                verdict = Verdict.AUTHENTIC
                confidence = 1.0 - adjusted_ai_score
                explanation = (
                    f"주파수 스펙트럼이 자연 이미지 패턴과 일치합니다. "
                    f"AI 점수: {adjusted_ai_score:.2%}"
                )
            else:
                verdict = Verdict.UNCERTAIN
                confidence = 0.5
                explanation = (
                    f"주파수 분석 결과가 명확하지 않습니다. "
                    f"AI 점수: {adjusted_ai_score:.2%}. 추가 분석이 필요합니다."
                )

            if is_likely_jpeg and self.jpeg_penalty > 0.0:
                explanation += " (JPEG 패턴 감지로 점수 감점 적용)"

            processing_time = time.time() - start_time

            return ToolResult(
                tool_name=self.name,
                verdict=verdict,
                confidence=confidence,
                evidence={
                    "ai_generation_score_raw": float(ai_score),
                    "ai_generation_score": float(adjusted_ai_score),
                    "jpeg_penalty": float(self.jpeg_penalty),
                    "grid_analysis": grid_analysis,
                    "gan_checkerboard_analysis": checkerboard_analysis,
                    "power_spectrum_slope_analysis": slope_analysis,
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
