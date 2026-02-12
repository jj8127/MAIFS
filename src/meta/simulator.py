"""
에이전트 시뮬레이터 (Path B)

문헌 기반 confusion matrix + Gaussian copula로
합성 에이전트 출력을 생성한다.

각 에이전트의 성능 특성:
    Frequency (FFT)  - GAN 아티팩트 강점, Diffusion/JPEG 약점
    Noise (PRNU/SRM) - 카메라 진위 강점, 고압축/리사이즈 약점
    FatFormer (CLIP)  - Diffusion 탐지 강점, 부분 조작 약점
    Spatial (ViT)     - 부분 조작 강점, AI 전체 생성 약점

에이전트 간 상관성:
    Frequency ↔ FatFormer: ρ=0.4 (주파수 도메인 공유)
    Noise ↔ 나머지:        ρ=0.1 (독립적 신호)
    Spatial ↔ 나머지:      ρ=0.15 (픽셀 수준)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm, beta as beta_dist


# 판정 카테고리 (순서 고정)
VERDICTS = ["authentic", "manipulated", "ai_generated", "uncertain"]
# 실제 레이블 (ground truth, uncertain 제외)
TRUE_LABELS = ["authentic", "manipulated", "ai_generated"]
# 에이전트 이름 (순서 고정)
AGENT_NAMES = ["frequency", "noise", "fatformer", "spatial"]

VERDICT_TO_IDX = {v: i for i, v in enumerate(VERDICTS)}
TRUE_LABEL_TO_IDX = {v: i for i, v in enumerate(TRUE_LABELS)}


@dataclass
class SimulatedOutput:
    """단일 샘플의 시뮬레이션 출력"""
    true_label: str                      # ground truth
    agent_verdicts: Dict[str, str]       # {agent_name: verdict}
    agent_confidences: Dict[str, float]  # {agent_name: confidence}
    sub_type: Optional[str] = None       # GAN, Diffusion, Splicing 등

    def to_dict(self) -> dict:
        return {
            "true_label": self.true_label,
            "sub_type": self.sub_type,
            "agent_verdicts": dict(self.agent_verdicts),
            "agent_confidences": dict(self.agent_confidences),
        }


@dataclass
class AgentProfile:
    """에이전트 성능 프로파일"""
    name: str
    # confusion matrix: confusion_matrices[true_label] → [P(auth), P(manip), P(ai_gen), P(uncertain)]
    confusion_matrices: Dict[str, np.ndarray]
    # confidence 분포 파라미터: (alpha, beta) for Beta distribution
    confidence_correct: Tuple[float, float] = (8.0, 2.0)    # 정답 시 ~0.80
    confidence_wrong: Tuple[float, float] = (2.0, 5.0)      # 오답 시 ~0.29
    confidence_uncertain: Tuple[float, float] = (3.0, 3.0)  # uncertain 시 ~0.50


def _build_default_profiles() -> Dict[str, AgentProfile]:
    """문헌 기반 기본 에이전트 프로파일 생성"""

    # --- Frequency Agent (FFT) ---
    # 강점: GAN upsampling 아티팩트 (Durall et al. 2020)
    # 약점: Diffusion 생성, JPEG 압축과 혼동
    freq_cm = {
        "authentic": np.array([0.80, 0.05, 0.10, 0.05]),
        "manipulated": np.array([0.10, 0.55, 0.15, 0.20]),
        "ai_generated": np.array([0.08, 0.07, 0.75, 0.10]),
    }

    # --- Noise Agent (PRNU/SRM) ---
    # 강점: 카메라 센서 노이즈 진위 (Chen et al. 2008, Fridrich & Kodovsky 2012)
    # 약점: 고압축, 리사이즈에 취약
    noise_cm = {
        "authentic": np.array([0.88, 0.02, 0.05, 0.05]),
        "manipulated": np.array([0.15, 0.60, 0.10, 0.15]),
        "ai_generated": np.array([0.10, 0.10, 0.65, 0.15]),
    }

    # --- FatFormer Agent (CLIP ViT-L/14 + DWT) ---
    # 강점: Diffusion 탐지, 교차 일반화 (Liu et al. CVPR 2024)
    # 약점: 부분 조작 미탐지
    fat_cm = {
        "authentic": np.array([0.85, 0.02, 0.08, 0.05]),
        "manipulated": np.array([0.20, 0.40, 0.20, 0.20]),
        "ai_generated": np.array([0.03, 0.02, 0.90, 0.05]),
    }

    # --- Spatial Agent (ViT) ---
    # 강점: 부분 조작 영역 탐지 (Guillaro et al. 2023)
    # 약점: AI 전체 생성 약함
    spatial_cm = {
        "authentic": np.array([0.82, 0.08, 0.05, 0.05]),
        "manipulated": np.array([0.05, 0.80, 0.05, 0.10]),
        "ai_generated": np.array([0.15, 0.15, 0.55, 0.15]),
    }

    return {
        "frequency": AgentProfile(
            name="frequency",
            confusion_matrices=freq_cm,
            confidence_correct=(8.0, 2.0),
            confidence_wrong=(2.0, 5.0),
            confidence_uncertain=(3.0, 3.0),
        ),
        "noise": AgentProfile(
            name="noise",
            confusion_matrices=noise_cm,
            confidence_correct=(9.0, 2.0),     # 노이즈 에이전트는 정답 시 더 확신
            confidence_wrong=(2.0, 4.0),
            confidence_uncertain=(2.5, 3.5),
        ),
        "fatformer": AgentProfile(
            name="fatformer",
            confusion_matrices=fat_cm,
            confidence_correct=(10.0, 2.0),    # FatFormer는 정답 시 매우 확신
            confidence_wrong=(2.5, 5.0),
            confidence_uncertain=(3.0, 3.0),
        ),
        "spatial": AgentProfile(
            name="spatial",
            confusion_matrices=spatial_cm,
            confidence_correct=(7.0, 2.0),
            confidence_wrong=(2.0, 4.5),
            confidence_uncertain=(3.0, 3.5),
        ),
    }


# 에이전트 간 상관 행렬 (Gaussian copula)
# 순서: frequency, noise, fatformer, spatial
DEFAULT_CORRELATION = np.array([
    [1.00, 0.10, 0.40, 0.15],   # frequency
    [0.10, 1.00, 0.10, 0.15],   # noise
    [0.40, 0.10, 1.00, 0.12],   # fatformer (주파수 도메인 공유)
    [0.15, 0.15, 0.12, 1.00],   # spatial
])


class AgentSimulator:
    """
    Gaussian copula 기반 에이전트 시뮬레이터

    Usage:
        sim = AgentSimulator(seed=42)
        dataset = sim.generate(n_samples=10000)
        # dataset: List[SimulatedOutput]
    """

    def __init__(
        self,
        profiles: Optional[Dict[str, AgentProfile]] = None,
        correlation: Optional[np.ndarray] = None,
        label_distribution: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        """
        Args:
            profiles: 에이전트별 성능 프로파일 (None이면 기본값)
            correlation: 에이전트 간 상관 행렬 (4×4)
            label_distribution: 레이블 비율 (None이면 균등)
            seed: 난수 시드
        """
        self.profiles = profiles or _build_default_profiles()
        self.correlation = correlation if correlation is not None else DEFAULT_CORRELATION.copy()
        self.label_dist = label_distribution or {
            "authentic": 1 / 3,
            "manipulated": 1 / 3,
            "ai_generated": 1 / 3,
        }
        self.rng = np.random.default_rng(seed)

        # Cholesky 분해 (copula 샘플링용)
        self._chol = np.linalg.cholesky(self.correlation)

    def generate(
        self,
        n_samples: int = 10000,
        sub_type_distribution: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[SimulatedOutput]:
        """
        합성 에이전트 출력 생성

        Args:
            n_samples: 생성할 샘플 수
            sub_type_distribution: 레이블별 서브타입 분포
                e.g. {"ai_generated": {"gan": 0.5, "diffusion": 0.5}}

        Returns:
            List[SimulatedOutput]: 시뮬레이션 결과
        """
        if sub_type_distribution is None:
            sub_type_distribution = {
                "ai_generated": {"gan": 0.5, "diffusion": 0.5},
                "manipulated": {"splicing": 0.4, "copy_move": 0.3, "inpainting": 0.3},
            }

        # 레이블 샘플링
        labels = list(self.label_dist.keys())
        probs = np.array([self.label_dist[l] for l in labels])
        probs /= probs.sum()
        true_labels = self.rng.choice(labels, size=n_samples, p=probs)

        # Gaussian copula: 상관 정규 변수 생성
        z = self.rng.standard_normal((n_samples, len(AGENT_NAMES)))
        z_corr = z @ self._chol.T  # 상관 적용
        u = norm.cdf(z_corr)       # 균일 분포로 변환

        outputs = []
        for i in range(n_samples):
            true_label = true_labels[i]

            # 서브타입 결정
            sub_type = None
            if true_label in sub_type_distribution:
                st_dist = sub_type_distribution[true_label]
                st_keys = list(st_dist.keys())
                st_probs = np.array([st_dist[k] for k in st_keys])
                st_probs /= st_probs.sum()
                sub_type = self.rng.choice(st_keys, p=st_probs)

            verdicts = {}
            confidences = {}

            for j, agent_name in enumerate(AGENT_NAMES):
                profile = self.profiles[agent_name]

                # 서브타입에 따른 confusion matrix 조정
                cm_row = self._get_cm_row(profile, true_label, sub_type)

                # Copula 기반 verdict 샘플링
                verdict = self._sample_verdict(cm_row, u[i, j])
                verdicts[agent_name] = verdict

                # Confidence 샘플링
                confidence = self._sample_confidence(
                    profile, true_label, verdict, u[i, j]
                )
                confidences[agent_name] = confidence

            outputs.append(SimulatedOutput(
                true_label=true_label,
                agent_verdicts=verdicts,
                agent_confidences=confidences,
                sub_type=sub_type,
            ))

        return outputs

    def _get_cm_row(
        self,
        profile: AgentProfile,
        true_label: str,
        sub_type: Optional[str],
    ) -> np.ndarray:
        """서브타입을 고려한 confusion matrix 행 반환"""
        base_row = profile.confusion_matrices[true_label].copy()

        if sub_type is None:
            return base_row

        # 서브타입에 따른 성능 조정
        if profile.name == "frequency":
            if sub_type == "gan":
                # GAN: 주파수 에이전트 강화 (AI_GEN 확률 상승)
                base_row[VERDICT_TO_IDX["ai_generated"]] *= 1.15
            elif sub_type == "diffusion":
                # Diffusion: 주파수 에이전트 약화
                base_row[VERDICT_TO_IDX["ai_generated"]] *= 0.70
                base_row[VERDICT_TO_IDX["uncertain"]] *= 1.40

        elif profile.name == "fatformer":
            if sub_type == "diffusion":
                # FatFormer는 Diffusion에 특히 강함
                base_row[VERDICT_TO_IDX["ai_generated"]] *= 1.10
            elif sub_type == "inpainting":
                # Inpainting: FatFormer 약화
                base_row[VERDICT_TO_IDX["manipulated"]] *= 1.20
                base_row[VERDICT_TO_IDX["ai_generated"]] *= 0.80

        elif profile.name == "noise":
            if sub_type in ("splicing", "copy_move"):
                # 노이즈 불일치가 더 뚜렷
                base_row[VERDICT_TO_IDX["manipulated"]] *= 1.15
            elif sub_type == "diffusion":
                # AI 생성: 노이즈 패턴 부재로 탐지 가능하지만 약간 약함
                base_row[VERDICT_TO_IDX["ai_generated"]] *= 0.90

        elif profile.name == "spatial":
            if sub_type == "splicing":
                # 공간 에이전트는 스플라이싱에 매우 강함
                base_row[VERDICT_TO_IDX["manipulated"]] *= 1.20
            elif sub_type in ("gan", "diffusion"):
                # AI 전체 생성: 공간 에이전트 약함
                base_row[VERDICT_TO_IDX["ai_generated"]] *= 0.85

        # 정규화
        base_row = np.maximum(base_row, 0.01)
        base_row /= base_row.sum()
        return base_row

    def _sample_verdict(self, cm_row: np.ndarray, u: float) -> str:
        """CDF 역변환으로 verdict 샘플링"""
        cdf = np.cumsum(cm_row)
        idx = np.searchsorted(cdf, u)
        idx = min(idx, len(VERDICTS) - 1)
        return VERDICTS[idx]

    def _sample_confidence(
        self,
        profile: AgentProfile,
        true_label: str,
        predicted_verdict: str,
        u: float,
    ) -> float:
        """판정 정확성에 따른 confidence 샘플링"""
        if predicted_verdict == "uncertain":
            a, b = profile.confidence_uncertain
        elif predicted_verdict == true_label:
            a, b = profile.confidence_correct
        else:
            a, b = profile.confidence_wrong

        # Beta 분포에서 quantile 함수로 샘플링 (copula와 부분 상관)
        # u를 약간 교란하여 confidence와 verdict 간 완전 종속 방지
        u_conf = (u + self.rng.uniform(0, 0.3)) % 1.0
        confidence = float(beta_dist.ppf(u_conf, a, b))
        return np.clip(confidence, 0.05, 0.99)

    def generate_split(
        self,
        n_train: int = 8000,
        n_val: int = 1000,
        n_test: int = 1000,
        sub_type_split: bool = True,
    ) -> Tuple[List[SimulatedOutput], List[SimulatedOutput], List[SimulatedOutput]]:
        """
        학습/검증/테스트 분할 생성

        Args:
            n_train: 학습 샘플 수
            n_val: 검증 샘플 수
            n_test: 테스트 샘플 수
            sub_type_split: True면 train과 test의 서브타입을 다르게 설정
                (생성기 교차 검증: train=SD1.5+BigGAN, test=SDXL+StyleGAN3)

        Returns:
            (train, val, test) 튜플
        """
        if sub_type_split:
            # 생성기 교차 split
            train_sub = {
                "ai_generated": {"gan": 0.5, "diffusion": 0.5},
                "manipulated": {"splicing": 0.4, "copy_move": 0.3, "inpainting": 0.3},
            }
            test_sub = {
                "ai_generated": {"gan": 0.4, "diffusion": 0.6},  # 분포 시프트
                "manipulated": {"splicing": 0.3, "copy_move": 0.4, "inpainting": 0.3},
            }
        else:
            train_sub = None
            test_sub = None

        train = self.generate(n_train, sub_type_distribution=train_sub)
        val = self.generate(n_val, sub_type_distribution=train_sub)
        test = self.generate(n_test, sub_type_distribution=test_sub)

        return train, val, test

    def get_profile_summary(self) -> Dict[str, Dict[str, float]]:
        """각 에이전트의 예상 정확도 요약"""
        summary = {}
        for name, profile in self.profiles.items():
            accs = {}
            for label in TRUE_LABELS:
                cm_row = profile.confusion_matrices[label]
                # 정답 확률
                if label in VERDICT_TO_IDX:
                    accs[label] = float(cm_row[VERDICT_TO_IDX[label]])
            # 전체 평균 정확도 (균등 레이블 가정)
            accs["overall"] = float(np.mean(list(accs.values())))
            summary[name] = accs
        return summary
