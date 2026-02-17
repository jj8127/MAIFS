"""
Path A(실데이터)용 에이전트 출력 수집기.

역할:
    1) 실제 이미지를 4개 도구(Frequency/CAT-Net, Noise, FatFormer, Spatial)에 통과
    2) SimulatedOutput 포맷과 동일한 구조로 변환
    3) Path A 메타 학습/평가 파이프라인에서 재사용 가능하게 분할/저장 제공
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from ..tools.catnet_tool import CATNetAnalysisTool
from ..tools.fatformer_tool import FatFormerTool
from ..tools.noise_tool import NoiseAnalysisTool
from ..tools.spatial_tool import SpatialAnalysisTool
from ..tools.base_tool import ToolResult

from .simulator import (
    AGENT_NAMES,
    VERDICTS,
    VERDICT_TO_IDX,
    AgentProfile,
    AgentSimulator,
    SimulatedOutput,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class CollectedRecord:
    """원본 경로/정답/도구 출력을 함께 저장하기 위한 직렬화 레코드."""

    image_path: str
    true_label: str
    sub_type: str
    agent_verdicts: Dict[str, str]
    agent_confidences: Dict[str, float]
    evidence_digest: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "true_label": self.true_label,
            "sub_type": self.sub_type,
            "agent_verdicts": dict(self.agent_verdicts),
            "agent_confidences": dict(self.agent_confidences),
            "evidence_digest": dict(self.evidence_digest),
        }


def _iter_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


# evidence 표준 스키마: 모든 도구에서 공통으로 기록할 키.
# 누락값은 None으로 채워져 JSON 직렬화가 보장된다.
# 2채널 feature (value + is_present) 구성 시 None 여부로 is_present를 결정한다.
EVIDENCE_SCHEMA: Dict[str, None] = {
    # 공통
    "backend": None,
    "catnet_available": None,
    "catnet_error": None,
    # CATNet / Compression
    "compression_artifact_score": None,
    "manipulation_ratio": None,
    "mean_intensity": None,
    "max_intensity": None,
    # Noise / MVSS
    "mvss_score": None,
    # FatFormer — 2채널 feature에 핵심
    "fake_probability": None,
    "real_probability": None,
    # Fallback 진단
    "fallback_mode": None,
    "fallback_raw_verdict": None,
    "fallback_raw_confidence": None,
    # 일반 에러
    "error": None,
}

# 숫자 값으로 변환하는 키 (evidence_to_feature_vector에서 사용)
EVIDENCE_NUMERIC_KEYS: List[str] = [
    "manipulation_ratio",
    "mean_intensity",
    "max_intensity",
    "mvss_score",
    "compression_artifact_score",
    "fake_probability",
    "real_probability",
]


def evidence_to_feature_vector(digest: Dict[str, Any]) -> List[float]:
    """
    evidence digest를 (value, is_present) 2채널 float 벡터로 변환.

    - value: 실제 측정값 (None이면 0.0)
    - is_present: 1.0 = 측정됨, 0.0 = 측정 안 됨 (None 또는 비숫자)

    Returns:
        길이 = len(EVIDENCE_NUMERIC_KEYS) * 2 의 float 리스트
    """
    vec: List[float] = []
    for key in EVIDENCE_NUMERIC_KEYS:
        val = digest.get(key)
        if val is not None and isinstance(val, (int, float)):
            vec.append(float(val))
            vec.append(1.0)
        else:
            vec.append(0.0)
            vec.append(0.0)
    return vec


def _safe_evidence_digest(result: ToolResult) -> Dict[str, Any]:
    """
    큰 텐서/마스크를 피하고 EVIDENCE_SCHEMA 기준 표준 키만 저장.
    누락 키는 None으로 초기화되어 스키마가 항상 완전하게 유지된다.
    """
    out: Dict[str, Any] = dict(EVIDENCE_SCHEMA)  # None으로 초기화
    for k, v in (result.evidence or {}).items():
        if k not in out:
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


class AgentOutputCollector:
    """
    Path A용 실제 에이전트 출력 수집기.

    class_specs 예시:
        {
            "authentic": {"path": "datasets/CASIA2_subset/Au", "sub_type": "casia_au", "max_samples": 60},
            "manipulated": {"path": "datasets/CASIA2_subset/Tp", "sub_type": "casia_tp", "max_samples": 60},
            "ai_generated": {"path": "datasets/GenImage_subset/BigGAN/val/ai", "sub_type": "biggan", "max_samples": 60},
        }
    """

    def __init__(
        self,
        device: str = "cuda",
        noise_backend: str = "mvss",
        spatial_backend: str = "mesorch",
        seed: int = 42,
    ):
        self.device = device
        self.noise_backend = noise_backend
        self.spatial_backend = spatial_backend
        self.rng = np.random.default_rng(seed)

        self._freq_tool: Optional[CATNetAnalysisTool] = None
        self._noise_tool: Optional[NoiseAnalysisTool] = None
        self._fat_tool: Optional[FatFormerTool] = None
        self._spatial_tool: Optional[SpatialAnalysisTool] = None

    def _ensure_tools(self) -> None:
        if self._freq_tool is None:
            self._freq_tool = CATNetAnalysisTool(device=self.device)
        if self._noise_tool is None:
            self._noise_tool = NoiseAnalysisTool(device=self.device, backend=self.noise_backend)
        if self._fat_tool is None:
            self._fat_tool = FatFormerTool(device=self.device)
        if self._spatial_tool is None:
            self._spatial_tool = SpatialAnalysisTool(device=self.device, backend=self.spatial_backend)

    def _infer_agents(self, image: np.ndarray) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, Any]]]:
        self._ensure_tools()
        if self._freq_tool is None:
            raise RuntimeError("CATNetAnalysisTool 초기화 실패 — _ensure_tools() 확인 필요")
        if self._noise_tool is None:
            raise RuntimeError("NoiseAnalysisTool 초기화 실패 — _ensure_tools() 확인 필요")
        if self._fat_tool is None:
            raise RuntimeError("FatFormerTool 초기화 실패 — _ensure_tools() 확인 필요")
        if self._spatial_tool is None:
            raise RuntimeError("SpatialAnalysisTool 초기화 실패 — _ensure_tools() 확인 필요")

        r_freq = self._freq_tool(image)
        r_noise = self._noise_tool(image)
        r_fat = self._fat_tool(image)
        r_spatial = self._spatial_tool(image)

        verdicts = {
            "frequency": r_freq.verdict.value,
            "noise": r_noise.verdict.value,
            "fatformer": r_fat.verdict.value,
            "spatial": r_spatial.verdict.value,
        }
        confidences = {
            "frequency": float(r_freq.confidence),
            "noise": float(r_noise.confidence),
            "fatformer": float(r_fat.confidence),
            "spatial": float(r_spatial.confidence),
        }
        digest = {
            "frequency": _safe_evidence_digest(r_freq),
            "noise": _safe_evidence_digest(r_noise),
            "fatformer": _safe_evidence_digest(r_fat),
            "spatial": _safe_evidence_digest(r_spatial),
        }
        return verdicts, confidences, digest

    def _sample_paths(self, image_paths: Sequence[Path], max_samples: int) -> List[Path]:
        if max_samples <= 0 or len(image_paths) <= max_samples:
            return list(image_paths)
        idx = self.rng.choice(len(image_paths), size=max_samples, replace=False)
        return [image_paths[i] for i in sorted(idx)]

    def collect(
        self,
        class_specs: Dict[str, Dict[str, Any]],
        verbose_every: int = 25,
    ) -> Tuple[List[SimulatedOutput], List[CollectedRecord], List[Dict[str, str]]]:
        samples: List[SimulatedOutput] = []
        records: List[CollectedRecord] = []
        errors: List[Dict[str, str]] = []

        for true_label, spec in class_specs.items():
            data_path = Path(str(spec.get("path", "")))
            if not data_path.exists():
                errors.append(
                    {
                        "label": true_label,
                        "path": str(data_path),
                        "error": "dataset path does not exist",
                    }
                )
                continue

            sub_type = str(spec.get("sub_type", true_label))
            max_samples = int(spec.get("max_samples", 0))
            image_paths = _iter_images(data_path)
            image_paths = self._sample_paths(image_paths, max_samples=max_samples)

            print(f"  [{true_label}] selected {len(image_paths)} samples from {data_path}")
            for i, image_path in enumerate(image_paths, 1):
                try:
                    image = _load_rgb(image_path)
                    verdicts, confidences, digest = self._infer_agents(image)
                    sample = SimulatedOutput(
                        true_label=true_label,
                        sub_type=sub_type,
                        agent_verdicts=verdicts,
                        agent_confidences=confidences,
                    )
                    samples.append(sample)
                    records.append(
                        CollectedRecord(
                            image_path=str(image_path),
                            true_label=true_label,
                            sub_type=sub_type,
                            agent_verdicts=verdicts,
                            agent_confidences=confidences,
                            evidence_digest=digest,
                        )
                    )
                except Exception as e:  # noqa: PERF203
                    errors.append(
                        {
                            "label": true_label,
                            "path": str(image_path),
                            "error": str(e),
                        }
                    )

                if verbose_every > 0 and i % verbose_every == 0:
                    print(f"    progress: {i}/{len(image_paths)}")

        return samples, records, errors

    @staticmethod
    def save_jsonl(records: Sequence[CollectedRecord], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(in_path: Path) -> Tuple[List[SimulatedOutput], List[CollectedRecord]]:
        """
        save_jsonl로 저장한 파일을 다시 로드해 샘플/레코드로 복원한다.

        주의:
            - verdict 값이 비정상인 경우 'uncertain'으로 보정한다.
            - confidence는 [0,1] 구간으로 클리핑한다.
        """
        samples: List[SimulatedOutput] = []
        records: List[CollectedRecord] = []
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)

                verdict_src = d.get("agent_verdicts", {})
                conf_src = d.get("agent_confidences", {})
                digest_src = d.get("evidence_digest", {})

                verdicts: Dict[str, str] = {}
                confidences: Dict[str, float] = {}
                digest: Dict[str, Dict[str, Any]] = {}
                for agent in AGENT_NAMES:
                    v = str(verdict_src.get(agent, "uncertain"))
                    if v not in VERDICTS:
                        v = "uncertain"
                    verdicts[agent] = v

                    c = float(conf_src.get(agent, 0.0))
                    c = min(1.0, max(0.0, c))
                    confidences[agent] = c

                    e = digest_src.get(agent, {}) if isinstance(digest_src, dict) else {}
                    digest[agent] = e if isinstance(e, dict) else {}

                true_label = str(d.get("true_label", "uncertain"))
                sub_type = str(d.get("sub_type", true_label))
                image_path = str(d.get("image_path", ""))

                sample = SimulatedOutput(
                    true_label=true_label,
                    sub_type=sub_type,
                    agent_verdicts=verdicts,
                    agent_confidences=confidences,
                )
                record = CollectedRecord(
                    image_path=image_path,
                    true_label=true_label,
                    sub_type=sub_type,
                    agent_verdicts=verdicts,
                    agent_confidences=confidences,
                    evidence_digest=digest,
                )
                samples.append(sample)
                records.append(record)

        return samples, records

    @staticmethod
    def stratified_split(
        samples: Sequence[SimulatedOutput],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
        return_indices: bool = False,
    ) -> "Union[Tuple[List[SimulatedOutput], List[SimulatedOutput], List[SimulatedOutput]], Tuple[List[int], List[int], List[int]]]":
        """
        라벨별 층화 분할.

        Args:
            return_indices: True면 원본 samples의 인덱스 리스트 반환.
                            False(기본값)면 기존 동작(SimulatedOutput 리스트 반환).
                            evidence_digest와 정합이 필요할 때 True로 사용.
        """
        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
        if np.any(ratios < 0):
            raise ValueError("split ratios must be >= 0")
        if float(ratios.sum()) <= 0:
            raise ValueError("sum of split ratios must be > 0")
        ratios /= ratios.sum()

        # 원본 인덱스를 함께 유지
        by_label: Dict[str, List[Tuple[int, SimulatedOutput]]] = {}
        for i, s in enumerate(samples):
            by_label.setdefault(s.true_label, []).append((i, s))

        rng = np.random.default_rng(seed)
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []
        train_s: List[SimulatedOutput] = []
        val_s: List[SimulatedOutput] = []
        test_s: List[SimulatedOutput] = []

        for label, group in by_label.items():
            arr = list(group)
            rng.shuffle(arr)
            n = len(arr)
            if n == 0:
                continue

            n_train = int(round(n * ratios[0]))
            n_val = int(round(n * ratios[1]))
            if n >= 3:
                n_train = max(1, min(n - 2, n_train))
                n_val = max(1, min(n - n_train - 1, n_val))
                n_test = n - n_train - n_val
            else:
                n_train = min(1, n)
                n_val = min(1, max(0, n - n_train))
                n_test = max(0, n - n_train - n_val)

            for idx, s in arr[:n_train]:
                train_idx.append(idx); train_s.append(s)
            for idx, s in arr[n_train:n_train + n_val]:
                val_idx.append(idx); val_s.append(s)
            for idx, s in arr[n_train + n_val:n_train + n_val + n_test]:
                test_idx.append(idx); test_s.append(s)
            print(f"  split[{label}] train={n_train}, val={n_val}, test={n_test}")

        if return_indices:
            return train_idx, val_idx, test_idx
        return train_s, val_s, test_s

    @staticmethod
    def stratified_kfold_split(
        samples: Sequence[SimulatedOutput],
        k_folds: int = 5,
        test_fold: int = 0,
        val_fold: int = 1,
        seed: int = 42,
        return_indices: bool = False,
    ) -> "Union[Tuple[List[SimulatedOutput], List[SimulatedOutput], List[SimulatedOutput]], Tuple[List[int], List[int], List[int]]]":
        """
        라벨별로 동일한 셔플 순서를 만든 뒤 k-fold로 분할하여 train/val/test를 구성한다.

        Args:
            test_fold: 테스트로 사용할 fold index
            val_fold: 검증으로 사용할 fold index (test와 동일하면 자동으로 다음 fold 사용)
            return_indices: True면 원본 samples의 인덱스 리스트 반환.
                            evidence_digest와 정합이 필요할 때 True로 사용.
        """
        k = int(k_folds)
        if k < 3:
            raise ValueError("k_folds must be >= 3 for train/val/test split")

        # 원본 인덱스를 함께 유지
        by_label: Dict[str, List[Tuple[int, SimulatedOutput]]] = {}
        for i, s in enumerate(samples):
            by_label.setdefault(s.true_label, []).append((i, s))

        rng = np.random.default_rng(seed)
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []
        train_s: List[SimulatedOutput] = []
        val_s: List[SimulatedOutput] = []
        test_s: List[SimulatedOutput] = []

        for label, group in by_label.items():
            arr = list(group)
            n = len(arr)
            if n == 0:
                continue
            if n < k:
                raise ValueError(f"not enough samples for k-fold split: label={label}, n={n}, k={k}")

            rng.shuffle(arr)
            # np.array_split은 (index, sample) 튜플 배열을 지원하지 않으므로 인덱스만 분할
            positions = list(range(n))
            fold_positions = [list(x) for x in np.array_split(positions, k)]

            t = int(test_fold) % k
            v = int(val_fold) % k
            if v == t:
                v = (t + 1) % k

            n_train = 0
            n_val = 0
            n_test = 0
            for fold_i, pos_list in enumerate(fold_positions):
                for pos in pos_list:
                    orig_idx, s = arr[pos]
                    if fold_i == t:
                        test_idx.append(orig_idx); test_s.append(s)
                        n_test += 1
                    elif fold_i == v:
                        val_idx.append(orig_idx); val_s.append(s)
                        n_val += 1
                    else:
                        train_idx.append(orig_idx); train_s.append(s)
                        n_train += 1

            print(
                f"  kfold[{label}] train={n_train}, val={n_val}, test={n_test} "
                f"(k={k}, test_fold={t}, val_fold={v})"
            )

        if return_indices:
            return train_idx, val_idx, test_idx
        return train_s, val_s, test_s


def _confidence_entropy(confidences: np.ndarray, eps: float = 1e-12) -> float:
    """confidence 벡터를 정규화해 엔트로피를 계산한다."""
    c = np.asarray(confidences, dtype=np.float64)
    c = np.clip(c, eps, None)
    c = c / float(c.sum())
    h = -float(np.sum(c * np.log(c)))
    h_max = float(np.log(len(c))) if len(c) > 1 else 1.0
    if h_max <= 0.0:
        return 0.0
    return h / h_max


def build_proxy_image_features(
    samples: Sequence[SimulatedOutput],
    feature_set: str = "base20",
    records: "Optional[Sequence[CollectedRecord]]" = None,
) -> np.ndarray:
    """
    Path A 프록시 이미지 특징.

    feature_set:
        - "base20"      : agent별 verdict one-hot(4) + confidence(1) = 20-dim
        - "enhanced36"  : base20 + 집계/충돌 통계 17-dim = 37-dim
                          (이름은 "36"이나 실제 차원은 37 — verdict_ratio 4 + 13 scalars)
        - "risk52"      : enhanced37 + 리스크 중심 통계 15-dim = 52-dim
        - "evidence_2ch": enhanced37 + evidence (value, is_present) 2채널
                          = 37 + len(EVIDENCE_NUMERIC_KEYS)*2 = 37+14 = 51-dim
                          records 인자가 필요하며, samples와 1:1 대응이어야 함.

    Args:
        samples: SimulatedOutput 리스트
        feature_set: 특징 구성 방식
        records: evidence_2ch 사용 시 필수. samples와 동일 인덱스로 정렬된
                 CollectedRecord 리스트. stratified_split(return_indices=True)로
                 인덱스 정합을 보장해야 함.
    """
    feature_set = str(feature_set).lower().strip()
    valid_sets = {"base20", "enhanced36", "risk52", "evidence_2ch"}
    if feature_set not in valid_sets:
        raise ValueError(f"unsupported proxy feature_set: {feature_set}")

    if feature_set == "evidence_2ch":
        if records is None:
            raise ValueError(
                "records 인자가 필요합니다 (feature_set='evidence_2ch'). "
                "stratified_split(return_indices=True)로 index 정합을 보장한 뒤 사용하세요."
            )
        if len(records) != len(samples):
            raise ValueError(
                f"samples({len(samples)})와 records({len(records)}) 길이가 다릅니다. "
                "split 후 동일 인덱스 재정렬이 필요합니다."
            )

    rows: List[List[float]] = []
    for row_i, s in enumerate(samples):
        base_feat: List[float] = []
        verdict_idx: List[int] = []
        conf_vals: List[float] = []

        for agent in AGENT_NAMES:
            v = s.agent_verdicts[agent]
            base_feat.extend([1.0 if v == vv else 0.0 for vv in VERDICTS])
            conf = float(s.agent_confidences[agent])
            base_feat.append(conf)
            verdict_idx.append(VERDICT_TO_IDX[v])
            conf_vals.append(conf)

        if feature_set == "base20":
            rows.append(base_feat)
            continue

        # enhanced36 / risk52 공통 통계
        conf_arr = np.asarray(conf_vals, dtype=np.float64)
        verdict_counts = np.bincount(verdict_idx, minlength=len(VERDICTS)).astype(np.float64)
        verdict_ratio = verdict_counts / float(len(AGENT_NAMES))
        majority_ratio = float(np.max(verdict_ratio))
        uncertain_ratio = float(verdict_ratio[VERDICT_TO_IDX["uncertain"]])

        pair_disagree: List[float] = []
        pair_conflict: List[float] = []
        pair_conf_gap: List[float] = []
        for i in range(len(AGENT_NAMES)):
            for j in range(i + 1, len(AGENT_NAMES)):
                disagree = 1.0 if verdict_idx[i] != verdict_idx[j] else 0.0
                gap = abs(conf_arr[i] - conf_arr[j])
                pair_disagree.append(disagree)
                pair_conf_gap.append(gap)
                pair_conflict.append(disagree * gap)

        conf_sorted = np.sort(conf_arr)[::-1]
        top2_gap = float(conf_sorted[0] - conf_sorted[1]) if len(conf_sorted) >= 2 else 0.0

        extra_feat = [
            *verdict_ratio.tolist(),                    # 4 (VERDICTS = 4)
            float(np.mean(conf_arr)),                   # 1
            float(np.std(conf_arr)),                    # 1
            float(np.min(conf_arr)),                    # 1
            float(np.max(conf_arr)),                    # 1
            float(np.max(conf_arr) - np.min(conf_arr)), # 1
            float(_confidence_entropy(conf_arr)),       # 1
            majority_ratio,                             # 1
            uncertain_ratio,                            # 1
            float(np.mean(pair_disagree)),              # 1
            float(np.mean(pair_conflict)),              # 1
            float(np.max(pair_conflict)),               # 1
            float(np.mean(pair_conf_gap)),              # 1
            top2_gap,                                   # 1
        ]  # total 17-dim (verdict_ratio 4 + 13 scalars)
        if feature_set == "enhanced36":
            rows.append(base_feat + extra_feat)
            continue

        # risk52: enhanced36 + 16-dim downside/instability 지표
        verdict_entropy = 0.0
        v_safe = np.clip(verdict_ratio, 1e-12, 1.0)
        if len(v_safe) > 1:
            verdict_entropy = float(-np.sum(v_safe * np.log(v_safe)) / np.log(len(v_safe)))

        non_uncertain_ratio = float(1.0 - uncertain_ratio)
        pair_disagree_arr = np.asarray(pair_disagree, dtype=np.float64)
        pair_conflict_arr = np.asarray(pair_conflict, dtype=np.float64)
        pair_conf_gap_arr = np.asarray(pair_conf_gap, dtype=np.float64)

        conf_q25, _, conf_q75 = np.quantile(conf_arr, [0.25, 0.5, 0.75])
        conf_iqr = float(conf_q75 - conf_q25)
        conf_cv = float(np.std(conf_arr, ddof=0) / max(float(np.mean(conf_arr)), 1e-6))

        uncertain_mask = np.array([idx == VERDICT_TO_IDX["uncertain"] for idx in verdict_idx], dtype=bool)
        if np.any(uncertain_mask):
            uncertain_conf_mean = float(np.mean(conf_arr[uncertain_mask]))
        else:
            uncertain_conf_mean = 0.0

        non_uncertain_mask = ~uncertain_mask
        if np.any(non_uncertain_mask):
            non_uncertain_conf_mean = float(np.mean(conf_arr[non_uncertain_mask]))
            non_uncertain_conf_min = float(np.min(conf_arr[non_uncertain_mask]))
        else:
            non_uncertain_conf_mean = 0.0
            non_uncertain_conf_min = 0.0

        sorted_ratio = np.sort(verdict_ratio)[::-1]
        consensus_margin = float(sorted_ratio[0] - sorted_ratio[1]) if len(sorted_ratio) >= 2 else float(sorted_ratio[0])

        risk_feat = [
            verdict_entropy,                             # 1
            non_uncertain_ratio,                         # 1
            float(1.0 - np.mean(pair_disagree_arr)),    # 1
            float(np.std(pair_disagree_arr, ddof=0)),   # 1
            float(np.std(pair_conflict_arr, ddof=0)),   # 1
            float(np.std(pair_conf_gap_arr, ddof=0)),   # 1
            float(conf_q25),                             # 1
            float(conf_q75),                             # 1
            conf_iqr,                                    # 1
            conf_cv,                                     # 1
            non_uncertain_conf_min,                      # 1
            non_uncertain_conf_mean,                     # 1
            uncertain_conf_mean,                         # 1
            consensus_margin,                            # 1
            float(np.max(pair_conf_gap_arr)),            # 1
        ]  # total 15-dim

        if feature_set == "risk52":
            rows.append(base_feat + extra_feat + risk_feat)
            continue

        # evidence_2ch: enhanced37 + (value, is_present) 2채널 14-dim = 51-dim
        if feature_set == "evidence_2ch":
            # evidence_digest는 {agent_name: {evidence_key: value}} 중첩 구조.
            # evidence_to_feature_vector는 평탄 dict를 기대하므로 먼저 merge한다.
            # 같은 키가 여러 에이전트에 있으면 먼저 나온 non-None 값을 사용한다.
            per_agent: Dict[str, Dict[str, Any]] = (
                records[row_i].evidence_digest if records is not None else {}  # type: ignore[index]
            )
            flat_digest: Dict[str, Any] = {}
            for agent_ev in per_agent.values():
                for k, v in agent_ev.items():
                    if k not in flat_digest or flat_digest[k] is None:
                        flat_digest[k] = v
            ev_feat = evidence_to_feature_vector(flat_digest)
            rows.append(base_feat + extra_feat + ev_feat)
            continue

        # fallback (이 경로는 정상 경우 도달하지 않음)
        rows.append(base_feat + extra_feat + risk_feat)

    if not rows:
        if feature_set == "base20":
            dim = 20
        elif feature_set == "enhanced36":
            dim = 37  # base20(20) + extra_feat(17)
        elif feature_set == "evidence_2ch":
            dim = 37 + len(EVIDENCE_NUMERIC_KEYS) * 2  # 37 + 14 = 51
        else:
            dim = 52  # enhanced37(37) + risk_feat(15)
        return np.zeros((0, dim), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def build_empirical_simulator(train_data: Sequence[SimulatedOutput], seed: int = 42) -> AgentSimulator:
    """
    Path A 수집 결과로 에이전트별 confusion matrix를 추정해 OracleWeightComputer에 제공.
    """
    true_labels = ["authentic", "manipulated", "ai_generated"]
    profiles: Dict[str, AgentProfile] = {}

    for agent in AGENT_NAMES:
        cm: Dict[str, np.ndarray] = {}
        for tlabel in true_labels:
            # Laplace smoothing으로 zero-probability 회피
            counts = np.ones(len(VERDICTS), dtype=np.float64)
            rows = [s for s in train_data if s.true_label == tlabel]
            for s in rows:
                counts[VERDICT_TO_IDX[s.agent_verdicts[agent]]] += 1.0
            cm[tlabel] = counts / counts.sum()

        profiles[agent] = AgentProfile(
            name=agent,
            confusion_matrices=cm,
            confidence_correct=(8.0, 2.0),
            confidence_wrong=(2.0, 5.0),
            confidence_uncertain=(3.0, 3.0),
        )

    return AgentSimulator(profiles=profiles, seed=seed)
