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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _safe_evidence_digest(result: ToolResult) -> Dict[str, Any]:
    """큰 텐서/마스크를 피하고, 디버깅에 필요한 핵심 값만 저장."""
    keep_keys = {
        "backend",
        "error",
        "catnet_available",
        "compression_artifact_score",
        "manipulation_ratio",
        "mean_intensity",
        "max_intensity",
        "mvss_score",
    }
    out: Dict[str, Any] = {}
    for k, v in (result.evidence or {}).items():
        if k not in keep_keys:
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
        assert self._freq_tool is not None
        assert self._noise_tool is not None
        assert self._fat_tool is not None
        assert self._spatial_tool is not None

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
    ) -> Tuple[List[SimulatedOutput], List[SimulatedOutput], List[SimulatedOutput]]:
        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
        if np.any(ratios < 0):
            raise ValueError("split ratios must be >= 0")
        if float(ratios.sum()) <= 0:
            raise ValueError("sum of split ratios must be > 0")
        ratios /= ratios.sum()

        by_label: Dict[str, List[SimulatedOutput]] = {}
        for s in samples:
            by_label.setdefault(s.true_label, []).append(s)

        rng = np.random.default_rng(seed)
        train: List[SimulatedOutput] = []
        val: List[SimulatedOutput] = []
        test: List[SimulatedOutput] = []

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
                # 소수 샘플 방어: 순서대로 train/val/test에 배치
                n_train = min(1, n)
                n_val = min(1, max(0, n - n_train))
                n_test = max(0, n - n_train - n_val)

            train.extend(arr[:n_train])
            val.extend(arr[n_train:n_train + n_val])
            test.extend(arr[n_train + n_val:n_train + n_val + n_test])
            print(f"  split[{label}] train={n_train}, val={n_val}, test={n_test}")

        return train, val, test

    @staticmethod
    def stratified_kfold_split(
        samples: Sequence[SimulatedOutput],
        k_folds: int = 5,
        test_fold: int = 0,
        val_fold: int = 1,
        seed: int = 42,
    ) -> Tuple[List[SimulatedOutput], List[SimulatedOutput], List[SimulatedOutput]]:
        """
        라벨별로 동일한 셔플 순서를 만든 뒤 k-fold로 분할하여 train/val/test를 구성한다.

        - test_fold: 테스트로 사용할 fold index
        - val_fold : 검증으로 사용할 fold index (test와 동일하면 자동으로 다음 fold 사용)
        """
        k = int(k_folds)
        if k < 3:
            raise ValueError("k_folds must be >= 3 for train/val/test split")

        by_label: Dict[str, List[SimulatedOutput]] = {}
        for s in samples:
            by_label.setdefault(s.true_label, []).append(s)

        rng = np.random.default_rng(seed)
        train: List[SimulatedOutput] = []
        val: List[SimulatedOutput] = []
        test: List[SimulatedOutput] = []

        for label, group in by_label.items():
            arr = list(group)
            n = len(arr)
            if n == 0:
                continue
            if n < k:
                raise ValueError(f"not enough samples for k-fold split: label={label}, n={n}, k={k}")

            rng.shuffle(arr)
            folds = [list(x) for x in np.array_split(arr, k)]
            t = int(test_fold) % k
            v = int(val_fold) % k
            if v == t:
                v = (t + 1) % k

            n_train = 0
            n_val = 0
            n_test = 0
            for i, bucket in enumerate(folds):
                if i == t:
                    test.extend(bucket)
                    n_test += len(bucket)
                elif i == v:
                    val.extend(bucket)
                    n_val += len(bucket)
                else:
                    train.extend(bucket)
                    n_train += len(bucket)

            print(
                f"  kfold[{label}] train={n_train}, val={n_val}, test={n_test} "
                f"(k={k}, test_fold={t}, val_fold={v})"
            )

        return train, val, test


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
) -> np.ndarray:
    """
    Path A 프록시 이미지 특징.

    feature_set:
        - "base20": agent별 verdict one-hot(4) + confidence(1) = 20-dim
        - "enhanced36": base20 + 집계/충돌 통계 16-dim = 36-dim
    """
    feature_set = str(feature_set).lower().strip()
    if feature_set not in {"base20", "enhanced36"}:
        raise ValueError(f"unsupported proxy feature_set: {feature_set}")

    rows: List[List[float]] = []
    for s in samples:
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

        # enhanced36 추가 통계
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
            *verdict_ratio.tolist(),                    # 4
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
        ]  # total 16-dim
        rows.append(base_feat + extra_feat)

    if not rows:
        dim = 20 if feature_set == "base20" else 36
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
