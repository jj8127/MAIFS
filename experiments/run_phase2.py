"""
DAAC Phase 2 실험 파이프라인 (Path B: 시뮬레이션)

Phase 2 (Adaptive Routing):
    - SyntheticImageEncoder로 "image feature"를 합성 생성
    - MetaRouter가 image feature → agent weights(4-dim)를 예측
    - Phase 1의 43-dim 메타 특징에 router weights를 concat하여 (47-dim)
      메타 분류기 성능 변화를 평가한다.

실행:
    cd /home/dsu/Desktop/MAIFS
    .venv-qwen/bin/python -m experiments.run_phase2

또는:
    .venv-qwen/bin/python experiments/run_phase2.py [config_path]
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

# 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.meta.simulator import (
    AgentSimulator,
    AgentProfile,
    AGENT_NAMES,
    TRUE_LABELS,
    VERDICTS,
    _build_default_profiles,
)
from src.meta.features import MetaFeatureExtractor
from src.meta.baselines import MajorityVoteBaseline, COBRABaseline
from src.meta.trainer import MetaTrainer
from src.meta.evaluate import MetaEvaluator
from src.meta.router import (
    SyntheticImageEncoder,
    SyntheticImageEncoderConfig,
    OracleWeightComputer,
    OracleWeightConfig,
    MetaRouter,
    MetaRouterConfig,
)


def load_config(config_path: str = None) -> dict:
    """YAML 설정 로드"""
    if config_path is None:
        config_path = _project_root / "experiments" / "configs" / "phase2.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _normalize_cm_row(row: list) -> np.ndarray:
    """confusion matrix 한 행을 안전하게 정규화"""
    arr = np.array(row, dtype=float).reshape(-1)
    if arr.size != len(VERDICTS):
        raise ValueError(f"confusion row size must be {len(VERDICTS)}")
    arr = np.clip(arr, 0.0, None)
    if float(arr.sum()) <= 0.0:
        raise ValueError("confusion row sum must be > 0")
    arr /= arr.sum()
    return arr


def _to_beta_pair(value, default_pair):
    """Beta 분포 파라미터를 (a, b) 튜플로 보정"""
    if value is None:
        return default_pair
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            a = float(value[0])
            b = float(value[1])
            if a > 0 and b > 0:
                return (a, b)
    except (TypeError, ValueError):
        pass
    return default_pair


def build_simulator_from_config(config: dict) -> AgentSimulator:
    """설정 기반으로 AgentSimulator 생성 (프로파일/상관/레이블 분포 반영)"""
    sim_cfg = config["simulation"]

    default_profiles = _build_default_profiles()
    profile_overrides = sim_cfg.get("agent_profiles", {}) or {}
    applied_profile_overrides = 0

    for agent_name, override in profile_overrides.items():
        if agent_name not in default_profiles:
            print(f"[WARN] Unknown agent profile override ignored: {agent_name}")
            continue

        base = default_profiles[agent_name]
        merged_cm = {label: base.confusion_matrices[label].copy() for label in TRUE_LABELS}

        cm_override = override.get("confusion_matrices", {}) or {}
        for label, row in cm_override.items():
            if label not in TRUE_LABELS:
                print(f"[WARN] Unknown true label in profile override: {agent_name}.{label}")
                continue
            try:
                merged_cm[label] = _normalize_cm_row(row)
            except ValueError as e:
                print(f"[WARN] Invalid confusion row ignored ({agent_name}.{label}): {e}")

        default_profiles[agent_name] = AgentProfile(
            name=agent_name,
            confusion_matrices=merged_cm,
            confidence_correct=_to_beta_pair(
                override.get("confidence_correct"), base.confidence_correct
            ),
            confidence_wrong=_to_beta_pair(override.get("confidence_wrong"), base.confidence_wrong),
            confidence_uncertain=_to_beta_pair(
                override.get("confidence_uncertain"), base.confidence_uncertain
            ),
        )
        applied_profile_overrides += 1

    corr_cfg = config.get("correlation_matrix")
    correlation = None
    if corr_cfg is not None:
        correlation = np.array(corr_cfg, dtype=float)
        if correlation.shape != (len(AGENT_NAMES), len(AGENT_NAMES)):
            raise ValueError(
                f"correlation_matrix shape must be {(len(AGENT_NAMES), len(AGENT_NAMES))}, "
                f"got {correlation.shape}"
            )

    label_distribution = sim_cfg.get("label_distribution")

    if applied_profile_overrides > 0:
        print(f"  적용된 agent profile override 수: {applied_profile_overrides}")
    if correlation is not None:
        print("  correlation_matrix 설정 반영")
    if label_distribution:
        print(f"  label_distribution 설정 반영: {label_distribution}")

    return AgentSimulator(
        profiles=default_profiles,
        correlation=correlation,
        label_distribution=label_distribution,
        seed=sim_cfg["seed"],
    )


def _coerce_hidden_layer_sizes(model_configs: dict) -> None:
    """YAML(list) → sklearn/torch가 기대하는 tuple로 보정"""
    if not model_configs:
        return
    mlp = model_configs.get("mlp")
    if isinstance(mlp, dict) and "hidden_layer_sizes" in mlp:
        h = mlp["hidden_layer_sizes"]
        if isinstance(h, list):
            mlp["hidden_layer_sizes"] = tuple(h)


def _build_router_from_config(config: dict, simulator: AgentSimulator):
    router_cfg = (config.get("router") or {})
    enc_cfg = router_cfg.get("image_encoder") or {}
    oracle_cfg = router_cfg.get("oracle") or {}
    model_cfg = router_cfg.get("model") or {}

    encoder = SyntheticImageEncoder(
        SyntheticImageEncoderConfig(
            feature_dim=int(enc_cfg.get("feature_dim", 32)),
            prototype_scale=float(enc_cfg.get("prototype_scale", 3.0)),
            noise_std=float(enc_cfg.get("noise_std", 1.0)),
            seed=int(enc_cfg.get("seed", config["simulation"]["seed"])),
        )
    )

    oracle = OracleWeightComputer(
        simulator=simulator,
        config=OracleWeightConfig(
            power=float(oracle_cfg.get("power", 2.0)),
            eps=float(oracle_cfg.get("eps", 1e-6)),
            label_smoothing=float(oracle_cfg.get("label_smoothing", 0.0)),
            confidence_power=float(oracle_cfg.get("confidence_power", 0.0)),
            uncertain_penalty=float(oracle_cfg.get("uncertain_penalty", 1.0)),
            entropy_power=float(oracle_cfg.get("entropy_power", 0.0)),
            verdict_power=float(oracle_cfg.get("verdict_power", 0.0)),
        ),
    )

    hls = model_cfg.get("hidden_layer_sizes", [64, 64])
    if isinstance(hls, list):
        hls = tuple(int(v) for v in hls)
    elif isinstance(hls, tuple):
        hls = tuple(int(v) for v in hls)
    else:
        hls = (64, 64)

    router = MetaRouter(
        config=MetaRouterConfig(
            hidden_layer_sizes=hls,
            activation=str(model_cfg.get("activation", "relu")),
            solver=str(model_cfg.get("solver", "adam")),
            alpha=float(model_cfg.get("alpha", 1e-4)),
            learning_rate_init=float(model_cfg.get("learning_rate_init", 1e-3)),
            max_iter=int(model_cfg.get("max_iter", 500)),
            early_stopping=bool(model_cfg.get("early_stopping", True)),
            validation_fraction=float(model_cfg.get("validation_fraction", 0.15)),
            n_iter_no_change=int(model_cfg.get("n_iter_no_change", 20)),
            random_state=int(model_cfg.get("random_state", 42)),
            regressor=str(model_cfg.get("regressor", "mlp")),
            ridge_alpha=float(model_cfg.get("ridge_alpha", 1.0)),
            gbr_n_estimators=int(model_cfg.get("gbr_n_estimators", 200)),
            gbr_learning_rate=float(model_cfg.get("gbr_learning_rate", 0.05)),
            gbr_max_depth=int(model_cfg.get("gbr_max_depth", 3)),
            gbr_subsample=float(model_cfg.get("gbr_subsample", 1.0)),
        ),
        eps=float(oracle_cfg.get("eps", 1e-6)),
    )

    return encoder, oracle, router


def run_phase2(config: dict = None) -> dict:
    if config is None:
        config = load_config()

    start_time = time.time()
    results = {"timestamp": datetime.now().isoformat(), "config": config}

    sim_cfg = config["simulation"]
    eval_cfg = config.get("evaluation", {})

    print("=" * 70)
    print("DAAC Phase 2: Adaptive Routing (Path B: Synthetic)")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: 합성 데이터 생성
    # ---------------------------------------------------------------
    print("\n[Step 1] 합성 에이전트 출력 생성...")
    simulator = build_simulator_from_config(config)

    train_data, val_data, test_data = simulator.generate_split(
        n_train=sim_cfg["n_train"],
        n_val=sim_cfg["n_val"],
        n_test=sim_cfg["n_test"],
        sub_type_split=sim_cfg.get("cross_generator_split", True),
    )
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # ---------------------------------------------------------------
    # Step 2: Phase 1 메타 특징(43-dim) 추출
    # ---------------------------------------------------------------
    print("\n[Step 2] Phase 1 43-dim 메타 특징 추출...")
    extractor = MetaFeatureExtractor()
    X_train_meta, y_train = extractor.extract_dataset(train_data)
    X_val_meta, y_val = extractor.extract_dataset(val_data)
    X_test_meta, y_test = extractor.extract_dataset(test_data)
    print(f"  X_train_meta: {X_train_meta.shape}, X_test_meta: {X_test_meta.shape}")

    # ---------------------------------------------------------------
    # Step 3: 베이스라인 평가 (Majority Vote, COBRA)
    # ---------------------------------------------------------------
    print("\n[Step 3] 베이스라인 평가...")
    evaluator = MetaEvaluator(
        n_bootstrap=int(eval_cfg.get("n_bootstrap", 200)),
        ci_level=float(eval_cfg.get("ci_level", 0.95)),
    )

    mv = MajorityVoteBaseline()
    y_pred_mv = mv.predict(test_data)
    eval_mv = evaluator.evaluate(y_test, y_pred_mv, model_name="majority_vote")
    print(evaluator.format_result(eval_mv))

    cobra_cfg = config.get("cobra", {}) or {}
    cobra = COBRABaseline(
        trust_scores=cobra_cfg.get("trust_scores"),
        algorithm=cobra_cfg.get("algorithm", "drwa"),
    )
    y_pred_cobra = cobra.predict(test_data)
    y_proba_cobra = cobra.predict_proba(test_data)
    eval_cobra = evaluator.evaluate(y_test, y_pred_cobra, y_proba_cobra, "cobra")
    print(evaluator.format_result(eval_cobra))

    results["baselines"] = {
        "majority_vote": {"macro_f1": eval_mv.macro_f1, "balanced_accuracy": eval_mv.balanced_accuracy},
        "cobra": {
            "macro_f1": eval_cobra.macro_f1,
            "balanced_accuracy": eval_cobra.balanced_accuracy,
            "auroc": eval_cobra.auroc,
            "ece": eval_cobra.ece,
            "brier": eval_cobra.brier,
        },
    }

    # ---------------------------------------------------------------
    # Step 4: Router 학습 (image_feature → weights)
    # ---------------------------------------------------------------
    print("\n[Step 4] Meta-Router 학습 (Synthetic image features)...")
    encoder, oracle, router = _build_router_from_config(config, simulator)

    X_train_img = encoder.encode_dataset(train_data)
    X_val_img = encoder.encode_dataset(val_data)
    X_test_img = encoder.encode_dataset(test_data)

    W_train_oracle = oracle.compute_dataset(train_data)
    W_val_oracle = oracle.compute_dataset(val_data)
    W_test_oracle = oracle.compute_dataset(test_data)

    router.fit(X_train_img, W_train_oracle)

    m_train = router.evaluate(X_train_img, W_train_oracle)
    m_val = router.evaluate(X_val_img, W_val_oracle)
    m_test = router.evaluate(X_test_img, W_test_oracle)

    print(f"  Router metrics (train): mse={m_train.mse_weights:.6f} kl={m_train.kl_weights:.6f} top1={m_train.top1_match:.4f}")
    print(f"  Router metrics (val)  : mse={m_val.mse_weights:.6f} kl={m_val.kl_weights:.6f} top1={m_val.top1_match:.4f}")
    print(f"  Router metrics (test) : mse={m_test.mse_weights:.6f} kl={m_test.kl_weights:.6f} top1={m_test.top1_match:.4f}")

    results["router"] = {
        "image_feature_dim": int(X_train_img.shape[1]),
        "metrics": {
            "train": m_train.__dict__,
            "val": m_val.__dict__,
            "test": m_test.__dict__,
        },
    }

    W_train_pred = router.predict_weights(X_train_img)
    W_val_pred = router.predict_weights(X_val_img)
    W_test_pred = router.predict_weights(X_test_img)

    # ---------------------------------------------------------------
    # Step 5: 메타 분류기 학습/평가 (Phase 1 vs Phase 2)
    # ---------------------------------------------------------------
    print("\n[Step 5] 메타 분류기 학습/평가 (Phase 1 vs Phase 2)...")
    model_configs = config.get("models", {}) or {}
    _coerce_hidden_layer_sizes(model_configs)

    # Phase 1 (43-dim)
    print("\n  [Phase 1 baseline] 43-dim meta features")
    trainer_p1 = MetaTrainer(model_configs=model_configs)
    trainer_p1.train_all(X_train_meta, y_train, X_val_meta, y_val)

    p1_results = {}
    for model_name in trainer_p1.models:
        y_pred = trainer_p1.predict(model_name, X_test_meta)
        y_proba = trainer_p1.predict_proba(model_name, X_test_meta)
        er = evaluator.evaluate(y_test, y_pred, y_proba, model_name=f"phase1_{model_name}")
        print(evaluator.format_result(er))
        p1_results[model_name] = {
            "macro_f1": er.macro_f1,
            "balanced_accuracy": er.balanced_accuracy,
            "auroc": er.auroc,
            "ece": er.ece,
            "brier": er.brier,
        }

    # Phase 2 (43+4=47 dim)
    X_train_aug = np.concatenate([X_train_meta, W_train_pred], axis=1)
    X_val_aug = np.concatenate([X_val_meta, W_val_pred], axis=1)
    X_test_aug = np.concatenate([X_test_meta, W_test_pred], axis=1)

    print("\n  [Phase 2] 47-dim meta features (43 + router weights)")
    trainer_p2 = MetaTrainer(model_configs=model_configs)
    trainer_p2.train_all(X_train_aug, y_train, X_val_aug, y_val)

    p2_results = {}
    for model_name in trainer_p2.models:
        y_pred = trainer_p2.predict(model_name, X_test_aug)
        y_proba = trainer_p2.predict_proba(model_name, X_test_aug)
        er = evaluator.evaluate(y_test, y_pred, y_proba, model_name=f"phase2_{model_name}")
        print(evaluator.format_result(er))
        p2_results[model_name] = {
            "macro_f1": er.macro_f1,
            "balanced_accuracy": er.balanced_accuracy,
            "auroc": er.auroc,
            "ece": er.ece,
            "brier": er.brier,
        }

    results["phase1_meta"] = p1_results
    results["phase2_meta"] = p2_results

    # ---------------------------------------------------------------
    # Step 6: Best(Phase2) vs Best(Phase1) 통계 비교
    # ---------------------------------------------------------------
    print("\n[Step 6] Best Phase2 vs Best Phase1 비교...")
    best_p1 = max(p1_results, key=lambda k: p1_results[k]["macro_f1"])
    best_p2 = max(p2_results, key=lambda k: p2_results[k]["macro_f1"])

    y_pred_p1 = trainer_p1.predict(best_p1, X_test_meta)
    y_pred_p2 = trainer_p2.predict(best_p2, X_test_aug)

    comp = evaluator.compare(
        y_true=y_test,
        y_pred_a=y_pred_p1,
        y_pred_b=y_pred_p2,
        model_a=f"phase1_{best_p1}",
        model_b=f"phase2_{best_p2}",
    )
    print(evaluator.format_comparison(comp))

    results["phase2_vs_phase1_best"] = {
        "phase1_best": best_p1,
        "phase2_best": best_p2,
        "f1_diff": comp.f1_diff,
        "mcnemar_pvalue": comp.mcnemar_pvalue,
        "significant": comp.significant,
        "ci_lower": comp.f1_ci_lower,
        "ci_upper": comp.f1_ci_upper,
    }

    # ---------------------------------------------------------------
    # 결과 저장
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = float(elapsed)
    print(f"\n총 실행 시간: {elapsed:.1f}초")

    output_cfg = config.get("output", {}) or {}
    save_dir = Path(output_cfg.get("save_dir", "experiments/results/phase2"))
    save_dir = _project_root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    result_path = save_dir / f"phase2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"\n결과 저장: {result_path}")

    return results


if __name__ == "__main__":
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    config = load_config(config_path)
    run_phase2(config)
