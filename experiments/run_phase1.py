"""
DAAC Phase 1 실험 파이프라인 (Path B: 시뮬레이션)

실행:
    cd /home/dsu/Desktop/MAIFS
    .venv-qwen/bin/python -m experiments.run_phase1

또는:
    .venv-qwen/bin/python experiments/run_phase1.py

단계:
    1. 시뮬레이터로 합성 에이전트 출력 생성
    2. 43-dim 메타 특징 추출
    3. 베이스라인 (Majority Vote, COBRA) 평가
    4. 메타 분류기 (LogReg, GBM, MLP) 학습/평가
    5. Ablation (A1~A6) 실행
    6. Go/No-Go 판정
    7. 결과 저장
"""
import sys
import os
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

from src.meta.simulator import AgentSimulator
from src.meta.features import MetaFeatureExtractor, ABLATION_CONFIGS
from src.meta.baselines import MajorityVoteBaseline, COBRABaseline
from src.meta.trainer import MetaTrainer
from src.meta.evaluate import MetaEvaluator
from src.meta.ablation import AblationRunner


def load_config(config_path: str = None) -> dict:
    """YAML 설정 로드"""
    if config_path is None:
        config_path = _project_root / "experiments" / "configs" / "phase1.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_phase1(config: dict = None) -> dict:
    """
    Phase 1 전체 파이프라인 실행

    Returns:
        dict: 실험 결과 요약
    """
    if config is None:
        config = load_config()

    start_time = time.time()
    results = {"timestamp": datetime.now().isoformat(), "config": config}

    sim_cfg = config["simulation"]
    eval_cfg = config.get("evaluation", {})

    print("=" * 70)
    print("DAAC Phase 1: Disagreement Pattern Analysis (Path B)")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: 합성 데이터 생성
    # ---------------------------------------------------------------
    print("\n[Step 1] 합성 에이전트 출력 생성...")
    simulator = AgentSimulator(seed=sim_cfg["seed"])

    train_data, val_data, test_data = simulator.generate_split(
        n_train=sim_cfg["n_train"],
        n_val=sim_cfg["n_val"],
        n_test=sim_cfg["n_test"],
        sub_type_split=sim_cfg.get("cross_generator_split", True),
    )

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 에이전트 성능 요약
    profile_summary = simulator.get_profile_summary()
    results["agent_profiles"] = profile_summary
    print("\n  에이전트 예상 정확도:")
    for agent, accs in profile_summary.items():
        print(f"    {agent:12s}: overall={accs['overall']:.2f}  "
              f"auth={accs['authentic']:.2f} manip={accs['manipulated']:.2f} "
              f"ai_gen={accs['ai_generated']:.2f}")

    # ---------------------------------------------------------------
    # Step 2: 메타 특징 추출
    # ---------------------------------------------------------------
    print("\n[Step 2] 43-dim 메타 특징 추출...")
    extractor = MetaFeatureExtractor()
    X_train, y_train = extractor.extract_dataset(train_data)
    X_val, y_val = extractor.extract_dataset(val_data)
    X_test, y_test = extractor.extract_dataset(test_data)

    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"  Feature dim: {extractor.dim}")

    # 클래스 분포 확인
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        counts = np.bincount(y, minlength=3)
        print(f"  {split_name} 분포: auth={counts[0]}, manip={counts[1]}, ai_gen={counts[2]}")

    # ---------------------------------------------------------------
    # Step 3: 베이스라인 평가
    # ---------------------------------------------------------------
    print("\n[Step 3] 베이스라인 평가...")
    evaluator = MetaEvaluator(
        n_bootstrap=eval_cfg.get("n_bootstrap", 1000),
        ci_level=eval_cfg.get("ci_level", 0.95),
    )

    # Majority Vote
    mv = MajorityVoteBaseline()
    y_pred_mv = mv.predict(test_data)
    eval_mv = evaluator.evaluate(y_test, y_pred_mv, model_name="majority_vote")
    print(evaluator.format_result(eval_mv))

    # COBRA
    cobra_cfg = config.get("cobra", {})
    cobra = COBRABaseline(
        trust_scores=cobra_cfg.get("trust_scores"),
        algorithm=cobra_cfg.get("algorithm", "drwa"),
    )
    y_pred_cobra = cobra.predict(test_data)
    y_proba_cobra = cobra.predict_proba(test_data)
    eval_cobra = evaluator.evaluate(y_test, y_pred_cobra, y_proba_cobra, "cobra")
    print(evaluator.format_result(eval_cobra))

    results["baselines"] = {
        "majority_vote": {
            "macro_f1": eval_mv.macro_f1,
            "balanced_accuracy": eval_mv.balanced_accuracy,
            "per_class_f1": eval_mv.per_class_f1,
        },
        "cobra": {
            "macro_f1": eval_cobra.macro_f1,
            "balanced_accuracy": eval_cobra.balanced_accuracy,
            "auroc": eval_cobra.auroc,
            "ece": eval_cobra.ece,
            "per_class_f1": eval_cobra.per_class_f1,
        },
    }

    # ---------------------------------------------------------------
    # Step 4: 메타 분류기 학습/평가
    # ---------------------------------------------------------------
    print("\n[Step 4] 메타 분류기 학습...")
    model_configs = config.get("models", {})
    # tuple 변환 (YAML은 list로 로드)
    if "mlp" in model_configs and "hidden_layer_sizes" in model_configs["mlp"]:
        model_configs["mlp"]["hidden_layer_sizes"] = tuple(
            model_configs["mlp"]["hidden_layer_sizes"]
        )

    trainer = MetaTrainer(model_configs=model_configs)
    train_results = trainer.train_all(X_train, y_train, X_val, y_val)

    for name, tr in train_results.items():
        print(f"  {name:25s}: train_acc={tr.train_accuracy:.4f}, val_acc={tr.val_accuracy:.4f}")

    # 테스트 평가
    print("\n[Step 4b] 메타 분류기 테스트 평가...")
    meta_results = {}
    for model_name in trainer.models:
        y_pred = trainer.predict(model_name, X_test)
        y_proba = trainer.predict_proba(model_name, X_test)
        eval_result = evaluator.evaluate(y_test, y_pred, y_proba, model_name)
        print(evaluator.format_result(eval_result))
        meta_results[model_name] = {
            "macro_f1": eval_result.macro_f1,
            "balanced_accuracy": eval_result.balanced_accuracy,
            "auroc": eval_result.auroc,
            "ece": eval_result.ece,
            "brier": eval_result.brier,
            "per_class_f1": eval_result.per_class_f1,
        }

    results["meta_classifiers"] = meta_results

    # ---------------------------------------------------------------
    # Step 5: COBRA vs 최고 메타 분류기 비교
    # ---------------------------------------------------------------
    print("\n[Step 5] COBRA vs 메타 분류기 통계 비교...")
    best_meta_name = max(meta_results, key=lambda k: meta_results[k]["macro_f1"])
    y_pred_best = trainer.predict(best_meta_name, X_test)

    comparison = evaluator.compare(
        y_test, y_pred_cobra, y_pred_best,
        model_a="cobra", model_b=best_meta_name,
    )
    print(evaluator.format_comparison(comparison))

    results["cobra_vs_best"] = {
        "model_a": comparison.model_a,
        "model_b": comparison.model_b,
        "f1_diff": comparison.f1_diff,
        "mcnemar_pvalue": comparison.mcnemar_pvalue,
        "significant": comparison.significant,
        "ci_lower": comparison.f1_ci_lower,
        "ci_upper": comparison.f1_ci_upper,
    }

    # ---------------------------------------------------------------
    # Step 6: Ablation 실행
    # ---------------------------------------------------------------
    print("\n[Step 6] Ablation 실행...")
    abl_cfg = config.get("ablation", {})
    runner = AblationRunner(
        models=abl_cfg.get("models"),
        evaluator=evaluator,
    )
    ablation_summary = runner.run(
        train_data, val_data, test_data,
        run_a6=abl_cfg.get("run_a6_agent_removal", True),
    )
    report = runner.print_summary(ablation_summary)

    # Ablation 결과를 dict로 변환
    ablation_dict = {}
    for aid, ares in ablation_summary.results.items():
        ablation_dict[aid] = {
            "feature_dim": ares.feature_dim,
            "best_model": ares.best_model,
            "best_f1": ares.best_f1,
            "models": {
                mn: {"macro_f1": er.macro_f1, "balanced_accuracy": er.balanced_accuracy}
                for mn, er in ares.model_results.items()
            },
        }
    results["ablation"] = ablation_dict

    # ---------------------------------------------------------------
    # Step 7: Go/No-Go 종합 판정
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Go/No-Go 종합 판정")
    print("=" * 70)

    go_cfg = config.get("go_no_go", {})

    # C1: Full(A5) > COBRA (McNemar p < alpha)
    c1_passed = comparison.significant and comparison.f1_diff > 0
    print(f"\n  [{'PASS' if c1_passed else 'FAIL':4s}] C1: A5(Full) > COBRA")
    print(f"         F1 diff: {comparison.f1_diff:+.4f}, "
          f"McNemar p={comparison.mcnemar_pvalue:.4f}")

    # C2: 교차 데이터셋 성능 유지 (시뮬레이션에서는 train/test 분포 차이로 근사)
    a5_full = ablation_summary.results.get("A5_full")
    if a5_full:
        # train F1 vs test F1 차이
        best_model_name = a5_full.best_model
        train_trainer = MetaTrainer(model_configs=model_configs)
        train_trainer.train(best_model_name, X_train, y_train)
        y_pred_train = train_trainer.predict(best_model_name, X_train)
        from sklearn.metrics import f1_score
        train_f1 = f1_score(y_train, y_pred_train, average="macro", zero_division=0)
        test_f1 = a5_full.best_f1
        f1_drop = train_f1 - test_f1
        c2_threshold = go_cfg.get("c2_cross_dataset_drop_threshold", 0.05)
        c2_passed = f1_drop < c2_threshold
        print(f"\n  [{'PASS' if c2_passed else 'FAIL':4s}] C2: 교차 데이터셋 성능 유지")
        print(f"         Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, "
              f"Drop: {f1_drop:.4f} (threshold: {c2_threshold})")
    else:
        c2_passed = False
        print(f"\n  [FAIL] C2: A5_full 결과 없음")

    # C3: A3 (disagreement only) > random + margin
    a3 = ablation_summary.results.get("A3_disagreement_only")
    random_f1 = 1.0 / 3.0
    c3_margin = go_cfg.get("c3_disagreement_margin", 0.05)
    if a3:
        c3_passed = a3.best_f1 > (random_f1 + c3_margin)
        print(f"\n  [{'PASS' if c3_passed else 'FAIL':4s}] C3: A3(disagreement) > random")
        print(f"         A3 F1: {a3.best_f1:.4f}, "
              f"Threshold: {random_f1 + c3_margin:.4f} (random={random_f1:.4f} + {c3_margin})")
    else:
        c3_passed = False
        print(f"\n  [FAIL] C3: A3 결과 없음")

    overall_go = c1_passed and c2_passed and c3_passed
    print(f"\n  {'=' * 50}")
    print(f"  OVERALL: {'>>> GO >>> Phase 2 착수' if overall_go else '>>> NO-GO >>> 재설계 필요'}")
    print(f"  {'=' * 50}")

    results["go_no_go"] = {
        "c1_full_vs_cobra": {"passed": c1_passed, "f1_diff": comparison.f1_diff, "p": comparison.mcnemar_pvalue},
        "c2_cross_dataset": {"passed": c2_passed},
        "c3_disagreement": {"passed": c3_passed, "a3_f1": a3.best_f1 if a3 else None},
        "overall": overall_go,
    }

    # ---------------------------------------------------------------
    # Step 8: 특징 중요도 분석
    # ---------------------------------------------------------------
    print("\n[Step 8] 특징 중요도 분석...")
    feature_names = extractor.feature_names

    for model_name in trainer.models:
        importance = trainer.get_feature_importance(model_name)
        if importance is not None:
            print(f"\n  {model_name} Top-10 features:")
            top_idx = np.argsort(importance)[::-1][:10]
            for rank, idx in enumerate(top_idx, 1):
                print(f"    {rank:2d}. {feature_names[idx]:35s}: {importance[idx]:.4f}")

    # ---------------------------------------------------------------
    # 결과 저장
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    print(f"\n총 실행 시간: {elapsed:.1f}초")

    output_cfg = config.get("output", {})
    save_dir = Path(output_cfg.get("save_dir", "experiments/results/phase1"))
    save_dir = _project_root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # JSON 결과 저장
    result_path = save_dir / f"phase1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # numpy 타입을 JSON 호환으로 변환
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

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert, ensure_ascii=False)
    print(f"\n결과 저장: {result_path}")

    # Ablation 리포트 저장
    report_path = save_dir / "ablation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Ablation 리포트: {report_path}")

    return results


if __name__ == "__main__":
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = load_config(config_path)
    run_phase1(config)
