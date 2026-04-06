"""
Phase 1: MNV2 ECE 측정 + Error Pattern 분석 + w(x) 분포 시각화

사용자 결정사항:
- Dustbin: ai_generated는 제외하지 않고 dustbin class로 처리
- 정답확률: c(x) = P(y_true|x), NOT MSP
- Ablation: Temperature Scaling 유무 모두 테스트
- γ = {1, 2, 3} 비교
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── 설정 ──
RESULT_DIR = Path(__file__).parent / "results" / "backbone_eval"
DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]
TIMESTAMP = "20260319_070725"
OUTPUT_DIR = Path(__file__).parent / "results" / "phase1_ewct"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GAMMAS = [1, 2, 3]
N_BINS = 15  # ECE bin 수
CLASSES = ["authentic", "manipulated", "ai_generated"]


def load_dataset(ds_name):
    """JSONL 파일 로드"""
    path = RESULT_DIR / f"mobilenetv2_dualstream_{ds_name}_{TIMESTAMP}.jsonl"
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def compute_ece(confidences, accuracies, n_bins=15):
    """Expected Calibration Error (binned)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:  # 첫 bin은 0도 포함
            mask = (confidences >= lo) & (confidences <= hi)

        n = mask.sum()
        if n == 0:
            bin_stats.append({"lo": lo, "hi": hi, "n": 0, "avg_conf": 0, "avg_acc": 0, "gap": 0})
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        gap = abs(avg_acc - avg_conf)
        ece += (n / len(confidences)) * gap

        bin_stats.append({
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "n": int(n),
            "avg_conf": round(float(avg_conf), 4),
            "avg_acc": round(float(avg_acc), 4),
            "gap": round(float(gap), 4)
        })

    return float(ece), bin_stats


def compute_msp_ece(samples):
    """MSP 기반 ECE: confidence = max(scores), correct = (pred == true)"""
    confs = np.array([s["confidence"] for s in samples])
    accs = np.array([1.0 if s["pred_label"] == s["true_label"] else 0.0 for s in samples])
    return compute_ece(confs, accs, N_BINS)


def compute_gtprob_ece(samples):
    """정답확률 기반 ECE: confidence = P(y_true|x)"""
    confs = np.array([s["scores"][s["true_label"]] for s in samples])
    accs = np.array([1.0 if s["pred_label"] == s["true_label"] else 0.0 for s in samples])
    return compute_ece(confs, accs, N_BINS)


def compute_confusion_matrix(samples):
    """3x3 confusion matrix"""
    cm = defaultdict(lambda: defaultdict(int))
    for s in samples:
        cm[s["true_label"]][s["pred_label"]] += 1
    return {t: dict(p) for t, p in cm.items()}


def compute_error_analysis(samples):
    """오분류 케이스 상세 분석"""
    errors = [s for s in samples if s["pred_label"] != s["true_label"]]
    correct = [s for s in samples if s["pred_label"] == s["true_label"]]

    # 오분류 패턴별 분류
    error_patterns = defaultdict(list)
    for s in errors:
        pattern = f"{s['true_label']}→{s['pred_label']}"
        error_patterns[pattern].append({
            "msp": s["confidence"],
            "gt_prob": s["scores"][s["true_label"]],
            "scores": s["scores"]
        })

    # 오분류 vs 정분류 confidence 비교
    error_msp = [s["confidence"] for s in errors]
    correct_msp = [s["confidence"] for s in correct]
    error_gtprob = [s["scores"][s["true_label"]] for s in errors]
    correct_gtprob = [s["scores"][s["true_label"]] for s in correct]

    return {
        "n_total": len(samples),
        "n_errors": len(errors),
        "error_rate": round(len(errors) / len(samples), 4),
        "error_patterns": {
            k: {
                "count": len(v),
                "msp_mean": round(np.mean([x["msp"] for x in v]), 4),
                "msp_std": round(np.std([x["msp"] for x in v]), 4),
                "gtprob_mean": round(np.mean([x["gt_prob"] for x in v]), 4),
                "gtprob_std": round(np.std([x["gt_prob"] for x in v]), 4),
            }
            for k, v in sorted(error_patterns.items(), key=lambda x: -len(x[1]))
        },
        "confidence_comparison": {
            "error_msp": {"mean": round(np.mean(error_msp), 4), "std": round(np.std(error_msp), 4)} if error_msp else None,
            "correct_msp": {"mean": round(np.mean(correct_msp), 4), "std": round(np.std(correct_msp), 4)},
            "error_gtprob": {"mean": round(np.mean(error_gtprob), 4), "std": round(np.std(error_gtprob), 4)} if error_gtprob else None,
            "correct_gtprob": {"mean": round(np.mean(correct_gtprob), 4), "std": round(np.std(correct_gtprob), 4)},
        }
    }


def compute_wx_distribution(samples, gamma):
    """w(x) = (1 - P(y_true|x))^γ 분포 계산"""
    gt_probs = np.array([s["scores"][s["true_label"]] for s in samples])
    wx = (1 - gt_probs) ** gamma

    # 정분류 vs 오분류 분리
    correct_mask = np.array([s["pred_label"] == s["true_label"] for s in samples])
    error_mask = ~correct_mask

    result = {
        "gamma": gamma,
        "all": _stats(wx),
        "correct": _stats(wx[correct_mask]) if correct_mask.sum() > 0 else None,
        "error": _stats(wx[error_mask]) if error_mask.sum() > 0 else None,
    }

    # 클래스별 분포
    for cls in CLASSES:
        cls_mask = np.array([s["true_label"] == cls for s in samples])
        if cls_mask.sum() > 0:
            result[f"class_{cls}"] = _stats(wx[cls_mask])

    # w(x) > threshold 비율 (학습 시 실질적으로 기여하는 샘플 비율)
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        result[f"pct_above_{threshold}"] = round(float((wx > threshold).mean()), 4)

    return result


def _stats(arr):
    """기본 통계량"""
    if len(arr) == 0:
        return None
    return {
        "n": int(len(arr)),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
        "max": round(float(arr.max()), 4),
    }


def msp_vs_gtprob_comparison(samples):
    """MSP vs 정답확률 차이가 큰 케이스 (confident but wrong) 분석"""
    cases = []
    for s in samples:
        msp = s["confidence"]
        gt_prob = s["scores"][s["true_label"]]
        diff = msp - gt_prob  # MSP - 정답확률: 양수이면 "자신감 높지만 오답 방향"

        if abs(diff) > 0.3:  # 유의미한 차이만
            cases.append({
                "true": s["true_label"],
                "pred": s["pred_label"],
                "msp": round(msp, 4),
                "gt_prob": round(gt_prob, 4),
                "diff": round(diff, 4),
                "w_msp_g1": round(1 - msp, 4),
                "w_gt_g1": round(1 - gt_prob, 4),
            })

    # "confident but wrong" 케이스 (MSP 높지만 정답확률 낮음)
    confident_wrong = [c for c in cases if c["diff"] > 0.3 and c["true"] != c["pred"]]

    return {
        "n_large_diff": len(cases),
        "n_confident_wrong": len(confident_wrong),
        "confident_wrong_examples": sorted(confident_wrong, key=lambda x: -x["diff"])[:10],
        "summary": "정답확률 w(x)는 confident-but-wrong 케이스에서 높은 가중치 부여 → 전문가 재학습에 핵심"
    }


def main():
    all_results = {}

    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds}")
        print(f"{'='*60}")

        samples = load_dataset(ds)
        n = len(samples)

        # 1. 기본 accuracy
        n_correct = sum(1 for s in samples if s["pred_label"] == s["true_label"])
        accuracy = n_correct / n
        print(f"\n[1] Accuracy: {accuracy:.4f} ({n_correct}/{n})")

        # 2. Confusion Matrix
        cm = compute_confusion_matrix(samples)
        print(f"\n[2] Confusion Matrix:")
        for true_cls in CLASSES:
            if true_cls in cm:
                row = cm[true_cls]
                counts = [f"{row.get(pred_cls, 0):4d}" for pred_cls in CLASSES]
                print(f"  {true_cls:15s} → {' | '.join(counts)}")

        # 3. ECE (MSP vs 정답확률)
        msp_ece, msp_bins = compute_msp_ece(samples)
        gt_ece, gt_bins = compute_gtprob_ece(samples)
        print(f"\n[3] ECE:")
        print(f"  MSP-based:     {msp_ece:.4f}")
        print(f"  정답확률-based: {gt_ece:.4f}")

        # 4. Error Analysis
        error_analysis = compute_error_analysis(samples)
        print(f"\n[4] Error Analysis:")
        print(f"  Error rate: {error_analysis['error_rate']:.4f} ({error_analysis['n_errors']}/{error_analysis['n_total']})")
        for pattern, info in error_analysis["error_patterns"].items():
            print(f"  {pattern}: {info['count']}건, MSP={info['msp_mean']:.3f}±{info['msp_std']:.3f}, GT={info['gtprob_mean']:.3f}±{info['gtprob_std']:.3f}")

        # 5. MSP vs 정답확률 comparison
        comparison = msp_vs_gtprob_comparison(samples)
        print(f"\n[5] MSP vs 정답확률 비교:")
        print(f"  |diff| > 0.3인 케이스: {comparison['n_large_diff']}건")
        print(f"  Confident-but-wrong: {comparison['n_confident_wrong']}건")
        if comparison["confident_wrong_examples"]:
            print(f"  대표 예시:")
            for ex in comparison["confident_wrong_examples"][:3]:
                print(f"    {ex['true']}→{ex['pred']}: MSP={ex['msp']}, GT_prob={ex['gt_prob']}, w(MSP)={ex['w_msp_g1']}, w(GT)={ex['w_gt_g1']}")

        # 6. w(x) distribution per gamma
        print(f"\n[6] w(x) = (1-P(y_true))^γ 분포:")
        wx_results = {}
        for gamma in GAMMAS:
            wx = compute_wx_distribution(samples, gamma)
            wx_results[f"gamma_{gamma}"] = wx
            print(f"  γ={gamma}: mean={wx['all']['mean']:.4f}, median={wx['all']['median']:.4f}, "
                  f"pct>0.3={wx['pct_above_0.3']:.1%}, pct>0.5={wx['pct_above_0.5']:.1%}")
            if wx["error"]:
                print(f"         error: mean={wx['error']['mean']:.4f}, median={wx['error']['median']:.4f}")
            if wx["correct"]:
                print(f"         correct: mean={wx['correct']['mean']:.4f}, median={wx['correct']['median']:.4f}")

        all_results[ds] = {
            "n_samples": n,
            "accuracy": round(accuracy, 4),
            "confusion_matrix": cm,
            "ece": {
                "msp": {"ece": round(msp_ece, 4), "bins": msp_bins},
                "gt_prob": {"ece": round(gt_ece, 4), "bins": gt_bins},
            },
            "error_analysis": error_analysis,
            "msp_vs_gtprob": comparison,
            "wx_distribution": wx_results,
        }

    # ── 전체 요약 ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY ACROSS DATASETS")
    print(f"{'='*60}")

    print(f"\n{'Dataset':12s} | {'Acc':6s} | {'ECE_MSP':8s} | {'ECE_GT':8s} | {'Errors':6s} | {'w(γ=2) mean':12s} | {'w(γ=2) err':12s}")
    print("-" * 80)
    for ds in DATASETS:
        r = all_results[ds]
        w2 = r["wx_distribution"]["gamma_2"]
        w2_err = w2["error"]["mean"] if w2["error"] else "N/A"
        w2_err_str = f"{w2_err:.4f}" if isinstance(w2_err, float) else w2_err
        print(f"{ds:12s} | {r['accuracy']:.4f} | {r['ece']['msp']['ece']:.4f}   | {r['ece']['gt_prob']['ece']:.4f}   | {r['error_analysis']['n_errors']:4d}   | {w2['all']['mean']:.4f}       | {w2_err_str}")

    # ── Temperature Scaling 적용 가능성 판단 ──
    print(f"\n[Temperature Scaling 필요성 판단]")
    for ds in DATASETS:
        ece_msp = all_results[ds]["ece"]["msp"]["ece"]
        if ece_msp > 0.05:
            print(f"  {ds}: ECE={ece_msp:.4f} > 0.05 → TS 권장")
        else:
            print(f"  {ds}: ECE={ece_msp:.4f} ≤ 0.05 → TS 선택적")

    # ── 저장 ──
    output_path = OUTPUT_DIR / "mnv2_phase1_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_path}")

    # ── w(x) 분포 히스토그램 데이터 (matplotlib 없이도 분석 가능하도록) ──
    print(f"\n[w(x) 히스토그램 데이터 (γ=2, 10-bin)]")
    for ds in DATASETS:
        samples = load_dataset(ds)
        gt_probs = np.array([s["scores"][s["true_label"]] for s in samples])
        wx = (1 - gt_probs) ** 2
        hist, edges = np.histogram(wx, bins=10, range=(0, 1))
        pcts = hist / len(wx) * 100
        print(f"  {ds}:")
        for i in range(10):
            bar = "█" * int(pcts[i] / 2)
            print(f"    [{edges[i]:.1f}-{edges[i+1]:.1f}): {hist[i]:4d} ({pcts[i]:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
