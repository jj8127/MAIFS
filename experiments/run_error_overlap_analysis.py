"""
Error Overlap Analysis: MNV2 vs CLIP
두 모델의 에러 패턴 분석:
1. 에러 overlap 비율 (Jaccard, 집합 기반)
2. 에러 클래스 분포 (어떤 클래스에서 주로 틀리는가)
3. 에러 타입 분류 (both_wrong / only_mnv2 / only_clip / both_correct)
4. 에러 이미지의 confidence 분포
5. 클래스별 에러 overlap 세부 분석
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
RESULTS = BASE / "experiments" / "results" / "backbone_eval"

# 가장 최신 파일 (070725 = MNV2 15ep, 061834 = CLIP ft4)
DATASETS = ["base", "dsC", "opensdi", "aigenproxy"]

MNV2_FILES = {ds: RESULTS / f"mobilenetv2_dualstream_{ds}_20260319_070725.jsonl" for ds in DATASETS}
CLIP_FILES = {ds: RESULTS / f"mobileclip_s2_finetuned_{ds}_20260319_061834.jsonl" for ds in DATASETS}

# ── 유틸 ──────────────────────────────────────────────────────────────
def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            key = r["image_path"]
            records[key] = r
    return records


def analyze_overlap(mnv2_recs, clip_recs, dataset_name):
    common_keys = set(mnv2_recs.keys()) & set(clip_recs.keys())
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  |  공통 이미지: {len(common_keys)}")
    print(f"{'='*60}")

    # 에러 집합
    mnv2_errors = {k for k in common_keys if mnv2_recs[k]["pred_label"] != mnv2_recs[k]["true_label"]}
    clip_errors  = {k for k in common_keys if clip_recs[k]["pred_label"]  != clip_recs[k]["true_label"]}

    both_wrong   = mnv2_errors & clip_errors
    only_mnv2    = mnv2_errors - clip_errors
    only_clip    = clip_errors - mnv2_errors
    both_correct = common_keys - mnv2_errors - clip_errors

    total = len(common_keys)
    union = mnv2_errors | clip_errors
    jaccard = len(both_wrong) / len(union) if union else 0.0

    print(f"\n[에러 집합 통계]")
    print(f"  MNV2 에러:         {len(mnv2_errors):4d} / {total} ({100*len(mnv2_errors)/total:.1f}%)")
    print(f"  CLIP 에러:         {len(clip_errors):4d} / {total} ({100*len(clip_errors)/total:.1f}%)")
    print(f"  둘 다 틀림:        {len(both_wrong):4d} / {total} ({100*len(both_wrong)/total:.1f}%)")
    print(f"  MNV2만 틀림:       {len(only_mnv2):4d} / {total} ({100*len(only_mnv2)/total:.1f}%)")
    print(f"  CLIP만 틀림:       {len(only_clip):4d} / {total} ({100*len(only_clip)/total:.1f}%)")
    print(f"  둘 다 맞음:        {len(both_correct):4d} / {total} ({100*len(both_correct)/total:.1f}%)")
    print(f"  Jaccard(에러겹침): {jaccard:.4f}  (1=완전겹침, 0=완전독립)")

    # 에러 클래스 분포 ─ true_label별
    def class_dist(error_set):
        dist = defaultdict(int)
        for k in error_set:
            dist[mnv2_recs[k]["true_label"]] += 1
        return dict(dist)

    print(f"\n[에러 true_label 분포]")
    print(f"  {'클래스':<20} {'MNV2에러':>10} {'CLIP에러':>10} {'공통에러':>10}")
    for cls in ["authentic", "manipulated", "ai_generated"]:
        mnv2_c = sum(1 for k in mnv2_errors if mnv2_recs[k]["true_label"] == cls)
        clip_c  = sum(1 for k in clip_errors  if clip_recs[k]["true_label"]  == cls)
        both_c  = sum(1 for k in both_wrong  if mnv2_recs[k]["true_label"] == cls)
        print(f"  {cls:<20} {mnv2_c:>10} {clip_c:>10} {both_c:>10}")

    # 혼동 패턴: true → pred
    print(f"\n[MNV2 에러 혼동 패턴 (true→pred)]")
    conf_mnv2 = defaultdict(int)
    for k in mnv2_errors:
        conf_mnv2[(mnv2_recs[k]["true_label"], mnv2_recs[k]["pred_label"])] += 1
    for (t, p), cnt in sorted(conf_mnv2.items(), key=lambda x: -x[1]):
        print(f"  {t} → {p}: {cnt}")

    print(f"\n[CLIP 에러 혼동 패턴 (true→pred)]")
    conf_clip = defaultdict(int)
    for k in clip_errors:
        conf_clip[(clip_recs[k]["true_label"], clip_recs[k]["pred_label"])] += 1
    for (t, p), cnt in sorted(conf_clip.items(), key=lambda x: -x[1]):
        print(f"  {t} → {p}: {cnt}")

    # 독립 에러 심층 분석 (ASSIST 가능성 평가)
    print(f"\n[ASSIST 가능성 평가]")
    print(f"  MNV2만 틀리는 이미지 {len(only_mnv2)}개 → CLIP이 맞춘다 → CLIP-ASSIST 불필요, CLIP이 보완 가능")
    print(f"  CLIP만 틀리는 이미지 {len(only_clip)}개 → MNV2가 맞춘다 → MNV2-ASSIST 불필요, MNV2가 보완 가능")
    print(f"  둘 다 틀리는 이미지  {len(both_wrong)}개 → 어떤 보완 모델이 필요한 대상")

    # 둘 다 틀리는 이미지의 클래스별 분포 상세
    if both_wrong:
        print(f"\n[둘 다 틀리는 이미지 - 클래스별 혼동 패턴]")
        conf_both = defaultdict(int)
        for k in both_wrong:
            true_l = mnv2_recs[k]["true_label"]
            pred_mnv2 = mnv2_recs[k]["pred_label"]
            pred_clip  = clip_recs[k]["pred_label"]
            conf_both[(true_l, pred_mnv2, pred_clip)] += 1
        for (t, pm, pc), cnt in sorted(conf_both.items(), key=lambda x: -x[1]):
            print(f"  true={t}, MNV2→{pm}, CLIP→{pc}: {cnt}개")

    return {
        "dataset": dataset_name,
        "total": total,
        "mnv2_errors": len(mnv2_errors),
        "clip_errors": len(clip_errors),
        "both_wrong": len(both_wrong),
        "only_mnv2": len(only_mnv2),
        "only_clip": len(only_clip),
        "both_correct": len(both_correct),
        "jaccard": jaccard,
        "mnv2_error_rate": len(mnv2_errors) / total,
        "clip_error_rate": len(clip_errors) / total,
        "both_wrong_rate": len(both_wrong) / total,
        "independent_error_rate": (len(only_mnv2) + len(only_clip)) / total,
    }


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    print("MNV2 vs CLIP 에러 Overlap 분석")
    print("목적: ASSIST 모델 필요성 및 설계 방향 결정")

    all_results = []

    for ds in DATASETS:
        mnv2_path = MNV2_FILES[ds]
        clip_path  = CLIP_FILES[ds]

        if not mnv2_path.exists():
            print(f"[SKIP] {ds}: MNV2 파일 없음 ({mnv2_path})")
            continue
        if not clip_path.exists():
            print(f"[SKIP] {ds}: CLIP 파일 없음 ({clip_path})")
            continue

        mnv2_recs = load_jsonl(mnv2_path)
        clip_recs  = load_jsonl(clip_path)
        result = analyze_overlap(mnv2_recs, clip_recs, ds)
        all_results.append(result)

    # ── 전체 요약 ──────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*60}")
        print("전체 데이터셋 요약")
        print(f"{'='*60}")
        print(f"\n{'데이터셋':<12} {'MNV2에러%':>10} {'CLIP에러%':>10} {'공통에러%':>10} {'Jaccard':>10} {'독립에러%':>10}")
        for r in all_results:
            print(f"{r['dataset']:<12} "
                  f"{100*r['mnv2_error_rate']:>9.1f}% "
                  f"{100*r['clip_error_rate']:>9.1f}% "
                  f"{100*r['both_wrong_rate']:>9.1f}% "
                  f"{r['jaccard']:>10.4f} "
                  f"{100*r['independent_error_rate']:>9.1f}%")

        avg_jaccard = sum(r["jaccard"] for r in all_results) / len(all_results)
        avg_indep   = sum(r["independent_error_rate"] for r in all_results) / len(all_results)
        print(f"\n평균 Jaccard:     {avg_jaccard:.4f}")
        print(f"평균 독립에러율:  {100*avg_indep:.1f}%")
        print()
        print("[해석 가이드]")
        print("  Jaccard > 0.7 → 에러가 거의 겹침 → ASSIST 모델도 비슷해질 위험")
        print("  Jaccard < 0.3 → 에러가 독립적 → ASSIST 아이디어 효과적")
        print("  독립에러율 높음 → 두 모델이 서로 보완 가능한 구조")
        print()

        # ASSIST 방향 판단
        print("[ASSIST 설계 방향 결론]")
        if avg_jaccard > 0.6:
            print("  ⚠ Jaccard 높음 → 동일 이미지를 틀릴 가능성 큼")
            print("  → ASSIST 모델은 다른 입력 도메인 (DCT/frequency) 활용 권장")
            print("  → 또는 binary specialist 구조 (authentic vs NOT) 고려")
        elif avg_jaccard > 0.3:
            print("  ~ Jaccard 중간 → 부분적 보완 가능")
            print("  → confidence 기반 hard example mining으로 ASSIST 학습 시도 가능")
        else:
            print("  ✓ Jaccard 낮음 → 에러가 독립적 → ASSIST 아이디어 효과적")
            print("  → 에러 집합 기반 직접 학습 가능")

        # 결과 저장
        out_path = BASE / "experiments" / "results" / "error_overlap_analysis.json"
        with open(out_path, "w") as f:
            json.dump({
                "description": "MNV2 vs CLIP per-image error overlap analysis",
                "summary": {
                    "avg_jaccard": avg_jaccard,
                    "avg_independent_error_rate": avg_indep,
                },
                "per_dataset": all_results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
