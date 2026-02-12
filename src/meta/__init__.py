"""
DAAC (Disagreement-Aware Adaptive Consensus) 메타 분류기 모듈

에이전트 간 불일치 패턴을 메타 특징으로 활용하여
단순 가중 투표 대비 개선된 합의를 도출한다.

Phase 1: Disagreement Pattern Analysis (시뮬레이션 → 실데이터)
Phase 2: Adaptive Routing (조건부)

모듈 구성:
    simulator  - confusion matrix 기반 합성 에이전트 출력 생성
    features   - 43-dim 메타 특징 추출
    baselines  - majority vote, COBRA 래퍼 베이스라인
    trainer    - LogReg, XGBoost, MLP 메타 분류기 학습
    evaluate   - Macro-F1, AUROC, ECE, McNemar, bootstrap CI
    ablation   - 6가지 feature set ablation 실행기
"""
from .features import MetaFeatureExtractor
from .simulator import AgentSimulator
from .baselines import MajorityVoteBaseline, COBRABaseline
from .trainer import MetaTrainer
from .evaluate import MetaEvaluator
from .ablation import AblationRunner

__all__ = [
    "MetaFeatureExtractor",
    "AgentSimulator",
    "MajorityVoteBaseline",
    "COBRABaseline",
    "MetaTrainer",
    "MetaEvaluator",
    "AblationRunner",
]
