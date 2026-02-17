"""
Path A repeated split 실행기 유틸 단위 테스트.
"""

from __future__ import annotations

import math

from experiments.run_phase2_patha_repeated import _pooled_mcnemar_from_runs
from experiments.run_phase2_patha_repeated import _resolve_kfold_split_seeds
from experiments.run_phase2_patha_repeated import _resolve_split_seeds
from experiments.run_phase2_patha_repeated import _resolve_test_folds


def test_resolve_split_seeds_auto_range():
    seeds = _resolve_split_seeds(raw="", n_repeats=4, split_seed_start=300)
    assert seeds == [300, 301, 302, 303]


def test_resolve_split_seeds_from_csv():
    seeds = _resolve_split_seeds(raw="10, 12,13", n_repeats=99, split_seed_start=0)
    assert seeds == [10, 12, 13]


def test_resolve_test_folds_default():
    folds = _resolve_test_folds(raw="", k_folds=5)
    assert folds == [0, 1, 2, 3, 4]


def test_resolve_test_folds_from_csv_with_wrap():
    folds = _resolve_test_folds(raw="4,5,7", k_folds=5)
    assert folds == [4, 0, 2]


def test_resolve_kfold_split_seeds_default_single():
    seeds = _resolve_kfold_split_seeds(raw="", default_seed=300)
    assert seeds == [300]


def test_resolve_kfold_split_seeds_from_csv():
    seeds = _resolve_kfold_split_seeds(raw="300,301", default_seed=999)
    assert seeds == [300, 301]


def test_pooled_mcnemar_from_runs():
    pooled = _pooled_mcnemar_from_runs(
        [
            {"mcnemar_b": 10, "mcnemar_c": 14},
            {"mcnemar_b": 3, "mcnemar_c": 7},
        ]
    )
    assert pooled is not None
    b, c, stat, pval = pooled
    assert b == 13
    assert c == 21
    assert math.isclose(stat, 1.4411764705882353, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(pval, 0.22994905679421346, rel_tol=1e-9, abs_tol=1e-12)
