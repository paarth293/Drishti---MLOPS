"""Tests for metrics utilities."""

from __future__ import annotations

import math

from app.utils.metrics import auc_pr, coverage_conformal, f1_window, ndcg_at_k


def test_f1_window_matches_expected() -> None:
    truth = [False, False, True, False, False, True, False]
    preds = [False, True, False, False, True, False, False]
    score = f1_window(truth, preds, window_tolerance=1)
    assert math.isclose(score, 1.0)


def test_auc_pr_simple_case() -> None:
    pairs = [(1, 0.9), (0, 0.8), (1, 0.7), (0, 0.4)]
    result = auc_pr(pairs)
    assert 0 <= result <= 1


def test_conformal_coverage() -> None:
    assert math.isclose(coverage_conformal(violations=5, total=10), 0.5)
    assert coverage_conformal(violations=0, total=0) == 0.0


def test_ndcg_at_k() -> None:
    true_causes = ["checkout", "inventory"]
    ranking = [("checkout", 0.9), ("payment", 0.5), ("inventory", 0.4)]
    score = ndcg_at_k(true_causes, ranking, k=3)
    assert 0 < score <= 1
