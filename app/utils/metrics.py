"""Evaluation metrics utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import log2


def f1_window(y_true: Sequence[bool], y_pred: Sequence[bool], window_tolerance: int) -> float:
    """Compute F1 score allowing matches within a tolerance window.

    Args:
        y_true: Ground truth sequence of anomaly indicators.
        y_pred: Predicted anomaly indicators.
        window_tolerance: Allowable distance between predictions and ground-truth.

    Returns:
        F1 score computed with tolerance.

    Raises:
        ValueError: When the input sequences differ in length.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    matched_true = set()
    true_positive = 0
    for idx, predicted in enumerate(y_pred):
        if not predicted:
            continue
        start = max(0, idx - window_tolerance)
        end = min(len(y_true), idx + window_tolerance + 1)
        window = range(start, end)
        for j in window:
            if y_true[j] and j not in matched_true:
                matched_true.add(j)
                true_positive += 1
                break
    predicted_positive = sum(y_pred)
    actual_positive = sum(y_true)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def auc_pr(y_true_scores: Iterable[tuple[int, float]]) -> float:
    """Compute the area under the precision-recall curve.

    Args:
        y_true_scores: Iterable of (label, score) pairs sorted descending.

    Returns:
        Area under the precision-recall curve.
    """

    pairs = sorted(list(y_true_scores), key=lambda item: item[1], reverse=True)
    if not pairs:
        return 0.0
    positives = sum(label for label, _ in pairs)
    if positives == 0:
        return 0.0
    precision_recall = []
    tp = 0
    fp = 0
    for label, _score in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / positives
        precision_recall.append((recall, precision))
    auc = 0.0
    prev_recall = 0.0
    for recall, precision in precision_recall:
        auc += precision * (recall - prev_recall)
        prev_recall = recall
    return auc


def coverage_conformal(violations: int, total: int) -> float:
    """Compute conformal prediction coverage.

    Args:
        violations: Number of threshold violations.
        total: Total number of observations.

    Returns:
        Empirical coverage rate.
    """

    if total <= 0:
        return 0.0
    return 1 - (violations / total)


def ndcg_at_k(true_causes: Iterable[str], ranking: Sequence[tuple[str, float]], k: int) -> float:
    """Compute Normalised Discounted Cumulative Gain at K.

    Args:
        true_causes: Iterable of ground-truth root causes.
        ranking: Ranked list of (service, score) tuples.
        k: Cut-off rank.

    Returns:
        NDCG@k score in [0, 1].
    """

    true_set = set(true_causes)
    if not ranking or k <= 0:
        return 0.0
    k = min(k, len(ranking))
    dcg = 0.0
    for idx in range(k):
        label = ranking[idx][0]
        gain = 1.0 if label in true_set else 0.0
        dcg += gain / log2(idx + 2)
    ideal_hits = min(len(true_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
