from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def to_intervals(arr: Sequence[int] | np.ndarray) -> list[tuple[int, int]]:
    """Convert a binary sequence into inclusive ``(start, end)`` intervals."""

    values = np.asarray(arr, dtype=np.int64).reshape(-1)
    intervals: list[tuple[int, int]] = []
    in_interval = False
    start = 0
    for t, value in enumerate(values):
        if value == 1 and not in_interval:
            start = int(t)
            in_interval = True
        elif value == 0 and in_interval:
            intervals.append((start, int(t - 1)))
            in_interval = False
    if in_interval:
        intervals.append((start, int(values.shape[0] - 1)))
    return intervals


def _overlap_len(left: tuple[int, int], right: tuple[int, int]) -> int:
    start = max(int(left[0]), int(right[0]))
    end = min(int(left[1]), int(right[1]))
    return max(0, end - start + 1)


def _total_overlap(interval: tuple[int, int], others: Sequence[tuple[int, int]]) -> int:
    return int(sum(_overlap_len(interval, other) for other in others))


def _fallback_etapr(
    labels: np.ndarray,
    alarms: np.ndarray,
    *,
    theta_p: float = 0.5,
    theta_r: float = 0.1,
) -> dict[str, float]:
    attacks = to_intervals(labels)
    predictions = to_intervals(alarms)

    if attacks:
        recall_terms = []
        for attack in attacks:
            length = attack[1] - attack[0] + 1
            portion = _total_overlap(attack, predictions) / max(length, 1)
            detected = 1.0 if portion >= theta_r else 0.0
            recall_terms.append((detected + detected * portion) / 2.0)
        etar = float(np.mean(recall_terms))
    else:
        etar = 0.0

    if predictions:
        weights_raw = np.asarray([sqrt(pred[1] - pred[0] + 1) for pred in predictions], dtype=np.float64)
        weights = weights_raw / max(float(weights_raw.sum()), 1e-12)
        precision_terms = []
        for pred in predictions:
            length = pred[1] - pred[0] + 1
            portion = _total_overlap(pred, attacks) / max(length, 1)
            detected = 1.0 if portion >= theta_p else 0.0
            precision_terms.append((detected + detected * portion) / 2.0)
        etap = float(np.sum(np.asarray(precision_terms, dtype=np.float64) * weights))
    else:
        etap = 0.0

    etaf1 = 2.0 * etap * etar / max(etap + etar, 1e-10)
    return {"eTaP": etap, "eTaR": etar, "eTaF1": etaf1}


def etapr_metrics(labels: np.ndarray, alarms: np.ndarray) -> dict[str, float]:
    """Return eTaPR metrics as fractions, using faster-etapr when available."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    pred = np.asarray(alarms, dtype=np.int64).reshape(-1)
    try:
        from faster_etapr import evaluate_from_time_series

        result = evaluate_from_time_series(
            anomalies=y.tolist(),
            predictions=pred.tolist(),
            theta_p=0.5,
            theta_r=0.1,
            delta=0,
        )
        return {
            "eTaP": float(result["eTaP"]),
            "eTaR": float(result["eTaR"]),
            "eTaF1": float(result["eTaF1"]),
        }
    except Exception:
        return _fallback_etapr(y, pred, theta_p=0.5, theta_r=0.1)


def false_positive_alarms(
    labels: np.ndarray,
    alarms: np.ndarray,
    *,
    expand_steps: int = 6,
) -> int:
    """Count predicted alarm intervals that do not overlap expanded attacks."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = int(y.shape[0])
    attack_intervals = to_intervals(y)
    expanded = [
        (max(0, start - int(expand_steps)), min(n - 1, end + int(expand_steps)))
        for start, end in attack_intervals
    ]
    count = 0
    for pred in to_intervals(alarms):
        if all(_overlap_len(pred, attack) == 0 for attack in expanded):
            count += 1
    return int(count)


def scenario_detection_rate(labels: np.ndarray, alarms: np.ndarray) -> float:
    """Fraction of attack intervals with at least one alarm inside the interval."""

    attacks = to_intervals(labels)
    if not attacks:
        return 0.0
    pred = np.asarray(alarms, dtype=np.int64).reshape(-1)
    detected = 0
    for start, end in attacks:
        if np.any(pred[start : end + 1] == 1):
            detected += 1
    return float(detected / len(attacks))


def compute_detection_metrics(
    labels: np.ndarray,
    system_alarm: np.ndarray,
    *,
    expand_steps: int = 6,
) -> dict[str, float]:
    """Compute point, eTaPR, FPA, and scenario metrics as report percentages."""

    y = (np.asarray(labels, dtype=np.float64).reshape(-1) >= 0.5).astype(np.int64)
    pred = (np.asarray(system_alarm, dtype=np.float64).reshape(-1) >= 0.5).astype(np.int64)
    if y.shape[0] != pred.shape[0]:
        raise ValueError(f"labels/system_alarm length mismatch: {y.shape[0]} vs {pred.shape[0]}")

    point_precision = precision_score(y, pred, zero_division=0) * 100.0
    point_recall = recall_score(y, pred, zero_division=0) * 100.0
    point_f1 = f1_score(y, pred, zero_division=0) * 100.0
    etap = etapr_metrics(y, pred)
    attacks = to_intervals(y)
    return {
        "point_precision": float(point_precision),
        "point_recall": float(point_recall),
        "point_f1": float(point_f1),
        "eTaP": float(etap["eTaP"] * 100.0),
        "eTaR": float(etap["eTaR"] * 100.0),
        "eTaF1": float(etap["eTaF1"] * 100.0),
        "FPA": float(false_positive_alarms(y, pred, expand_steps=int(expand_steps))),
        "scenario_detection_rate": float(scenario_detection_rate(y, pred) * 100.0),
        "attack_interval_count": float(len(attacks)),
    }
