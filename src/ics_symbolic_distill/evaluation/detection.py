from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def residual_scores(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    residual = np.asarray(targets, dtype=np.float64) - np.asarray(preds, dtype=np.float64)
    if residual.ndim < 2:
        raise ValueError("preds/targets must have at least batch and feature dimensions")
    return np.mean(residual.reshape(residual.shape[0], -1) ** 2, axis=1)


def static_quantile_threshold(scores: np.ndarray, *, alpha: float = 0.01) -> float:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot fit threshold on empty score array")
    q = float(np.clip(1.0 - float(alpha), 0.0, 1.0))
    return float(np.quantile(arr, q))


def detection_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = (np.asarray(labels, dtype=np.float64).reshape(-1) >= 0.5).astype(np.int64)
    if s.shape[0] != y.shape[0]:
        raise ValueError(f"scores/labels length mismatch: {s.shape[0]} vs {y.shape[0]}")

    alarms = (s > float(threshold)).astype(np.int64)
    tp = int(((alarms == 1) & (y == 1)).sum())
    fp = int(((alarms == 1) & (y == 0)).sum())
    tn = int(((alarms == 0) & (y == 0)).sum())
    fn = int(((alarms == 0) & (y == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)

    out = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }

    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))

    attack_starts = np.flatnonzero((y[1:] == 1) & (y[:-1] == 0)) + 1
    if y.size > 0 and y[0] == 1:
        attack_starts = np.concatenate(([0], attack_starts))
    delays: list[int] = []
    for start in attack_starts:
        end = start
        while end < y.size and y[end] == 1:
            end += 1
        alarm_offsets = np.flatnonzero(alarms[start:end] == 1)
        if alarm_offsets.size > 0:
            delays.append(int(alarm_offsets[0]))
    if delays:
        out["detection_delay"] = float(np.mean(delays))
    return out
