from __future__ import annotations

import numpy as np

from ics_symbolic_distill.detection.metrics import (
    compute_detection_metrics,
    etapr_metrics,
    false_positive_alarms,
    scenario_detection_rate,
)


def test_false_positive_alarm_expansion() -> None:
    labels = np.zeros(40, dtype=int)
    labels[10:13] = 1
    alarms = np.zeros(40, dtype=int)
    alarms[5:6] = 1  # overlaps expanded attack [4, 18]
    alarms[30:32] = 1
    assert false_positive_alarms(labels, alarms, expand_steps=6) == 1


def test_scenario_detection_rate_detected_and_missed() -> None:
    labels = np.zeros(20, dtype=int)
    labels[2:5] = 1
    labels[10:13] = 1
    alarms = np.zeros(20, dtype=int)
    alarms[3] = 1
    assert scenario_detection_rate(labels, alarms) == 0.5


def test_etapr_fallback_perfect_and_empty_predictions() -> None:
    labels = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=int)
    perfect = etapr_metrics(labels, labels)
    assert perfect["eTaP"] > 0.99
    assert perfect["eTaR"] > 0.99
    assert perfect["eTaF1"] > 0.99

    empty = etapr_metrics(labels, np.zeros_like(labels))
    assert empty["eTaP"] == 0.0
    assert empty["eTaR"] == 0.0
    assert empty["eTaF1"] == 0.0


def test_compute_detection_metrics_percentages() -> None:
    labels = np.array([0, 1, 1, 0], dtype=int)
    alarms = np.array([0, 1, 0, 0], dtype=int)
    metrics = compute_detection_metrics(labels, alarms)
    assert metrics["point_precision"] == 100.0
    assert metrics["point_recall"] == 50.0
    assert metrics["point_f1"] > 60.0
