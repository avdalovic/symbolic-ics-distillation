#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.evaluation import (
    detection_metrics,
    residual_scores,
    static_quantile_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate residual-threshold detection.")
    parser.add_argument("--val-preds", required=True)
    parser.add_argument("--val-targets", required=True)
    parser.add_argument("--test-preds", required=True)
    parser.add_argument("--test-targets", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    val_scores = residual_scores(np.load(args.val_preds), np.load(args.val_targets))
    test_scores = residual_scores(np.load(args.test_preds), np.load(args.test_targets))
    labels = np.load(args.test_labels)
    threshold = static_quantile_threshold(val_scores, alpha=float(args.alpha))
    metrics = detection_metrics(test_scores, labels, threshold)

    text = json.dumps(metrics, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
