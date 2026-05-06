#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.training import load_experiment_config, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ICS anomaly model.")
    parser.add_argument("--experiment", required=True, help="Path to an experiment YAML.")
    args = parser.parse_args()

    cfg, _ = load_experiment_config(args.experiment)
    result = train_model(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
