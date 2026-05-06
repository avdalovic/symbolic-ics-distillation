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

from ics_symbolic_distill.inference import export_model_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model prediction arrays.")
    parser.add_argument("--checkpoint", default=None, help="Path to a .pth checkpoint.")
    parser.add_argument("--config", default=None, help="Path to resolved_config.yaml.")
    parser.add_argument("--experiment", default=None, help="Path to an experiment YAML.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--normal-only", action="store_true")
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    result = export_model_predictions(
        checkpoint=args.checkpoint,
        config=args.config,
        experiment=args.experiment,
        split=args.split,
        normal_only=bool(args.normal_only),
        out=args.out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
