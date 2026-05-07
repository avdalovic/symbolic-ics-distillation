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

from ics_symbolic_distill.distillation import prepare_distillation_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare raw-unit, named distillation arrays from GRU and MLP model exports."
    )
    parser.add_argument("--gru-export", required=True, help="Path to the GRU export directory.")
    parser.add_argument("--mlp-export", required=True, help="Path to the MLP export directory.")
    parser.add_argument("--out", required=True, help="Output directory for distillation arrays.")
    parser.add_argument(
        "--dt-model-step",
        type=float,
        default=1.0,
        help="Time interval between downsampled model steps for rate features. Defaults to 1 when unknown.",
    )
    args = parser.parse_args()

    result = prepare_distillation_data(
        gru_export=args.gru_export,
        mlp_export=args.mlp_export,
        out=args.out,
        dt_model_step=args.dt_model_step,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
