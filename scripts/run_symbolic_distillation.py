#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder for symbolic distillation.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    target = cfg.get("distill", {}).get("target", "unspecified")
    print(f"Symbolic distillation is not implemented yet. Config target: {target}")


if __name__ == "__main__":
    main()
