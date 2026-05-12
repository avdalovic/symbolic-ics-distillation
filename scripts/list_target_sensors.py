#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_target_sensors(distill_dir: str | Path) -> list[str]:
    path = Path(distill_dir) / "distill_target_columns.json"
    targets = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise ValueError(f"Expected a JSON string list in {path}")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Print predicted target sensors from a distillation export directory.")
    parser.add_argument("--distill-dir", default="artifacts/model_exports/swat/distillation/val20_overlap")
    parser.add_argument("--format", choices=["space", "json", "lines"], default="space")
    args = parser.parse_args()

    targets = load_target_sensors(args.distill_dir)
    if args.format == "json":
        print(json.dumps(targets))
    elif args.format == "lines":
        print("\n".join(targets))
    else:
        print(" ".join(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
