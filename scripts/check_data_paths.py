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

from ics_symbolic_distill.training import load_experiment_config


def _load_config(path: Path):
    payload = OmegaConf.load(path)
    if "experiment" in payload:
        cfg, _ = load_experiment_config(path)
        return cfg
    if "dataset" in payload:
        return payload
    raise ValueError("Config must be an experiment YAML or a dataset YAML")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check configured raw data paths.")
    parser.add_argument("--config", required=True, help="Experiment or dataset YAML path.")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_config(cfg_path)

    train_path = Path(str(cfg.dataset.train_csv)).expanduser()
    test_path = Path(str(cfg.dataset.test_csv)).expanduser()
    if not train_path.is_absolute():
        train_path = REPO_ROOT / train_path
    if not test_path.is_absolute():
        test_path = REPO_ROOT / test_path

    checks = [
        ("train_csv", train_path),
        ("test_csv", test_path),
    ]
    missing = 0
    for label, path in checks:
        exists = path.exists()
        missing += int(not exists)
        status = "OK" if exists else "MISSING"
        print(f"{status} {label}: {path}")

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
