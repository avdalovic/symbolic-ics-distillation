from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def get_device(requested: str | None) -> torch.device:
    name = str(requested or "cpu").lower()
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[Device] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)
