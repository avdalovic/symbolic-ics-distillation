from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _config_root_from_experiment(experiment_path: Path) -> Path:
    if experiment_path.parent.name != "experiment":
        return experiment_path.parent.parent
    return experiment_path.parent.parent


def _merge_named(root: Path, section: str, name: str | None, cfg: DictConfig) -> DictConfig:
    if not name:
        return cfg
    path = root / section / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing {section} config: {path}")
    return OmegaConf.merge(cfg, OmegaConf.load(path))


def load_experiment_config(path: str | Path) -> tuple[DictConfig, Path]:
    exp_path = Path(path).expanduser().resolve()
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    payload = OmegaConf.load(exp_path)
    exp = payload.get("experiment", payload)
    root = _config_root_from_experiment(exp_path)

    cfg = OmegaConf.create()
    cfg = _merge_named(root, "dataset", exp.get("dataset_cfg"), cfg)
    cfg = _merge_named(root, "model", exp.get("model_cfg"), cfg)
    cfg = _merge_named(root, "train", exp.get("train_cfg"), cfg)
    cfg = _merge_named(root, "export", exp.get("export_cfg"), cfg)
    cfg = _merge_named(root, "evaluation", exp.get("evaluation_cfg"), cfg)

    overrides = exp.get("overrides") or []
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([str(x) for x in overrides]))

    cfg = OmegaConf.merge(cfg, {"experiment": OmegaConf.to_container(exp, resolve=True)})
    return cfg, exp_path


def load_resolved_config(path: str | Path) -> tuple[DictConfig, Path]:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return OmegaConf.load(cfg_path), cfg_path


def save_resolved_config(cfg: DictConfig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return out


def to_container(cfg: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved config must be a mapping")
    return payload
