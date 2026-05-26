from __future__ import annotations

from collections.abc import Iterable, Sequence

from omegaconf import DictConfig

WADI_PROBLEMATIC_FEATURES: tuple[str, ...] = (
    "2B_AIT_002_PV",
    "2_LS_001_AL",
    "2_LS_002_AL",
    "2_P_001_STATUS",
    "2_P_002_STATUS",
    "LEAK_DIFF_PRESSURE",
    "PLANT_START_STOP_LOG",
    "TOTAL_CONS_REQUIRED_FLOW",
)

WADI_NAN_ONLY_FEATURES: tuple[str, ...] = (
    "2_LS_001_AL",
    "2_LS_002_AL",
    "2_P_001_STATUS",
    "2_P_002_STATUS",
)

WADI_OPERATIONAL_NONPROCESS_FEATURES: tuple[str, ...] = (
    "LEAK_DIFF_PRESSURE",
    "PLANT_START_STOP_LOG",
    "TOTAL_CONS_REQUIRED_FLOW",
)

WADI_DROP_POLICIES: dict[str, tuple[str, ...]] = {
    "none": tuple(),
    "nan_only": WADI_NAN_ONLY_FEATURES,
    "nan_and_operational_keep_2bait002": (
        *WADI_NAN_ONLY_FEATURES,
        *WADI_OPERATIONAL_NONPROCESS_FEATURES,
    ),
    "full_problematic": WADI_PROBLEMATIC_FEATURES,
}


def is_wadi_dataset(cfg: DictConfig) -> bool:
    return str(cfg.dataset.get("name", "")).strip().lower() == "wadi"


def resolve_wadi_drop_columns(cfg: DictConfig) -> set[str]:
    if not is_wadi_dataset(cfg):
        return set()

    drop_cols: set[str] = set()
    policy = cfg.dataset.get("drop_policy")
    if policy is not None:
        policy_name = str(policy).strip().lower()
        if policy_name not in WADI_DROP_POLICIES:
            valid = ", ".join(sorted(WADI_DROP_POLICIES.keys()))
            raise ValueError(f"Unknown WADI drop_policy={policy}. Valid: {valid}")
        drop_cols.update(WADI_DROP_POLICIES[policy_name])

    if bool(cfg.dataset.get("drop_problematic_features", True)):
        drop_cols.update(WADI_PROBLEMATIC_FEATURES)

    configured = cfg.dataset.get("drop_columns")
    if configured is None:
        return drop_cols

    if isinstance(configured, str):
        drop_cols.update(tok.strip() for tok in configured.split(",") if tok.strip())
        return drop_cols

    if isinstance(configured, Sequence):
        drop_cols.update(str(col).strip() for col in configured if str(col).strip())
        return drop_cols

    return drop_cols


def apply_drop_columns(columns: Iterable[str], drop_cols: set[str]) -> list[str]:
    if not drop_cols:
        return [str(c) for c in columns]
    return [str(c) for c in columns if str(c) not in drop_cols]
