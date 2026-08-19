from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ics_symbolic_distill.data.ics_metadata import is_actuator
from ics_symbolic_distill.detection.swat1s_delta_sampling import coverage_stratified_indices


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_batadal_module():
    path = REPO_ROOT / "scripts" / "run_batadal_delta_full.py"
    spec = importlib.util.spec_from_file_location("run_batadal_delta_full", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_batadal_metadata_split_uses_status_tags_as_actuators() -> None:
    assert is_actuator("BATADAL", "S_PU1")
    assert is_actuator("BATADAL", "S_V2")
    assert not is_actuator("BATADAL", "L_T1")
    assert not is_actuator("BATADAL", "F_PU1")
    assert not is_actuator("BATADAL", "P_J280")


def test_batadal_processed_header_and_identifier_safety() -> None:
    csv_path = REPO_ROOT / "data" / "batadal" / "processed" / "train.csv"
    if not csv_path.exists():
        pytest.skip("BATADAL processed data is not present in this checkout")
    mod = load_batadal_module()
    train = pd.read_csv(csv_path, nrows=4)
    features = mod.process_columns(train)

    assert len(features) >= 40
    assert all(name.isidentifier() for name in features)
    assert "P_J280" in features
    assert any(is_actuator("BATADAL", name) for name in features)
    assert any(not is_actuator("BATADAL", name) for name in features)


def test_batadal_sample_size_all_uses_fit_pool() -> None:
    mod = load_batadal_module()
    assert mod.sample_size_for_fit("all", 7008) == 7008
    assert mod.sample_size_for_fit("20000", 7008) == 7008
    assert mod.sample_size_for_fit("100", 7008) == 100


def test_batadal_coverage_stratified_can_select_all_rows() -> None:
    n = 96
    p = 43
    x_current = np.zeros((n, p), dtype=float)
    x_next = x_current.copy()
    x_next[20, 10] = 1.0
    y_delta = np.sin(np.linspace(0.0, 4.0, n))

    idx, audit = coverage_stratified_indices(
        target="L_T1",
        y_delta_fit_pool=y_delta,
        x_current_fit_pool=x_current,
        x_next_fit_pool=x_next,
        actuator_indices=[10],
        sample_size=n,
        seed=1337,
        transition_radius=5,
    )

    np.testing.assert_array_equal(idx, np.arange(n))
    assert audit.sample_size_actual == n


def test_batadal_selected_equation_schema_constant() -> None:
    mod = load_batadal_module()
    required = {
        "target",
        "variable_type",
        "target_mode",
        "equation",
        "sympy_format",
        "complexity",
        "score",
        "holdout_r2",
        "residual_tail_ratio",
    }
    assert required.issubset(set(mod.SELECTED_EQUATION_COLUMNS))
