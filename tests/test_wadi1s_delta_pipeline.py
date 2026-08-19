from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ics_symbolic_distill.data.ics_metadata import is_actuator
from ics_symbolic_distill.detection.swat1s_delta_sampling import coverage_stratified_indices, reconstruct_next_from_delta


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_wadi_full_module():
    path = REPO_ROOT / "scripts" / "run_wadi_1sec_delta_full.py"
    spec = importlib.util.spec_from_file_location("run_wadi_1sec_delta_full", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wadi_header_has_sensor_and_actuator_columns() -> None:
    csv_path = REPO_ROOT / "data" / "wadi" / "raw" / "wadi_train.csv"
    if not csv_path.exists():
        pytest.skip("WADI raw data is not present in this checkout")
    header = pd.read_csv(csv_path, nrows=2)
    features = header.drop(columns=[c for c in ["Row", "Attack"] if c in header.columns])
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    assert len(numeric_cols) >= 100
    assert any(is_actuator("WADI", col) for col in numeric_cols)
    assert any(not is_actuator("WADI", col) for col in numeric_cols)


def test_coverage_stratified_sampling_handles_wadi_dimensions() -> None:
    n = 500
    p = 123
    x_current = np.zeros((n, p), dtype=float)
    x_next = x_current.copy()
    x_next[50, 100] = 1.0
    x_next[300, 115] = 1.0
    y_delta = np.sin(np.linspace(0.0, 20.0, n))

    idx_a, audit_a = coverage_stratified_indices(
        target="1_FIT_001_PV",
        y_delta_fit_pool=y_delta,
        x_current_fit_pool=x_current,
        x_next_fit_pool=x_next,
        actuator_indices=[100, 115],
        sample_size=120,
        seed=1337,
    )
    idx_b, audit_b = coverage_stratified_indices(
        target="1_FIT_001_PV",
        y_delta_fit_pool=y_delta,
        x_current_fit_pool=x_current,
        x_next_fit_pool=x_next,
        actuator_indices=[100, 115],
        sample_size=120,
        seed=1337,
    )

    np.testing.assert_array_equal(idx_a, idx_b)
    assert audit_a.to_dict() == audit_b.to_dict()
    assert idx_a.size == np.unique(idx_a).size
    assert idx_a.min() >= 0
    assert idx_a.max() < n


def test_wadi_delta_reconstruction() -> None:
    current = np.array([10.0, 11.0, 12.0])
    delta_hat = np.array([0.25, -0.5, 0.0])
    np.testing.assert_allclose(reconstruct_next_from_delta(current, delta_hat), np.array([10.25, 10.5, 12.0]))


def test_wadi_selected_equation_schema_constant() -> None:
    mod = load_wadi_full_module()
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


def test_wadi_variable_name_mapping_roundtrips_digit_prefixed_tags() -> None:
    mod = load_wadi_full_module()
    feature_columns = ["1_FIT_001_PV", "2_LS_201_AL", "TOTAL_CONS_REQUIRED_FLOW"]
    original_to_safe, safe_to_original, safe_feature_columns = mod.make_variable_name_mapping(feature_columns)

    assert safe_feature_columns == ["V0", "V1", "V2"]
    assert original_to_safe["1_FIT_001_PV"] == "V0"
    assert safe_to_original["V1"] == "2_LS_201_AL"

    original_eq = "1_FIT_001_PV + 2_LS_201_AL * 0.5 + TOTAL_CONS_REQUIRED_FLOW"
    safe_eq = mod.equation_original_to_safe(original_eq, original_to_safe)
    assert safe_eq == "V0 + V1 * 0.5 + V2"
    assert mod.equation_safe_to_original(safe_eq, safe_to_original) == original_eq


def test_wadi_sample_audit_target_column_is_resume_safe() -> None:
    mod = load_wadi_full_module()
    table = pd.DataFrame(
        {
            "sample_policy": ["coverage_stratified"],
            "target": ["stale_target"],
            "sample_size_actual": [7],
        }
    )

    normalized = mod.with_target_first_column(table, "1_FIT_001_PV")

    assert normalized.columns.tolist()[0] == "target"
    assert normalized.columns.tolist().count("target") == 1
    assert normalized.loc[0, "target"] == "1_FIT_001_PV"


def test_wadi_single_target_command_is_isolated() -> None:
    mod = load_wadi_full_module()
    args = mod.parse_args([
        "--out",
        "artifacts/tmp_wadi",
        "--target-parallel-jobs",
        "2",
        "--pysr-procs",
        "1",
    ])

    cmd = mod.build_single_target_command(args, "1_FIT_001_PV")

    assert "--single-target" in cmd
    assert "1_FIT_001_PV" in cmd
    assert cmd[cmd.index("--target-parallel-jobs") + 1] == "1"



def test_wadi_default_parallelism_is_conservative() -> None:
    mod = load_wadi_full_module()
    args = mod.parse_args([])

    assert args.target_parallel_jobs == 2
    assert args.target_wall_timeout_minutes == 75.0
