from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import sympy as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "distill_sensor_mlp.py"
spec = importlib.util.spec_from_file_location("distill_sensor_mlp", SCRIPT_PATH)
distill_sensor_mlp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(distill_sensor_mlp)

LINEAR_SCRIPT_PATH = REPO_ROOT / "scripts" / "linear_lit101_baselines.py"
linear_spec = importlib.util.spec_from_file_location("linear_lit101_baselines", LINEAR_SCRIPT_PATH)
linear_lit101_baselines = importlib.util.module_from_spec(linear_spec)
assert linear_spec.loader is not None
linear_spec.loader.exec_module(linear_lit101_baselines)


def test_subsample_indices_are_sorted_and_seeded() -> None:
    first = distill_sensor_mlp.subsample_indices(100, 10, 0)
    second = distill_sensor_mlp.subsample_indices(100, 10, 0)
    assert first.shape == (10,)
    assert np.all(first[:-1] <= first[1:])
    np.testing.assert_array_equal(first, second)


def test_target_aliases() -> None:
    assert distill_sensor_mlp.normalize_target_source("mlp") == "mlp_delta"
    assert distill_sensor_mlp.normalize_target_source("actual") == "actual_delta"
    assert distill_sensor_mlp.normalize_target_source("mlp_next") == "mlp_next"
    assert distill_sensor_mlp.normalize_target_source("actual_next") == "actual_next"


def test_features_in_equation_uses_tag_delimiters() -> None:
    equation = "0.19*FIT101 - 0.20*FIT201 + square(MV101) + PIT502"
    features = ["FIT101", "FIT201", "MV101", "P101", "PIT502"]
    assert distill_sensor_mlp.features_in_equation(equation, features) == [
        "FIT101",
        "FIT201",
        "MV101",
        "PIT502",
    ]


def test_temp_equation_file_drops_pysr_output_directory_conflict() -> None:
    params = {
        "temp_equation_file": True,
        "output_directory": "artifacts/symbolic_equations/example",
        "niterations": 1,
    }
    resolved = distill_sensor_mlp.resolve_pysr_param_conflicts(params)
    assert resolved["temp_equation_file"] is True
    assert "output_directory" not in resolved
    assert params["output_directory"] == "artifacts/symbolic_equations/example"


def test_json_serialization_handles_sympy_numpy_and_nan(tmp_path: Path) -> None:
    x = sp.Symbol("FIT101")
    path = tmp_path / "payload.json"
    distill_sensor_mlp.write_json(
        path,
        {
            "expr": x + 1,
            "scalar": np.float32(1.5),
            "array": np.array([1, 2]),
            "nan": float("nan"),
            "path": Path("abc"),
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["expr"] == "FIT101 + 1"
    assert payload["scalar"] == 1.5
    assert payload["array"] == [1, 2]
    assert payload["nan"] is None
    assert payload["path"] == "abc"


def _write_minimal_distill_dir(root: Path) -> None:
    feature_columns = ["FIT101", "LIT101", "FIT201"]
    target_columns = ["LIT101"]
    sensor_idx = [1]
    (root / "distill_feature_columns.json").write_text(json.dumps(feature_columns), encoding="utf-8")
    (root / "distill_target_columns.json").write_text(json.dumps(target_columns), encoding="utf-8")
    (root / "distill_sensor_idx.json").write_text(json.dumps(sensor_idx), encoding="utf-8")
    x = np.array([[10.0, 100.0, 8.0], [11.0, 101.0, 9.0]], dtype=np.float32)
    mlp_delta = np.array([[0.5], [0.25]], dtype=np.float32)
    actual_delta = np.array([[1.0], [-1.0]], dtype=np.float32)
    np.save(root / "distill_inputs_current_raw.npy", x)
    np.save(root / "distill_pred_delta_raw_mlp.npy", mlp_delta)
    np.save(root / "distill_actual_delta_raw.npy", actual_delta)


def test_target_source_derives_next_from_current_plus_delta(tmp_path: Path) -> None:
    _write_minimal_distill_dir(tmp_path)
    mlp = distill_sensor_mlp.load_distillation_arrays(tmp_path, "mlp_next")
    actual = distill_sensor_mlp.load_distillation_arrays(tmp_path, "actual_next")
    np.testing.assert_allclose(mlp["y_all"][:, 0], np.array([100.5, 101.25], dtype=np.float32))
    np.testing.assert_allclose(actual["y_all"][:, 0], np.array([101.0, 100.0], dtype=np.float32))
    assert mlp["target_source_method"] == "derived_current_plus_delta"
    assert actual["target_source_method"] == "derived_current_plus_delta"


def test_operator_set_restricted_excludes_unary_ops() -> None:
    config = distill_sensor_mlp.operator_config("restricted")
    assert config["binary_operators"] == ["+", "-", "*", "/"]
    assert config["unary_operators"] == []
    assert "square" not in " ".join(config["unary_operators"])
    assert "abs" not in " ".join(config["unary_operators"])


def test_sample_size_all_uses_full_pool() -> None:
    pool = np.arange(7)
    assert distill_sensor_mlp.parse_sample_size("all") is None
    np.testing.assert_array_equal(distill_sensor_mlp.choose_sample_indices(pool, None, 0), pool)
    np.testing.assert_array_equal(distill_sensor_mlp.choose_sample_indices(pool, 0, 0), pool)


def test_linear_baseline_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = 0.19 * x[:, 0] - 0.20 * x[:, 1] + 0.009
    train_idx = np.arange(150)
    holdout_idx = np.arange(150, 200)
    result = linear_lit101_baselines.fit_linear_baseline(
        x,
        y,
        ["FIT101", "FIT201"],
        train_idx,
        holdout_idx,
        model_type="ols",
    )
    assert abs(result["coefficients"]["FIT101"] - 0.19) < 1e-10
    assert abs(result["coefficients"]["FIT201"] + 0.20) < 1e-10
    assert abs(result["intercept"] - 0.009) < 1e-10
    assert result["holdout_mse"] < 1e-20
