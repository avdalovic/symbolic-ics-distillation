from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


report_cross_sensor = load_script("report_cross_sensor.py")
linear_sensor_baselines = load_script("linear_sensor_baselines.py")
list_target_sensors = load_script("list_target_sensors.py")
parse_geco_swat_model = load_script("parse_geco_swat_model.py")


def test_support_config_loads_expected_sensors() -> None:
    config = report_cross_sensor.load_support_config(REPO_ROOT / "configs" / "swat_sensor_local_support.json")
    assert "LIT101" in config
    assert "FIT101" in config["LIT101"]["local_features"]
    assert config["FIT201"]["expected_patterns"]["mlp_delta"] == ["P101"]


def test_feature_extraction_avoids_substring_false_positives() -> None:
    features = ["P101", "P1012", "FIT101", "FIT10", "PIT502"]
    equation = "P1012 + FIT101 - PIT502"
    assert report_cross_sensor.extract_equation_features(equation, features) == ["P1012", "FIT101", "PIT502"]


def test_off_process_feature_detection() -> None:
    status, off = report_cross_sensor.classify_feature_support(["FIT101", "PIT502"], ["FIT101", "FIT201"])
    assert status == "partially_local"
    assert off == ["PIT502"]
    status, off = report_cross_sensor.classify_feature_support(["PIT502"], ["FIT101", "FIT201"])
    assert status == "off_process"
    assert off == ["PIT502"]
    status, off = report_cross_sensor.classify_feature_support(["FIT101"], None)
    assert status == "unknown_support"
    assert off == []


def test_report_aggregation_on_fake_audit_dir(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    run_dir = audit_root / "LIT101_actual_delta_restricted_unconstrained"
    run_dir.mkdir(parents=True)
    metadata = {
        "best_equation": "FIT101 - FIT201",
        "selected_features": ["FIT101", "LIT101", "FIT201", "PIT502"],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "complexity": 1,
                "loss": 3.0,
                "equation": "0.1",
                "score": 0.0,
                "holdout_mse": 3.0,
                "holdout_r2_against_constant": 0.0,
            },
            {
                "complexity": 3,
                "loss": 1.0,
                "equation": "FIT101 - FIT201",
                "score": 0.5,
                "holdout_mse": 1.0,
                "holdout_r2_against_constant": 0.8,
            },
            {
                "complexity": 5,
                "loss": 0.8,
                "equation": "FIT101 - PIT502",
                "score": 0.4,
                "holdout_mse": 0.9,
                "holdout_r2_against_constant": 0.82,
            },
        ]
    ).to_csv(run_dir / "pareto_front_scored.csv", index=False)
    support = {
        "LIT101": {"local_features": ["LIT101", "FIT101", "FIT201"], "notes": ""},
    }
    rows, narratives = report_cross_sensor.summarize_runs(audit_root, support, {}, sensors=["LIT101"])
    actual_delta = [row for row in rows if row["sensor"] == "LIT101" and row["target_source"] == "actual_delta"][0]
    assert actual_delta["best_local_physical_equation"] == "FIT101 - FIT201"
    assert "off=['PIT502']" in actual_delta["notes"]
    assert "actual_delta" in "\n".join(narratives["LIT101"])


def test_report_handles_missing_local_support(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    run_dir = audit_root / "AIT201_actual_delta_restricted_unconstrained"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"best_equation": "AIT201 - PIT502", "selected_features": ["AIT201", "PIT502"]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "complexity": 3,
                "loss": 1.0,
                "equation": "AIT201 - PIT502",
                "score": 0.5,
                "holdout_mse": 1.0,
                "holdout_r2_against_constant": 0.2,
            }
        ]
    ).to_csv(run_dir / "pareto_front_scored.csv", index=False)
    rows, narratives = report_cross_sensor.summarize_runs(audit_root, {}, {}, sensors=["AIT201"])
    row = [item for item in rows if item["sensor"] == "AIT201" and item["target_source"] == "actual_delta"][0]
    assert row["score_selected_equation"] == "AIT201 - PIT502"
    assert "unknown_support" in row["notes"]
    assert "local support unknown" in "\n".join(narratives["AIT201"])


def test_target_sensor_listing_comes_from_distill_targets(tmp_path: Path) -> None:
    (tmp_path / "distill_target_columns.json").write_text(json.dumps(["FIT101", "LIT101"]), encoding="utf-8")
    assert list_target_sensors.load_target_sensors(tmp_path) == ["FIT101", "LIT101"]


def test_parse_geco_model_best_effort(tmp_path: Path) -> None:
    model = {
        "CI": {
            "LIT101": {
                "equation": "Sum",
                "combination": ["LIT101", "FIT101", "FIT201"],
                "parameters": [1.0, 0.2, -0.2, 0.0],
            }
        }
    }
    path = tmp_path / "SWaT.model"
    path.write_text(json.dumps(model), encoding="utf-8")
    parsed = parse_geco_swat_model.parse_geco_model(path)
    assert parsed["LIT101"]["geco_reference_available"] is True
    assert parsed["LIT101"]["geco_variables"] == ["LIT101", "FIT101", "FIT201"]


def test_generalized_linear_baseline_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.5
    train_idx = np.arange(150)
    holdout_idx = np.arange(150, 200)
    result = linear_sensor_baselines.fit_linear_baseline(
        x,
        y,
        ["FIT101", "FIT201"],
        train_idx,
        holdout_idx,
        model_type="ols",
    )
    assert abs(result["coefficients"]["FIT101"] - 2.0) < 1e-10
    assert abs(result["coefficients"]["FIT201"] + 3.0) < 1e-10
    assert abs(result["intercept"] - 0.5) < 1e-10
    assert result["holdout_mse"] < 1e-20
    assert result["holdout_rmse"] < 1e-10
    assert result["holdout_mae"] < 1e-10


def test_full_runner_has_resume_checks() -> None:
    text = (REPO_ROOT / "scripts" / "run_full_sensor_audit.sh").read_text(encoding="utf-8")
    assert "RESUME" in text
    assert "pareto_front_scored.csv" in text
    assert "SKIP existing" in text
    assert "list_target_sensors.py" in text
