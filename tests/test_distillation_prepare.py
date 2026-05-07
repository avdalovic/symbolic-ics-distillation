from __future__ import annotations

import numpy as np

from ics_symbolic_distill.data.normalization import (
    inverse_normalize_features,
    inverse_normalize_targets,
)
from ics_symbolic_distill.distillation import (
    align_mlp_to_gru_overlap,
    build_name_mapping,
    compute_temporal_summary_features,
    squeeze_horizon_one,
)
from ics_symbolic_distill.training.config import load_experiment_config


def test_safe_std_inverse_normalization_uses_saved_scale() -> None:
    stats = {
        "normalization_mode": "zscore",
        "mean": np.asarray([10.0, -2.0, 5.0], dtype=np.float32),
        "std": np.asarray([0.01, 2.0, 0.5], dtype=np.float32),
        "sensor_idx": [0, 2],
    }
    norm = np.asarray([[0.0, 1.0, -2.0], [3.0, -1.0, 4.0]], dtype=np.float32)
    raw = inverse_normalize_features(norm, stats)
    expected = np.asarray([[10.0, 0.0, 4.0], [10.03, -4.0, 7.0]], dtype=np.float32)
    np.testing.assert_allclose(raw, expected)

    target_norm = np.asarray([[[0.0, -2.0]], [[3.0, 4.0]]], dtype=np.float32)
    target_raw = inverse_normalize_targets(target_norm, stats, sensor_idx=[0, 2])
    np.testing.assert_allclose(target_raw[:, 0, :], expected[:, [0, 2]])


def test_target_columns_map_to_features_through_sensor_idx() -> None:
    feature_columns = ["FIT101", "LIT101", "MV101", "FIT201", "P101"]
    target_columns = ["FIT101", "LIT101", "FIT201"]
    sensor_idx = [0, 1, 3]
    mapping = build_name_mapping(feature_columns, target_columns, sensor_idx)
    assert mapping["target_name_to_feature_index"]["FIT201"] == 3
    assert feature_columns[mapping["target_index_to_feature_index"]["2"]] == "FIT201"


def test_squeeze_horizon_one() -> None:
    arr = np.zeros((4, 1, 25), dtype=np.float32)
    squeezed = squeeze_horizon_one(arr, "targets")
    assert squeezed.shape == (4, 25)


def test_mlp_gru_overlap_alignment_takes_last_samples() -> None:
    mlp = np.arange(20, dtype=np.float32).reshape(10, 2)
    aligned = align_mlp_to_gru_overlap(mlp, 6, "mlp")
    assert aligned.shape == (6, 2)
    np.testing.assert_array_equal(aligned, mlp[-6:])


def test_temporal_feature_names_and_shape() -> None:
    feature_columns = ["FIT101", "LIT101", "FIT201", "MV101", "P101"]
    window = np.arange(2 * 5 * 12, dtype=np.float32).reshape(2, 5, 12)
    features, names, mapping = compute_temporal_summary_features(
        window,
        feature_columns,
        dt_model_step=1.0,
    )
    assert features.shape == (2, 5 * 11)
    assert names[0] == "FIT101_current"
    assert "LIT101_mean_10" in names
    assert "FIT201_rate_5" in names
    assert mapping["temporal_feature_name_to_index"]["MV101_delta_1"] == names.index("MV101_delta_1")


def test_explicit_name_mapping_contains_key_swat_tags() -> None:
    feature_columns = ["FIT101", "LIT101", "FIT201", "MV101", "P101"]
    target_columns = ["FIT101", "LIT101", "FIT201"]
    mapping = build_name_mapping(feature_columns, target_columns, sensor_idx=[0, 1, 2])
    for name in ["LIT101", "FIT101", "FIT201", "MV101", "P101"]:
        assert name in mapping["feature_name_to_index"]
        assert mapping["current_feature_safe_names"][name].startswith("x_")
    assert mapping["delta_target_safe_names"]["LIT101_delta"] == "y_LIT101_delta"


def test_val20_configs_load_with_distinct_checkpoint_dirs() -> None:
    gru_cfg, _ = load_experiment_config("configs/experiment/swat_gru_val20.yaml")
    mlp_cfg, _ = load_experiment_config("configs/experiment/swat_mlp_current_val20.yaml")
    assert float(gru_cfg.dataset.val_ratio) == 0.20
    assert float(mlp_cfg.dataset.val_ratio) == 0.20
    assert str(gru_cfg.train.ckpt_dir).endswith("artifacts/checkpoints/swat/gru_h1_val20")
    assert str(mlp_cfg.train.ckpt_dir).endswith("artifacts/checkpoints/swat/mlp_current_h1_val20")
