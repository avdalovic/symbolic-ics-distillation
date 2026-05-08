from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ics_symbolic_distill.attribution import (
    absolute_pearson_correlation,
    align_last_n,
    build_target_attribution_summary,
    compute_gradient_attribution,
    convert_norm_grad_to_raw_sensitivity,
    detect_floored_channels,
    delta_sensitivity_from_raw_next,
    rank_top_features_excluding,
    rank_top_features,
)


def test_chain_rule_conversion_from_normalized_gradient_to_raw_sensitivity() -> None:
    norm_grad = np.asarray([[2.0, 4.0, 8.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    safe_std = np.asarray([2.0, 4.0, 10.0], dtype=np.float32)
    raw = convert_norm_grad_to_raw_sensitivity(norm_grad, safe_std, sensor_idx=[0, 2])
    expected = np.asarray([[2.0, 2.0, 1.6], [5.0, 2.5, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(raw, expected)


def test_delta_sensitivity_subtracts_self_sensor_only() -> None:
    raw_next = np.ones((2, 4), dtype=np.float32)
    delta = delta_sensitivity_from_raw_next(raw_next, sensor_idx=[1, 3])
    expected = np.ones((2, 4), dtype=np.float32)
    expected[0, 1] = 0.0
    expected[1, 3] = 0.0
    np.testing.assert_allclose(delta, expected)


def test_ranking_marks_self_feature_correctly() -> None:
    matrix = np.asarray([[0.2, 0.5, 0.1], [0.7, 0.1, 0.9]], dtype=np.float32)
    rankings = rank_top_features(
        matrix,
        feature_columns=["FIT101", "LIT101", "FIT201"],
        target_columns=["FIT101", "FIT201"],
        sensor_idx=[0, 2],
        top_k=2,
    )
    first = rankings["targets"][0]["top_features"]
    second = rankings["targets"][1]["top_features"]
    assert first[0]["feature"] == "LIT101"
    assert first[0]["is_self_feature"] is False
    assert second[0]["feature"] == "FIT201"
    assert second[0]["is_self_feature"] is True


def test_pearson_correlation_handles_constant_features_without_nan() -> None:
    features = np.asarray(
        [
            [1.0, 0.0, 3.0],
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 1.0],
            [1.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [
            [0.0, 5.0],
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
        ],
        dtype=np.float32,
    )
    corr = absolute_pearson_correlation(features, targets)
    assert corr.shape == (2, 3)
    assert np.isfinite(corr).all()
    assert corr[0, 0] == 0.0
    assert corr[1].max() == 0.0
    assert corr[0, 1] > 0.99


def test_alignment_from_mlp_export_to_gru_overlap_uses_last_samples() -> None:
    arr = np.arange(9899 * 2, dtype=np.float32).reshape(9899, 2)
    aligned, start = align_last_n(arr, 9840)
    assert start == 59
    assert aligned.shape == (9840, 2)
    np.testing.assert_array_equal(aligned[0], arr[59])
    np.testing.assert_array_equal(aligned[-1], arr[-1])


def test_sensor_idx_mapping_matches_target_columns() -> None:
    feature_columns = ["FIT101", "LIT101", "MV101", "FIT201"]
    target_columns = ["FIT101", "FIT201"]
    sensor_idx = [0, 3]
    assert all(target_columns[j] == feature_columns[sensor_idx[j]] for j in range(len(target_columns)))


def test_floored_channel_detection() -> None:
    stats = {
        "normalization_mode": "zscore",
        "std_floor": 0.01,
        "std": np.asarray([0.01, 0.02, 0.0100000001], dtype=np.float64),
    }
    floored = detect_floored_channels(stats, ["P101", "FIT101", "P102"], tolerance=1e-8)
    assert floored["indices"] == [0, 2]
    assert floored["channels"] == ["P101", "P102"]
    assert "std_target / std_input" in floored["reason"]


def test_ranking_with_excluded_features() -> None:
    matrix = np.asarray([[10.0, 9.0, 8.0, 7.0]], dtype=np.float32)
    rankings = rank_top_features_excluding(
        matrix,
        feature_columns=["P101", "FIT101", "P102", "LIT101"],
        target_columns=["LIT101"],
        sensor_idx=[3],
        exclude_indices=[0, 2],
        excluded_reason="floored",
        top_k=2,
    )
    top = rankings["targets"][0]["top_features"]
    assert [item["feature"] for item in top] == ["FIT101", "LIT101"]
    assert rankings["excluded_features"] == ["P101", "P102"]


def test_lit101_summary_schema_contains_required_fields() -> None:
    feature_columns = ["FIT101", "LIT101", "MV101", "P101"]
    target_columns = ["LIT101"]
    sensor_idx = [1]
    matrices = {
        "grad_norm_next": np.asarray([[0.2, 1.0, 0.4, 10.0]], dtype=np.float32),
        "raw_next_sensitivity": np.asarray([[0.2, 1.0, 0.4, 10.0]], dtype=np.float32),
        "raw_delta_sensitivity": np.asarray([[0.2, 0.1, 0.4, 10.0]], dtype=np.float32),
        "corr_mlp_pred_delta": np.asarray([[0.8, 0.2, 0.7, 0.0]], dtype=np.float32),
    }
    summary = build_target_attribution_summary(
        target_name="LIT101",
        matrices=matrices,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sensor_idx=sensor_idx,
        floored={"indices": [3], "channels": ["P101"], "reason": "floored"},
        watched_features=["LIT101", "FIT101", "MV101", "P101"],
    )
    for key in [
        "top10_normalized_next_gradient_features",
        "top10_raw_next_sensitivity_features",
        "top10_raw_next_sensitivity_nonfloored_features",
        "top10_raw_delta_sensitivity_features",
        "top10_raw_delta_sensitivity_nonfloored_features",
        "top10_corr_mlp_pred_delta_features",
        "top10_corr_mlp_pred_delta_nonfloored_features",
        "ranks",
        "interpretation",
        "recommendation",
    ]:
        assert key in summary
    assert summary["top10_raw_delta_sensitivity_features"][0]["feature"] == "P101"
    assert summary["top10_raw_delta_sensitivity_nonfloored_features"][0]["feature"] == "MV101"
    assert "corr_mlp_pred_delta_nonfloored" in summary["recommendation"]


def test_attribution_doc_does_not_recommend_raw_sensitivity_for_topk_when_floored() -> None:
    text = (Path(__file__).resolve().parents[1] / "docs" / "mlp_attribution_sanity.md").read_text(
        encoding="utf-8"
    )
    lower = " ".join(text.lower().split())
    assert "raw sensitivity matrices are therefore diagnostic outputs" in lower
    assert "not the recommended" in lower
    assert "attribution-guided variables from mlp-predicted delta correlation" in lower


def test_per_sample_gradient_uses_mean_abs_not_abs_mean() -> None:
    class SquareModel(torch.nn.Module):
        def forward(self, x_window: torch.Tensor) -> torch.Tensor:
            return x_window[:, 0:1, :] ** 2

    inputs = np.asarray([[[-1.0]], [[1.0]]], dtype=np.float32)
    result = compute_gradient_attribution(
        SquareModel(),
        inputs,
        {"std": np.asarray([1.0], dtype=np.float32)},
        sensor_idx=[0],
        sample_size=None,
        seed=0,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert result["grad_norm_next"].shape == (1, 1)
    np.testing.assert_allclose(result["grad_norm_next"], np.asarray([[2.0]], dtype=np.float32))

    signed_grads = np.asarray([-2.0, 2.0], dtype=np.float32)
    assert abs(float(signed_grads.mean())) == 0.0
    assert float(np.abs(signed_grads).mean()) == 2.0
