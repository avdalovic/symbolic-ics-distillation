from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.run_localization_experiment import (
    AttackRecord,
    CusumParams,
    DatasetBundle,
    build_dependency_graph,
    localization_tables,
    nearest_distance,
    run_cusum_reset_blocks,
    threshold_crossings,
)


def _toy_bundle(attacks: list[AttackRecord]) -> DatasetBundle:
    return DatasetBundle(
        dataset="TOY",
        arrays={},
        selected_rows=[],
        residual_cache={},
        feature_columns=["a", "b", "c", "d", "y"],
        attacks=attacks,
        s=1.0,
        g=1.0,
        sample_period_seconds=1,
        source_note="toy",
    )


def test_exact_alert_has_zero_distance() -> None:
    graph, _ = build_dependency_graph([{"target": "y", "equation": "a + b", "sympy_format": "a + b"}], ["a", "b", "y"])
    dist, nearest = nearest_distance(graph.to_undirected(), "a", ["a"])
    assert dist == 0
    assert nearest == "a"


def test_direct_dependency_has_distance_one() -> None:
    graph, _ = build_dependency_graph([{"target": "y", "equation": "a + b", "sympy_format": "a + b"}], ["a", "b", "y"])
    dist, nearest = nearest_distance(graph.to_undirected(), "y", ["a"])
    assert dist == 1
    assert nearest == "a"


def test_multi_target_attack_uses_minimum_distance() -> None:
    graph, _ = build_dependency_graph([{"target": "y", "equation": "a + b", "sympy_format": "a + b"}], ["a", "b", "c", "y"])
    dist, nearest = nearest_distance(graph.to_undirected(), "y", ["c", "a"])
    assert dist == 1
    assert nearest == "a"


def test_undetected_attack_is_infinity_not_omitted() -> None:
    graph, _ = build_dependency_graph([{"target": "y", "equation": "a", "sympy_format": "a"}], ["a", "y"])
    bundle = _toy_bundle([AttackRecord("TOY", "1", 0, 2, ("a",), "fixture")])
    per_attack, per_alert = localization_tables(
        bundle,
        graph.to_undirected(),
        pd.DataFrame(columns=["attack_id", "variable", "new_threshold_crossing"]),
    )
    assert len(per_attack) == 1
    assert per_alert.empty
    assert per_attack.iloc[0]["detected"] is False or per_attack.iloc[0]["detected"] == False
    assert math.isinf(float(per_attack.iloc[0]["nearest_distance"]))


def test_disconnected_alert_is_infinity_not_omitted() -> None:
    graph, _ = build_dependency_graph([{"target": "y", "equation": "a", "sympy_format": "a"}], ["a", "b", "y"])
    graph.add_node("d")
    bundle = _toy_bundle([AttackRecord("TOY", "1", 0, 2, ("a",), "fixture")])
    alerts = pd.DataFrame(
        [
            {
                "attack_id": "1",
                "variable": "d",
                "new_threshold_crossing": True,
                "first_crossing_time": 1,
                "detection_delay_seconds": 1,
                "max_threshold_ratio": 2.0,
            }
        ]
    )
    per_attack, per_alert = localization_tables(bundle, graph.to_undirected(), alerts)
    assert len(per_alert) == 1
    assert math.isinf(float(per_alert.iloc[0]["distance_to_nearest_attacked_tag"]))
    assert math.isinf(float(per_attack.iloc[0]["nearest_distance"]))


def test_graph_construction_uses_only_selected_equations_not_attack_targets() -> None:
    selected = [{"target": "y", "equation": "a + b", "sympy_format": "a + b"}]
    graph, edges = build_dependency_graph(selected, ["a", "b", "c", "y"])
    assert set(map(tuple, edges[["predictor", "target"]].to_numpy())) == {("a", "y"), ("b", "y")}
    assert "c" in graph.nodes
    assert ("c", "y") not in graph.edges


def test_graph_construction_is_deterministic() -> None:
    selected = [
        {"target": "y", "equation": "b + a", "sympy_format": "b + a"},
        {"target": "c", "equation": "y", "sympy_format": "y"},
    ]
    graph1, edges1 = build_dependency_graph(selected, ["a", "b", "c", "y"])
    graph2, edges2 = build_dependency_graph(list(reversed(selected)), ["y", "c", "b", "a"])
    assert sorted(graph1.edges()) == sorted(graph2.edges())
    pd.testing.assert_frame_equal(edges1, edges2)


def test_threshold_crossing_requires_new_crossing() -> None:
    cusum = np.asarray([0.0, 1.0, 2.0, 2.5, 1.0, 2.1])
    crossings = threshold_crossings(cusum, threshold=2.0)
    np.testing.assert_array_equal(crossings, np.asarray([False, False, True, False, False, True]))


def test_cusum_resets_at_file_boundaries() -> None:
    params = CusumParams(delta=0.0, threshold=1.0, growth_cap=10.0, max_calib_cusum=1.0, s=1.0, g=1.0)
    reset = run_cusum_reset_blocks(np.asarray([2.0, 2.0]), params, [1, 1])
    np.testing.assert_allclose(reset, np.asarray([2.0, 2.0]))
