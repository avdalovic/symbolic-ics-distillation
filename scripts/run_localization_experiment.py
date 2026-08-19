#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.data.ics_metadata import AttackWindow, get_attack_windows, is_actuator
from ics_symbolic_distill.detection import CusumParams, evaluate_equation, fit_cusum_params, run_cusum
from ics_symbolic_distill.detection.symbolic_eval import equation_features
from ics_symbolic_distill.detection.swat1s_delta_sampling import reconstruct_next_from_delta


OPERATING_POINTS = {
    "SWAT": (1.20, 15.0),
    "WADI": (1.32, 25.0),
    "BATADAL": (1.39, 2.16),
    "HAI": (2.5, 12.0),
}
SWAT_GECO_EXCLUSIONS = {"AIT201", "AIT202", "AIT203", "P201"}
HAI_GECO_IGNORED = {
    "P1_PCV02Z",
    "P2_SIT01",
    "P2_SIT02",
    "P2_VT01",
    "P2_VXT02",
    "P2_VXT03",
    "P2_VYT02",
}
BOOTSTRAP_SEED = 20250813


@dataclass(frozen=True)
class AttackRecord:
    dataset: str
    attack_id: str
    start: int
    end: int
    attacked_tags: tuple[str, ...]
    source: str
    official_start: int | str | None = None
    official_end: int | str | None = None


@dataclass
class DatasetBundle:
    dataset: str
    arrays: dict[str, Any]
    selected_rows: list[dict[str, Any]]
    residual_cache: dict[str, dict[str, np.ndarray]]
    feature_columns: list[str]
    attacks: list[AttackRecord]
    s: float
    g: float
    sample_period_seconds: int
    source_note: str


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SWAT = _load_module(REPO_ROOT / "scripts" / "run_swat_1sec_delta_posthoc_ablation.py", "swat_posthoc_for_localization")
WADI = _load_module(REPO_ROOT / "scripts" / "run_wadi_1sec_delta_posthoc_ablation.py", "wadi_posthoc_for_localization")
BATADAL = _load_module(REPO_ROOT / "scripts" / "run_batadal_delta_full.py", "batadal_for_localization")
HAI = _load_module(REPO_ROOT / "scripts" / "run_hai_1sec_delta_full.py", "hai_for_localization")


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "inf"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sorted_edges(path: Path, graph: nx.DiGraph) -> None:
    rows = [{"predictor": u, "target": v} for u, v in sorted(graph.edges())]
    pd.DataFrame(rows, columns=["predictor", "target"]).to_csv(path, index=False)


def graph_stats(graph: nx.DiGraph) -> dict[str, Any]:
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected))
    return {
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "n_connected_components": int(len(components)),
        "largest_component_size": int(max((len(c) for c in components), default=0)),
        "isolated_nodes": sorted(nx.isolates(undirected)),
    }


def build_dependency_graph(
    selected_rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    extra_nodes: Sequence[str] = (),
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Build dependency graph from final deployed equations only."""

    graph = nx.DiGraph()
    for name in sorted({str(col) for col in feature_columns} | {str(node) for node in extra_nodes}):
        graph.add_node(name)

    edge_rows: list[dict[str, str]] = []
    for row in sorted(selected_rows, key=lambda item: str(item["target"])):
        target = str(row["target"])
        graph.add_node(target)
        variable_type = str(row.get("variable_type", ""))
        source = str(row.get("source", ""))
        if variable_type in {"actuator", "persistence"} or source == "actuator_persistence":
            continue
        expr = str(row.get("sympy_format", row.get("equation", "")))
        for predictor in sorted(set(equation_features(expr, feature_columns))):
            if predictor == target:
                # Self terms are informative but do not improve localization
                # distances. GeCo's localization graph also focuses on
                # relationships between different process variables.
                continue
            graph.add_node(predictor)
            graph.add_edge(predictor, target)
            edge_rows.append({"predictor": predictor, "target": target})

    edge_df = pd.DataFrame(edge_rows, columns=["predictor", "target"]).drop_duplicates()
    if not edge_df.empty:
        edge_df = edge_df.sort_values(["predictor", "target"]).reset_index(drop=True)
    return graph, edge_df


def validate_attack_targets(attacks: Sequence[AttackRecord], feature_columns: Sequence[str]) -> None:
    feature_set = set(str(col) for col in feature_columns)
    missing_rows = []
    for attack in attacks:
        if not attack.attacked_tags:
            missing_rows.append((attack.attack_id, "<empty>"))
            continue
        for tag in attack.attacked_tags:
            if tag not in feature_set:
                missing_rows.append((attack.attack_id, tag))
    if missing_rows:
        detail = ", ".join(f"attack {attack_id}: {tag}" for attack_id, tag in missing_rows[:20])
        raise ValueError(f"Attack target tags missing from process variables: {detail}")


def unobserved_attack_targets(attacks: Sequence[AttackRecord], feature_columns: Sequence[str]) -> pd.DataFrame:
    """Official attack targets that are not present as logged state variables."""

    feature_set = set(str(col) for col in feature_columns)
    rows = []
    for attack in attacks:
        for tag in attack.attacked_tags:
            if tag not in feature_set:
                rows.append(
                    {
                        "attack_id": attack.attack_id,
                        "attacked_tag": tag,
                        "source": attack.source,
                        "reason": "official_attack_target_absent_from_logged_state_schema",
                    }
                )
    return pd.DataFrame(rows, columns=["attack_id", "attacked_tag", "source", "reason"])


def attack_targets_table(attacks: Sequence[AttackRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "attack_id": attack.attack_id,
                "start": attack.start,
                "end": attack.end,
                "attacked_tags": ";".join(attack.attacked_tags),
                "source": attack.source,
                "official_start": attack.official_start,
                "official_end": attack.official_end,
            }
            for attack in attacks
        ],
        columns=["attack_id", "start", "end", "attacked_tags", "source", "official_start", "official_end"],
    )


def rows_from_selected_csv(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    return [{str(k): v for k, v in row.items()} for row in df.to_dict("records")]


def swat_bundle(args: argparse.Namespace) -> DatasetBundle:
    arrays = SWAT.load_arrays(
        argparse.Namespace(
            experiment=args.swat_experiment,
            train_csv=args.swat_train_csv,
            test_csv=args.swat_test_csv,
        )
    )
    models = SWAT.models_from_selected(Path(args.swat_selected), exclude=SWAT_GECO_EXCLUSIONS)
    models += SWAT.make_actuator_persistence_models(arrays["feature_columns"], exclude=SWAT_GECO_EXCLUSIONS)
    cache = SWAT.residual_cache_for_models(arrays, models)
    rows = []
    for model in models:
        row = model.to_row()
        if model.source == "actuator_persistence":
            row["source"] = "actuator_persistence"
            row["target_mode"] = "actuator_persistence_next"
        else:
            row["source"] = "selected_sensor_delta"
            row["target_mode"] = "sensors_delta_actuators_next"
        rows.append(row)
    attacks = [
        AttackRecord(
            dataset="SWAT",
            attack_id=str(i),
            start=max(0, int(window.start) - 1),
            end=min(int(arrays["labels"].shape[0]) - 1, int(window.end) - 1),
            attacked_tags=tuple(str(tag) for tag in window.affected_tags),
            source="src/ics_symbolic_distill/data/ics_metadata.py::_SWAT_ATTACK_WINDOWS",
            official_start=window.start,
            official_end=window.end,
        )
        for i, window in enumerate(get_attack_windows("SWAT"), start=1)
    ]
    return DatasetBundle("SWAT", arrays, rows, cache, list(arrays["feature_columns"]), attacks, args.swat_s, args.swat_g, 1, "SWaT seed-0 no-holdout GeCo-matched plus actuator persistence")


def wadi_bundle(args: argparse.Namespace) -> DatasetBundle:
    arrays = WADI.FULL.load_wadi_1sec_arrays(argparse.Namespace(train_csv=args.wadi_train_csv, test_csv=args.wadi_test_csv))
    selected_rows = WADI.load_selected_equations(Path(args.wadi_selected))
    geco_exclusions, missing = WADI.resolve_exclusions(WADI.WADI_GECO_EXCLUSIONS, arrays["feature_columns"])
    rows, cache = WADI.build_variant_rows(
        arrays,
        selected_rows,
        variant="geco_matched_plus_actuator_persistence",
        geco_exclusions=geco_exclusions,
    )
    attacks = [
        AttackRecord(
            dataset="WADI",
            attack_id=str(i),
            start=max(0, int(window.start) - 1),
            end=min(int(arrays["labels"].shape[0]) - 1, int(window.end) - 1),
            attacked_tags=tuple(str(tag) for tag in window.affected_tags),
            source="src/ics_symbolic_distill/data/ics_metadata.py::_WADI_ATTACK_WINDOWS",
            official_start=window.start,
            official_end=window.end,
        )
        for i, window in enumerate(get_attack_windows("WADI"), start=1)
    ]
    note = "WADI seed-0 GeCo-matched plus actuator persistence"
    if missing:
        note += f"; unresolved published GeCo exclusion names in this processed schema: {','.join(sorted(missing))}"
    return DatasetBundle("WADI", arrays, rows, cache, list(arrays["feature_columns"]), attacks, args.wadi_s, args.wadi_g, 1, note)


def batadal_bundle(args: argparse.Namespace) -> DatasetBundle:
    selected_rows = rows_from_selected_csv(Path(args.batadal_selected))
    test_paths = [Path(args.batadal_test_csv)]
    extra_path = Path(args.batadal_test_csv_extra)
    if extra_path.exists() and extra_path not in test_paths:
        test_paths.append(extra_path)

    segments = []
    for test_path in test_paths:
        arrays = BATADAL.load_batadal_arrays(argparse.Namespace(train_csv=args.batadal_train_csv, test_csv=str(test_path)))
        rows, cache = BATADAL.build_variant_rows(arrays, selected_rows, "geco_matched_plus_actuator_persistence")
        segments.append((test_path, arrays, rows, cache))

    _, arrays0, rows0, cache0 = segments[0]
    target_order = [str(row["target"]) for row in rows0]
    for test_path, _, rows, _ in segments[1:]:
        if [str(row["target"]) for row in rows] != target_order:
            raise ValueError(f"BATADAL monitored row order differs for {test_path}")

    combined_cache: dict[str, dict[str, np.ndarray]] = {}
    for target in target_order:
        combined_cache[target] = {
            "train": np.asarray(cache0[target]["train"], dtype=np.float64),
            "test": np.concatenate([np.asarray(cache[target]["test"], dtype=np.float64) for _, _, _, cache in segments]),
        }

    labels = np.concatenate([np.asarray(arrays["labels"], dtype=np.int64) for _, arrays, _, _ in segments])
    block_lengths = [int(np.asarray(arrays["labels"]).shape[0]) for _, arrays, _, _ in segments]
    combined_arrays = {
        "feature_columns": list(arrays0["feature_columns"]),
        "labels": labels,
        "test_block_lengths": block_lengths,
    }

    attacks: list[AttackRecord] = []
    offset = 0
    for test_path, arrays, _, _ in segments:
        for window in arrays["attack_windows"]:
            attacks.append(
                AttackRecord(
                    dataset="BATADAL",
                    attack_id=str(window.attack_id),
                    start=offset + int(window.start),
                    end=offset + int(window.end),
                    attacked_tags=tuple(str(tag) for tag in window.affected_tags),
                    source=f"{test_path}::attack_id/attack_target",
                    official_start=int(window.start),
                    official_end=int(window.end),
                )
            )
        offset += int(np.asarray(arrays["labels"]).shape[0])

    note = "BATADAL seed-0 GeCo-matched plus actuator persistence"
    if len(test_paths) > 1:
        note += "; combined test_dataset04 and test_dataset_test with CUSUM reset per segment"
    return DatasetBundle("BATADAL", combined_arrays, rows0, combined_cache, list(arrays0["feature_columns"]), attacks, args.batadal_s, args.batadal_g, 3600, note)


def strip_hai_attack_point(value: str) -> str:
    text = str(value)
    return text.split(";", 1)[1] if ";" in text else text


def hai_bundle(args: argparse.Namespace) -> DatasetBundle:
    arrays = HAI.load_hai_arrays(argparse.Namespace(data_dir=args.hai_data_dir, sample_size=None))
    selected = rows_from_selected_csv(Path(args.hai_selected))
    if args.hai_use_geco_ignore_roster:
        selected = [row for row in selected if str(row["target"]) not in HAI_GECO_IGNORED]
    cache: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for row in selected:
        target = str(row["target"])
        variable_type = str(row.get("variable_type", "sensor"))
        rows.append(row)
        if variable_type == "persistence":
            cache[target] = {
                "train": HAI.persistence_residual(arrays, target, split="train"),
                "test": HAI.persistence_residual(arrays, target, split="test"),
            }
        else:
            equation = str(row.get("sympy_format", row.get("equation", "")))
            cache[target] = {
                "train": HAI.prediction_residual(arrays, target, equation, split="train"),
                "test": HAI.prediction_residual(arrays, target, equation, split="test"),
            }
    attacks_by_id = {int(item["id"]): item for item in arrays["attacks"]}
    attacks: list[AttackRecord] = []
    attack_ids = arrays["attack_ids"]
    timestamps = arrays["test_timestamps"]
    for attack_id in sorted(int(x) for x in np.unique(attack_ids) if int(x) > 0):
        idx = np.flatnonzero(attack_ids == attack_id)
        if idx.size == 0:
            continue
        meta = attacks_by_id.get(attack_id)
        if meta is None:
            raise ValueError(f"Missing HAI attacks.json metadata for attack id {attack_id}")
        tags = tuple(strip_hai_attack_point(str(item)) for item in meta.get("attack_point", []))
        attacks.append(
            AttackRecord(
                dataset="HAI",
                attack_id=str(attack_id),
                start=int(idx[0]),
                end=int(idx[-1]),
                attacked_tags=tags,
                source="data/hai/ipal/attacks.json::attack_point",
                official_start=int(meta.get("start")) if meta.get("start") is not None else int(timestamps[idx[0]]),
                official_end=int(meta.get("end")) if meta.get("end") is not None else int(timestamps[idx[-1]]),
            )
        )
    return DatasetBundle("HAI", arrays, rows, cache, list(arrays["feature_columns"]), attacks, args.hai_s, args.hai_g, 1, "HAI seed-0 safe grammar without GeCo ignored variables")


def fit_and_run_trace(bundle: DatasetBundle, target: str) -> dict[str, Any]:
    residuals = bundle.residual_cache[target]
    if bundle.dataset == "HAI":
        params = HAI.fit_cusum_params_sequences(
            HAI.train_residual_blocks(bundle.arrays, residuals["train"], fit_only=None),
            s=float(bundle.s),
            g=float(bundle.g),
        )
        cusum, _, _, _ = HAI.run_cusum_sequences(
            HAI.split_flat_by_blocks(residuals["test"], bundle.arrays["test_block_lengths"]),
            params,
        )
    else:
        params = fit_cusum_params(residuals["train"], s=float(bundle.s), g=float(bundle.g))
        if "test_block_lengths" in bundle.arrays:
            cusum = run_cusum_reset_blocks(residuals["test"], params, bundle.arrays["test_block_lengths"])
        else:
            cusum, _ = run_cusum(residuals["test"], params)
    alarm = (np.asarray(cusum, dtype=np.float64) >= float(params.threshold)).astype(np.int8)
    return {
        "target": target,
        "params": params,
        "cusum": np.asarray(cusum, dtype=np.float64),
        "alarm": alarm,
        "residual_test": np.asarray(residuals["test"], dtype=np.float64),
    }


def run_cusum_reset_blocks(residuals: np.ndarray, params: CusumParams, block_lengths: Sequence[int]) -> np.ndarray:
    blocks = []
    start = 0
    for length in block_lengths:
        end = start + int(length)
        block_cusum, _ = run_cusum(np.asarray(residuals)[start:end], params)
        blocks.append(np.asarray(block_cusum, dtype=np.float64))
        start = end
    if start != len(residuals):
        raise ValueError(f"CUSUM block lengths sum to {start}, but residual vector has length {len(residuals)}")
    return np.concatenate(blocks) if blocks else np.asarray([], dtype=np.float64)


def threshold_crossings(cusum: np.ndarray, threshold: float) -> np.ndarray:
    c = np.asarray(cusum, dtype=np.float64)
    above = c >= float(threshold)
    prev_below = np.ones_like(above, dtype=bool)
    if above.size > 1:
        prev_below[1:] = c[:-1] < float(threshold)
    return above & prev_below


def interval_overlaps(alarm: np.ndarray, start: int, end: int) -> bool:
    lo = max(0, int(start))
    hi = min(int(end), int(alarm.shape[0]) - 1)
    return bool(hi >= lo and np.any(alarm[lo : hi + 1] > 0))


def alert_nodes_for_attacks(bundle: DatasetBundle, traces: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for attack in bundle.attacks:
        lo = max(0, int(attack.start))
        hi = int(attack.end)
        for target, trace in sorted(traces.items()):
            cusum = trace["cusum"]
            threshold = float(trace["params"].threshold)
            hi_clip = min(hi, cusum.shape[0] - 1)
            if hi_clip < lo:
                continue
            crossings = threshold_crossings(cusum, threshold)
            local = np.flatnonzero(crossings[lo : hi_clip + 1])
            overlap = interval_overlaps(trace["alarm"], lo, hi_clip)
            if local.size == 0 and not overlap:
                continue
            first_crossing = int(lo + local[0]) if local.size else None
            segment = cusum[lo : hi_clip + 1]
            max_cusum = float(np.max(segment)) if segment.size else 0.0
            rows.append(
                {
                    "dataset": bundle.dataset,
                    "attack_id": attack.attack_id,
                    "variable": target,
                    "new_threshold_crossing": bool(local.size > 0),
                    "overlap_alarm": bool(overlap),
                    "first_crossing_time": first_crossing,
                    "first_crossing_index": first_crossing,
                    "detection_delay_seconds": None
                    if first_crossing is None
                    else int((first_crossing - lo) * bundle.sample_period_seconds),
                    "max_cusum": max_cusum,
                    "threshold": threshold,
                    "max_threshold_ratio": float(max_cusum / max(threshold, 1e-12)),
                }
            )
    return pd.DataFrame(rows)


def nearest_distance(graph: nx.Graph, node: str, targets: Sequence[str]) -> tuple[float, str | None]:
    if node in targets:
        return 0.0, node
    if node not in graph:
        return math.inf, None
    best = math.inf
    best_target: str | None = None
    for target in targets:
        if target not in graph:
            continue
        try:
            dist = nx.shortest_path_length(graph, node, target)
        except nx.NetworkXNoPath:
            continue
        if dist < best:
            best = float(dist)
            best_target = target
    return best, best_target


def localization_tables(
    bundle: DatasetBundle,
    graph: nx.Graph,
    alert_nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = alert_nodes[alert_nodes["new_threshold_crossing"] == True].copy()
    per_attack_rows = []
    per_alert_rows = []
    for attack in bundle.attacks:
        attack_alerts = primary[primary["attack_id"].astype(str) == str(attack.attack_id)].copy()
        alert_records = []
        for _, row in attack_alerts.iterrows():
            alert_node = str(row["variable"])
            dist, nearest = nearest_distance(graph, alert_node, attack.attacked_tags)
            alert_records.append((alert_node, dist, nearest, row))
            per_alert_rows.append(
                {
                    "dataset": bundle.dataset,
                    "attack_id": attack.attack_id,
                    "attacked_tags": ";".join(attack.attacked_tags),
                    "alert_node": alert_node,
                    "distance_to_nearest_attacked_tag": dist,
                    "nearest_attacked_tag": nearest,
                    "first_crossing_time": row["first_crossing_time"],
                    "detection_delay_seconds": row["detection_delay_seconds"],
                    "max_threshold_ratio": row["max_threshold_ratio"],
                }
            )
        detected = bool(alert_records)
        if detected:
            distances = [dist for _, dist, _, _ in alert_records]
            nearest_idx = int(np.argmin(distances))
            nearest_alert_node, nearest_dist, nearest_target, _ = alert_records[nearest_idx]
            mean_distance = float(np.mean(distances))
            median_distance = float(np.median(distances))
        else:
            nearest_alert_node = None
            nearest_target = None
            nearest_dist = math.inf
            mean_distance = math.inf
            median_distance = math.inf
        per_attack_rows.append(
            {
                "dataset": bundle.dataset,
                "attack_id": attack.attack_id,
                "attacked_tags": ";".join(attack.attacked_tags),
                "alert_nodes": ";".join(node for node, *_ in alert_records),
                "n_alert_nodes": len(alert_records),
                "detected": detected,
                "nearest_alert_node": nearest_alert_node,
                "nearest_attack_target": nearest_target,
                "nearest_distance": nearest_dist,
                "mean_alert_distance": mean_distance,
                "median_alert_distance": median_distance,
                "exact_target_alert": bool(nearest_dist == 0),
                "within_1_hop": bool(nearest_dist <= 1),
                "within_2_hops": bool(nearest_dist <= 2),
                "within_3_hops": bool(nearest_dist <= 3),
                "disconnected_alert_count": int(sum(math.isinf(dist) for _, dist, _, _ in alert_records)),
            }
        )
    return pd.DataFrame(per_attack_rows), pd.DataFrame(per_alert_rows)


def _percent(mask: np.ndarray) -> float:
    return float(np.mean(mask) * 100.0) if mask.size else float("nan")


def aggregate_from_tables(dataset: str, per_attack: pd.DataFrame, per_alert: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attack_rows: list[dict[str, Any]] = []
    distances = pd.to_numeric(per_attack["nearest_distance"], errors="coerce").to_numpy(dtype=np.float64)
    detected = per_attack["detected"].astype(bool).to_numpy()
    finite = np.isfinite(distances)
    n = len(per_attack)
    attack_rows.append(
        {
            "dataset": dataset,
            "evaluation": "end_to_end_all_attacks",
            "denominator": n,
            "pct_D_eq_0": _percent(distances == 0),
            "pct_D_le_1": _percent(distances <= 1),
            "pct_D_le_2": _percent(distances <= 2),
            "pct_D_le_3": _percent(distances <= 3),
            "pct_undetected": _percent(~detected),
            "pct_disconnected": _percent(detected & ~finite),
            "mean_D": float(np.mean(distances[finite])) if np.any(finite) else math.inf,
            "median_D": float(np.median(distances[finite])) if np.any(finite) else math.inf,
        }
    )
    detected_distances = distances[detected]
    detected_finite = np.isfinite(detected_distances)
    attack_rows.append(
        {
            "dataset": dataset,
            "evaluation": "conditional_detected_attacks",
            "denominator": int(np.sum(detected)),
            "pct_D_eq_0": _percent(detected_distances == 0),
            "pct_D_le_1": _percent(detected_distances <= 1),
            "pct_D_le_2": _percent(detected_distances <= 2),
            "pct_D_le_3": _percent(detected_distances <= 3),
            "pct_undetected": 0.0,
            "pct_disconnected": _percent(~detected_finite),
            "mean_D": float(np.mean(detected_distances[detected_finite])) if np.any(detected_finite) else math.inf,
            "median_D": float(np.median(detected_distances[detected_finite])) if np.any(detected_finite) else math.inf,
        }
    )
    alert_rows: list[dict[str, Any]] = []
    if per_alert.empty:
        alert_rows.append(
            {
                "dataset": dataset,
                "evaluation": "all_alert_nodes",
                "denominator": 0,
                "pct_d_eq_0": float("nan"),
                "pct_d_le_1": float("nan"),
                "pct_d_le_2": float("nan"),
                "pct_d_le_3": float("nan"),
                "pct_disconnected": float("nan"),
                "mean_d": math.inf,
                "median_d": math.inf,
            }
        )
    else:
        alert_dist = pd.to_numeric(per_alert["distance_to_nearest_attacked_tag"], errors="coerce").to_numpy(dtype=np.float64)
        alert_finite = np.isfinite(alert_dist)
        alert_rows.append(
            {
                "dataset": dataset,
                "evaluation": "all_alert_nodes",
                "denominator": len(per_alert),
                "pct_d_eq_0": _percent(alert_dist == 0),
                "pct_d_le_1": _percent(alert_dist <= 1),
                "pct_d_le_2": _percent(alert_dist <= 2),
                "pct_d_le_3": _percent(alert_dist <= 3),
                "pct_disconnected": _percent(~alert_finite),
                "mean_d": float(np.mean(alert_dist[alert_finite])) if np.any(alert_finite) else math.inf,
                "median_d": float(np.median(alert_dist[alert_finite])) if np.any(alert_finite) else math.inf,
            }
        )
    return attack_rows, alert_rows


def bootstrap_attack_metrics(
    dataset: str,
    per_attack: pd.DataFrame,
    per_alert: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if n_boot <= 0 or per_attack.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    attacks = per_attack["attack_id"].astype(str).tolist()
    rows = []
    for _ in range(int(n_boot)):
        sample = rng.choice(attacks, size=len(attacks), replace=True)
        attack_sample = pd.concat([per_attack[per_attack["attack_id"].astype(str) == attack_id] for attack_id in sample], ignore_index=True)
        alert_sample = pd.concat([per_alert[per_alert["attack_id"].astype(str) == attack_id] for attack_id in sample], ignore_index=True) if not per_alert.empty else per_alert
        attack_rows, alert_rows = aggregate_from_tables(dataset, attack_sample, alert_sample)
        rows.extend(attack_rows + alert_rows)
    boot = pd.DataFrame(rows)
    ci_rows = []
    metric_cols = [col for col in boot.columns if col not in {"dataset", "evaluation", "denominator"}]
    for (ds, evaluation), group in boot.groupby(["dataset", "evaluation"], sort=True):
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            ci_rows.append(
                {
                    "dataset": ds,
                    "evaluation": evaluation,
                    "metric": metric,
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "bootstrap_samples": int(values.shape[0]),
                    "bootstrap_seed": int(seed),
                }
            )
    return pd.DataFrame(ci_rows)


def write_dataset_outputs(
    bundle: DatasetBundle,
    graph: nx.DiGraph,
    edge_df: pd.DataFrame,
    traces: dict[str, dict[str, Any]],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slug = bundle.dataset.lower()
    nx.write_graphml(graph, out_dir / f"{slug}_dependency_graph.graphml")
    edge_df.to_csv(out_dir / f"{slug}_dependency_edges.csv", index=False)
    write_json(out_dir / f"{slug}_dependency_graph_stats.json", graph_stats(graph))
    attack_targets_table(bundle.attacks).to_csv(out_dir / f"{slug}_attack_targets.csv", index=False)
    alert_nodes = alert_nodes_for_attacks(bundle, traces)
    alert_nodes.to_csv(out_dir / f"{slug}_attack_alert_nodes.csv", index=False)
    per_attack, per_alert = localization_tables(bundle, graph.to_undirected(), alert_nodes)
    per_attack.to_csv(out_dir / f"{slug}_localization_per_attack.csv", index=False)
    per_alert.to_csv(out_dir / f"{slug}_localization_per_alert.csv", index=False)
    attack_agg, alert_agg = aggregate_from_tables(bundle.dataset, per_attack, per_alert)
    aggregate = pd.DataFrame(attack_agg)
    alert_aggregate = pd.DataFrame(alert_agg)
    aggregate.to_csv(out_dir / f"{slug}_localization_aggregate_attacks.csv", index=False)
    alert_aggregate.to_csv(out_dir / f"{slug}_localization_aggregate_alerts.csv", index=False)
    bootstrap = bootstrap_attack_metrics(bundle.dataset, per_attack, per_alert, n_boot=500, seed=BOOTSTRAP_SEED)
    bootstrap.to_csv(out_dir / f"{slug}_localization_bootstrap_ci.csv", index=False)
    return alert_nodes, per_attack, per_alert, aggregate


def swat_prediction_trace(bundle: DatasetBundle, target: str) -> dict[str, np.ndarray] | None:
    model = next((row for row in bundle.selected_rows if str(row["target"]) == target), None)
    if model is None or str(model.get("source")) == "actuator_persistence":
        return None
    idx = bundle.feature_columns.index(target)
    current = bundle.arrays["test_current"]
    nxt = bundle.arrays["test_next"]
    pred_delta = evaluate_equation(str(model.get("sympy_format", model.get("equation", ""))), bundle.feature_columns, current)
    pred_next = reconstruct_next_from_delta(current[:, idx], pred_delta)
    observed = nxt[:, idx].astype(np.float64)
    return {
        "observed": observed,
        "prediction": np.asarray(pred_next, dtype=np.float64),
        "residual": np.abs(observed - np.asarray(pred_next, dtype=np.float64)),
    }


def plot_swat_dpit301_graph(bundle: DatasetBundle, graph: nx.DiGraph, per_attack: pd.DataFrame, out_dir: Path) -> None:
    candidates = [attack for attack in bundle.attacks if "DPIT301" in attack.attacked_tags]
    if not candidates:
        return
    attack = candidates[0]
    row = per_attack[per_attack["attack_id"].astype(str) == str(attack.attack_id)]
    alert_nodes = set()
    if not row.empty:
        alert_nodes = {tag for tag in str(row.iloc[0]["alert_nodes"]).split(";") if tag}
    attacked = set(attack.attacked_tags)
    pos = nx.spring_layout(graph.to_undirected(), seed=1337, iterations=200)
    plt.figure(figsize=(11, 8.5))
    node_colors = ["#d8ecff" if n in alert_nodes else "white" for n in graph.nodes()]
    edge_colors = ["#b9b9b9"] * graph.number_of_edges()
    nx.draw_networkx_edges(graph, pos, alpha=0.25, width=0.7, arrows=True, edge_color=edge_colors, arrowsize=6)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, edgecolors="#555555", linewidths=0.8, node_size=520)
    nx.draw_networkx_nodes(graph, pos, nodelist=sorted(attacked), node_color="none", edgecolors="#d62728", linewidths=2.2, node_size=720)
    both = sorted(attacked & alert_nodes)
    if both:
        nx.draw_networkx_nodes(graph, pos, nodelist=both, node_color="#d8ecff", edgecolors="#d62728", linewidths=2.8, node_size=800)
    nx.draw_networkx_labels(graph, pos, font_size=6)
    plt.title(f"SWaT DPIT301 Attack {attack.attack_id}: ASID Dependency Graph", fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "swat_dpit301_dependency_graph.pdf")
    plt.savefig(out_dir / "swat_dpit301_dependency_graph.png", dpi=240)
    plt.close()


def plot_swat_dpit301_trace(bundle: DatasetBundle, traces: dict[str, dict[str, Any]], out_dir: Path) -> None:
    candidates = [attack for attack in bundle.attacks if "DPIT301" in attack.attacked_tags]
    if not candidates or "DPIT301" not in traces:
        return
    attack = candidates[0]
    pred = swat_prediction_trace(bundle, "DPIT301")
    if pred is None:
        return
    trace = traces["DPIT301"]
    pad = 500
    lo = max(0, attack.start - pad)
    hi = min(len(trace["cusum"]) - 1, attack.end + pad)
    x = np.arange(lo, hi + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(x, pred["observed"][lo : hi + 1], color="black", lw=0.9, label="observed")
    axes[0].plot(x, pred["prediction"][lo : hi + 1], color="#d62728", lw=0.9, label="prediction")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("DPIT301")
    axes[1].plot(x, pred["residual"][lo : hi + 1], color="#666666", lw=0.8)
    axes[1].set_ylabel("|err|")
    axes[2].plot(x, trace["cusum"][lo : hi + 1], color="#008b8b", lw=0.8)
    axes[2].axhline(float(trace["params"].threshold), color="#d62728", lw=0.8, ls="--")
    axes[2].set_ylabel("CUSUM")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.axvspan(attack.start, attack.end, color="#f4a6b7", alpha=0.35)
        ax.grid(alpha=0.18, lw=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "swat_dpit301_trace.pdf")
    plt.savefig(out_dir / "swat_dpit301_trace.png", dpi=240)
    plt.close()


def plot_aggregate_bars(aggregate_all: pd.DataFrame, out_dir: Path) -> None:
    primary = aggregate_all[aggregate_all["evaluation"] == "end_to_end_all_attacks"].copy()
    if primary.empty:
        return
    metrics = [("pct_D_eq_0", "<=0"), ("pct_D_le_1", "<=1"), ("pct_D_le_2", "<=2"), ("pct_D_le_3", "<=3")]
    rows = []
    for _, row in primary.iterrows():
        for metric, label in metrics:
            rows.append({"dataset": row["dataset"], "radius": label, "percentage": row[metric]})
    plot_df = pd.DataFrame(rows)
    plot_df.to_csv(out_dir / "aggregate_localization_plot_values.csv", index=False)
    datasets = primary["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.18
    plt.figure(figsize=(8.5, 3.8))
    for i, (_, label) in enumerate(metrics):
        values = [float(primary.loc[primary["dataset"] == ds, metrics[i][0]].iloc[0]) for ds in datasets]
        plt.bar(x + (i - 1.5) * width, values, width=width, label=label)
    plt.xticks(x, datasets)
    plt.ylabel("attack scenarios [%]")
    plt.ylim(0, 100)
    plt.legend(title="nearest alert")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "aggregate_localization_all_attacks.pdf")
    plt.savefig(out_dir / "aggregate_localization_all_attacks.png", dpi=240)
    plt.close()

    detected = aggregate_all[aggregate_all["evaluation"] == "conditional_detected_attacks"].copy()
    if detected.empty:
        return
    plt.figure(figsize=(8.5, 3.8))
    datasets = detected["dataset"].tolist()
    x = np.arange(len(datasets))
    for i, (metric, label) in enumerate(metrics):
        values = [float(detected.loc[detected["dataset"] == ds, metric].iloc[0]) for ds in datasets]
        plt.bar(x + (i - 1.5) * width, values, width=width, label=label)
    plt.xticks(x, datasets)
    plt.ylabel("detected scenarios [%]")
    plt.ylim(0, 100)
    plt.legend(title="nearest alert")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "aggregate_localization_detected_attacks.pdf")
    plt.savefig(out_dir / "aggregate_localization_detected_attacks.png", dpi=240)
    plt.close()


def traces_for_bundle(bundle: DatasetBundle) -> dict[str, dict[str, Any]]:
    traces = {}
    for row in bundle.selected_rows:
        target = str(row["target"])
        traces[target] = fit_and_run_trace(bundle, target)
    return traces


def run_dataset(bundle: DatasetBundle, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attack_target_nodes = sorted({tag for attack in bundle.attacks for tag in attack.attacked_tags})
    validate_attack_targets(bundle.attacks, sorted(set(bundle.feature_columns) | set(attack_target_nodes)))
    unobserved_targets = unobserved_attack_targets(bundle.attacks, bundle.feature_columns)
    unobserved_targets.to_csv(out_dir / f"{bundle.dataset.lower()}_unobserved_attack_targets.csv", index=False)
    graph, edge_df = build_dependency_graph(bundle.selected_rows, bundle.feature_columns, extra_nodes=attack_target_nodes)
    traces = traces_for_bundle(bundle)
    alert_nodes, per_attack, per_alert, aggregate = write_dataset_outputs(bundle, graph, edge_df, traces, out_dir)
    if bundle.dataset == "SWAT":
        plot_swat_dpit301_graph(bundle, graph, per_attack, out_dir)
        plot_swat_dpit301_trace(bundle, traces, out_dir)
    print(f"\n{bundle.dataset} inspectable examples")
    print(per_attack.head(5).to_string(index=False))
    write_json(
        out_dir / f"{bundle.dataset.lower()}_localization_run_config.json",
        {
            "dataset": bundle.dataset,
            "S": bundle.s,
            "G": bundle.g,
            "sample_period_seconds": bundle.sample_period_seconds,
            "source_note": bundle.source_note,
            "selected_targets": sorted(str(row["target"]) for row in bundle.selected_rows),
            "unobserved_official_attack_targets": unobserved_targets.to_dict("records"),
            "graph_stats": graph_stats(graph),
        },
    )
    return aggregate, pd.read_csv(out_dir / f"{bundle.dataset.lower()}_localization_aggregate_alerts.csv"), per_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASID-ICS alert localization experiment.")
    parser.add_argument("--out", default="artifacts/localization")
    parser.add_argument("--datasets", default="SWAT,WADI,BATADAL,HAI")
    parser.add_argument("--swat-experiment", default="configs/experiment/swat_mlp_current_val20.yaml")
    parser.add_argument("--swat-train-csv", default="data/swat/raw/swat_train.csv")
    parser.add_argument("--swat-test-csv", default="data/swat/raw/swat_test.csv")
    parser.add_argument("--swat-selected", default="results/swat/selected_equations.csv")
    parser.add_argument("--wadi-train-csv", default="data/wadi/raw/wadi_train.csv")
    parser.add_argument("--wadi-test-csv", default="data/wadi/raw/wadi_test.csv")
    parser.add_argument("--wadi-selected", default="results/wadi/selected_equations.csv")
    parser.add_argument("--batadal-train-csv", default="data/batadal/processed/train.csv")
    parser.add_argument("--batadal-test-csv", default="data/batadal/processed/test_dataset04.csv")
    parser.add_argument("--batadal-test-csv-extra", default="data/batadal/processed/test_dataset_test.csv")
    parser.add_argument("--batadal-selected", default="results/batadal/selected_equations.csv")
    parser.add_argument("--hai-data-dir", default="data/hai/ipal")
    parser.add_argument("--hai-selected", default="artifacts/experiments/hai_baseline_seed0/selected_equations.csv")
    parser.add_argument("--hai-use-geco-ignore-roster", action="store_true", default=True)
    parser.add_argument("--swat-s", type=float, default=OPERATING_POINTS["SWAT"][0])
    parser.add_argument("--swat-g", type=float, default=OPERATING_POINTS["SWAT"][1])
    parser.add_argument("--wadi-s", type=float, default=OPERATING_POINTS["WADI"][0])
    parser.add_argument("--wadi-g", type=float, default=OPERATING_POINTS["WADI"][1])
    parser.add_argument("--batadal-s", type=float, default=OPERATING_POINTS["BATADAL"][0])
    parser.add_argument("--batadal-g", type=float, default=OPERATING_POINTS["BATADAL"][1])
    parser.add_argument("--hai-s", type=float, default=OPERATING_POINTS["HAI"][0])
    parser.add_argument("--hai-g", type=float, default=OPERATING_POINTS["HAI"][1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = {token.strip().upper() for token in str(args.datasets).split(",") if token.strip()}
    builders = {
        "SWAT": swat_bundle,
        "WADI": wadi_bundle,
        "BATADAL": batadal_bundle,
        "HAI": hai_bundle,
    }
    aggregate_rows = []
    alert_aggregate_rows = []
    per_attack_frames = []
    for dataset in ["SWAT", "WADI", "BATADAL", "HAI"]:
        if dataset not in requested:
            continue
        print(f"\n=== {dataset} localization ===", flush=True)
        bundle = builders[dataset](args)
        aggregate, alert_aggregate, per_attack = run_dataset(bundle, out_dir)
        aggregate_rows.append(aggregate)
        alert_aggregate_rows.append(alert_aggregate)
        per_attack_frames.append(per_attack)
    aggregate_all = pd.concat(aggregate_rows, ignore_index=True) if aggregate_rows else pd.DataFrame()
    alert_aggregate_all = pd.concat(alert_aggregate_rows, ignore_index=True) if alert_aggregate_rows else pd.DataFrame()
    aggregate_all.to_csv(out_dir / "localization_aggregate_attacks.csv", index=False)
    alert_aggregate_all.to_csv(out_dir / "localization_aggregate_alerts.csv", index=False)
    if per_attack_frames:
        pd.concat(per_attack_frames, ignore_index=True).to_csv(out_dir / "localization_per_attack_all_datasets.csv", index=False)
    plot_aggregate_bars(aggregate_all, out_dir)

    primary = aggregate_all[aggregate_all["evaluation"] == "end_to_end_all_attacks"].copy()
    if not primary.empty:
        display = primary[
            [
                "dataset",
                "denominator",
                "pct_D_eq_0",
                "pct_D_le_1",
                "pct_D_le_2",
                "pct_D_le_3",
                "mean_D",
                "pct_disconnected",
            ]
        ].rename(
            columns={
                "dataset": "Dataset",
                "denominator": "Scenarios",
                "pct_D_eq_0": "D=0",
                "pct_D_le_1": "D<=1",
                "pct_D_le_2": "D<=2",
                "pct_D_le_3": "D<=3",
                "mean_D": "Mean hops (finite)",
                "pct_disconnected": "Disconnected",
            }
        )
        detected_counts = []
        for df in per_attack_frames:
            detected_counts.append(int(df["detected"].astype(bool).sum()))
        display.insert(2, "Detected", detected_counts)
        print("\nDataset | Scenarios | Detected | D=0 | D<=1 | D<=2 | D<=3 | Mean hops (detected) | Disconnected")
        print(display.to_string(index=False))

    if not alert_aggregate_all.empty:
        alert_display = alert_aggregate_all[
            [
                "dataset",
                "denominator",
                "pct_d_eq_0",
                "pct_d_le_1",
                "pct_d_le_2",
                "pct_d_le_3",
                "mean_d",
                "median_d",
                "pct_disconnected",
            ]
        ].rename(
            columns={
                "dataset": "Dataset",
                "denominator": "Alert nodes",
                "pct_d_eq_0": "d=0",
                "pct_d_le_1": "d<=1",
                "pct_d_le_2": "d<=2",
                "pct_d_le_3": "d<=3",
                "mean_d": "Mean",
                "median_d": "Median",
                "pct_disconnected": "Disconnected",
            }
        )
        print("\nDataset | Alert nodes | d=0 | d<=1 | d<=2 | d<=3 | Mean | Median | Disconnected")
        print(alert_display.to_string(index=False))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
