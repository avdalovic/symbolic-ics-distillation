#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.detection.symbolic_eval import evaluate_equation


def _load_day1_module():
    path = REPO_ROOT / "scripts" / "run_swat_1sec_pysr.py"
    spec = importlib.util.spec_from_file_location("run_swat_1sec_pysr", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DAY1 = _load_day1_module()

SENSOR_NAMES = {
    "FIT101",
    "LIT101",
    "AIT201",
    "AIT202",
    "AIT203",
    "FIT201",
    "DPIT301",
    "FIT301",
    "LIT301",
    "AIT401",
    "AIT402",
    "FIT401",
    "LIT401",
    "AIT501",
    "AIT502",
    "AIT503",
    "AIT504",
    "FIT501",
    "FIT502",
    "FIT503",
    "FIT504",
    "PIT501",
    "PIT502",
    "PIT503",
    "FIT601",
}


def is_actuator(tag: str) -> bool:
    return DAY1.is_actuator("SWAT", tag)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def load_swat(args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    train, _, _, _, feature_columns, _ = DAY1.load_swat_1sec_arrays(args)
    return train, feature_columns


def pareto_path(args: argparse.Namespace, target: str) -> Path:
    if target in SENSOR_NAMES:
        return Path(args.sensor_next_pareto_root) / f"{target}_actual_next" / "pareto_front_scored.csv"
    return Path(args.actuator_pareto_root) / f"{target}_actual_next" / "pareto_front_scored.csv"


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def is_numeric_constant(text: str) -> bool:
    try:
        float(str(text).strip())
    except Exception:
        return False
    return True


def sorted_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(-np.inf)
    out["complexity"] = pd.to_numeric(out["complexity"], errors="coerce").fillna(np.inf)
    return out.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=False).rename(columns={"index": "pareto_row"})


def bad_fraction(equation: str, feature_columns: list[str], x_check: np.ndarray) -> float:
    pred = evaluate_equation(equation, feature_columns, x_check)
    if pred.size == 0:
        return 1.0
    return float(np.mean(~np.isfinite(pred)))


def sympy_expr(expr: str, feature_columns: list[str]) -> sp.Expr | None:
    symbols = {name: sp.Symbol(name) for name in feature_columns}
    symbols.update({"square": lambda x: x**2, "cube": lambda x: x**3, "abs_op": sp.Abs, "abs": sp.Abs, "Abs": sp.Abs})
    try:
        return sp.sympify(str(expr).replace("^", "**"), locals=symbols)
    except Exception:
        return None


def direct_denominator_symbols(expr: sp.Expr, feature_columns: list[str]) -> set[str]:
    denom = sp.denom(sp.together(expr))
    if denom == 1:
        return set()
    factors = sp.factor(denom).as_ordered_factors()
    out: set[str] = set()
    for factor in factors:
        if isinstance(factor, sp.Symbol):
            out.add(str(factor))
        elif isinstance(factor, sp.Pow) and isinstance(factor.base, sp.Symbol):
            out.add(str(factor.base))
    return {name for name in out if name in feature_columns}


def division_safety_reason(
    *,
    target: str,
    expr_text: str,
    feature_columns: list[str],
    zero_value_features: set[str],
) -> str | None:
    expr = sympy_expr(expr_text, feature_columns)
    if expr is None:
        return "parse_failed"
    denom_symbols = direct_denominator_symbols(expr, feature_columns)
    if target == "P302" and "DPIT301" in denom_symbols:
        return "p302_divides_by_dpit301_expression"
    zero_symbols = sorted(denom_symbols & zero_value_features)
    if zero_symbols:
        return "division_by_zero_valued_variable:" + ",".join(zero_symbols)
    return None


def select_for_target(
    *,
    target: str,
    df: pd.DataFrame,
    feature_columns: list[str],
    x_check: np.ndarray,
    zero_value_features: set[str],
    r2_threshold: float,
    in_scope: bool,
    nan_threshold: float,
) -> dict[str, Any]:
    fallback_used = False
    unsafe_reasons: list[str] = []
    selected = None
    selected_nan_fraction = 1.0
    selected_division_reason = None
    candidates = sorted_candidates(df)
    for _, row in candidates.iterrows():
        complexity = safe_float(row.get("complexity"))
        equation = str(row.get("equation", ""))
        expr = str(row.get("sympy_format", equation))
        if complexity <= 1.0 and is_numeric_constant(equation):
            fallback_used = True
            unsafe_reasons.append("skipped_numeric_constant")
            continue
        div_reason = division_safety_reason(
            target=target,
            expr_text=expr,
            feature_columns=feature_columns,
            zero_value_features=zero_value_features,
        )
        if div_reason is not None:
            fallback_used = True
            unsafe_reasons.append(div_reason)
            continue
        frac = bad_fraction(expr, feature_columns, x_check)
        if frac > nan_threshold:
            fallback_used = True
            unsafe_reasons.append(f"nan_fraction>{nan_threshold}:{frac:.6g}")
            continue
        selected = row
        selected_nan_fraction = frac
        selected_division_reason = div_reason
        break

    nan_unfixable = False
    if selected is None:
        selected = candidates.iloc[0]
        expr = str(selected.get("sympy_format", selected.get("equation", "")))
        selected_nan_fraction = bad_fraction(expr, feature_columns, x_check)
        selected_division_reason = division_safety_reason(
            target=target,
            expr_text=expr,
            feature_columns=feature_columns,
            zero_value_features=zero_value_features,
        )
        nan_unfixable = selected_nan_fraction > nan_threshold

    fit_r2 = safe_float(selected.get("fit_r2_against_constant"))
    holdout_r2 = safe_float(selected.get("holdout_r2_against_constant"))
    threshold_pass = bool(np.isfinite(fit_r2) and np.isfinite(holdout_r2) and fit_r2 > r2_threshold and holdout_r2 > r2_threshold)
    quality_pass = bool(in_scope and threshold_pass and selected_nan_fraction <= nan_threshold and selected_division_reason is None)
    fail_reasons: list[str] = []
    if not in_scope:
        fail_reasons.append("out_of_scope")
    if not np.isfinite(fit_r2) or fit_r2 <= r2_threshold:
        fail_reasons.append(f"fit_r2<={r2_threshold}")
    if not np.isfinite(holdout_r2) or holdout_r2 <= r2_threshold:
        fail_reasons.append(f"holdout_r2<={r2_threshold}")
    if selected_nan_fraction > nan_threshold:
        fail_reasons.append("nan_unsafe_unfixable")
    if selected_division_reason is not None:
        fail_reasons.append(selected_division_reason)

    variable_type = "actuator" if is_actuator(target) else "sensor"
    return {
        "target": target,
        "variable_type": variable_type,
        "target_source": "actual_next",
        "selected_complexity": safe_float(selected.get("complexity")),
        "selected_score": safe_float(selected.get("score")),
        "selected_equation": str(selected.get("equation", "")),
        "sympy_format": str(selected.get("sympy_format", selected.get("equation", ""))),
        "fit_r2": fit_r2,
        "holdout_r2": holdout_r2,
        "nan_fraction": selected_nan_fraction,
        "quality_pass": quality_pass,
        "threshold_pass": threshold_pass,
        "in_scope": bool(in_scope),
        "nan_safe": bool(selected_nan_fraction <= nan_threshold),
        "fallback_used": bool(fallback_used),
        "unsafe_reasons_seen": ";".join(dict.fromkeys(unsafe_reasons)),
        "nan_unsafe_unfixable": bool(nan_unfixable),
        "division_safety_reason": selected_division_reason or "",
        "quality_fail_reason": ";".join(fail_reasons),
        "pareto_row": int(selected.get("pareto_row", -1)),
        "r2_threshold": float(r2_threshold),
    }


def build_selected_equations(
    *,
    args: argparse.Namespace,
    feature_columns: list[str],
    train: np.ndarray,
    x_check: np.ndarray,
    experiment_name: str,
    sensors_only: bool,
    r2_threshold: float,
    out_dir: Path,
) -> pd.DataFrame:
    min_abs = np.nanmin(np.abs(train.astype(np.float64)), axis=0)
    zero_value_features = {feature_columns[idx] for idx, value in enumerate(min_abs) if np.isfinite(value) and value < 1e-10}
    rows = []
    for target in feature_columns:
        in_scope = (target in SENSOR_NAMES) if sensors_only else True
        path = pareto_path(args, target)
        if not path.exists():
            rows.append(
                {
                    "target": target,
                    "variable_type": "actuator" if is_actuator(target) else "sensor",
                    "target_source": "actual_next",
                    "quality_pass": False,
                    "threshold_pass": False,
                    "in_scope": in_scope,
                    "quality_fail_reason": "missing_pareto",
                    "r2_threshold": float(r2_threshold),
                }
            )
            continue
        df = pd.read_csv(path)
        row = select_for_target(
            target=target,
            df=df,
            feature_columns=feature_columns,
            x_check=x_check,
            zero_value_features=zero_value_features,
            r2_threshold=float(r2_threshold),
            in_scope=in_scope,
            nan_threshold=float(args.nan_threshold),
        )
        row["experiment_name"] = experiment_name
        row["pareto_csv"] = str(path)
        rows.append(row)
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "selected_equations.csv", index=False)
    return table


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_experiment(
    *,
    args: argparse.Namespace,
    name: str,
    label: str,
    sensors_only: bool,
    r2_threshold: float,
    feature_columns: list[str],
    train: np.ndarray,
    x_check: np.ndarray,
) -> dict[str, Any]:
    started = time.time()
    out_dir = Path(args.out_root) / name
    selected = build_selected_equations(
        args=args,
        feature_columns=feature_columns,
        train=train,
        x_check=x_check,
        experiment_name=name,
        sensors_only=sensors_only,
        r2_threshold=r2_threshold,
        out_dir=out_dir,
    )
    run_command(
        [
            sys.executable,
            "scripts/evaluate_swat_1sec_detection.py",
            "--selected-equations",
            str(out_dir / "selected_equations.csv"),
            "--out",
            str(out_dir),
            "--s",
            str(args.s),
            "--g",
            str(args.g),
            "--fpa-expand-steps",
            str(args.fpa_expand_steps),
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/per_attack_analysis_swat_1sec.py",
            "--detection-dir",
            str(out_dir),
            "--out",
            str(out_dir / "per_attack_analysis.csv"),
        ]
    )
    results = json.loads((out_dir / "detection_results.json").read_text(encoding="utf-8"))
    details = pd.read_csv(out_dir / "per_sensor_details.csv") if (out_dir / "per_sensor_details.csv").exists() else pd.DataFrame()
    attack = pd.read_csv(out_dir / "per_attack_analysis.csv")
    metrics = results["metrics"]
    elapsed = time.time() - started
    summary = {
        "experiment": name,
        "label": label,
        "sensors_only": sensors_only,
        "r2_threshold": float(r2_threshold),
        "elapsed_seconds": elapsed,
        "variables_denominator": 25 if sensors_only else 51,
        "quality_pass": int(selected["quality_pass"].sum()),
        "monitored": int(results["num_monitored"]),
        "metrics": metrics,
        "selected": selected,
        "details": details,
        "attack": attack,
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "grid_summary.json", {k: v for k, v in summary.items() if k not in {"selected", "details", "attack"}})
    return summary


def fmt_pct(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):.1f}%"


def fmt_num(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):.1f}"


def print_detection_grid(summaries: list[dict[str, Any]]) -> None:
    ge = {"monitored": "47/51", "F1": 86.2, "eTaF1": 70.2, "FPA": 0, "Scen": 78.1}
    labels = [s["label"] for s in summaries]
    rows = []
    rows.append(["Variables included", *[f"{s['monitored']}/{s['variables_denominator']}" for s in summaries], ge["monitored"]])
    rows.append(["Quality pass", *[str(s["quality_pass"]) for s in summaries], "47"])
    rows.append(["Precision", *[fmt_pct(s["metrics"]["point_precision"]) for s in summaries], "-"])
    rows.append(["Recall", *[fmt_pct(s["metrics"]["point_recall"]) for s in summaries], "-"])
    rows.append(["F1", *[fmt_num(s["metrics"]["point_f1"]) for s in summaries], fmt_num(ge["F1"])])
    rows.append(["eTaP", *[fmt_num(s["metrics"]["eTaP"]) for s in summaries], "-"])
    rows.append(["eTaR", *[fmt_num(s["metrics"]["eTaR"]) for s in summaries], "-"])
    rows.append(["eTaF1", *[fmt_num(s["metrics"]["eTaF1"]) for s in summaries], fmt_num(ge["eTaF1"])])
    rows.append(["FPA", *[str(int(round(s["metrics"]["FPA"]))) for s in summaries], str(ge["FPA"])])
    rows.append(["Scen", *[fmt_pct(s["metrics"]["scenario_detection_rate"]) for s in summaries], fmt_pct(ge["Scen"])])
    table = pd.DataFrame(rows, columns=["metric", *labels, "GeCo"])
    print("=== Detection Experiment Grid ===")
    print(table.to_string(index=False))


def print_fpa_contributors(summaries: list[dict[str, Any]]) -> None:
    print("\nTop FPA contributors per experiment:")
    for summary in summaries:
        details = summary["details"]
        if details.empty:
            print(f"  {summary['label']}: (none)")
            continue
        top = details.sort_values(["num_alarm_intervals", "benign_alarm_rate"], ascending=False).head(3)
        items = [
            f"[{row['target']}, {int(row['num_alarm_intervals'])}, {float(row['benign_alarm_rate']):.4f}]"
            for _, row in top.iterrows()
        ]
        print(f"  {summary['label']}: " + ", ".join(items))


def print_attack_comparison(summaries: list[dict[str, Any]]) -> None:
    attacks = sorted(set().union(*[set(s["attack"]["attack_id"].astype(int)) for s in summaries]))
    rows = []
    for attack_id in attacks:
        row = {"Attack": attack_id}
        affected = ""
        include = False
        for summary in summaries:
            hit = summary["attack"][summary["attack"]["attack_id"].astype(int) == attack_id]
            if hit.empty:
                category = ""
            else:
                category = str(hit.iloc[0]["category"])
                affected = str(hit.iloc[0]["affected_tags"])
            row[summary["label"]] = category
            if category == "detection_failure":
                include = True
        row["Affected"] = affected
        if include:
            rows.append(row)
    print("\nPer-attack comparison (detection failures only):")
    if rows:
        cols = ["Attack", "Affected", *[s["label"] for s in summaries]]
        print(pd.DataFrame(rows)[cols].to_string(index=False))
    else:
        print("  (none)")


def print_threshold_impact(exp_a: dict[str, Any], exp_c: dict[str, Any]) -> None:
    a_pass = set(exp_a["selected"].loc[exp_a["selected"]["quality_pass"] == True, "target"].astype(str))
    c_table = exp_c["selected"]
    added = c_table[(c_table["quality_pass"] == True) & (~c_table["target"].astype(str).isin(a_pass))]
    print("\n=== R2 threshold impact ===")
    print("Variables passing at 0.2 but not 0.3:")
    if added.empty:
        print("  (none)")
        return
    cols = ["target", "fit_r2", "holdout_r2", "selected_equation"]
    print(added[cols].rename(columns={"selected_equation": "equation"}).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWaT 1-second detection experiment grid.")
    parser.add_argument("--experiment", default="configs/experiment/swat_mlp_current_val20.yaml")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--sensor-next-pareto-root", default="artifacts/swat_1sec/pareto_fronts_next")
    parser.add_argument("--actuator-pareto-root", default="artifacts/swat_1sec/pareto_fronts")
    parser.add_argument("--out-root", default="artifacts/swat_1sec/detection_grid")
    parser.add_argument("--nan-check-size", type=int, default=2000)
    parser.add_argument("--nan-threshold", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--s", type=float, default=1.42)
    parser.add_argument("--g", type=float, default=5.98)
    parser.add_argument("--fpa-expand-steps", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    print("Loading SWaT 1-second training data for grid selection")
    train, feature_columns = load_swat(args)
    rng = np.random.default_rng(int(args.seed))
    n_check = min(int(args.nan_check_size), int(train.shape[0]))
    x_check = train[np.sort(rng.choice(train.shape[0], size=n_check, replace=False))].astype(np.float64, copy=False)

    experiments = [
        ("exp_a_sensors_actuators_r2_0p3", "Exp A", False, 0.3),
        ("exp_b_sensors_only_r2_0p3", "Exp B", True, 0.3),
        ("exp_c_sensors_actuators_r2_0p2", "Exp C", False, 0.2),
        ("exp_d_sensors_only_r2_0p2", "Exp D", True, 0.2),
    ]
    summaries = []
    for name, label, sensors_only, threshold in experiments:
        print(f"\n=== Running {label}: sensors_only={sensors_only} r2>{threshold} ===", flush=True)
        summaries.append(
            run_experiment(
                args=args,
                name=name,
                label=label,
                sensors_only=sensors_only,
                r2_threshold=threshold,
                feature_columns=feature_columns,
                train=train,
                x_check=x_check,
            )
        )
    out_root = Path(args.out_root)
    rows = []
    for summary in summaries:
        m = summary["metrics"]
        rows.append(
            {
                "experiment": summary["experiment"],
                "label": summary["label"],
                "variables_included": summary["monitored"],
                "variables_denominator": summary["variables_denominator"],
                "quality_pass": summary["quality_pass"],
                "precision": m["point_precision"],
                "recall": m["point_recall"],
                "f1": m["point_f1"],
                "etap": m["eTaP"],
                "etar": m["eTaR"],
                "etaf1": m["eTaF1"],
                "fpa": m["FPA"],
                "scen": m["scenario_detection_rate"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "grid_summary.csv", index=False)
    print_detection_grid(summaries)
    print_fpa_contributors(summaries)
    print_attack_comparison(summaries)
    print_threshold_impact(summaries[0], summaries[2])
    write_json(
        out_root / "run_detection_grid_summary.json",
        {
            "elapsed_seconds": time.time() - started,
            "summary_csv": str(out_root / "grid_summary.csv"),
            "experiments": [
                {k: v for k, v in summary.items() if k not in {"selected", "details", "attack"}}
                for summary in summaries
            ],
        },
    )
    print(f"\nWrote {out_root / 'grid_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
