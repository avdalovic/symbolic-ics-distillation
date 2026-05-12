#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

TARGET_SOURCES = ["actual_next", "actual_delta", "mlp_next", "mlp_delta"]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_support_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    payload = read_json(config_path)
    for sensor, config in payload.items():
        if "local_features" not in config:
            raise ValueError(f"{sensor} missing local_features")
    return payload


def load_target_sensors(distill_dir: str | Path) -> list[str]:
    path = Path(distill_dir) / "distill_target_columns.json"
    return [str(item) for item in read_json(path)]


def load_geco_lookup(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    model_path = Path(path)
    if not model_path.exists():
        return {}
    payload = read_json(model_path)
    ci = payload.get("CI", {})
    if not isinstance(ci, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for target, entry in ci.items():
        if not isinstance(entry, dict):
            continue
        variables = [str(item) for item in entry.get("combination", [])]
        if len(variables) <= 1:
            notes = "constant/trivial or persistence-only template"
        elif variables and variables[0] == str(target):
            notes = "linear template with persistence/self and additional variables"
        else:
            notes = f"{entry.get('equation', 'unknown')} template"
        out[str(target)] = {
            "geco_reference_available": True,
            "geco_variables": variables,
            "geco_equation_type": entry.get("equation"),
            "geco_notes": notes,
        }
    return out


def extract_equation_features(equation: str, feature_columns: Sequence[str]) -> list[str]:
    text = str(equation)
    used: list[str] = []
    for name in feature_columns:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(name))}(?![A-Za-z0-9_])", text):
            used.append(str(name))
    return used


def parse_feature_cell(value: Any, feature_columns: Sequence[str] | None = None, equation: str | None = None) -> list[str]:
    if feature_columns is not None and equation is not None:
        return extract_equation_features(equation, feature_columns)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def classify_feature_support(features: Sequence[str], local_features: Sequence[str] | None) -> tuple[str, list[str]]:
    used = set(str(item) for item in features)
    if local_features is None:
        return ("constant" if not used else "unknown_support"), []
    local = set(str(item) for item in local_features)
    off_process = sorted(used - local)
    if not used:
        return "constant", []
    if not off_process:
        return "local", []
    if used & local:
        return "partially_local", off_process
    return "off_process", off_process


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def sort_metric_column(df: pd.DataFrame) -> str:
    if "holdout_mse" in df.columns and df["holdout_mse"].notna().any():
        return "holdout_mse"
    return "loss"


def select_rows(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    local_features: Sequence[str] | None,
) -> dict[str, pd.Series | None]:
    if df.empty:
        return {"lowest_loss": None, "score_selected": None, "simplest_nonconstant": None, "local_physical": None}

    rows: dict[str, pd.Series | None] = {}
    rows["lowest_loss"] = df.loc[df["loss"].astype(float).idxmin()] if "loss" in df.columns else df.iloc[0]

    best_equation = str(metadata.get("best_equation", ""))
    score_match = df[df["equation"].astype(str) == best_equation] if "equation" in df.columns else pd.DataFrame()
    if not score_match.empty:
        rows["score_selected"] = score_match.iloc[0]
    elif "score" in df.columns and df["score"].notna().any():
        rows["score_selected"] = df.loc[df["score"].astype(float).idxmax()]
    else:
        rows["score_selected"] = None

    nonconstant = df[df["features_used"].map(lambda values: len(values) > 0)]
    rows["simplest_nonconstant"] = (
        nonconstant.sort_values(["complexity", "loss"], ascending=[True, True]).iloc[0] if not nonconstant.empty else None
    )

    if local_features is None:
        rows["local_physical"] = None
    else:
        local = df[
            df["features_used"].map(
                lambda values: len(values) > 0 and set(str(item) for item in values).issubset(set(local_features))
            )
        ]
        rows["local_physical"] = (
            local.sort_values([sort_metric_column(local), "complexity"], ascending=[True, True]).iloc[0]
            if not local.empty
            else None
        )
    return rows


def row_value(row: pd.Series | None, key: str) -> Any:
    if row is None:
        return None
    value = row.get(key)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def equation_text(row: pd.Series | None) -> str:
    value = row_value(row, "equation")
    return "" if value is None else str(value)


def features_text(row: pd.Series | None) -> str:
    if row is None:
        return ""
    features = row.get("features_used", [])
    return ", ".join(features) if features else ""


def run_dir_name(sensor: str, target_source: str, operator_set: str = "restricted", mode: str = "unconstrained") -> str:
    return f"{sensor}_{target_source}_{operator_set}_{mode}"


def load_run(audit_root: Path, sensor: str, target_source: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None, Path]:
    run_dir = audit_root / run_dir_name(sensor, target_source)
    csv_path = run_dir / "pareto_front_scored.csv"
    if not csv_path.exists():
        csv_path = run_dir / "pareto_front.csv"
    metadata_path = run_dir / "metadata.json"
    if not csv_path.exists() or not metadata_path.exists():
        return None, None, run_dir
    metadata = read_json(metadata_path)
    df = pd.read_csv(csv_path)
    feature_columns = metadata.get("selected_features") or []
    df = df.copy()
    df["features_used"] = [
        extract_equation_features(str(eq), feature_columns) for eq in df.get("equation", pd.Series([], dtype=str))
    ]
    return df, metadata, run_dir


def load_linear_lookup(path: str | Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = (str(row["sensor"]), str(row["target_source"]), str(row["feature_set"]), str(row["model_type"]))
        lookup[key] = row.to_dict()
    return lookup


def coefficient_scale_warning(linear_lookup: dict[tuple[str, str, str, str], dict[str, Any]]) -> str | None:
    row = linear_lookup.get(("LIT101", "actual_delta", "local_support", "ols"))
    if not row:
        return None
    try:
        coefficients = json.loads(row.get("coefficients", "{}"))
    except Exception:
        return None
    fit101 = abs(float(coefficients.get("FIT101", 0.0)))
    fit201 = abs(float(coefficients.get("FIT201", 0.0)))
    if fit101 > 1.0 or fit201 > 1.0:
        return (
            "Coefficient scale differs by roughly 10x from the GeCo reference. "
            "Verify sampling interval/downsampling before claiming coefficient-level agreement."
        )
    return None


def summarize_runs(
    audit_root: Path,
    support_config: dict[str, Any],
    linear_lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    sensors: Sequence[str],
    geco_lookup: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    summary_rows: list[dict[str, Any]] = []
    narratives: dict[str, list[str]] = {sensor: [] for sensor in sensors}
    geco_lookup = geco_lookup or {}

    for sensor in sensors:
        has_support = sensor in support_config
        local_features = support_config[sensor]["local_features"] if has_support else None
        notes = support_config.get(sensor, {}).get("notes", "")
        geco = geco_lookup.get(sensor, {"geco_reference_available": False, "geco_variables": [], "geco_notes": ""})
        if has_support:
            narratives[sensor].append(f"Process context: {notes}")
        else:
            narratives[sensor].append("Process context: no hand-written local-support entry; local/off-process labels are unknown.")
        if geco.get("geco_reference_available"):
            narratives[sensor].append(
                f"GeCo reference variables: {geco.get('geco_variables', [])}; {geco.get('geco_notes', '')}"
            )
        for target_source in TARGET_SOURCES:
            df, metadata, run_dir = load_run(audit_root, sensor, target_source)
            if df is None or metadata is None:
                summary_rows.append(
                    {
                        "sensor": sensor,
                        "target_source": target_source,
                        "simplest_nonconstant_equation": "",
                        "score_selected_equation": "",
                        "lowest_loss_equation": "",
                        "best_local_physical_equation": "",
                        "best_local_features": "",
                        "best_local_complexity": "",
                        "best_local_holdout_r2": "",
                        "score_selected_holdout_r2": "",
                        "lowest_loss_holdout_r2": "",
                        "OLS_local_holdout_r2": "",
                        "Ridge_local_holdout_r2": "",
                        "geco_reference_available": bool(geco.get("geco_reference_available", False)),
                        "geco_variables": ", ".join(geco.get("geco_variables", [])),
                        "geco_notes": geco.get("geco_notes", ""),
                        "notes": f"missing run outputs in {run_dir}",
                    }
                )
                narratives[sensor].append(f"- {target_source}: missing run outputs.")
                continue

            selections = select_rows(df, metadata, local_features)
            local = selections["local_physical"]
            score = selections["score_selected"]
            lowest = selections["lowest_loss"]
            simplest = selections["simplest_nonconstant"]
            score_features = score.get("features_used", []) if score is not None else []
            lowest_features = lowest.get("features_used", []) if lowest is not None else []
            score_class, score_off = classify_feature_support(score_features, local_features)
            lowest_class, lowest_off = classify_feature_support(lowest_features, local_features)
            ols = linear_lookup.get((sensor, target_source, "local_support", "ols"), {})
            ridge = linear_lookup.get((sensor, target_source, "local_support", "ridge"), {})
            local_r2 = row_value(local, "holdout_r2_against_constant")
            score_r2 = row_value(score, "holdout_r2_against_constant")
            lowest_r2 = row_value(lowest, "holdout_r2_against_constant")
            summary_rows.append(
                {
                    "sensor": sensor,
                    "target_source": target_source,
                    "simplest_nonconstant_equation": equation_text(simplest),
                    "score_selected_equation": equation_text(score),
                    "lowest_loss_equation": equation_text(lowest),
                    "best_local_physical_equation": equation_text(local),
                    "best_local_features": features_text(local),
                    "best_local_complexity": row_value(local, "complexity"),
                    "best_local_holdout_r2": local_r2,
                    "score_selected_holdout_r2": score_r2,
                    "lowest_loss_holdout_r2": lowest_r2,
                    "OLS_local_holdout_r2": safe_float(ols.get("holdout_r2")) if ols else "",
                    "Ridge_local_holdout_r2": safe_float(ridge.get("holdout_r2")) if ridge else "",
                    "geco_reference_available": bool(geco.get("geco_reference_available", False)),
                    "geco_variables": ", ".join(geco.get("geco_variables", [])),
                    "geco_notes": geco.get("geco_notes", ""),
                    "notes": (
                        f"score={score_class}"
                        + (f" off={score_off}" if score_off else "")
                        + f"; lowest={lowest_class}"
                        + (f" off={lowest_off}" if lowest_off else "")
                        + "; PySR input was unconstrained over all selected/current features"
                    ),
                }
            )
            if local_features is None:
                local_phrase = "local support unknown"
            else:
                local_phrase = "no local-only equation found" if local is None else f"local R2={local_r2}"
            narratives[sensor].append(
                f"- {target_source}: {local_phrase}; score-selected support is {score_class}; "
                f"lowest-loss support is {lowest_class}."
            )
        narratives[sensor].append(
            "The main PySR runs are unconstrained over all 51 current raw SWaT features; "
            "local support is used only for post-hoc interpretation."
        )
    return summary_rows, narratives


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    columns = [
        "sensor",
        "target_source",
        "simplest_nonconstant_equation",
        "score_selected_equation",
        "lowest_loss_equation",
        "best_local_physical_equation",
        "best_local_features",
        "best_local_complexity",
        "best_local_holdout_r2",
        "score_selected_holdout_r2",
        "lowest_loss_holdout_r2",
        "OLS_local_holdout_r2",
        "Ridge_local_holdout_r2",
        "geco_reference_available",
        "geco_variables",
        "geco_notes",
        "notes",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            text = "" if value is None else str(value)
            text = text.replace("|", "\\|").replace("\n", " ")
            if "equation" in column and text:
                text = f"`{text}`"
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_report(
    audit_root: Path,
    summary_rows: list[dict[str, Any]],
    narratives: dict[str, list[str]],
    linear_lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> str:
    lines = [
        "# Cross-Sensor Symbolic Audit",
        "",
        f"Audit root: `{audit_root}`",
        "",
        "PySR was run unconstrained over all 51 current raw SWaT features for every completed sensor and target. "
        "Process-local feature sets and GeCo references are used only after the fact to label and interpret equations.",
        "Some targets may lack local-support labels; those rows are marked `unknown_support`.",
        "",
        "## Summary",
        "",
        *markdown_table(summary_rows),
        "",
        "## Per-Sensor Notes",
        "",
    ]
    for sensor, sensor_lines in narratives.items():
        lines.extend([f"### {sensor}", "", *sensor_lines, ""])

    lines.extend(
        [
            "## LIT101 Special Check",
            "",
            "For LIT101, actual delta should qualitatively behave like inflow minus outflow, and actual next should behave like persistence plus inflow minus outflow.",
        ]
    )
    warning = coefficient_scale_warning(linear_lookup)
    if warning:
        lines.append(warning)
    lit_rows = [row for row in summary_rows if row["sensor"] == "LIT101"]
    for row in lit_rows:
        lines.append(
            f"- {row['target_source']}: local equation `{row['best_local_physical_equation']}`; "
            f"score-selected `{row['score_selected_equation']}`; local holdout R2={row['best_local_holdout_r2']}."
        )

    lines.extend(
        [
            "",
            "## Equation Selection Policy",
            "",
            "PySR score-selected equations are not automatically treated as final physical equations. "
            "Final candidates are selected by local support, holdout performance, simplicity, and physical plausibility. "
            "Equations using off-process features are flagged as possible proxy fits.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a cross-sensor symbolic audit report.")
    parser.add_argument("--audit-root", default="artifacts/symbolic_equations/swat/full_sensor_audit")
    parser.add_argument("--distill-dir", default="artifacts/model_exports/swat/distillation/val20_overlap")
    parser.add_argument("--support-config", default="configs/swat_sensor_local_support.json")
    parser.add_argument("--geco-model", default="artifacts/geco_templates/SWaT.model")
    parser.add_argument(
        "--linear-baselines",
        default="artifacts/symbolic_equations/swat/full_sensor_audit/linear_sensor_baselines.csv",
    )
    parser.add_argument("--out", default="artifacts/symbolic_equations/swat/full_sensor_audit/cross_sensor_report.md")
    parser.add_argument("--summary-csv", default="artifacts/symbolic_equations/swat/full_sensor_audit/cross_sensor_summary.csv")
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    support_config = load_support_config(args.support_config)
    sensors = load_target_sensors(args.distill_dir)
    geco_lookup = load_geco_lookup(args.geco_model)
    linear_lookup = load_linear_lookup(args.linear_baselines)
    summary_rows, narratives = summarize_runs(
        audit_root,
        support_config,
        linear_lookup,
        sensors=sensors,
        geco_lookup=geco_lookup,
    )

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    report = render_report(audit_root, summary_rows, narratives, linear_lookup)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWrote {out_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
