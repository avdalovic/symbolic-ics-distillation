#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

TARGET_SOURCES = ["actual_next", "actual_delta", "mlp_next", "mlp_delta"]
ALLOWED_LOCAL_FEATURES = {"LIT101", "FIT101", "FIT201", "MV101", "P101"}
GECO_REFERENCE = "0.19*FIT101 - 0.20*FIT201 + 0.009"


def parse_feature_support(value) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_summary(row: pd.Series | None, label: str) -> str:
    if row is None:
        return f"- {label}: missing"
    features = parse_feature_support(row.get("equation_features"))
    off_process = [name for name in features if name not in ALLOWED_LOCAL_FEATURES]
    return (
        f"- {label}: complexity={row.get('complexity')} loss={row.get('loss')} "
        f"holdout_mse={row.get('holdout_mse')} holdout_rmse={row.get('holdout_rmse')} "
        f"holdout_r2={row.get('holdout_r2_against_constant')} "
        f"features={features if features else '(constant)'} "
        f"off_process={off_process if off_process else 'no'}\n"
        f"  equation: `{row.get('equation')}`"
    )


def select_rows(df: pd.DataFrame, metadata: dict) -> dict[str, pd.Series | None]:
    rows: dict[str, pd.Series | None] = {}
    if df.empty:
        return {"lowest_loss": None, "score_selected": None, "simplest_nonconstant": None, "local_physical": None}

    rows["lowest_loss"] = df.loc[df["loss"].astype(float).idxmin()]

    best_equation = str(metadata.get("best_equation", ""))
    score_match = df[df["equation"].astype(str) == best_equation]
    if not score_match.empty:
        rows["score_selected"] = score_match.iloc[0]
    elif "score" in df.columns:
        rows["score_selected"] = df.loc[df["score"].astype(float).idxmax()]
    else:
        rows["score_selected"] = None

    nonconstant = df[df["equation_features"].map(lambda value: len(parse_feature_support(value)) > 0)]
    if not nonconstant.empty:
        rows["simplest_nonconstant"] = nonconstant.sort_values(["complexity", "loss"], ascending=[True, True]).iloc[0]
    else:
        rows["simplest_nonconstant"] = None

    local = df[
        df["equation_features"].map(
            lambda value: 0 < len(parse_feature_support(value)) and set(parse_feature_support(value)).issubset(ALLOWED_LOCAL_FEATURES)
        )
    ]
    if not local.empty:
        sort_column = "holdout_mse" if "holdout_mse" in local.columns and local["holdout_mse"].notna().any() else "loss"
        rows["local_physical"] = local.sort_values([sort_column, "complexity"], ascending=[True, True]).iloc[0]
    else:
        rows["local_physical"] = None
    return rows


def load_run(audit_dir: Path, target_source: str, operator_set: str = "restricted") -> tuple[pd.DataFrame | None, dict | None]:
    run_dir = audit_dir / f"{target_source}_{operator_set}_unconstrained"
    csv_path = run_dir / "pareto_front_scored.csv"
    if not csv_path.exists():
        csv_path = run_dir / "pareto_front.csv"
    metadata_path = run_dir / "metadata.json"
    if not csv_path.exists() or not metadata_path.exists():
        return None, None
    return pd.read_csv(csv_path), read_json(metadata_path)


def format_linear_table(audit_dir: Path) -> list[str]:
    path = audit_dir / "lit101_linear_baselines.csv"
    if not path.exists():
        return ["Linear coefficient baselines: missing"]
    df = pd.read_csv(path)
    wanted = [
        ("actual_delta", "physics_delta_support"),
        ("actual_next", "physics_next_support"),
        ("mlp_delta", "physics_delta_support"),
        ("mlp_next", "physics_next_support"),
    ]
    lines = [
        "| target_source | feature_set | model | holdout_r2 | holdout_mse | coefficients | intercept |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for target_source, feature_set in wanted:
        subset = df[(df["target_source"] == target_source) & (df["feature_set"] == feature_set)]
        for _, row in subset.iterrows():
            lines.append(
                f"| {target_source} | {feature_set} | {row['model_type']} | "
                f"{row['holdout_r2']:.6g} | {row['holdout_mse']:.6g} | "
                f"`{row['coefficients']}` | {row['intercept']:.6g} |"
            )
    return lines


def interpret(audit_dir: Path) -> list[str]:
    path = audit_dir / "lit101_linear_baselines.csv"
    if not path.exists():
        return ["Interpretation: run the linear baselines before drawing coefficient conclusions."]
    df = pd.read_csv(path)
    lines = ["Interpretation:"]
    for target_source, feature_set in [
        ("actual_delta", "physics_delta_support"),
        ("actual_next", "physics_next_support"),
        ("mlp_delta", "physics_delta_support"),
        ("mlp_next", "physics_next_support"),
    ]:
        subset = df[
            (df["target_source"] == target_source)
            & (df["feature_set"] == feature_set)
            & (df["model_type"] == "ols")
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"- {target_source}: OLS on {feature_set} gives holdout_r2={row['holdout_r2']:.6g}; "
            f"coefficients `{row['coefficients']}`."
        )
    lines.append(
        "- Compare the local-physical PySR row against the score-selected row; a better score-selected equation "
        "with off-process features is evidence of proxy fitting rather than a clean LIT101 physical law."
    )
    lines.append(f"- GeCo reference for delta: `{GECO_REFERENCE}`.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Report LIT101 target/operator audit outputs.")
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--sensor", default="LIT101")
    parser.add_argument("--operator-set", default="restricted")
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    report_lines = [f"# {args.sensor} Target Audit", "", f"Audit dir: `{audit_dir}`", ""]
    comparison_rows = [
        "| target | local physical equation | score-selected equation | holdout R2 | features | notes |",
        "|---|---|---|---:|---|---|",
    ]

    for target_source in TARGET_SOURCES:
        df, metadata = load_run(audit_dir, target_source, args.operator_set)
        report_lines.extend([f"## {target_source}", ""])
        if df is None or metadata is None:
            report_lines.extend(["Missing run outputs.", ""])
            comparison_rows.append(f"| {target_source} | missing | missing |  |  | missing |")
            continue
        selections = select_rows(df, metadata)
        for label, key in [
            ("best lowest-loss equation", "lowest_loss"),
            ("PySR score-selected equation", "score_selected"),
            ("simplest non-constant equation", "simplest_nonconstant"),
            ("best local physical equation", "local_physical"),
        ]:
            report_lines.append(row_summary(selections[key], label))
        report_lines.append("")

        local = selections["local_physical"]
        score = selections["score_selected"]
        row_for_r2 = local if local is not None else score
        features = parse_feature_support(local.get("equation_features")) if local is not None else []
        off_process = [name for name in parse_feature_support(score.get("equation_features")) if name not in ALLOWED_LOCAL_FEATURES] if score is not None else []
        comparison_rows.append(
            f"| {target_source} | `{local.get('equation') if local is not None else 'missing'}` | "
            f"`{score.get('equation') if score is not None else 'missing'}` | "
            f"{row_for_r2.get('holdout_r2_against_constant') if row_for_r2 is not None else ''} | "
            f"{features} | score off-process: {off_process if off_process else 'no'} |"
        )

    report_lines.extend(["## LIT101 Comparison", "", *comparison_rows, ""])
    report_lines.extend(["## Linear Coefficient Baselines", "", *format_linear_table(audit_dir), ""])
    report_lines.extend(["## Interpretation", "", *interpret(audit_dir), ""])

    report = "\n".join(str(line) for line in report_lines)
    out_path = audit_dir / "lit101_target_audit_report.md"
    out_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
