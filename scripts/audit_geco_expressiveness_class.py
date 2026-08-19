#!/usr/bin/env python3
"""Audit which ASID-ICS equations fall outside GeCo's template class.

GeCo's published models (``*.model`` JSON) admit exactly two equation shapes,
both capped at ``max_formel_length = 3`` non-self inputs:

    Sum      x_target[t+1] = a0*x_target + sum_i a_i*x_i + c        (affine, no interactions)
    Product  x_target[t+1] = a0*x_target + a1*(prod_i x_i) + c      (one product monomial)

ASID-ICS learns a delta equation, so the comparable object is the reconstructed
next-state predictor ``x_target[t] + F(x[t])``. This script expands that
predictor into a polynomial and decides whether it is representable in GeCo's
class.

The classifier is deliberately GENEROUS to GeCo: a product monomial over any
subset of inputs counts as representable, even though the published models only
ever emit products over exactly three inputs. An equation reported as OUTSIDE is
therefore outside GeCo's class under the most permissive reading.

Outside-class reasons:
    input_budget      more than 3 distinct non-self inputs
    power             some variable appears with exponent >= 2
    mixed_structure   an interaction monomial coexists with other non-self terms
    multi_interaction two or more distinct interaction monomials
    self_interaction  the target multiplies another variable
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
GECO_MAX_INPUTS = 3


def safe_symbol_map(names: list[str]) -> dict[str, str]:
    """Map raw ICS tags to sympy-safe names (WADI/HAI tags may lead with digits)."""
    return {name: f"v{idx}" for idx, name in enumerate(sorted(names, key=len, reverse=True))}


def rewrite(expr: str, mapping: dict[str, str]) -> str:
    out = str(expr)
    for raw, safe in mapping.items():
        out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(raw)}(?![A-Za-z0-9_])", safe, out)
    return out


def classify_next_state(
    *, target: str, equation: str, feature_names: list[str], is_delta: bool,
    binary_vars: set[str] | None = None, constant_vars: set[str] | None = None,
) -> dict[str, Any]:
    """Classify the reconstructed next-state predictor against GeCo's class.

    ``binary_vars`` are variables taking values in {0, 1} on benign training
    data. For those, ``x**k == x``, so a syntactic power is NOT evidence of
    extra expressiveness and is reduced before classification.

    ``constant_vars`` are variables with a single value on benign training data.
    Equations touching them are flagged (``involves_constant_var``) because any
    structure fitted over a dead channel is degenerate rather than physical.
    """
    binary_vars = binary_vars or set()
    constant_vars = constant_vars or set()
    result: dict[str, Any] = {
        "parse_ok": False,
        "in_geco_class": None,
        "geco_form": None,
        "outside_reason": "",
        "n_nonself_inputs": None,
        "max_exponent": None,
        "n_interaction_monomials": 0,
        "monomials": "",
        "involves_constant_var": False,
        "binary_power_reduced": False,
    }
    mapping = safe_symbol_map(list(dict.fromkeys(list(feature_names) + [target])))
    syms = {safe: sp.Symbol(safe) for safe in mapping.values()}
    try:
        expr = sp.sympify(rewrite(equation, mapping), locals=syms)
        next_expr = sp.expand(syms[mapping[target]] + expr) if is_delta else sp.expand(expr)
        binary_syms = {syms[mapping[v]] for v in binary_vars if v in mapping}
        if binary_syms:
            reduced = next_expr.replace(
                lambda e: (
                    isinstance(e, sp.Pow) and e.base in binary_syms
                    and getattr(e.exp, "is_Integer", False) and e.exp > 1
                ),
                lambda e: e.base,
            )
            if reduced != next_expr:
                result["binary_power_reduced"] = True
                next_expr = sp.expand(reduced)
    except Exception:
        # Conservative: unparsed equations are treated as representable so the
        # "outside" count is never inflated by parser failures.
        result["outside_reason"] = "parse_failure"
        result["in_geco_class"] = True
        return result

    result["parse_ok"] = True
    self_sym = syms[mapping[target]]
    free = list(next_expr.free_symbols)
    if not free:
        result.update(in_geco_class=True, geco_form="constant", n_nonself_inputs=0, max_exponent=0)
        return result

    try:
        poly = sp.Poly(next_expr, *free)
    except Exception:
        result["outside_reason"] = "non_polynomial"
        result["in_geco_class"] = False
        return result

    inv = {v: k for k, v in mapping.items()}
    gens = list(poly.gens)
    nonself_inputs: set[str] = set()
    max_exp = 0
    interaction_monos: list[str] = []
    linear_nonself: list[str] = []
    has_self_linear = False
    self_interaction = False
    mono_strings: list[str] = []

    for monomial, _coeff in zip(poly.monoms(), poly.coeffs()):
        parts = {gens[i]: e for i, e in enumerate(monomial) if e > 0}
        if not parts:
            continue
        degree = sum(parts.values())
        max_exp = max(max_exp, max(parts.values()))
        factors = sorted((inv.get(str(g), str(g)), e) for g, e in parts.items())
        names = [n for n, _ in factors]
        mono_strings.append("*".join(f"{n}^{e}" if e > 1 else n for n, e in factors))
        for g in parts:
            if g != self_sym:
                nonself_inputs.add(inv.get(str(g), str(g)))
        if degree == 1:
            if self_sym in parts:
                has_self_linear = True
            else:
                linear_nonself.append(mono_strings[-1])
        else:
            interaction_monos.append(mono_strings[-1])
            if self_sym in parts:
                self_interaction = True

    result["n_nonself_inputs"] = len(nonself_inputs)
    result["max_exponent"] = int(max_exp)
    result["involves_constant_var"] = bool(
        (nonself_inputs | {target}) & set(constant_vars)
    )
    result["n_interaction_monomials"] = len(interaction_monos)
    result["monomials"] = " + ".join(sorted(mono_strings))  # deterministic order

    reasons: list[str] = []
    if len(nonself_inputs) > GECO_MAX_INPUTS:
        reasons.append("input_budget")
    if max_exp >= 2:
        reasons.append("power")
    if self_interaction:
        reasons.append("self_interaction")
    if len(interaction_monos) >= 2:
        reasons.append("multi_interaction")
    if len(interaction_monos) == 1 and linear_nonself:
        reasons.append("mixed_structure")

    if reasons:
        result.update(in_geco_class=False, outside_reason=";".join(sorted(set(reasons))))
    else:
        result.update(
            in_geco_class=True,
            geco_form="Product" if interaction_monos else "Sum",
        )
    _ = has_self_linear
    return result


def audit_selection(
    *, dataset: str, seed: int, selected_csv: Path, feature_names: list[str],
    drop_targets: set[str] | None = None,
    binary_vars: set[str] | None = None, constant_vars: set[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(selected_csv)
    df["target"] = df["target"].astype(str)
    if drop_targets:
        df = df[~df["target"].isin(drop_targets)]
    rows: list[dict[str, Any]] = []
    for _, row in df.drop_duplicates("target").iterrows():
        target = str(row["target"])
        equation = str(row.get("sympy_format") or row.get("equation") or "")
        target_mode = str(row.get("target_mode", ""))
        vtype = str(row.get("variable_type", "sensor"))
        is_delta = "delta" in target_mode and vtype != "persistence"
        if vtype == "persistence" or equation.strip() == "" or equation.strip() == target:
            rows.append(
                {
                    "dataset": dataset, "seed": seed, "target": target, "variable_type": vtype,
                    "equation": row.get("equation"), "in_geco_class": True, "geco_form": "persistence",
                    "outside_reason": "", "n_nonself_inputs": 0, "max_exponent": 0,
                    "n_interaction_monomials": 0, "monomials": "", "parse_ok": True,
                    "complexity": row.get("complexity"), "holdout_r2": row.get("holdout_r2"),
                }
            )
            continue
        cls = classify_next_state(
            target=target, equation=equation, feature_names=feature_names, is_delta=is_delta,
            binary_vars=binary_vars, constant_vars=constant_vars,
        )
        rows.append(
            {
                "dataset": dataset, "seed": seed, "target": target, "variable_type": vtype,
                "equation": row.get("equation"), **cls,
                "complexity": row.get("complexity"), "holdout_r2": row.get("holdout_r2"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "scripts" / "geco_expressiveness_sources.json"))
    parser.add_argument("--out-dir", default="paper_artifacts/expressiveness_v1")
    args = parser.parse_args()

    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = Path(args.out_dir) / "variable_stats.csv"
    vstats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()

    per_rows = []
    for dataset, cfg in spec.items():
        feats = pd.read_csv(cfg["feature_list"])[cfg["feature_column"]].astype(str).tolist()
        drop = set(cfg.get("drop_targets", []))
        binary_vars: set[str] = set()
        constant_vars: set[str] = set()
        if len(vstats):
            sub = vstats[vstats["dataset"] == dataset]
            binary_vars = set(sub.loc[sub["is_binary"].astype(bool), "variable"].astype(str))
            constant_vars = set(sub.loc[sub["is_constant"].astype(bool), "variable"].astype(str))
        for seed, path in cfg["selected"].items():
            per_rows.append(
                audit_selection(
                    dataset=dataset, seed=int(seed), selected_csv=Path(path),
                    feature_names=feats, drop_targets=drop,
                    binary_vars=binary_vars, constant_vars=constant_vars,
                )
            )
    per = pd.concat(per_rows, ignore_index=True)
    per.to_csv(out_dir / "equation_class_per_target.csv", index=False)

    summary = (
        per.groupby(["dataset", "seed"])
        .apply(
            lambda g: pd.Series(
                {
                    "selected": len(g),
                    "outside_geco_class": int((~g["in_geco_class"].astype(bool)).sum()),
                    "pct_outside": round(100 * (~g["in_geco_class"].astype(bool)).mean(), 2),
                    "outside_nondegenerate": int(
                        (~g["in_geco_class"].astype(bool) & ~g["involves_constant_var"].astype(bool)).sum()
                    ),
                    "reasons": ";".join(sorted({r for r in g.loc[~g["in_geco_class"].astype(bool), "outside_reason"] if r})),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "equation_class_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(f"\nWrote {out_dir}/equation_class_per_target.csv and equation_class_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
