from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sympy as sp

LOGGER = logging.getLogger(__name__)
_PRINTED_PARETO_COLUMNS = False


def _candidate_run_dirs(sensor: str, target_source: str, audit_root: str | Path) -> list[Path]:
    root = Path(audit_root)
    exact = [
        root / f"{sensor}_{target_source}",
        root / f"{sensor}_{target_source}_restricted_unconstrained",
    ]
    globbed = sorted(path for path in root.glob(f"{sensor}_{target_source}*") if path.is_dir())
    out: list[Path] = []
    for path in [*exact, *globbed]:
        if path not in out:
            out.append(path)
    return out


def _find_pareto_csv(run_dir: Path) -> Path | None:
    preferred = [
        run_dir / "pareto_front_scored.csv",
        run_dir / "equations.csv",
        run_dir / "hall_of_fame.csv",
    ]
    for path in preferred:
        if path.exists():
            return path
    for path in sorted(run_dir.glob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if "equation" in header and "loss" in header:
            return path
    return None


def _coerce_float_series(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def load_pareto_front(
    sensor: str,
    target_source: str,
    audit_root: str | Path,
) -> tuple[pd.Series | None, pd.DataFrame | None]:
    """Load a PySR Pareto front and return the score-selected row plus DataFrame."""

    global _PRINTED_PARETO_COLUMNS
    for run_dir in _candidate_run_dirs(sensor, target_source, audit_root):
        if not run_dir.exists():
            continue
        csv_path = _find_pareto_csv(run_dir)
        if csv_path is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            LOGGER.warning("Failed to read Pareto CSV %s: %s", csv_path, exc)
            continue
        if df.empty or "equation" not in df.columns:
            LOGGER.warning("Pareto CSV is empty or missing equation column: %s", csv_path)
            continue
        if not _PRINTED_PARETO_COLUMNS:
            print(f"Pareto columns from {csv_path}: {list(df.columns)}")
            _PRINTED_PARETO_COLUMNS = True
        df = df.copy()
        df["_source_csv"] = str(csv_path)
        df["_run_dir"] = str(run_dir)
        if "score" in df.columns:
            df["_selection_score"] = _coerce_float_series(df["score"], default=-np.inf)
        else:
            loss = _coerce_float_series(df.get("loss", pd.Series(np.inf, index=df.index)), default=np.inf)
            complexity = _coerce_float_series(df.get("complexity", pd.Series(0.0, index=df.index)), default=0.0)
            df["score"] = -np.log(loss.astype(float) + 1e-10) - 0.01 * complexity.astype(float)
            df["_selection_score"] = df["score"]
        df["_row_index"] = df.index.astype(int)
        selected_idx = df["_selection_score"].astype(float).idxmax()
        selected = df.loc[selected_idx]

        if sensor == "LIT101" and target_source in {"actual_next", "mlp_next"}:
            max_complexity = pd.to_numeric(df.get("complexity", pd.Series([], dtype=float)), errors="coerce").max()
            print(
                f"LIT101 {target_source} selected row={int(selected['_row_index'])} "
                f"complexity={selected.get('complexity')} loss={selected.get('loss')} "
                f"score={selected.get('score')} max_complexity={max_complexity}"
            )
            print(f"LIT101 {target_source} equation: {selected.get('equation')}")
            if target_source == "actual_next":
                equation = str(selected.get("equation", ""))
                if not all(name in equation for name in ("LIT101", "FIT101", "FIT201")):
                    LOGGER.warning(
                        "LIT101 actual_next selected equation lacks one or more expected local terms. "
                        "Top scored alternatives:\n%s",
                        df.sort_values("_selection_score", ascending=False)
                        .head(5)[["equation", "complexity", "loss", "score"]]
                        .to_string(index=False),
                    )
        return selected, df

    LOGGER.warning("Missing Pareto front for sensor=%s target_source=%s under %s", sensor, target_source, audit_root)
    return None, None


def _prepare_expression(equation: str) -> str:
    expr = str(equation).strip()
    expr = expr.replace("^", "**")
    return expr


def _symbol_locals(feature_names: Sequence[str]) -> tuple[dict[str, Any], list[sp.Symbol]]:
    feature_symbols = [sp.Symbol(str(name)) for name in feature_names]
    locals_map: dict[str, Any] = {
        str(name): symbol for name, symbol in zip(feature_names, feature_symbols)
    }
    for idx, symbol in enumerate(feature_symbols):
        locals_map[f"x{idx}"] = symbol
    locals_map.update(
        {
            "square": lambda x: x**2,
            "cube": lambda x: x**3,
            "abs_op": sp.Abs,
            "abs": sp.Abs,
            "Abs": sp.Abs,
        }
    )
    return locals_map, feature_symbols


def evaluate_equation(equation_str: str, feature_names: Sequence[str], x: np.ndarray) -> np.ndarray:
    """Evaluate a PySR equation on raw current features.

    Failures are reported as an all-NaN vector so callers can skip a symbolic
    sensor without aborting a full detector run.
    """

    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"X must be [N, F], got {values.shape}")
    if values.shape[1] != len(feature_names):
        raise ValueError(f"X has {values.shape[1]} features but {len(feature_names)} feature names were provided")
    n = values.shape[0]
    locals_map, feature_symbols = _symbol_locals(feature_names)
    try:
        expr = sp.sympify(_prepare_expression(equation_str), locals=locals_map)
    except Exception as exc:
        LOGGER.warning("Failed to parse symbolic equation %r: %s", equation_str, exc)
        return np.full(n, np.nan, dtype=np.float64)

    allowed = set(feature_symbols)
    unknown = sorted(str(symbol) for symbol in expr.free_symbols if symbol not in allowed)
    if unknown:
        LOGGER.warning("Equation contains unknown variables %s: %s", unknown, equation_str)
        return np.full(n, np.nan, dtype=np.float64)

    try:
        fn = sp.lambdify(feature_symbols, expr, modules="numpy")
        result = fn(*[values[:, idx] for idx in range(values.shape[1])])
        arr = np.asarray(result, dtype=np.float64)
        if arr.shape == ():
            arr = np.full(n, float(arr), dtype=np.float64)
        else:
            arr = arr.reshape(-1)
            if arr.shape[0] == 1 and n != 1:
                arr = np.full(n, float(arr[0]), dtype=np.float64)
        if arr.shape[0] != n:
            LOGGER.warning("Equation returned shape %s for N=%d: %s", arr.shape, n, equation_str)
            return np.full(n, np.nan, dtype=np.float64)
        return arr.astype(np.float64)
    except Exception as exc:
        LOGGER.warning("Failed to evaluate symbolic equation %r: %s", equation_str, exc)
        return np.full(n, np.nan, dtype=np.float64)


def equation_features(equation: str, feature_names: Sequence[str]) -> list[str]:
    """Return feature tags used by an equation, avoiding substring matches."""

    used = []
    text = str(equation)
    for name in feature_names:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(name))}(?![A-Za-z0-9_])", text):
            used.append(str(name))
    return used
