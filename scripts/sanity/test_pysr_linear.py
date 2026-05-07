#!/usr/bin/env python
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "artifacts" / "symbolic_equations" / "sanity" / "linear"
VARIABLE_NAMES = ["x1", "x2", "x3"]


def _filter_supported_params(params: dict) -> dict:
    signature = inspect.signature(PySRRegressor.__init__)
    supported = {
        name
        for name, value in signature.parameters.items()
        if name != "self" and value.kind in {value.POSITIONAL_OR_KEYWORD, value.KEYWORD_ONLY}
    }
    return {key: value for key, value in params.items() if key in supported}


def _sanitize_for_print(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        if callable(value):
            out[key] = repr(value)
        elif isinstance(value, dict):
            out[key] = {k: repr(v) if callable(v) else v for k, v in value.items()}
        else:
            out[key] = value
    return out


def _make_model(output_directory: Path, seed: int) -> PySRRegressor:
    requested = {
        "binary_operators": ["+", "*", "-"],
        "unary_operators": ["inv(x) = 1/x"],
        "extra_sympy_mappings": {"inv": lambda x: 1 / x},
        "maxsize": 20,
        "niterations": 300,
        "populations": 30,
        "procs": 4,
        "parallelism": "serial",
        "random_state": seed,
        "deterministic": True,
        "model_selection": "score",
        "output_directory": str(output_directory),
        "verbosity": 1,
        "progress": False,
    }
    params = _filter_supported_params(requested)
    output_directory.mkdir(parents=True, exist_ok=True)
    print("PySRRegressor parameters actually used:")
    print(json.dumps(_sanitize_for_print(params), indent=2, sort_keys=True))
    return PySRRegressor(**params)


def _fit(model: PySRRegressor, X: np.ndarray, y: np.ndarray) -> PySRRegressor:
    fit_signature = inspect.signature(model.fit)
    if "variable_names" in fit_signature.parameters:
        return model.fit(X, y, variable_names=VARIABLE_NAMES)
    return model.fit(pd.DataFrame(X, columns=VARIABLE_NAMES), y)


def _equation_columns(model: PySRRegressor) -> pd.DataFrame:
    equations = model.equations_.copy()
    wanted = [col for col in ["equation", "complexity", "loss", "score"] if col in equations.columns]
    return equations[wanted] if wanted else equations


def _best_equation(model: PySRRegressor) -> tuple[str, str]:
    best = model.get_best()
    equation = str(best.get("equation", best))
    sympy_text = str(best.get("sympy_format", equation))
    return equation, sympy_text


def _variables_in_expression(expr_text: str) -> set[str]:
    expr_text = re.sub(r"\binv\(", "1/(", str(expr_text))
    symbols = {name: sp.Symbol(name) for name in VARIABLE_NAMES}
    try:
        expr = sp.sympify(expr_text, locals=symbols)
    except Exception:
        return {name for name in VARIABLE_NAMES if name in str(expr_text)}
    return {str(symbol) for symbol in expr.free_symbols}


def main() -> int:
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(5000, 3)).astype(np.float32)
    y = (0.99 * X[:, 0] + 0.2 * X[:, 1] - 0.18 * X[:, 2] + rng.normal(0.0, 0.01, 5000)).astype(
        np.float32
    )

    X_test = rng.uniform(-2.0, 2.0, size=(1000, 3)).astype(np.float32)
    y_test = (
        0.99 * X_test[:, 0]
        + 0.2 * X_test[:, 1]
        - 0.18 * X_test[:, 2]
        + rng.normal(0.0, 0.01, 1000)
    ).astype(np.float32)

    model = _make_model(OUT_DIR, seed=0)
    _fit(model, X, y)

    equations = _equation_columns(model)
    print("\nPareto front:")
    print(equations.to_string(index=False))

    best_equation, best_sympy = _best_equation(model)
    print("\nSelected best equation:")
    print(best_equation)

    preds = np.asarray(model.predict(X_test), dtype=np.float64)
    test_mse = float(np.mean((preds - y_test.astype(np.float64)) ** 2))
    print(f"\nlinear_test_mse={test_mse:.12g}")

    front_texts = []
    for _, row in model.equations_.iterrows():
        front_texts.append(str(row.get("sympy_format", row.get("equation", ""))))
    best_has_all = _variables_in_expression(best_sympy) == set(VARIABLE_NAMES)
    front_has_all = any(_variables_in_expression(text) == set(VARIABLE_NAMES) for text in front_texts)
    passed = (best_has_all or front_has_all) and test_mse < 5e-4
    print(
        json.dumps(
            {
                "best_contains_all_variables": best_has_all,
                "front_contains_all_variables": front_has_all,
                "test_mse_below_threshold": test_mse < 5e-4,
                "passed": passed,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
