from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import sympy as sp


@dataclass(frozen=True)
class StateDependenceResult:
    target: str
    state_dependent: bool
    reason: str
    simplified_next: str | None = None


def _replace_feature_names(expr: str, feature_names: Sequence[str]) -> tuple[str, list[sp.Symbol]]:
    """Replace process tags with parser-safe symbols.

    Several ICS tags begin with digits (for example WADI's ``2_FIT_003_PV``),
    which makes direct ``sympify`` brittle. We therefore substitute each known
    process tag with ``v<i>`` before parsing and keep a stable symbol map.
    """

    text = str(expr).replace("^", "**")
    symbols = [sp.Symbol(f"v{idx}") for idx, _ in enumerate(feature_names)]
    for idx, name in sorted(enumerate(feature_names), key=lambda item: len(str(item[1])), reverse=True):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(str(name))}(?![A-Za-z0-9_])"
        text = re.sub(pattern, f"v{idx}", text)
    return text, symbols


def state_dependence_for_delta_equation(
    *,
    target: str,
    equation: str,
    feature_names: Sequence[str],
) -> StateDependenceResult:
    """Return whether ``x_target + delta_hat`` depends on any state variable.

    ASID-ICS learns delta equations. A candidate such as ``41.7 - F_PU7`` is a
    valid delta equation syntactically, but the reconstructed next-state
    predictor is ``F_PU7 + 41.7 - F_PU7 = 41.7``. Such constant-next predictors
    are stable on benign data yet do not express process dynamics, so the
    selection guard rejects them before choosing the highest-score candidate.

    Parse failures are treated conservatively as state-dependent so this guard
    never excludes an equation merely because the parser could not simplify it.
    """

    names = [str(name) for name in feature_names]
    if str(target) not in names:
        return StateDependenceResult(str(target), True, "target_not_in_feature_names")
    target_idx = names.index(str(target))
    safe_expr, symbols = _replace_feature_names(str(equation), names)
    locals_map = {f"v{idx}": symbol for idx, symbol in enumerate(symbols)}
    locals_map.update(
        {
            "square": lambda x: x**2,
            "cube": lambda x: x**3,
            "abs_op": sp.Abs,
            "abs": sp.Abs,
            "Abs": sp.Abs,
        }
    )
    try:
        delta_expr = sp.sympify(safe_expr, locals=locals_map)
        next_expr = sp.simplify(symbols[target_idx] + delta_expr)
    except Exception as exc:
        return StateDependenceResult(str(target), True, f"parse_failed:{type(exc).__name__}")
    free = next_expr.free_symbols
    if free:
        return StateDependenceResult(str(target), True, "next_state_depends_on_state", str(next_expr))
    return StateDependenceResult(str(target), False, "constant_next_state", str(next_expr))


def passes_state_dependence_guard(
    *,
    target: str,
    equation: str,
    feature_names: Sequence[str],
    enabled: bool = True,
) -> bool:
    if not enabled:
        return True
    return state_dependence_for_delta_equation(
        target=target,
        equation=equation,
        feature_names=feature_names,
    ).state_dependent
