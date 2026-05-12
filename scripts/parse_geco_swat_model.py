#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _entry_notes(entry: dict[str, Any]) -> str:
    variables = [str(item) for item in entry.get("combination", [])]
    equation_type = str(entry.get("equation", ""))
    if not variables:
        return "no variables parsed"
    if len(variables) == 1:
        return "constant/trivial or persistence-only template"
    first = variables[0]
    if equation_type.lower() == "sum" and first in variables:
        if len(variables) == 2:
            return "mostly persistence/self with one additional variable"
        return "linear template with persistence/self and additional variables"
    return f"{equation_type} template"


def parse_geco_model(path: str | Path) -> dict[str, dict[str, Any]]:
    model_path = Path(path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    ci = payload.get("CI", {})
    if not isinstance(ci, dict):
        raise ValueError(f"Expected CI object in {model_path}")
    parsed: dict[str, dict[str, Any]] = {}
    for target, entry in ci.items():
        if not isinstance(entry, dict):
            continue
        variables = [str(item) for item in entry.get("combination", [])]
        parsed[str(target)] = {
            "geco_reference_available": True,
            "geco_equation_type": entry.get("equation"),
            "geco_variables": variables,
            "geco_parameters": entry.get("parameters", []),
            "geco_notes": _entry_notes(entry),
        }
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse GeCo SWaT model metadata for post-hoc reporting.")
    parser.add_argument("--model", default="artifacts/geco_templates/SWaT.model")
    parser.add_argument("--targets", nargs="*", default=None)
    args = parser.parse_args()

    parsed = parse_geco_model(args.model)
    if args.targets:
        parsed = {target: parsed.get(target, {"geco_reference_available": False}) for target in args.targets}
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
