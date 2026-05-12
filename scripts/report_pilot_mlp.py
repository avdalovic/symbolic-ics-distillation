#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNS = [
    ("LIT101 MLP delta unconstrained", "LIT101_mlp_delta_unconstrained"),
    ("LIT101 MLP delta top-8", "LIT101_mlp_delta_topk8"),
    ("FIT101 MLP delta top-8", "FIT101_mlp_delta_topk8"),
    ("FIT201 MLP delta top-8", "FIT201_mlp_delta_topk8"),
    ("LIT301 MLP delta top-8", "LIT301_mlp_delta_topk8"),
    ("DPIT301 MLP delta top-8", "DPIT301_mlp_delta_topk8"),
    ("LIT101 actual delta unconstrained", "LIT101_actual_delta_unconstrained"),
]


def load_metadata(root: Path, run_name: str) -> dict | None:
    path = root / run_name / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_run(label: str, metadata: dict | None) -> str:
    if metadata is None:
        return f"{label}: missing"
    used = ", ".join(metadata.get("equation_used_features") or [])
    selected = ", ".join(metadata.get("selected_features") or [])
    return (
        f"{label}\n"
        f"  equation: {metadata.get('best_equation')}\n"
        f"  best_loss: {metadata.get('best_loss')} sample_mse: {metadata.get('sample_mse')} "
        f"full_mse: {metadata.get('full_mse')}\n"
        f"  equation features: {used if used else '(none detected)'}\n"
        f"  input features: {selected}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Report MLP PySR pilot equations.")
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    root = Path(args.out_root)
    print("MLP PySR Pilot Report")
    print(f"Output root: {root}")
    print()
    for label, run_name in RUNS:
        print(render_run(label, load_metadata(root, run_name)))
        print()

    print("LIT101 three-way comparison")
    print("GeCo reference: 0.19*FIT101 - 0.20*FIT201 + 0.009")
    for label, run_name in [
        ("MLP delta unconstrained", "LIT101_mlp_delta_unconstrained"),
        ("MLP delta top-8", "LIT101_mlp_delta_topk8"),
        ("Actual delta unconstrained", "LIT101_actual_delta_unconstrained"),
    ]:
        metadata = load_metadata(root, run_name)
        if metadata is None:
            print(f"{label}: missing")
            continue
        print(
            f"{label}: equation={metadata.get('best_equation')} "
            f"loss={metadata.get('best_loss')} full_mse={metadata.get('full_mse')} "
            f"features={metadata.get('equation_used_features')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
