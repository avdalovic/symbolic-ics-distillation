#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.data.hai import (  # noqa: E402
    EXPECTED_ATTACK_COUNT,
    EXPECTED_TEST_FILES,
    EXPECTED_TRAIN_FILES,
    iter_jsonl_gzip,
    load_attacks,
    load_hai_sequences,
    infer_variable_types,
    sequence_key_differences,
    sequence_manifest_rows,
    shared_state_keys,
    timestamp_overlap_rows,
    validate_expected_generated_files,
    write_json,
)


DATA_DIR = REPO_ROOT / "data" / "hai" / "ipal"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "experiments" / "hai_protocol"


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def sanitize_sequence_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["path"] = rel(Path(str(item["path"])))
        out.append(item)
    return out


def validate_jsonl_files(data_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in validate_expected_generated_files(data_dir):
        count = 0
        for obj in iter_jsonl_gzip(path):
            if not isinstance(obj.get("state"), dict):
                raise ValueError(f"{path}: line {count + 1} lacks object field 'state'")
            count += 1
        rows.append({"file": path.name, "json_lines": count, "valid_json": True})
    return rows


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", text))


def _numbers_near(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = float(match.group(1))
            except Exception:
                continue
            if math.isfinite(value):
                return value
    return None


def safe_json_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def parse_geco_model(path: Path | None, data_variables: set[str]) -> dict[str, Any]:
    if path is None:
        return {
            "provided": False,
            "path": "",
            "s_scale": None,
            "growth_factor": None,
            "rows": [{"category": "not_provided", "variable": "", "in_transcribed_data": ""}],
        }
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
        settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
    except Exception:
        settings = {}
    s_scale = safe_json_float(settings.get("threshold_factor"))
    growth = safe_json_float(settings.get("cusum_factor"))
    if s_scale is None:
        s_scale = _numbers_near(
            [
                r"(?:threshold_factor|S_scale|threshold_scale|scale|s)\s*['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                r"['\"]S['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            ],
            text,
        )
    if growth is None:
        growth = _numbers_near(
            [
                r"(?:cusum_factor|G_cap|growth_factor|growth|cap|g)\s*['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
                r"['\"]G['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            ],
            text,
        )
    ignore_tokens: set[str] = set()
    target_tokens: set[str] = set()
    model_tokens: set[str] = _tokens(text)
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if "ignore" in lower or "exclude" in lower:
            ignore_tokens |= _tokens(stripped)
        target_match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:=|:)", stripped)
        if target_match:
            target_tokens.add(target_match.group(1))
    rows: list[dict[str, Any]] = []
    for category, values in [
        ("equation_target", sorted(target_tokens)),
        ("ignore_list", sorted(ignore_tokens)),
        ("model_token", sorted(model_tokens & data_variables)),
    ]:
        for token in values:
            rows.append(
                {
                    "category": category,
                    "variable": token,
                    "in_transcribed_data": bool(token in data_variables),
                }
            )
    mentioned = (target_tokens | ignore_tokens | model_tokens) & data_variables
    for variable in sorted(data_variables - mentioned):
        rows.append(
            {
                "category": "data_variable_absent_from_geco_model",
                "variable": variable,
                "in_transcribed_data": True,
            }
        )
    return {
        "provided": True,
        "path": rel(Path(path)),
        "s_scale": s_scale,
        "growth_factor": growth,
        "rows": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local HAI 21.03 IPAL transcription.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--geco-model", default=None, help="Optional path to GeCo HAI.model for key compatibility checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_rows = validate_jsonl_files(data_dir)
    train = load_hai_sequences(data_dir, split="train")
    test = load_hai_sequences(data_dir, split="test")
    sequences = train + test
    attacks = load_attacks(data_dir / "attacks.json")
    if len(train) != len(EXPECTED_TRAIN_FILES) or len(test) != len(EXPECTED_TEST_FILES):
        raise ValueError(f"expected 3 train and 5 test sequences, found {len(train)} and {len(test)}")
    if len(attacks) != EXPECTED_ATTACK_COUNT:
        raise ValueError(f"expected {EXPECTED_ATTACK_COUNT} attacks, found {len(attacks)}")

    feature_columns = shared_state_keys(sequences)
    variable_audit = infer_variable_types(sequences, feature_columns)
    seq_manifest = sanitize_sequence_manifest(sequence_manifest_rows(sequences))
    key_rows = sequence_key_differences(sequences)
    overlap_rows = timestamp_overlap_rows(sequences)
    geco = parse_geco_model(Path(args.geco_model) if args.geco_model else None, set(feature_columns))

    pd.DataFrame(seq_manifest).to_csv(out_dir / "sequence_manifest.csv", index=False)
    variable_audit.to_csv(out_dir / "variable_audit.csv", index=False)
    pd.DataFrame(geco["rows"]).to_csv(out_dir / "geco_key_compatibility.csv", index=False)

    audit = {
        "data_dir": rel(data_dir),
        "train_sequence_count": len(train),
        "test_sequence_count": len(test),
        "attack_count": len(attacks),
        "expected_attack_count": EXPECTED_ATTACK_COUNT,
        "state_key_count_intersection": len(feature_columns),
        "state_keys_consistent": all(not row["missing_from_union"] and not row["extra_relative_to_intersection"] for row in key_rows),
        "key_differences": key_rows,
        "timestamp_overlaps": overlap_rows,
        "jsonl_validation": jsonl_rows,
        "missing_values": int(variable_audit["missing_count"].sum()),
        "nonfinite_values": int(variable_audit["nonfinite_count"].sum()),
        "variable_type_counts": variable_audit["inferred_type"].value_counts().to_dict(),
        "selected_model_counts": variable_audit["selected_model"].value_counts().to_dict(),
        "geco_model": {
            "provided": bool(geco["provided"]),
            "path": geco["path"],
            "s_scale": geco["s_scale"],
            "growth_factor": geco["growth_factor"],
            "all_equation_targets_exist": bool(
                all(row["in_transcribed_data"] for row in geco["rows"] if row["category"] == "equation_target")
            ),
            "all_ignore_list_variables_exist": bool(
                all(row["in_transcribed_data"] for row in geco["rows"] if row["category"] == "ignore_list")
            ),
        },
    }
    write_json(out_dir / "hai_protocol_audit.json", audit)
    print(f"HAI protocol audit written to {rel(out_dir)}")
    print(f"Sequences: {len(train)} train, {len(test)} test")
    print(f"Attacks: {len(attacks)}")
    print(f"Variables: {len(feature_columns)}")
    if args.geco_model:
        print(
            "GeCo model point:",
            f"S={geco['s_scale']}" if geco["s_scale"] is not None else "S=not_found",
            f"G={geco['growth_factor']}" if geco["growth_factor"] is not None else "G=not_found",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
