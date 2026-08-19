from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ics_symbolic_distill.data.hai import (
    EXPECTED_STATE_FILES,
    CusumParams,
    assert_no_cross_sequence_pairs,
    fit_cusum_params_sequences,
    infer_variable_types,
    load_state_gz,
    make_pair_arrays,
    run_cusum_sequences,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_state(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_gzip_jsonl_loading(tmp_path: Path) -> None:
    path = tmp_path / "train1.state.gz"
    write_state(
        path,
        [
            {"timestamp": 10, "state": {"A": 1.0, "B": 0}, "malicious": False},
            {"timestamp": 11, "state": {"A": 2.5, "B": 1}, "malicious": 3},
        ],
    )
    seq = load_state_gz(path, split="train")
    assert seq.name == "train1.state.gz"
    assert seq.timestamps.tolist() == [10, 11]
    assert seq.attack_ids.tolist() == [0, 3]
    assert seq.labels.tolist() == [0, 1]
    assert seq.frame["A"].tolist() == [1.0, 2.5]


def test_no_cross_file_delta_pairs(tmp_path: Path) -> None:
    p1 = tmp_path / "train1.state.gz"
    p2 = tmp_path / "train2.state.gz"
    write_state(
        p1,
        [
            {"timestamp": 1, "state": {"A": 1.0}, "malicious": False},
            {"timestamp": 2, "state": {"A": 2.0}, "malicious": False},
        ],
    )
    write_state(
        p2,
        [
            {"timestamp": 100, "state": {"A": 100.0}, "malicious": False},
            {"timestamp": 101, "state": {"A": 101.0}, "malicious": False},
        ],
    )
    sequences = [load_state_gz(p1, split="train"), load_state_gz(p2, split="train")]
    pairs = make_pair_arrays(sequences, ["A"])
    assert_no_cross_sequence_pairs(sequences, pairs)
    assert pairs.flatten_current().reshape(-1).tolist() == [1.0, 100.0]
    assert pairs.flatten_next().reshape(-1).tolist() == [2.0, 101.0]


def test_cusum_resets_at_file_boundaries() -> None:
    params = CusumParams(delta=0.0, threshold=5.0, growth_cap=100.0, max_calib_cusum=5.0, s=1.0, g=1.0)
    _, alarms, _, blocks = run_cusum_sequences([np.asarray([4.0]), np.asarray([4.0])], params)
    assert alarms.tolist() == [0, 0]
    assert [block.tolist() for block in blocks] == [[0], [0]]


def test_sequence_cusum_fit_resets_for_training_max() -> None:
    params = fit_cusum_params_sequences([np.asarray([0.0, 10.0]), np.asarray([0.0, 10.0])], s=1.0, g=1.0)
    flat_wrong_train_max = 10.0
    assert params.max_calib_cusum < flat_wrong_train_max


def test_variable_typing_uses_values_not_names(tmp_path: Path) -> None:
    path = tmp_path / "train1.state.gz"
    write_state(
        path,
        [
            {"timestamp": 1, "state": {"looks_binary": 0, "continuous_tag": 1.0, "constant": 7}, "malicious": False},
            {"timestamp": 2, "state": {"looks_binary": 1, "continuous_tag": 1.5, "constant": 7}, "malicious": False},
            {"timestamp": 3, "state": {"looks_binary": 1, "continuous_tag": 3.2, "constant": 7}, "malicious": False},
        ],
    )
    table = infer_variable_types([load_state_gz(path, split="train")])
    by_var = {row["variable"]: row for row in table.to_dict("records")}
    assert by_var["looks_binary"]["selected_model"] == "persistence"
    assert by_var["continuous_tag"]["selected_model"] == "symbolic"
    assert by_var["constant"]["selected_model"] == "excluded"


def test_exact_geco_point_is_evaluated() -> None:
    run_hai = load_script("scripts/run_hai_1sec_delta_full.py")
    points = run_hai.make_detection_points(8.02147642080159, 1.4376682451456457)
    assert (8.02147642080159, 1.4376682451456457, "geco_operating_point") in points


def test_geco_model_json_keys_are_parsed(tmp_path: Path) -> None:
    run_hai = load_script("scripts/run_hai_1sec_delta_full.py")
    model = tmp_path / "HAI.model"
    model.write_text(
        json.dumps({"settings": {"threshold_factor": 8.02147642080159, "cusum_factor": 1.4376682451456457}}),
        encoding="utf-8",
    )
    assert run_hai.parse_geco_point(str(model)) == (8.02147642080159, 1.4376682451456457, "geco_model")


def test_output_schema_constants() -> None:
    run_hai = load_script("scripts/run_hai_1sec_delta_full.py")
    assert {"target", "equation", "sympy_format", "holdout_r2", "residual_tail_ratio"}.issubset(
        set(run_hai.SELECTED_EQUATION_COLUMNS)
    )
    assert {"Precision", "Recall", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"}


def test_resume_target_result(tmp_path: Path) -> None:
    run_hai = load_script("scripts/run_hai_1sec_delta_full.py")
    target = "P1_LIT01"
    out = run_hai.pareto_dir(tmp_path, target)
    out.mkdir(parents=True)
    (out / "pareto_front_scored.csv").write_text("equation,loss,score,complexity\n0,1,0,1\n", encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({"status": "completed", "error": "", "elapsed_seconds": 1.2}), encoding="utf-8")
    result = run_hai.target_result_from_artifacts(tmp_path, target, returncode=0)
    assert result["status"] == "completed"
    assert result["elapsed_seconds"] == 1.2


def test_setup_manifest_generation_with_fixtures(tmp_path: Path, monkeypatch) -> None:
    setup = load_script("scripts/setup_hai_21_03.py")
    data_dir = tmp_path / "data" / "hai" / "ipal"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_files = []
    for name in ["train1.csv.gz", "train2.csv.gz", "train3.csv.gz", "test1.csv.gz", "test2.csv.gz", "test3.csv.gz", "test4.csv.gz", "test5.csv.gz"]:
        path = source_dir / name
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("time,A,attack,attack_P1,attack_P2,attack_P3\n")
            handle.write("2020-01-01 00:00:00,1,0,0,0,0\n")
        source_files.append(path)
    for name in EXPECTED_STATE_FILES:
        write_state(
            data_dir / name,
            [
                {"timestamp": 1, "state": {"A": 1.0}, "malicious": False},
                {"timestamp": 2, "state": {"A": 2.0}, "malicious": False},
            ],
        )
    (data_dir / "attacks.json").write_text(json.dumps([{"id": i, "start": i, "end": i, "attack_point": []} for i in range(1, 51)]), encoding="utf-8")
    monkeypatch.setattr(setup, "DATA_DIR", data_dir)
    monkeypatch.setattr(setup, "MANIFEST_PATH", tmp_path / "data" / "hai" / "SOURCE_MANIFEST.json")
    monkeypatch.setattr(setup, "HAI_REPO", tmp_path / "hai")
    monkeypatch.setattr(setup, "IPAL_REPO", tmp_path / "ipal")
    monkeypatch.setattr(setup, "git_sha", lambda repo: "fixture-sha")
    manifest = setup.build_manifest(source_files, {"fixture": "fixture"}, ["python", "transcribe.py"])
    assert manifest["observed"]["attack_count"] == 50
    assert setup.MANIFEST_PATH.exists()


def test_no_raw_hai_data_tracked_by_git() -> None:
    proc = subprocess.run(
        ["git", "ls-files", "data/hai"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    tracked = proc.stdout.splitlines()
    assert not [path for path in tracked if path.endswith((".state.gz", ".csv.gz"))]
