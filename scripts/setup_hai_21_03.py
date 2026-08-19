#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.data.hai import (  # noqa: E402
    EXPECTED_ATTACK_COUNT,
    EXPECTED_RAW_FILES,
    EXPECTED_STATE_FILES,
    HAI_RELEASE,
    file_info,
    load_attacks,
    load_hai_sequences,
    sequence_manifest_rows,
    validate_expected_generated_files,
    write_json,
)


HAI_URL = "https://github.com/icsdataset/hai"
IPAL_URL = "https://github.com/ipal-ids/ipal_datasets"
UPSTREAM_ROOT = REPO_ROOT / "artifacts" / "datasets" / "upstream"
HAI_REPO = UPSTREAM_ROOT / "hai"
IPAL_REPO = UPSTREAM_ROOT / "ipal_datasets"
DATA_DIR = REPO_ROOT / "data" / "hai" / "ipal"
MANIFEST_PATH = REPO_ROOT / "data" / "hai" / "SOURCE_MANIFEST.json"


def rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout.strip()


def git_sha(repo: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=repo).strip()


def clone_or_update(url: str, dest: Path, sparse_path: str, *, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force and dest.exists() and not (dest / ".git").exists():
        raise RuntimeError(f"{dest} exists but is not a Git repository")
    if not (dest / ".git").exists():
        run(["git", "clone", "--filter=blob:none", "--sparse", url, str(dest)], cwd=REPO_ROOT)
    else:
        run(["git", "remote", "set-url", "origin", url], cwd=dest)
        run(["git", "fetch", "--depth=1", "origin"], cwd=dest)
        run(["git", "pull", "--ff-only"], cwd=dest)
    run(["git", "sparse-checkout", "set", sparse_path], cwd=dest)


def verify_source_files(hai_repo: Path) -> list[Path]:
    source_dir = hai_repo / "hai-21.03"
    paths = [source_dir / name for name in EXPECTED_RAW_FILES]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing HAI 21.03 files: {missing}")
    for path in paths:
        if path.stat().st_size < 1024:
            raise ValueError(f"{rel(path)} is too small and may be a placeholder")
        with gzip.open(path, "rb") as handle:
            first = handle.read(32)
        if first.startswith(b"version https://git-lfs"):
            raise ValueError(f"{rel(path)} appears to be a Git LFS pointer")
    return paths


def transcribe(source_files: list[Path], ipal_repo: Path, data_dir: Path) -> dict[str, str]:
    converter_dir = ipal_repo / "HAI"
    raw_dir = converter_dir / "raw"
    generated_dir = converter_dir / "ipal"
    raw_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        shutil.copy2(source, raw_dir / source.name)
    for name in EXPECTED_STATE_FILES:
        out = generated_dir / name
        if out.exists():
            out.unlink()
    attacks_out = converter_dir / "attacks.json"
    if attacks_out.exists():
        attacks_out.unlink()
    command = [sys.executable, "transcribe.py"]
    run(command, cwd=converter_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for name in EXPECTED_STATE_FILES:
        src = generated_dir / name
        dst = data_dir / name
        if not src.exists():
            raise FileNotFoundError(f"converter did not emit {src}")
        shutil.copy2(src, dst)
        mapping[rel(src)] = rel(dst)
    if not attacks_out.exists():
        raise FileNotFoundError(f"converter did not emit {attacks_out}")
    shutil.copy2(attacks_out, data_dir / "attacks.json")
    mapping[rel(attacks_out)] = rel(data_dir / "attacks.json")
    return mapping


def compact_file_info(path: Path) -> dict[str, Any]:
    info = file_info(path)
    info["path"] = rel(path)
    return info


def build_manifest(source_files: list[Path], generated_mapping: dict[str, str] | None, converter_command: list[str]) -> dict[str, Any]:
    generated_files = validate_expected_generated_files(DATA_DIR)
    attacks = load_attacks(DATA_DIR / "attacks.json")
    if len(attacks) != EXPECTED_ATTACK_COUNT:
        raise ValueError(f"expected {EXPECTED_ATTACK_COUNT} attacks, found {len(attacks)}")
    sequences = load_hai_sequences(DATA_DIR)
    sequence_rows = sequence_manifest_rows(sequences)
    for row in sequence_rows:
        row["path"] = rel(Path(row["path"]))
    manifest = {
        "hai_release": HAI_RELEASE,
        "setup_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hai_repository": {
            "url": HAI_URL,
            "commit_sha": git_sha(HAI_REPO),
            "dataset_directory": "hai-21.03",
        },
        "ipal_repository": {
            "url": IPAL_URL,
            "commit_sha": git_sha(IPAL_REPO),
            "converter_directory": "HAI",
        },
        "expected": {
            "training_files": 3,
            "test_files": 5,
            "attack_count": EXPECTED_ATTACK_COUNT,
        },
        "observed": {
            "training_files": sum(path.name.startswith("train") for path in generated_files),
            "test_files": sum(path.name.startswith("test") for path in generated_files),
            "attack_count": len(attacks),
        },
        "converter_command": converter_command,
        "source_files": [compact_file_info(path) for path in source_files],
        "generated_files": [compact_file_info(path) for path in generated_files + [DATA_DIR / "attacks.json"]],
        "source_to_generated_mapping": generated_mapping or {},
        "sequences": sequence_rows,
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and transcribe official HAI 21.03 through IPAL.")
    parser.add_argument("--download", action="store_true", help="Clone or update official upstream repositories.")
    parser.add_argument("--transcribe", action="store_true", help="Run ipal_datasets/HAI/transcribe.py.")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing local HAI transcription and refresh the manifest.")
    parser.add_argument("--force", action="store_true", help="Overwrite generated local transcription files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.force and not (args.download or args.transcribe or args.validate_only):
        args.download = True
        args.transcribe = True
    if not (args.download or args.transcribe or args.validate_only):
        raise SystemExit("Choose --download, --transcribe, --validate-only, or --force.")

    if args.download:
        clone_or_update(HAI_URL, HAI_REPO, "hai-21.03", force=bool(args.force))
        clone_or_update(IPAL_URL, IPAL_REPO, "HAI", force=bool(args.force))
    if not HAI_REPO.exists() or not IPAL_REPO.exists():
        raise FileNotFoundError("Run with --download first; upstream repositories are missing.")

    source_files = verify_source_files(HAI_REPO)
    mapping: dict[str, str] | None = None
    if args.transcribe:
        if DATA_DIR.exists() and args.force:
            for path in DATA_DIR.glob("*"):
                if path.is_file():
                    path.unlink()
        mapping = transcribe(source_files, IPAL_REPO, DATA_DIR)

    validate_expected_generated_files(DATA_DIR)
    manifest = build_manifest(
        source_files,
        mapping,
        ["python", "transcribe.py"],
    )
    print("HAI 21.03 setup validated")
    print(f"HAI SHA:  {manifest['hai_repository']['commit_sha']}")
    print(f"IPAL SHA: {manifest['ipal_repository']['commit_sha']}")
    print(f"Manifest: {rel(MANIFEST_PATH)}")
    print(f"Rows: {[row['rows'] for row in manifest['sequences']]}")
    print(f"Attacks: {manifest['observed']['attack_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
