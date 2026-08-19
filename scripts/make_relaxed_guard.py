#!/usr/bin/env python3
"""Make a copy of the selection code with the tail-ratio cutoff as a variable.

The shipped candidate filter has a fixed limit of 50 for the residual tail ratio.
To test a limit above 50, you must make this limit a variable. This script makes
two copies of the selection code. In the copies, the limit reads the environment
variable ASID_TAIL_MAX.

The copies are mechanical. The script changes only the numeric limit. It does not
change the filter logic. Run scripts/overnight/task3d_tail_frontier_both.py after
this script.
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INJECT = "import os as _os\n_TAIL_MAX = float(_os.environ.get('ASID_TAIL_MAX', '50.0'))\n"


def patch(src: Path, dst: Path, needle: str, extra: tuple[str, str] | None = None) -> int:
    text = src.read_text(encoding="utf-8")
    count = text.count(needle)
    if count == 0:
        raise SystemExit(f"pattern not found in {src}: {needle}")
    text = text.replace(needle, needle.replace("50.0", "_TAIL_MAX"))
    if extra:
        text = text.replace(*extra)
    lines = text.split("\n")
    idx = next((i for i, line in enumerate(lines) if line.startswith("from __future__")), -1)
    lines.insert(idx + 1, INJECT)
    dst.write_text("\n".join(lines), encoding="utf-8")
    return count


def main() -> int:
    s = REPO / "scripts"
    a = patch(s / "analyze_batadal_selection_ablation.py", s / "_relaxed_batadal_sel.py",
              "if tail_ratio > 50.0:")
    b = patch(s / "evaluate_global_selection_guard.py", s / "_relaxed_guard.py",
              "if tail > 50.0:",
              ('REPO_ROOT / "scripts" / "analyze_batadal_selection_ablation.py"',
               'REPO_ROOT / "scripts" / "_relaxed_batadal_sel.py"'))
    print(f"Made scripts/_relaxed_batadal_sel.py ({a} limit) and scripts/_relaxed_guard.py ({b} limits).")
    print("The copies are temporary. Delete them after you complete the sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
