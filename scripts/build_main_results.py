#!/usr/bin/env python3
"""Assemble the main results of the ACID paper into paper_artifacts/main_results/.

Outputs, keyed to the paper:

  main_table.csv         ACID at each plant's operating point
  comparison_table.csv   Table 2  - ACID and the six published IIDS baselines
  knowledge_swat.csv     Table 3  - ACID and the knowledge-based IIDS on SWaT
  operating_points.csv   Table 8  - ACID at its own and at GeCo's CUSUM parameters
  <plant>/equations.csv  the deployed equations (Tables 4, 14, 15 draw on these)
  <plant>/grid.csv       detection metrics over the S/G grid (Figure 4)
  <plant>/per_attack.csv per-attack outcome

ACID rows are read from the committed detection grids, so every table produced
here is consistent with the grids by construction. Baseline rows are the values
published in the GeCo evaluation (wolsing2025gecos, measured with the IPAL
framework) and are reproduced here verbatim for side-by-side comparison.

HAI is evaluated on the channel set that excludes the seven channels the GeCo
baseline also ignores, so both methods are measured on the same channels.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper_artifacts" / "main_results"
METRICS = ["Precision", "Recall", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"]

# Operating points: (ACID S, ACID G) and (GeCo S, GeCo G) per plant.
ACID_POINT = {"swat": (1.20, 15.0), "wadi": (1.20, 25.0), "batadal": (1.40, 2.00), "hai": (2.50, 12.0)}
GECO_POINT = {"swat": (1.42, 5.98), "wadi": (1.32, 9.74), "batadal": (1.39, 2.16),
              "hai": (8.02147642080159, 1.4376682451456457)}
LABEL = {"swat": "SWaT", "wadi": "WADI", "batadal": "BATADAL", "hai": "HAI"}

FILES = {
    "swat": ("results/swat/selected_equations.csv",
             "paper_artifacts/final_v2/extended_grid_swat.csv",
             "results/swat/per_attack.csv"),
    "wadi": ("results/wadi/selected_equations.csv",
             "paper_artifacts/final_v2/extended_grid_wadi.csv",
             "results/wadi/per_attack.csv"),
    "batadal": ("paper_artifacts/selected_models/batadal/selected_equations.csv",
                "paper_artifacts/selected_models/batadal/detection_grid.csv",
                "results/batadal/per_attack.csv"),
    "hai": ("paper_artifacts/selected_models/hai/selected_equations_guarded_r13.csv",
            "paper_artifacts/selected_models/hai_r13/without_geco_ignored_detection_grid.csv",
            "paper_artifacts/selected_models/hai_r13/without_geco_ignored_per_attack.csv"),
}

# Published baseline results (wolsing2025gecos, IPAL framework), as reported in
# Table 2 of the paper: Prec, Rec, F1, eTaP, eTaR, eTaF1, FPA, Scen.
PUBLISHED = {
    "SWaT": {
        "GeCo":      (94.8, 79.0, 86.2, 83.1, 60.7, 70.2, 4, 86.1),
        "SIMPLE":    (70.7, 86.7, 77.9, 58.7, 47.2, 52.3, 18, 75.0),
        "TABOR":     (81.5, 74.7, 77.9, 49.1, 18.9, 27.3, 27, 55.6),
        "Invariant": (97.3, 69.1, 80.8, 54.7, 29.8, 38.6, 182, 86.1),
        "Seq2SeqNN": (44.0, 10.9, 17.5, 42.8, 47.2, 44.9, 36, 75.0),
        "PASAD":     (32.4, 71.5, 44.6, 16.0, 4.9, 7.5, 14, 44.4),
    },
    "WADI": {
        "GeCo":      (92.6, 32.1, 47.7, 91.3, 56.3, 69.7, 0, 78.6),
        "SIMPLE":    (58.2, 43.6, 49.8, 57.0, 52.1, 54.4, 8, 64.3),
        "TABOR":     (19.1, 43.7, 26.6, 14.9, 13.0, 13.9, 3, 57.1),
        "Invariant": (90.0, 21.9, 35.2, 92.3, 32.6, 48.1, 2, 42.9),
        "Seq2SeqNN": (44.4, 13.4, 20.5, 45.4, 31.3, 37.1, 7, 64.3),
        "PASAD":     (16.4, 23.9, 19.5, 5.4, 4.3, 4.8, 3, 35.7),
    },
    "BATADAL": {
        "GeCo":      (93.8, 73.4, 82.3, 97.0, 88.1, 92.4, 0, 100.0),
        "SIMPLE":    (52.0, 43.3, 47.2, 49.0, 42.8, 45.7, 4, 71.4),
        "TABOR":     (78.5, 6.9, 12.7, 77.7, 14.3, 24.1, 2, 14.3),
        "Invariant": (27.2, 45.5, 34.0, 18.2, 74.9, 29.3, 865, 100.0),
        "Seq2SeqNN": (34.2, 5.6, 9.6, 27.0, 6.9, 11.0, 1, 14.3),
        "PASAD":     (20.1, 52.1, 29.1, 10.5, 21.5, 14.1, 32, 78.6),
    },
    "HAI": {
        "GeCo":      (75.4, 55.5, 63.9, 73.8, 65.4, 69.4, 8, 90.0),
        "SIMPLE":    (87.0, 39.8, 54.7, 86.0, 61.5, 71.7, 4, 88.0),
        "TABOR":     (4.8, 45.1, 8.7, 0.0, 0.0, 0.0, 4, 46.0),
        "Invariant": (77.1, 9.1, 16.2, 79.2, 25.2, 38.3, 12, 50.0),
        "Seq2SeqNN": (8.5, 4.6, 6.0, 7.7, 2.9, 4.2, 14, 26.0),
        "PASAD":     (3.3, 12.7, 5.3, 1.0, 2.0, 1.3, 51, 16.0),
    },
}

# Knowledge-based IIDS on SWaT (published in the GeCo evaluation), Table 3.
KNOWLEDGE_SWAT = (92.1, 69.2, 79.0, 75.7, 28.7, 41.6, 3, 52.8)


def grid_row(plant: str, s: float, g: float) -> dict:
    """Read one operating point from the plant's committed detection grid."""
    grid = pd.read_csv(REPO / FILES[plant][1])
    sel = grid[(grid["S"].round(6) == round(s, 6)) & (grid["G"].round(6) == round(g, 6))]
    if sel.empty:
        raise SystemExit(f"{plant}: no grid row at S={s}, G={g}")
    r = sel.iloc[0]
    out = {}
    for m in METRICS:
        v = r.get(m)
        if pd.isna(v):
            v = r.get({"Precision": "Prec", "Recall": "Rec"}.get(m, m))
        out[m] = round(float(v), 4)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # Per-plant files and the ACID main table
    acid_rows = []
    for plant, (eqs, grid, per_attack) in FILES.items():
        d = OUT / plant
        d.mkdir(parents=True, exist_ok=True)
        for name, rel in (("equations", eqs), ("grid", grid), ("per_attack", per_attack)):
            src = REPO / rel
            if not src.exists():
                raise SystemExit(f"missing source: {rel}")
            shutil.copyfile(src, d / f"{name}.csv")
        s, g = ACID_POINT[plant]
        acid_rows.append({"dataset": LABEL[plant], "method": "ACID", "S": s, "G": g,
                          **grid_row(plant, s, g)})
    main_table = pd.DataFrame(acid_rows)
    main_table.to_csv(OUT / "main_table.csv", index=False)

    # Table 2: ACID + published baselines
    comp_rows = []
    for plant in FILES:
        ds = LABEL[plant]
        acid = main_table[main_table["dataset"] == ds].iloc[0]
        comp_rows.append({"dataset": ds, "method": "ACID", "source": "this artifact",
                          **{m: float(acid[m]) for m in METRICS}})
        for method, vals in PUBLISHED[ds].items():
            comp_rows.append({"dataset": ds, "method": method, "source": "wolsing2025gecos",
                              **dict(zip(METRICS, vals))})
    pd.DataFrame(comp_rows).to_csv(OUT / "comparison_table.csv", index=False)

    # Table 3: knowledge-based comparison on SWaT
    acid_swat = main_table[main_table["dataset"] == "SWaT"].iloc[0]
    pd.DataFrame([
        {"method": "ACID", "source": "this artifact", **{m: float(acid_swat[m]) for m in METRICS}},
        {"method": "Knowledge", "source": "wolsing2025gecos", **dict(zip(METRICS, KNOWLEDGE_SWAT))},
    ]).to_csv(OUT / "knowledge_swat.csv", index=False)

    # Table 8: ACID at its own point, ACID at GeCo's point, GeCo published
    op_rows = []
    for plant in FILES:
        ds = LABEL[plant]
        s, g = ACID_POINT[plant]
        op_rows.append({"dataset": ds, "setting": "Ours", "S": s, "G": g,
                        "source": "this artifact", **grid_row(plant, s, g)})
        gs, gg = GECO_POINT[plant]
        op_rows.append({"dataset": ds, "setting": "At GeCo S/G", "S": round(gs, 2), "G": round(gg, 2),
                        "source": "this artifact", **grid_row(plant, gs, gg)})
        op_rows.append({"dataset": ds, "setting": "GeCo", "S": round(gs, 2), "G": round(gg, 2),
                        "source": "wolsing2025gecos", **dict(zip(METRICS, PUBLISHED[ds]["GeCo"]))})
    pd.DataFrame(op_rows).to_csv(OUT / "operating_points.csv", index=False)

    (OUT / "README.md").write_text(
        "# Main results\n\n"
        "Files in this directory, keyed to the paper:\n\n"
        "| File | Paper |\n|---|---|\n"
        "| `main_table.csv` | ACID rows of Table 2 |\n"
        "| `comparison_table.csv` | Table 2 |\n"
        "| `knowledge_swat.csv` | Table 3 |\n"
        "| `operating_points.csv` | Table 8 |\n"
        "| `<plant>/equations.csv` | deployed equations (Tables 4, 14, 15) |\n"
        "| `<plant>/grid.csv` | S/G sensitivity grids (Figure 4) |\n"
        "| `<plant>/per_attack.csv` | per-attack outcome |\n\n"
        "ACID rows are read from the committed detection grids. Rows marked\n"
        "`wolsing2025gecos` are the published baseline values, reproduced verbatim.\n"
        "`S` and `G` are the CUSUM scale and growth parameters. HAI is evaluated on\n"
        "the channel set that excludes the seven channels the GeCo baseline also\n"
        "ignores, so both methods are measured on the same channels.\n\n"
        "Rebuild with `python scripts/build_main_results.py`.\n",
        encoding="utf-8")

    pd.set_option("display.width", 200)
    print(main_table.to_string(index=False))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
