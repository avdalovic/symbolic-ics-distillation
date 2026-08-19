#!/usr/bin/env python3
"""Compile the paper-ready GeCo-expressiveness tables.

Reads the audit (equation_class_per_target.csv) and every restriction run
(restriction_seed*/restriction_summary.csv) and emits:

    table_expressiveness_audit.csv    per dataset x seed: selected / outside / %
    table_expressiveness_audit.tex    LaTeX version of the same
    table_restriction_ablation.csv    ASID vs GeCo-class-restricted detection
    showcase_out_of_class.csv         concrete out-of-class equations w/ GeCo counterpart
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ORDER = ["SWaT", "WADI", "BATADAL", "HAI_R13"]
LABEL = {"SWaT": "SWaT", "WADI": "WADI", "BATADAL": "BATADAL", "HAI_R13": "HAI (R13)"}
GECO_MODELS = {
    "SWaT": "artifacts/experiments/geco_model_inspection/SWaT.model",
    "WADI": "artifacts/experiments/geco_model_inspection/WADI.model",
    "BATADAL": "artifacts/experiments/geco_model_inspection/BATADAL.model",
    "HAI_R13": "artifacts/experiments/geco_model_inspection/HAI.model",
}


def geco_entry(dataset: str, target: str) -> str:
    path = GECO_MODELS.get(dataset)
    if not path or not Path(path).exists():
        return ""
    ci = json.loads(Path(path).read_text()).get("CI", {})
    e = ci.get(target)
    if e is None:
        return "target absent from GeCo model"
    comb, par = e["combination"], [round(float(x), 6) for x in e["parameters"]]
    if e["equation"] == "Sum":
        terms = [f"{par[0]}*{comb[0]}"] + [f"{par[i + 1]}*{comb[i + 1]}" for i in range(len(comb) - 1)] + [str(par[-1])]
        return "Sum: " + " + ".join(terms)
    return f"Product: {par[0]}*{comb[0]} + {par[1]}*({'*'.join(comb[1:])}) + {par[2]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="paper_artifacts/expressiveness_v1")
    args = ap.parse_args()
    root = Path(args.dir)

    per = pd.read_csv(root / "equation_class_per_target.csv")
    per["outside"] = ~per["in_geco_class"].astype(bool)

    audit = (
        per.groupby(["dataset", "seed"])
        .agg(selected=("target", "size"), outside=("outside", "sum"),
             nondegenerate=("outside", lambda s: 0))
        .reset_index()
    )
    nd = (
        per[per["outside"] & ~per["involves_constant_var"].astype(bool)]
        .groupby(["dataset", "seed"]).size().rename("nondegenerate").reset_index()
    )
    audit = audit.drop(columns=["nondegenerate"]).merge(nd, on=["dataset", "seed"], how="left").fillna({"nondegenerate": 0})
    audit["pct_outside"] = (100 * audit["outside"] / audit["selected"]).round(1)
    audit["dataset"] = pd.Categorical(audit["dataset"], ORDER, ordered=True)
    audit = audit.sort_values(["dataset", "seed"])
    audit.to_csv(root / "table_expressiveness_audit.csv", index=False)

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Dataset & Seed & Selected & Outside GeCo class & \% \\", r"\midrule"]
    for _, r in audit.iterrows():
        lines.append(f"{LABEL[str(r['dataset'])]} & {int(r['seed'])} & {int(r['selected'])} & "
                     f"{int(r['outside'])} & {r['pct_outside']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (root / "table_expressiveness_audit.tex").write_text("\n".join(lines), encoding="utf-8")

    # Rebuild from the per-dataset detection grids: restriction_summary.csv is
    # rewritten by each invocation and therefore only holds the last dataset run
    # for that seed.
    points = {"swat": {"headline": (1.20, 15.0), "geco": (1.42, 5.98)},
              "wadi": {"headline": (1.20, 25.0), "geco": (1.32, 9.74)},
              "batadal": {"headline": (1.40, 2.00), "geco": (1.39, 2.16)},
              "hai": {"headline": (2.50, 12.0), "geco": (8.02147642080159, 1.4376682451456457)}}
    frames = []
    for grid_path in sorted(root.glob("restriction_seed*/*/detection_grid_*.csv")):
        seed = int(grid_path.parent.parent.name.replace("restriction_seed", ""))
        ds = grid_path.parent.name
        arm = grid_path.stem.replace("detection_grid_", "")
        if ds not in points:
            continue
        grid = pd.read_csv(grid_path)
        for pname, (s, g) in points[ds].items():
            sel = grid[(grid["S"].round(6) == round(s, 6)) & (grid["G"].round(6) == round(g, 6))]
            if not len(sel):
                continue
            r = sel.iloc[0]
            frames.append(pd.DataFrame([{
                "dataset": ds.upper(), "seed": seed, "arm": arm, "point": pname,
                **{m: float(r[m]) for m in ["F1", "eTaF1", "FPA", "Scen"] if m in r},
            }]))
    if frames:
        rest = pd.concat(frames, ignore_index=True)
        wide = rest.pivot_table(index=["dataset", "seed", "point"], columns="arm",
                                values=["F1", "eTaF1", "FPA", "Scen"]).reset_index()
        wide.columns = ["_".join(c).rstrip("_") for c in wide.columns]
        if "eTaF1_baseline" in wide and "eTaF1_geco_class" in wide:
            wide["delta_eTaF1"] = (wide["eTaF1_geco_class"] - wide["eTaF1_baseline"]).round(3)
        wide.to_csv(root / "table_restriction_ablation.csv", index=False)
        print(wide.to_string(index=False))

    # Representative out-of-class equations. The paper's Table 15 examples are
    # always included; the remainder are the highest-holdout-R2 rows per plant.
    PAPER_TARGETS = {("SWaT", "FIT601"), ("WADI", "2_LS_301_AL"),
                     ("BATADAL", "P_J415"), ("HAI_R13", "P1_FT03Z")}
    pool = per[per["outside"] & ~per["involves_constant_var"].astype(bool)].copy()
    pool = pool.sort_values(["dataset", "holdout_r2"], ascending=[True, False])
    fixed = pool[[(d, t) in PAPER_TARGETS for d, t in zip(pool["dataset"], pool["target"])]]
    fixed = fixed.sort_values("seed").groupby(["dataset", "target"]).head(1)
    rest = pool.merge(fixed[["dataset", "target"]].drop_duplicates(), on=["dataset", "target"],
                      how="left", indicator=True)
    rest = rest[rest["_merge"] == "left_only"].drop(columns="_merge").groupby("dataset").head(2)
    show = pd.concat([fixed, rest]).sort_values(["dataset", "holdout_r2"], ascending=[True, False])
    show["geco_equation"] = [geco_entry(str(d), str(t)) for d, t in zip(show["dataset"], show["target"])]
    cols = ["dataset", "seed", "target", "holdout_r2", "outside_reason", "equation", "monomials", "geco_equation"]
    show[cols].to_csv(root / "showcase_out_of_class.csv", index=False)

    print("\n=== AUDIT ===")
    print(audit[["dataset", "seed", "selected", "outside", "nondegenerate", "pct_outside"]].to_string(index=False))
    print(f"\nWrote tables to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
