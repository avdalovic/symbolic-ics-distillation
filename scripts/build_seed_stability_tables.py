#!/usr/bin/env python3
"""Regenerate the retraining-stability tables of the ACID paper.

  Tables 11 and 12  seed014_stability_tables.csv
      Aggregated from the recorded per-seed detection points
      (seed014_points_guarded.csv). Continuous metrics: mean and sample
      standard deviation over the three retrainings. FPA, an integer count:
      median and [min-max].

  Tables 13 and 14  seed014_equation_stability_summary.csv,
                    seed014_equation_stability_per_target.csv,
                    seed014_showcase_equations.csv
      Recomputed from the three retrainings' committed equation selections.
      For each target selected in all three runs, the input-variable set of
      each run's equation is extracted and the mean pairwise Jaccard
      similarity over the three run pairs is reported.

Every input is committed, so this script reproduces the recorded tables
exactly; it exists so the aggregation itself is inspectable and re-runnable.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics_symbolic_distill.detection.symbolic_eval import equation_features  # noqa: E402

OUT = REPO / "paper_artifacts" / "seed_stability_v1"
CONT = ["Precision", "Recall", "F1", "eTaP", "eTaR", "eTaF1"]
ORDER = ["SWaT", "WADI", "BATADAL", "HAI_R13"]
SG = {("SWaT", "headline"): "1.20, 15.0", ("WADI", "headline"): "1.20, 25.0",
      ("BATADAL", "headline"): "1.40, 2.00", ("HAI_R13", "headline"): "2.50, 12.0",
      ("SWaT", "geco"): "1.42, 5.98", ("WADI", "geco"): "1.32, 9.74",
      ("BATADAL", "geco"): "1.39, 2.16", ("HAI_R13", "geco"): "8.02, 1.44"}
GECO_IGNORED_HAI = {"P1_PCV02Z", "P2_SIT01", "P2_SIT02", "P2_VT01", "P2_VXT02", "P2_VXT03", "P2_VYT02"}
SHOWCASE = [("SWaT", "LIT101"), ("SWaT", "LIT301"), ("SWaT", "DPIT301")]


def build_tables_11_12() -> pd.DataFrame:
    pts = pd.read_csv(OUT / "seed014_points_guarded.csv")
    rows = []
    for pt in ["headline", "geco"]:
        for ds in ORDER:
            g = pts[(pts.dataset == ds) & (pts.point == pt)].sort_values("seed")
            row = {"point": pt, "dataset": ds, "S_G": SG[(ds, pt)],
                   "seeds": "/".join(str(int(s)) for s in g["seed"]), "n": len(g),
                   "monitored": ";".join(f"{int(s)}:{int(m)}" for s, m in zip(g["seed"], g["num_monitored"]))}
            for m in CONT:
                row[f"{m}_mean"] = round(g[m].mean(), 2)
                row[f"{m}_std"] = round(g[m].std(ddof=1), 2)
                row[m] = f"{g[m].mean():.2f} ± {g[m].std(ddof=1):.2f}"
            f = g["FPA"].astype(float).values
            row.update(FPA_median=float(np.median(f)), FPA_min=float(f.min()), FPA_max=float(f.max()),
                       FPA=f"{np.median(f):.0f} [{f.min():.0f}–{f.max():.0f}]")
            s = g["Scen"].astype(float).values
            row.update(Scen_mean=round(s.mean(), 2), Scen_std=round(s.std(ddof=1), 2),
                       Scen=f"{s.mean():.2f} ± {s.std(ddof=1):.2f}",
                       Scen_median_range=f"{np.median(s):.2f} [{s.min():.2f}–{s.max():.2f}]")
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "seed014_stability_tables.csv", index=False)
    return df


def load_selection(spec: dict, dataset: str, seed: str) -> pd.DataFrame:
    df = pd.read_csv(REPO / spec[dataset]["selected"][seed])
    df["target"] = df["target"].astype(str)
    if dataset == "HAI_R13":
        df = df[~df["target"].isin(GECO_IGNORED_HAI)]
    return df.drop_duplicates("target").set_index("target")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def build_tables_13_14() -> pd.DataFrame:
    spec = json.loads((REPO / "scripts" / "geco_expressiveness_sources.json").read_text())
    vstats = pd.read_csv(REPO / "paper_artifacts" / "expressiveness_v1" / "variable_stats.csv")
    seeds = ["0", "1", "4"]
    summary, per_rows = [], []
    for ds in ORDER:
        sel = {s: load_selection(spec, ds, s) for s in seeds}
        sets = {s: set(sel[s].index) for s in seeds}
        inter = set.intersection(*sets.values())
        union = set.union(*sets.values())
        names = vstats.loc[vstats["dataset"] == ds, "variable"].astype(str).tolist()
        # WADI equations use the original tag names; restrict the universe to the
        # pipeline's variable-name mapping when it is available.
        mapping = REPO / "results" / "wadi" / "variable_name_mapping.csv"
        if ds == "WADI" and mapping.exists():
            names = pd.read_csv(mapping)["original_name"].astype(str).tolist()
        feats = {s: {t: set(equation_features(
            str(sel[s].loc[t].get("sympy_format") or sel[s].loc[t]["equation"]), names))
            for t in inter} for s in seeds}
        pair_t = [jaccard(sets[a], sets[b]) for a, b in itertools.combinations(seeds, 2)]
        ij, exact, same = [], [], 0
        for t in sorted(inter):
            pj = [jaccard(feats[a][t], feats[b][t]) for a, b in itertools.combinations(seeds, 2)]
            ij.append(np.mean(pj))
            eqs = [str(sel[s].loc[t]["equation"]).strip() for s in seeds]
            exact.append(np.mean([eqs[i] == eqs[j] for i, j in itertools.combinations(range(3), 2)]))
            ident = all(feats["0"][t] == feats[s][t] for s in ("1", "4"))
            same += ident
            per_rows.append({"dataset": ds, "target": t,
                             "mean_input_jaccard": round(float(np.mean(pj)), 4),
                             "identical_inputs_all3": bool(ident),
                             "exact_eq_pair_rate": round(float(exact[-1]), 4),
                             **{f"seed{s}_equation": str(sel[s].loc[t]["equation"]) for s in seeds},
                             **{f"seed{s}_inputs": ";".join(sorted(feats[s][t])) for s in seeds}})
        summary.append({"dataset": ds, "seeds": "0/1/4",
                        "counts": ";".join(f"{s}:{len(sets[s])}" for s in seeds),
                        "intersection": len(inter), "union": len(union),
                        "target_IoU": round(len(inter) / len(union), 4),
                        "mean_pair_target_jaccard": round(float(np.mean(pair_t)), 4),
                        "mean_input_jaccard": round(float(np.mean(ij)), 4),
                        "median_input_jaccard": round(float(np.median(ij)), 4),
                        "identical_input_set_targets": int(same),
                        "identical_input_set_pct": round(100 * same / len(inter), 1),
                        "exact_equation_pair_rate": round(float(np.mean(exact)), 4)})
    per = pd.DataFrame(per_rows)
    per.to_csv(OUT / "seed014_equation_stability_per_target.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "seed014_equation_stability_summary.csv", index=False)
    show = per[[(d, t) in SHOWCASE for d, t in zip(per["dataset"], per["target"])]]
    show.to_csv(OUT / "seed014_showcase_equations.csv", index=False)
    return pd.DataFrame(summary)


def main() -> int:
    t11 = build_tables_11_12()
    t13 = build_tables_13_14()
    pd.set_option("display.width", 220)
    print(t11[["point", "dataset", "eTaF1", "FPA"]].to_string(index=False))
    print()
    print(t13[["dataset", "intersection", "mean_input_jaccard", "median_input_jaccard"]].to_string(index=False))
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
