#!/usr/bin/env python3
"""Tail-ratio frontier in BOTH directions (tighten and relax), anchored on shipped.

The shipped candidate filter hardcodes `tail > 50 -> reject`, so no candidate above
50 can ever be selected. scripts/_relaxed_guard.py is a mechanical copy of the real
code with that cutoff parameterised (validated: identical candidate sets at T=50).

Unified rule per target and bound T:
    pool  = Pareto candidates with tail <= T that pass the state-dependence guard
            (+ the shipped equation itself when it complies with T)
    pick  = highest (score, -loss, -complexity); ties resolve to the shipped row
    empty = channel dropped from the roster

At T = shipped bound this returns the shipped selection exactly, so fidelity is
guaranteed; the run asserts the published eTaF1 and reports FAITHFUL / NOT.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
TAILS = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0]
SHIPPED_BOUND = 50.0
os.environ["ASID_TAIL_MAX"] = str(max(TAILS))   # enumerate up to the widest bound

def _imp(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

BASE = _imp("t3d_guard", REPO / "scripts" / "_relaxed_guard.py")
from ics_symbolic_distill.detection.selection_guards import state_dependence_for_delta_equation

HEADLINE = {"swat": (1.20, 15.0), "wadi": (1.20, 25.0), "batadal": (1.40, 2.00), "hai": (2.50, 12.0)}
PUBLISHED = {"swat": 67.6095, "wadi": 71.3327, "batadal": 88.1531, "hai": 64.3554}
CAND = {"swat": BASE.swat_candidate_rows, "wadi": BASE.wadi_candidate_rows,
        "batadal": BASE.batadal_candidate_rows, "hai": BASE.hai_candidate_rows}
IGN = {"P1_PCV02Z","P2_SIT01","P2_SIT02","P2_VT01","P2_VXT02","P2_VXT03","P2_VYT02"}
KEY = {"swat": "SWaT", "wadi": "WADI", "batadal": "BATADAL", "hai": "HAI_R13"}


def skey(r):
    s = BASE.finite_float(r.get("score"), float("-inf"))
    l = BASE.finite_float(r.get("loss"), float("inf"))
    c = BASE.finite_float(r.get("complexity"), float("inf"))
    return (s, -l if np.isfinite(l) else float("-inf"), -c)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["batadal"], choices=list(CAND))
    ap.add_argument("--out", default="paper_artifacts/overnight_v1")
    ap.add_argument("--swat-train", default="data/swat/raw/swat_train.csv")
    ap.add_argument("--swat-test", default="data/swat/raw/swat_test.csv")
    ap.add_argument("--wadi-train", default="data/wadi/raw/wadi_train.csv")
    ap.add_argument("--wadi-test", default="data/wadi/raw/wadi_test.csv")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((REPO / "scripts" / "geco_expressiveness_sources.json").read_text())
    rows, fid = [], []

    for ds in args.datasets:
        cfg = spec[KEY[ds]]
        feats = pd.read_csv(cfg["feature_list"])[cfg["feature_column"]].astype(str).tolist()
        shipped = pd.read_csv(cfg["selected"]["0"]); shipped["target"] = shipped["target"].astype(str)
        if ds == "hai":
            shipped = shipped[~shipped["target"].isin(IGN)].reset_index(drop=True)
        proot = REPO / cfg["pareto_root"]["0"]

        if ds == "batadal":
            arrays = BASE.BAT.load_batadal_arrays(argparse.Namespace(
                train_csv="data/batadal/processed/train.csv", test_csv="data/batadal/processed/test_dataset04.csv"))
        elif ds == "swat":
            arrays = BASE.SWAT.load_arrays(argparse.Namespace(
                experiment="configs/experiment/swat_mlp_current_val20.yaml",
                train_csv=str(args.swat_train), test_csv=str(args.swat_test)))
        elif ds == "wadi":
            arrays = BASE.WADI_FULL.load_wadi_1sec_arrays(argparse.Namespace(
                train_csv=str(args.wadi_train), test_csv=str(args.wadi_test)))
        else:
            arrays = BASE.HAI.load_hai_arrays(argparse.Namespace(
                data_dir="data/hai/ipal", output_dir="artifacts/experiments/hai_baseline_seed0",
                seed=0, sample_size=None, timeout_minutes=60.0, target_wall_timeout_minutes=75.0,
                max_complexity=15, niterations=400, max_workers=2, pysr_procs=1, resume=True,
                force=False, skip_pysr=True, skip_detection_eval=False, single_target=None,
                smoke_targets=None, geco_model=None, promote_results=False, enable_division=False,
                sample_index_source_dir=None))
        ev = {"batadal": lambda p, d: BASE.evaluate_batadal(p, d),
              "swat": lambda p, d: BASE.evaluate_swat(p, args, d),
              "wadi": lambda p, d: BASE.evaluate_wadi(p, args, d),
              "hai": lambda p, d: BASE.evaluate_hai(p, arrays)}[ds]

        cache: dict[str, list] = {}
        for _, row in shipped.iterrows():
            t = str(row["target"])
            if str(row.get("variable_type", "sensor")) == "persistence":
                continue
            cands = CAND[ds](arrays, proot, t, 15)
            cache[t] = [c for c in cands if state_dependence_for_delta_equation(
                target=t, equation=str(c.get("sympy_format") or c.get("equation") or ""),
                feature_names=feats).state_dependent]
        print(f"[{ds}] candidates enumerated up to tail<={max(TAILS)}")

        for T in TAILS:
            keep, changed, dropped = [], 0, []
            for _, row in shipped.iterrows():
                t = str(row["target"]); rd = row.to_dict()
                if str(row.get("variable_type", "sensor")) == "persistence":
                    keep.append(rd); continue
                st = BASE.finite_float(row.get("residual_tail_ratio"), float("inf"))
                if T >= SHIPPED_BOUND:
                    # RELAXING: the shipped selection already saw every candidate with
                    # tail <= SHIPPED_BOUND and chose this one. Only a NEWLY admitted
                    # candidate (SHIPPED_BOUND < tail <= T) may displace it. At T ==
                    # SHIPPED_BOUND that set is empty, so the shipped selection is
                    # reproduced exactly.
                    newly = [c for c in cache.get(t, [])
                             if SHIPPED_BOUND < BASE.finite_float(c.get("residual_tail_ratio"), float("inf")) <= T]
                    better = [c for c in newly if skey(c) > skey(rd)]
                    if better:
                        keep.append(max(better, key=skey)); changed += 1
                    else:
                        keep.append(rd)
                elif st <= T:
                    keep.append(rd)                       # complies: untouched
                else:
                    pool = [c for c in cache.get(t, [])
                            if BASE.finite_float(c.get("residual_tail_ratio"), float("inf")) <= T]
                    if pool:
                        keep.append(max(pool, key=skey)); changed += 1
                    else:
                        dropped.append(t)
            sel = pd.DataFrame(keep)
            d = out / "task3d_tmp" / ds; d.mkdir(parents=True, exist_ok=True)
            p = d / f"sel_T{T}.csv"; sel.to_csv(p, index=False)
            grid = ev(p, d); s, g = HEADLINE[ds]
            r = grid[(grid["S"].round(6) == round(s, 6)) & (grid["G"].round(6) == round(g, 6))]
            rec = {"dataset": ds.upper(), "tail_bound": T, "monitors": len(sel),
                   "changed": changed, "dropped": len(dropped)}
            if len(r):
                rr = r.iloc[0]
                rec.update({m: float(rr[m]) for m in ["F1", "eTaF1", "FPA", "Scen"] if m in rr})
            rows.append(rec)
            print(f"  T<={T:>5}: monitors={len(sel):>3} changed={changed:>2} dropped={len(dropped):>2} "
                  f"eTaF1={rec.get('eTaF1', float('nan')):7.3f} FPA={rec.get('FPA', float('nan')):3.0f} "
                  f"Scen={rec.get('Scen', float('nan')):6.2f}")
            if T == 50.0:
                ok = abs(rec.get("eTaF1", -1) - PUBLISHED[ds]) < 0.05
                fid.append({"dataset": ds.upper(), "reproduced": rec.get("eTaF1"),
                            "published": PUBLISHED[ds], "faithful": bool(ok)})
                print(f"     FIDELITY @50: {rec.get('eTaF1'):.4f} vs {PUBLISHED[ds]:.4f} -> "
                      f"{'FAITHFUL' if ok else 'NOT FAITHFUL'}")

    df = pd.DataFrame(rows); # Fresh runs write to *_run.csv; the recorded full sweep stays untouched
    df.to_csv(out / "task3d_tail_frontier_run.csv", index=False)
    f = pd.DataFrame(fid); f.to_csv(out / "task3d_fidelity_run.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + df.to_string(index=False)); print("\n" + f.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
