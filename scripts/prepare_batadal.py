#!/usr/bin/env python3
"""
Label BATADAL datasets using attacks.json ground truth.
Works with the CSVs already downloaded from batadal.net.

Usage:
    python prepare_batadal.py
"""
import csv
import json
import os
from datetime import datetime

BATADAL_DIR = "data/batadal"
OUT_DIR = "data/batadal/processed"

ATTACKS = [
    {"id": "1", "start": "13/09/2016 23", "end": "16/09/2016 00", "target": "L_T7"},
    {"id": "2", "start": "26/09/2016 11", "end": "27/09/2016 10", "target": "L_T7"},
    {"id": "3", "start": "09/10/2016 09", "end": "11/10/2016 20", "target": "L_T1"},
    {"id": "4", "start": "29/10/2016 19", "end": "02/11/2016 16", "target": "L_T1"},
    {"id": "5", "start": "26/11/2016 17", "end": "29/11/2016 04", "target": "F_PU7"},
    {"id": "6", "start": "06/12/2016 07", "end": "10/12/2016 04", "target": "F_PU7"},
    {"id": "7", "start": "14/12/2016 15", "end": "19/12/2016 04", "target": "F_PU7"},
    {"id": "8", "start": "16/01/2017 09", "end": "19/01/2017 06", "target": "L_T3"},
    {"id": "9", "start": "30/01/2017 08", "end": "02/02/2017 00", "target": "L_T2"},
    {"id": "10", "start": "09/02/2017 03", "end": "10/02/2017 09", "target": "F_PU3"},
    {"id": "11", "start": "12/02/2017 01", "end": "13/02/2017 07", "target": "F_PU3"},
    {"id": "12", "start": "24/02/2017 05", "end": "28/02/2017 08", "target": "F_PU3"},
    {"id": "13", "start": "10/03/2017 14", "end": "13/03/2017 21", "target": "L_T7"},
    {"id": "14", "start": "25/03/2017 20", "end": "27/03/2017 01", "target": "L_T4"},
]


def parse_dt(s):
    s = s.strip()
    for fmt in ["%d/%m/%y %H", "%d/%m/%Y %H"]:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s}")


def build_attack_intervals():
    intervals = []
    for a in ATTACKS:
        s = parse_dt(a["start"])
        e = parse_dt(a["end"])
        intervals.append((s, e, a["id"], a["target"]))
    return intervals


def get_attack(ts, intervals):
    for s, e, aid, target in intervals:
        if s <= ts <= e:
            return aid, target
    return None, None


def process_csv(inpath, outpath, intervals):
    with open(inpath, "r") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]

    sensor_cols = [h for h in header if h not in ("DATETIME", "ATT_FLAG", "")]
    out_header = ["timestamp"] + sensor_cols + ["ATT_FLAG", "attack_id", "attack_target"]

    rows_written = 0
    n_attack = 0
    attack_ids_found = set()

    with open(inpath, "r") as fin, open(outpath, "w", newline="") as fout:
        reader = csv.reader(fin)
        next(reader)
        writer = csv.writer(fout)
        writer.writerow(out_header)

        for row in reader:
            row = [v.strip() for v in row]
            ts = parse_dt(row[0])
            aid, target = get_attack(ts, intervals)

            out_row = [int(ts)]
            if header[-1] == "ATT_FLAG":
                vals = row[1:-1]
            else:
                vals = row[1:]
            out_row.extend(vals)
            out_row.append(1 if aid else 0)
            out_row.append(aid if aid else "")
            out_row.append(target if target else "")

            writer.writerow(out_row)
            rows_written += 1
            if aid:
                n_attack += 1
                attack_ids_found.add(aid)

    return rows_written, n_attack, attack_ids_found


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    intervals = build_attack_intervals()

    ds03 = os.path.join(BATADAL_DIR, "BATADAL_dataset03.csv")
    ds04 = os.path.join(BATADAL_DIR, "BATADAL_dataset04.csv")

    test_candidates = [
        os.path.join(BATADAL_DIR, "BATADAL_test_dataset.csv"),
        os.path.join(BATADAL_DIR, "BATADAL_dataset_test.csv"),
    ]
    ds_test = None
    for c in test_candidates:
        if os.path.exists(c):
            ds_test = c
            break

    if os.path.exists(ds03):
        print(f"Processing {ds03}...")
        n, na, aids = process_csv(ds03, os.path.join(OUT_DIR, "train.csv"), intervals)
        print(f"  {n} rows, {na} attack rows, attacks: {sorted(aids) if aids else 'none'}")
    else:
        print(f"WARNING: {ds03} not found")

    if os.path.exists(ds04):
        print(f"Processing {ds04}...")
        n, na, aids = process_csv(ds04, os.path.join(OUT_DIR, "test_dataset04.csv"), intervals)
        print(f"  {n} rows, {na} attack rows, attacks: {sorted(aids)}")
    else:
        print(f"WARNING: {ds04} not found")

    if ds_test:
        print(f"Processing {ds_test}...")
        n, na, aids = process_csv(ds_test, os.path.join(OUT_DIR, "test_dataset_test.csv"), intervals)
        print(f"  {n} rows, {na} attack rows, attacks: {sorted(aids)}")
    else:
        print(f"WARNING: test dataset not found. Download and unzip:")
        print(f"  cd {BATADAL_DIR}")
        print(f"  wget https://www.batadal.net/data/BATADAL_test_dataset.zip")
        print(f"  unzip BATADAL_test_dataset.zip")

    with open(os.path.join(OUT_DIR, "attacks.json"), "w") as f:
        json.dump(ATTACKS, f, indent=2)
    print(f"\nSaved attacks.json with {len(ATTACKS)} attacks")
    print(f"Output in: {OUT_DIR}")


if __name__ == "__main__":
    main()