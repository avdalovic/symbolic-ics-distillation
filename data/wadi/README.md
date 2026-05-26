# WADI Data Setup

WADI is distributed by iTrust/SUTD and is not included in this repository.
Request access before using the full reproduction pipeline:

```text
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
```

Use the first WADI dataset release, `WADI.A1_9 Oct 2017`, and place:

```text
data/wadi/raw/WADI_14days.csv
data/wadi/raw/WADI_attackdata.csv
```

Then run:

```bash
python data/wadi/process_WADI.py
```

The processor writes the filenames expected by the ASID-ICS scripts:

```text
data/wadi/raw/wadi_train.csv
data/wadi/raw/wadi_test.csv
```

The processor removes the long WADI logger prefix from tag names, adds the
documented 15 attack windows, and applies the small training-set NaN patches
used by this artifact.
