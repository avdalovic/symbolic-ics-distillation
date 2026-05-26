# BATADAL Data Setup

BATADAL is public and small enough for this artifact to include the processed
CSV files used by the paper:

```text
data/batadal/processed/train.csv
data/batadal/processed/test_dataset04.csv
data/batadal/processed/test_dataset_test.csv
data/batadal/processed/attacks.json
```

These files are sufficient for the BATADAL rerun command in the root
`README.md`.

To regenerate the processed files from the public BATADAL CSVs, place the raw
downloads at:

```text
data/batadal/BATADAL_dataset03.csv
data/batadal/BATADAL_dataset04.csv
data/batadal/BATADAL_test_dataset.csv
```

Then run:

```bash
python scripts/prepare_batadal.py
```
