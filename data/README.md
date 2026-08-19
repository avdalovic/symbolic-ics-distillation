# Data Directory

This repository does not include raw SWaT or WADI data.

SWaT and WADI require formal access approval from iTrust/SUTD:

```text
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
```

After approval, follow:

```text
data/swat/README.md
data/wadi/README.md
```

The processors create the CSV filenames expected by the ASID-ICS scripts:

```text
data/swat/raw/swat_train.csv
data/swat/raw/swat_test.csv
data/wadi/raw/wadi_train.csv
data/wadi/raw/wadi_test.csv
```

BATADAL is public and the processed files used in the paper are included:

```text
data/batadal/processed/train.csv
data/batadal/processed/test_dataset04.csv
data/batadal/processed/test_dataset_test.csv
data/batadal/processed/attacks.json
```

Keep raw datasets and derived large arrays outside version control.
