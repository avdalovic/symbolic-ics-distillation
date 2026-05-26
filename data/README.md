# Data Directory

This repository does not include raw SWaT or WADI data.

Expected local paths:

```text
data/swat/raw/swat_train.csv
data/swat/raw/swat_test.csv
data/wadi/raw/wadi_train.csv
data/wadi/raw/wadi_test.csv
```

The default YAML configs expect those exact paths. If your authorized local copies use different filenames, update the dataset config instead of committing raw data.

Keep raw datasets and derived large arrays outside version control.
