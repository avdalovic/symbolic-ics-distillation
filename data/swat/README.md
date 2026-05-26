# SWaT Data Setup

SWaT is distributed by iTrust/SUTD and is not included in this repository.
Request access before using the full reproduction pipeline:

```text
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
```

Use the first SWaT dataset release, `SWaT.A1 & A2_Dec 2015`. From its
`Physical` directory, export or save these files as CSV:

```text
data/swat/raw/SWaT_Dataset_Normal_v1.csv
data/swat/raw/SWaT_Dataset_Attack_v0.csv
```

Then run:

```bash
python data/swat/process_SWaT.py
```

The processor writes the filenames expected by the ASID-ICS scripts:

```text
data/swat/raw/swat_train.csv
data/swat/raw/swat_test.csv
```

The processor removes whitespace from column names, converts process variables
to numeric values, and replaces the attack labels with the documented SWaT attack windows used by this artifact.
