# HAI 21.03 Dataset Setup

HAI 21.03 is downloaded from the official HAI repository and transcribed
with the official IPAL dataset converter. The raw CSV.GZ files and the
generated `.state.gz` files are local artifacts and are intentionally not
tracked by Git.

```bash
python scripts/setup_hai_21_03.py --download --transcribe
python scripts/setup_hai_21_03.py --validate-only
```

The setup command clones the upstream repositories into
`artifacts/datasets/upstream/`, runs `ipal_datasets/HAI/transcribe.py`,
and writes the generated files to `data/hai/ipal/`:

- `attacks.json`
- `train1.state.gz`, `train2.state.gz`, `train3.state.gz`
- `test1.state.gz`, `test2.state.gz`, `test3.state.gz`, `test4.state.gz`, `test5.state.gz`

The committed `SOURCE_MANIFEST.json` records upstream commit SHAs, file
hashes, row counts, timestamp ranges, and the converter command used for
the local transcription.
