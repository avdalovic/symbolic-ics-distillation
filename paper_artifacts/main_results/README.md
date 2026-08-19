# Main results

Files in this directory, keyed to the paper:

| File | Paper |
|---|---|
| `main_table.csv` | ACID rows of Table 2 |
| `comparison_table.csv` | Table 2 |
| `knowledge_swat.csv` | Table 3 |
| `operating_points.csv` | Table 8 |
| `<plant>/equations.csv` | deployed equations (Tables 4, 14, 15) |
| `<plant>/grid.csv` | S/G sensitivity grids (Figure 4) |
| `<plant>/per_attack.csv` | per-attack outcome |

ACID rows are read from the committed detection grids. Rows marked
`wolsing2025gecos` are the published baseline values, reproduced verbatim.
`S` and `G` are the CUSUM scale and growth parameters. HAI is evaluated on
the channel set that excludes the seven channels the GeCo baseline also
ignores, so both methods are measured on the same channels.

Rebuild with `python scripts/build_main_results.py`.
