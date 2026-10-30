# Results

This directory contains the archived measurements for Tables 3–6 and Figure 2.
New experiment outputs are written separately under
`runs/reproduction/experiment/results/`; see the
[reproduction guide](../docs/REPRODUCING.md#5-generate-the-results).

## Layout

| Path | Contents |
| --- | --- |
| [raw/benchmarks/](raw/benchmarks/) | 435 solver logs and 15 batch summary CSVs, grouped by dataset and method |
| [raw/table_6_figure_2/](raw/table_6_figure_2/) | 15 scalability logs, grouped by method |
| [tables_3_4_5_instance_results.csv](tables_3_4_5_instance_results.csv) | Per-instance timings, status, dimensions, and reduction ratios for the 435 benchmark runs |
| [Table3_Ablation/](Table3_Ablation/) | Folding and presolve ablation table CSV |
| [Table4_Preprocessing/](Table4_Preprocessing/) | Preprocessing time and reduction table CSV |
| [Table5_EndToEnd/](Table5_EndToEnd/) | Preprocessing and solve timing table CSV |
| [Table6_QAPScalability/](Table6_QAPScalability/) | Scalability table CSV and full-precision per-instance timings, reduction ratios, and solver status |
| [Figure2_QAPScalability/](Figure2_QAPScalability/) | Plot data CSV and Figure 2 PDF |

## Raw outputs and source records

```text
raw/benchmarks/<dataset>/<method>/logs/*.log
raw/benchmarks/<dataset>/<method>/*_summary.csv
raw/table_6_figure_2/<method>/*.log
```

The benchmark datasets are `Mittelmann`, `MIPLIB_Large`, and `QAP20`.
Logs shared by multiple tables are stored once.

The `raw_log` fields in the result CSVs contain paths relative to the
repository root. The `source_batch` fields identify batches in the shared
[experiment configuration](../scripts/experiment_config.json).

See [statistics and provenance](../docs/STATISTICS_AND_PROVENANCE.md) for
timing definitions, SGM10 aggregation, reduction ratios, and the solved-instance
criterion.
