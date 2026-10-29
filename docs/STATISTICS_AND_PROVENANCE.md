# Statistics and provenance

The definitions below follow Section 4.1 of the paper. Tables 3-5 report
benchmark-group statistics; Table 6 and Figure 2 report individual QAP instances
as described in Section 4.5.

## Timing definitions

All times are reported in seconds.

- **Preprocessing time** is wall-clock time excluding MPS parsing and CPU-GPU
  transfers of LP data. For GPU-Preprocessor, it includes all enabled folding
  and presolve stages on the GPU. For PSLP and Gurobi, it is measured on the CPU.
- **Solve time** is the time reported by HPR-LP-C 0.1.3 for solving the
  preprocessed LP, or the original LP when preprocessing is disabled.
- **Total time** is computed for each instance as

  ```text
  total time = preprocessing time + solve time
  ```

The HPR-LP baseline disables both folding and presolve. Its preprocessing time
is zero, so its total time equals its solve time.

Postsolve time is excluded for every method to make the comparison consistent,
because Gurobi does not provide a separate measurement of its postsolve time.

## Solved instances and the time limit

A run is counted as solved when HPR-LP-C satisfies its relative KKT stopping
criterion with tolerance `1e-6` within 1000 seconds, excluding preprocessing
time. Following Cederberg and Boyd (2026b), this criterion is evaluated on the
preprocessed LP, or on the original LP when preprocessing is disabled. The
archived `OPTIMAL` status records this solver stopping decision.

For unsolved runs, solve time is set to 1000 seconds when computing aggregate
time metrics. Their measured preprocessing time is retained, and their
statistical total time is preprocessing time plus 1000 seconds.

## Tables 3-5: benchmark-group statistics

The paper uses 49 Mittelmann instances, 18 MIPLIB Large LP relaxations, and
20 QAP20 LP relaxations. Statistics are computed separately for each group and
each preprocessing configuration, using all instances in that group.

For times `t_i` over `K` instances, the shifted geometric mean is

```text
SGM10(t) = exp((1/K) * sum_i log(t_i + 10)) - 10
```

The shift is 10 seconds. Preprocessing, solve, and total times are aggregated
separately, with the unsolved-run convention above. Each instance's total is
formed before applying SGM10.

Reduction ratios are

```text
r_m   = reduced rows    / original rows
r_n   = reduced columns / original columns
r_nnz = reduced nonzeros / original nonzeros
```

The reduced dimensions describe the LP produced by the selected preprocessing
configuration and passed to HPR-LP-C. Tables 3 and 4 report arithmetic means of
the per-instance ratios over the benchmark group. Smaller values indicate
stronger reductions. The no-preprocessing baseline has ratios of one. Times
and mean ratios are rounded to three decimals for display after aggregation.

## Table 6 and Figure 2: individual QAP instances

These results use LP relaxations derived from the leading `N`-by-`N` principal
submatrices of the QAPLIB instance `wil100`, for `N = 20, 40, 60, 80, 100`.
Table 6 reports per-instance times and reduction ratios, without averaging
across these five sizes. The timing definitions above also apply here.

Totals are computed from the unrounded preprocessing and solve times before
rounding the displayed values to three decimals. Figure 2 plots the
corresponding preprocessing times on a logarithmic vertical axis. The
full-precision CSV retains the underlying per-instance measurements.

## Mapping to the recorded outputs

The result-reconstruction script uses `preprocessing_time` for the complete
preprocessing measurement. Historical Gurobi batch summaries store this
measurement under `presolve_time`; the no-preprocessing summaries record zero
there. When both fields are present, `preprocessing_time` supplies the paper's
preprocessing time, including enabled folding and presolve stages.

The `solve_time` field supplies the HPR-LP-C solve time. Table totals are
reconstructed from these phase measurements using the definitions above. The
driver's broader `Total Time` diagnostic is a separate execution record.

The [benchmark instance CSV](../results/tables_3_4_5_instance_results.csv)
records the reported and statistical solve times, statistical totals,
dimensions, ratios, source logs, and batches. The
[scalability CSV](../results/Table6_QAPScalability/table_6_full_precision_results.csv)
records the per-instance measurements and source logs for Table 6. The
[reconstruction script](../scripts/rebuild_results.py) verifies the archived
table CSVs or reconstructs the outputs of a selected new run.

The [experiment configuration](../scripts/experiment_config.json) records
commands, output paths, and environment settings. Hardware and software
versions are in the [environment record](experiment_environment.json).
