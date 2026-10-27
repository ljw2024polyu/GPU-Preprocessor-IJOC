# Statistics and provenance

## Tables 3–5

Times are in seconds and are aggregated across instances using the shifted
geometric mean:

```text
SGM10 = exp(mean(log(t + 10))) - 10
```

Preprocessing time uses `preprocessing_time`, or `presolve_time` when the former
is unavailable. Non-`OPTIMAL` runs contribute 1000 seconds of solve time. Each
instance's total is preprocessing plus solve time, calculated before
aggregation. The solved count is the number of `OPTIMAL` statuses.

These times exclude MPS parsing, host/device transfers, postsolve, and
original-space validation. The solver's `total_time` field is a different
measurement.

Reduction ratios are reduced/original rows, columns, and nonzeros, averaged
arithmetically across instances. For these tables, dimensions come from the
first two model-dimension records in each log; a baseline with one record has
ratios of one. Later solver reductions are excluded.

## Table 6 and Figure 2

Total time is computed from unrounded preprocessing and solve times, then
rounded to three decimals. Figure 2 uses the preprocessing values displayed
in Table 6; the full-precision CSV retains the underlying measurements.

## Solver status and recovery

`OPTIMAL` describes the stopping criterion on the LP passed to the solver.
For preprocessed runs, validation after recovery in the original model is
separate. New-run CSVs retain `postsolve_original_kkt_warning`, and the
reconstruction report counts these warnings separately from solver status.

The Gurobi adapter does not recover original-space solutions. This is recorded
in the logs and in `gurobi_original_solution_not_recovered` in the scalability
full-precision CSV.

## Execution records

Result CSVs identify their source logs or batches. The shared
[experiment configuration](../scripts/experiment_config.json) defines commands,
output paths, and recorded environment settings. Hardware and software
versions are in the [environment record](experiment_environment.json).
