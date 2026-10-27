# Statistics and provenance

## Tables 3–5

The benchmark groups contain 49 Mittelmann, 18 MIPLIB Large, and 20 QAP20 instances. Each has five variants: no preprocessing, folding only, GPU-Preprocessor, PSLP, and Gurobi.

Time aggregates use `SGM10 = exp(mean(log(t + 10))) - 10`. Preprocessing time comes from `preprocessing_time`, falling back to `presolve_time`. Non-`OPTIMAL` runs contribute 1000 seconds of solve time. Per-instance total time is preprocessing plus this solve time, computed before aggregation.

These metrics exclude MPS parsing, host/device transfers, postsolve, and original-space validation. The solver's `total_time` field uses a different definition and is not used for the paper's total time.

Reduction ratios are arithmetic means of reduced/original rows, columns, and nonzeros. Dimensions come from the first two `problem information` records in each log; a baseline with one record uses it for both. The summary fields `reduced_average_*` concern later solver activity and are not used for these ratios.

## Table 6 and Figure 2

Total time is added at full precision, then rounded to three decimals. Figure 2 uses the preprocessing values displayed in Table 6.

`OPTIMAL` denotes the stopping criterion on the preprocessed LP. Original-space KKT warnings and Gurobi's original-solution recovery limitation are retained in the logs and full-precision CSV.

New experiment runs also retain `postsolve_original_kkt_warning` in the
benchmark summary and per-instance CSVs. The reconstruction report counts
these warnings separately. A completed run and a reduced-LP `OPTIMAL` status
do not certify that the recovered original-space primal and dual solution
both meet the requested tolerance.

## Execution records

Command templates preserve recorded algorithm settings, with executable and input paths replaced by the placeholders defined in the [reproduction guide](REPRODUCING.md#4-run-the-complete-selected-experiment-set). Device indices in the templates use the default logical device 0; the runner applies the user's `--device` selection at runtime. The scale index also records captured environment variables. An empty environment object means no separate environment record was captured.

The [environment record](experiment_environment.json) describes hardware and software versions.
