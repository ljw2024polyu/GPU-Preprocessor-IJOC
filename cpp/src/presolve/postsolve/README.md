# Postsolve implementation

`../gpu_postsolve.cu` owns the recovery vectors, index mapping, reverse replay,
and final dual projection. It dispatches each recorded operation to the files
below. The primal replay (including structural checkpoints) still precedes the
dual replay. Recovery follows the actual transformation order, not file order.

| File stem | Recovery responsibility |
| --- | --- |
| `postsolve_parallel_rows` | Split merged row duals |
| `postsolve_parallel_columns` | Split an aggregated column and its reduced cost |
| `postsolve_doubleton_equations` | Undo two-variable equality elimination |
| `postsolve_singleton_columns` | Undo singleton-column substitution |
| `postsolve_singleton_rows` | Restore a deleted singleton row's dual and the affected reduced cost |
| `postsolve_bound_strengthening` | Undo a row-derived bound change in dual recovery |
| `postsolve_fixed_variables` | Shared fixed-column records, including infinite-bound dual fixing |
| `postsolve_projected_variables` | Shared projection records for two-row and zero-cost product variable elimination |
| `postsolve_piecewise_linear_reformulation_absolute_value` | Expand the absolute-value reformulation |
| `postsolve_piecewise_linear_reformulation_variable_pair` | Expand the variable-pair reformulation |
| `postsolve_common` | Shared bound checks, dual projection, record helpers, and host utilities |

The `.cuh` handlers are inlined into the existing replay kernel, without an
extra kernel launch per record. The two structural `.cu` files retain their
own recovery kernels. Their `.hpp` files declare internal host entry points.

`../device_recovery.cu` and `../fixed_col_tape.cu` still create and manage
recovery records during presolve. Folding expansion remains in
`../../folding/folding_map.cu`. Shared records need not have one implementation
per presolve rule; this split does not extend the existing dual-recovery coverage.
