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
| `postsolve_fixed_variables` | Shared fixed-variable records, including infinite-bound dual fixing |
| `postsolve_projected_variables` | Shared projection records for two-row and zero-cost product variable elimination |
| `postsolve_piecewise_linear_reformulation_absolute_value` | Expand the absolute-value reformulation |
| `postsolve_piecewise_linear_reformulation_variable_pair` | Expand the variable-pair reformulation |
| `postsolve_common` | Shared bound checks, dual projection, record helpers, and host utilities |

The `.cuh` handlers are inlined into the replay kernels, without an
extra kernel launch per record. The two structural `.cu` files retain their
own recovery kernels. Their `.hpp` files declare internal host entry points.

`../device_recovery.cu` and `../fixed_variable_tape.cu` still create and manage
recovery records during presolve. Folding expansion remains in
`../../folding/folding_map.cu`. Shared recovery records can serve multiple
presolve rules.

## Ordered parallel replay

`parallel_replay.cuh` checks record dependencies to build temporary replay plans.
The primal pass, structural checkpoints, dual pass and final dual projection keep
their original order. All execution paths call the same scalar record handlers.
Forward presolve, stored records and public interfaces are unchanged.

- Homogeneous fixed-variable ranges require distinct increasing target columns.
- Homogeneous singleton-column ranges require disjoint targets/rows and no
  support dependency on eliminated columns.
- Other supported ranges use GPU read/write components. Each component keeps
  the original reverse record order. Original-matrix supports read by
  doubleton, singleton-row and parallel-row handlers are included in the
  dependency graph. Floating-point arithmetic remains in the scalar handlers.

Ranges below 1,024 records, unsupported layouts, scratch-memory limits and
allocation exhaustion retain serial replay. Row-implied bound recovery retains
its existing serial handler. Plans are temporary and scratch is reused within a postsolve call.
`GPUPREPROCESSOR_PARALLEL_FIXED_COLS=0`, `GPUPREPROCESSOR_PARALLEL_SUBCOL=0` and
`GPUPREPROCESSOR_PARALLEL_COMPONENTS=0` independently disable the corresponding
paths for paired correctness checks.

`test_parallel_postsolve` checks byte equality, actual path selection,
input-vector immutability, dependent records, original-matrix accesses,
host/device tapes, structural checkpoints and scratch-memory fallbacks.
