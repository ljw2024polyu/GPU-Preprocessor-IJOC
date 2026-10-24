# Solver integration and experiment execution

The `hprlp_solve` executable connects GPU-Preprocessor, PSLP, and optionally Gurobi presolve to HPR-LP. The reduced LP is solved by HPR-LP for every backend, including the Gurobi comparison.

| Component | Source / version | Included |
| --- | --- | --- |
| GPU-Preprocessor | Public main-branch snapshot linked in the root README | `cpp/` |
| HPR-LP | Official HPR-LP-C v0.1.3 release | `third_party/hprlp/` |
| PSLP | 0.0.11, Apache-2.0 | `third_party/pslp/` |
| Gurobi | 12.0.2 C API | Interface only; install the SDK/runtime and license separately |

HPR-LP is pinned to the official [v0.1.3 release](https://github.com/PolyU-IOR/HPR-LP-C/releases/tag/v0.1.3), commit `89d3cc06806b4d31d029f7b5c8591db58219ff6d`. The copied source files, headers, release tests, licenses, and native PSLP dependency are byte-identical to that release. Build rules and all IJOC adapters live under `integrations/hprlp/`; the release source is not patched.

The official release's built-in PSLP 0.0.8 dependency is retained for the native library. The external driver disables built-in presolve and selects GPU-Preprocessor, PSLP 0.0.11, Gurobi, or no preprocessing. The comparison PSLP symbols are prefixed at compile time so the two versions cannot be confused. Both public CLI names `pslp` and `pslp0011` select 0.0.11.

The external adapters hand a host copy of the reduced LP to the official solver API and perform postsolve afterward. They do not use the previous experiment implementation's modified solver or its device-only handoff. Timing fields for external preprocessing are recorded by the driver. Archived results remain unchanged; agreement of new official-release runs with those measurements requires a new benchmark comparison.

## Build

Use Linux, CMake 3.24 or newer, a C/C++17 compiler, a compatible NVIDIA driver and CUDA Toolkit, and zlib development headers (for example, `zlib1g-dev` on Ubuntu). The paper environment uses CUDA 13.3.33, GCC/G++ 11.4.0, and an H100 GPU; its full settings are in [experiment_environment.json](experiment_environment.json).

From the repository root:

```sh
cmake -S . -B build-experiments \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DGPUPREPROCESSOR_BUILD_HPRLP=ON \
  -DHPRLP_GUROBI=OFF
cmake --build build-experiments -j 4
```

Set `CMAKE_CUDA_COMPILER` and `CUDAToolkit_ROOT` if the intended CUDA installation is not selected automatically. Architecture `90` is for H100; use the architecture of the GPU running the experiment. The HPR-LP interface uses 32-bit sparse offsets, so leave `GPUPREPROCESSOR_USE_INT64_NNZ` disabled.

HPR-LP and PSLP are built from the bundled sources; no separate solver installation is required. HDF5 support is optional (`-DHPRLP_HDF5=ON` plus HDF5 development headers/libraries); the paper experiments use `.mps.gz` files and do not need it.

Run the integration tests on a GPU machine:

```sh
ctest --test-dir build-experiments --output-on-failure
```

Tests cover preprocessing, objective constants, original-space primal/dual recovery, and the folding/presolve switches. `ctest --test-dir build-experiments -L cpu --output-on-failure` runs only the CPU PSLP handoff and the release's restart-control test. This subset does not validate GPU solving. With testing enabled, `build-experiments/integrations/hprlp/hprlp_release_reference` is built from the unmodified release CLI for direct baseline comparison (use `--presolver none`). On managed servers, submit GPU tests and experiments through the site's scheduler.

## Backend selection

```sh
# HPR-LP without external preprocessing
build-experiments/bin/hprlp_solve -i examples/model.mps --presolver none

# Folding only, followed by HPR-LP
build-experiments/bin/hprlp_solve -i examples/model.mps \
  --presolver gpu --gpu-folding true --gpu-presolve false

# Folding and GPU presolve, followed by HPR-LP
build-experiments/bin/hprlp_solve -i examples/model.mps \
  --presolver gpu --gpu-folding true --gpu-presolve true

# PSLP 0.0.11 CPU presolve, followed by HPR-LP
build-experiments/bin/hprlp_solve -i examples/model.mps --presolver pslp
```

`pslp0011` is an alias for `pslp`, retained for the archived command templates. Both use PSLP 0.0.11. All four pipelines above require a GPU for the HPR-LP solve, but no Gurobi installation or license.

The external driver always prints final preprocessing, presolve, and solve fields. Add `--print-debug-info true` for detailed diagnostics and original/reduced model dimensions; the Section 4 runner already enables it. `--inspect-input` reads an MPS model and reports its dimensions on CPU without calling CUDA or solving.

## Gurobi (optional)

Install the Gurobi 12.0.2 distribution with its C headers and shared library. A working Gurobi license is required at runtime, including when only presolve is called. A Python-only `gurobipy` installation is insufficient for compiling this C API adapter.

```sh
export GUROBI_HOME=/path/to/gurobi1202/linux64
# Use this only if the license is outside Gurobi's standard search locations:
export GRB_LICENSE_FILE=/path/to/gurobi.lic

cmake -S . -B build-experiments \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DGPUPREPROCESSOR_BUILD_HPRLP=ON \
  -DHPRLP_GUROBI=ON -DGUROBI_ROOT="$GUROBI_HOME"
cmake --build build-experiments -j 4

build-experiments/bin/hprlp_solve -i examples/model.mps --presolver gurobi
```

Use the license procedure appropriate to your institution; see Gurobi's [license-file instructions](https://support.gurobi.com/hc/en-us/articles/360013417211-Where-do-I-place-the-Gurobi-license-file-gurobi-lic). Do not commit a license file, credentials, or proprietary binaries. Without a license, leave `HPRLP_GUROBI=OFF` and omit `gurobi` from the experiment methods. The separate MIPLIB input preparation script also uses Gurobi; its requirements are documented in [data/README.md](../data/README.md).

The adapter calls `GRBpresolvemodel` with automatic presolve (`Presolve=-1`) and no Gurobi time limit. The 1000-second limit in the experiment commands applies to HPR-LP solving. Unspecified Gurobi parameters retain their defaults. Gurobi does not expose the presolve recovery map through this interface: its results are reported in the reduced model space, as recorded in the archived logs. An unavailable backend or a presolve failure is reported as an error; the runner does not silently substitute another method.

## Running Section 4

Prepare inputs using [data/README.md](../data/README.md). The runner expects:

```text
INPUT_ROOT/
  Mittelmann/        # 49 Hans benchmark models
  MIPLIB_Large/      # 18 LP relaxations
  QAP20/             # 20 full QAP LPs
  QAP_wil100_scale/  # five scaled models, N = 20, 40, 60, 80, 100
```

Inspect all commands without running a solver or creating output files:

```sh
python3 scripts/run_experiments.py --input-root /path/to/benchmark-inputs \
  --suite all --methods none folding gpu pslp gurobi --dry-run
```

Select a single benchmark and two methods for an initial run:

```sh
python3 scripts/run_experiments.py --input-root /path/to/benchmark-inputs \
  --datasets qap20 --instances tho40 --methods gpu pslp \
  --output-dir runs/qap20-check
```

For the complete experiment set, including Gurobi:

```sh
python3 scripts/run_experiments.py --input-root /path/to/benchmark-inputs \
  --suite all --methods none folding gpu pslp gurobi \
  --output-dir runs/section4
```

Omitting `--methods` selects the four methods that do not require Gurobi. `--suite benchmarks` selects Tables 3-5; `--suite scalability --sizes 20` selects one scalability size. Use `--solver /path/to/hprlp_solve` to override the default executable and `--device 0` for the first GPU visible to the process.

The runner reads the [benchmark](tables_3_4_5_batch_index.csv) and [scalability](table_6_figure_2_log_index.csv) command records. The recorded tolerance is `1e-6`, the HPR-LP solve limit is 1000 seconds, and the check interval is 150. Recorded scalability environment settings are applied per run; inherited HPR-LP/GPU-Preprocessor tuning variables are cleared to avoid accidental overrides. Scheduler GPU visibility and license settings are preserved.

Each output directory must be new. It receives `commands.json`, raw logs, and benchmark summary CSVs. Output names follow the paper's table/log mapping. The runner stops on missing inputs, process failures, or unsupported/failed preprocessing, keeping completed logs for inspection. New runs never overwrite the archived `results/` directory.

After the selected run finishes, reconstruct its CSV tables and Figure 2:

```sh
python3 scripts/rebuild_results.py --run-dir runs/section4
python3 scripts/plot_figure2.py \
  --data runs/section4/results/Figure2_QAPScalability/figure_2_preprocessing_times.csv
```

This writes new CSVs and the figure under the run directory; the manuscript's LaTeX tables are unchanged. Reconstruction uses `commands.json` to require exactly the selected instances, methods, and sizes. It supports the default run without Gurobi, a smaller finished check, or the full five-method run; an unfinished selected run is an error. A benchmark-only run has no scale figure. Without `--run-dir`, reconstruction reads all archived outputs and checks them against the paper tables. The [reproduction guide](REPRODUCING.md) provides a single entry script for the complete workflow.
