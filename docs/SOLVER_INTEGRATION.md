# Solver interface

`hprlp_solve` connects external preprocessing to HPR-LP. Every backend uses
HPR-LP to solve the resulting LP, including the Gurobi comparison.

| Component | Version | Location |
| --- | --- | --- |
| GPU-Preprocessor | This project | `cpp/` |
| HPR-LP | Official HPR-LP-C v0.1.3 | `third_party/hprlp/` |
| PSLP | 0.0.11 | `third_party/pslp/` |
| Gurobi | 12.0.2 | Install separately; optional |

HPR-LP source is bundled from the official [v0.1.3 release](https://github.com/PolyU-IOR/HPR-LP-C/releases/tag/v0.1.3)
and retains its [MIT license](../third_party/hprlp/LICENSE). The source is
unchanged; the connection code is in `solver_interface/`. The driver disables
HPR-LP's built-in presolve and applies the selected external preprocessor.

## Build

Install the [required dependencies](REPRODUCING.md#requirements) yourself, then
run from the repository root:

```sh
bash scripts/reproduce.sh setup
```

The executable is `build-experiments/bin/hprlp_solve`. HPR-LP and PSLP are built
from the bundled sources; Gurobi support is disabled by default.

CMake detects the visible GPU architecture. To select different compilers,
set `CC`, `CXX`, `CUDACXX`, or `CUDAHOSTCXX` before building. Use
`CUDAToolkit_ROOT` to select the CUDA Toolkit and `BUILD_JOBS` to set build
parallelism. Run the included tests with `bash scripts/reproduce.sh test`.

## Backend selection

The following commands use the bundled model. For the GPU backend, folding
and presolve are enabled by default.

```sh
# No external preprocessing
./build-experiments/bin/hprlp_solve -i examples/model.mps --presolver none

# Folding only
./build-experiments/bin/hprlp_solve -i examples/model.mps --presolver gpu --gpu-presolve false

# Folding and GPU presolve
./build-experiments/bin/hprlp_solve -i examples/model.mps --presolver gpu

# PSLP presolve
./build-experiments/bin/hprlp_solve -i examples/model.mps --presolver pslp
```

Both `pslp` and the alias `pslp0011` select PSLP 0.0.11.

## Gurobi (optional)

Install the Gurobi 12.0.2 C SDK, including its headers and shared library, and
configure a full license. The `gurobipy` package used for data preparation
does not provide the C SDK required by this adapter.

Set these variables before building and keep them set when running the full
reproduction workflow:

```sh
export WITH_GUROBI=ON
export GUROBI_HOME=/path/to/gurobi1202/linux64
bash scripts/reproduce.sh setup
./build-experiments/bin/hprlp_solve -i examples/model.mps --presolver gurobi
```

If the license is outside Gurobi's standard search locations, set
`GRB_LICENSE_FILE` to its path; see the [license-file instructions](https://support.gurobi.com/hc/en-us/articles/360013417211-Where-do-I-place-the-Gurobi-license-file-gurobi-lic).

Gurobi uses automatic presolve (`Presolve=-1`) without a presolve time limit.
The solve time limit applies to the subsequent HPR-LP solve. This adapter
reports reduced-LP results and does not recover original-space solutions.

## Running selected experiments

Follow [REPRODUCING.md](REPRODUCING.md) to prepare the data and run the complete
experiment set. For a smaller selection, choose the datasets, instances, and
methods explicitly. This example runs GPU-Preprocessor and PSLP on tho40:

```sh
python3 scripts/run_experiments.py --input-root runs/reproduction/inputs \
  --datasets qap20 --instances tho40 --methods gpu pslp \
  --output-dir runs/qap20-check
```

Add `--dry-run` to preview the commands. For scalability, omit `--datasets`
and `--instances` and select the scale with `--suite scalability --sizes 20`.
Each run needs a new output directory. Generate its table rows with:

```sh
RUN_DIR="$PWD/runs/qap20-check" bash scripts/reproduce.sh results
```

The shared [experiment configuration](../scripts/experiment_config.json)
contains command arguments, output paths, and recorded environment settings.
For timing definitions, solver status, and original-space recovery checks,
see [statistics and provenance](STATISTICS_AND_PROVENANCE.md).

New runs use the bundled official solver release. Archived paper measurements
used a different solver implementation; compare new measurements with them
before replacing paper values.
