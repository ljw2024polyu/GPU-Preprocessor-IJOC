# Solver interface

`hprlp_solve` connects external preprocessing to HPR-LP. Every backend uses
HPR-LP to solve the resulting LP, including the Gurobi comparison.

| Component | Version | Location |
| --- | --- | --- |
| GPU-Preprocessor | This project | `cpp/` |
| HPR-LP | Official HPR-LP-C v0.1.3 | `third_party/hprlp/` |
| PSLP | 0.0.11 | `third_party/pslp/` |
| Gurobi | 12.0.2 | Install separately; optional |

This project uses [HPR-LP-C](https://github.com/PolyU-IOR/HPR-LP-C) v0.1.3
as its LP solver. The source is bundled unchanged from the official
[v0.1.3 release](https://github.com/PolyU-IOR/HPR-LP-C/releases/tag/v0.1.3),
[commit `89d3cc0`](https://github.com/PolyU-IOR/HPR-LP-C/commit/89d3cc06806b4d31d029f7b5c8591db58219ff6d),
and retains its [MIT license](../third_party/hprlp/LICENSE). The connection
code is in `solver_interface/`. The driver disables HPR-LP's built-in
presolve and applies the selected external preprocessor.

## Build

Install the [required dependencies](REPRODUCING.md#requirements) yourself, then
run from the repository root:

```sh
bash scripts/reproduce.sh setup
```

The executable is `build-experiments/bin/hprlp_solve`. HPR-LP and PSLP are built
from the bundled sources. See [Gurobi setup](#gurobi-optional) for its
requirements and automatic selection. Direct CMake builds retain the opt-in
`HPRLP_GUROBI=ON` option.

CMake detects the visible GPU architecture. To select different compilers,
set `CC`, `CXX`, `CUDACXX`, or `CUDAHOSTCXX` before building. Use
`CUDAToolkit_ROOT` to select the CUDA Toolkit and `BUILD_JOBS` to set build
parallelism. Run the included tests with `bash scripts/reproduce.sh test`.
After changing compilers or CUDA Toolkits, use
`bash scripts/reproduce.sh setup --fresh` to reset CMake's cached configuration.

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
export GUROBI_HOME=/path/to/gurobi1202/linux64
export GRB_LICENSE_FILE=/path/to/gurobi.lic
export WITH_GUROBI=ON
bash scripts/reproduce.sh setup
./build-experiments/bin/hprlp_solve -i examples/model.mps --presolver gurobi
```

`GRB_LICENSE_FILE` is optional when your license is in Gurobi's standard search
locations; see the [license-file instructions](https://support.gurobi.com/hc/en-us/articles/360013417211-Where-do-I-place-the-Gurobi-license-file-gurobi-lic).
Provide your own license outside this repository. A
[size-limited license](https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error)
is insufficient for these benchmarks.

Both `reproduce.sh` and `run_experiments.py` use the following Gurobi selection
policy:

| `WITH_GUROBI` | Behavior |
| --- | --- |
| `AUTO` (default) | Enable Gurobi when available; otherwise report the reason and continue without it. |
| `ON` | Require Gurobi and stop with an error if its requirements are unmet. |
| `OFF` | Skip Gurobi comparisons. |

The same policy applies when `gurobi` is explicitly selected with `--methods`.

Gurobi uses automatic presolve (`Presolve=-1`) without a presolve time limit.
The solve time limit applies to the subsequent HPR-LP solve.

Gurobi does not expose its postsolve (uncrush) mapping through the public API;
see [Gurobi's explanation](https://support.gurobi.com/hc/en-us/articles/360024738352-How-does-presolve-work).
Consequently, this adapter reports the reduced-LP solution computed by HPR-LP
without recovering a solution to the original model.

## Running selected experiments

After building and preparing the inputs as described in
[REPRODUCING.md](REPRODUCING.md), run a selected instance with:

```sh
python3 scripts/run_experiments.py --input-root runs/reproduction/inputs \
  --suite benchmarks --datasets hans --instances FOME13 --methods gpu \
  --output-dir runs/hans-fome13
```

This example runs GPU-Preprocessor on `FOME13` from the Mittelmann (Hans)
dataset, then solves the reduced LP with HPR-LP. The `--methods` option can be
set to `none`, `gpu`, `pslp`, or `gurobi`. Here, `none` disables preprocessing,
and `gpu` enables both folding and presolve. All methods use HPR-LP for solving;
`gurobi` requires the [Gurobi setup](#gurobi-optional).

Add `--dry-run` to preview the commands without running benchmark models;
Gurobi availability is still checked when it is selected.
Use a new output directory for each new experiment. To resume an interrupted
experiment, repeat the original command with `--resume` and the same output
directory. After the run finishes, generate result CSVs for the selected
instances and methods:

```sh
RUN_DIR="$PWD/runs/hans-fome13" bash scripts/reproduce.sh results
```

The shared [experiment configuration](../scripts/experiment_config.json)
contains command arguments, output paths, and recorded environment settings.
For timing definitions, aggregation, and solver status, see
[statistics and provenance](STATISTICS_AND_PROVENANCE.md).
