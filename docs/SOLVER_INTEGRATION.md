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
from the bundled sources. The reproduction script automatically enables Gurobi
when its separately installed SDK and license pass preflight; otherwise it
builds without the Gurobi adapter. Direct CMake builds retain the opt-in
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
WITH_GUROBI=ON bash scripts/reproduce.sh setup
./build-experiments/bin/hprlp_solve -i examples/model.mps --presolver gurobi
```

`GRB_LICENSE_FILE` is optional when your license is in Gurobi's standard search
locations; see the [license-file instructions](https://support.gurobi.com/hc/en-us/articles/360013417211-Where-do-I-place-the-Gurobi-license-file-gurobi-lic).
Provide your own license outside this repository. Both reproduction entry points
default to `WITH_GUROBI=AUTO`: they check the SDK header/runtime version and start an isolated,
quiet license probe with a trivial 2,001-variable LP. This rejects the
[size-limited license](https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error)
that is insufficient for these benchmarks. The check has a 30-second timeout and
does not print license contents or runtime licensing details.

Without a usable SDK/license, automatic mode prints a skip reason and continues
with the other four methods. `WITH_GUROBI=OFF` disables Gurobi without probing;
`WITH_GUROBI=ON` requires it and fails early if preflight fails. The `run`, `resume`,
`smoke` and `test` stages also check that the build was configured with the adapter.
The Python entry point applies these same checks when `gurobi` is explicitly
listed in `--methods`; selections without `gurobi` do not probe its license.
If the license is unavailable at test time, automatic mode skips Gurobi-specific
CTest checks while retaining the other regression tests. Data preparation and
result reconstruction do not activate the benchmark license probe.

Gurobi uses automatic presolve (`Presolve=-1`) without a presolve time limit.
The solve time limit applies to the subsequent HPR-LP solve. This adapter
reports reduced-LP results and does not recover original-space solutions.

## Running selected experiments

Follow [REPRODUCING.md](REPRODUCING.md) to prepare the data and run the complete
experiment set. For a smaller selection, choose the datasets, instances, and
methods explicitly. The Python entry point defaults to all experiment groups
and automatically checks Gurobi, just like `reproduce.sh run`.
This example runs only GPU-Preprocessor and PSLP on tho40:

```sh
python3 scripts/run_experiments.py --input-root runs/reproduction/inputs \
  --suite benchmarks --datasets qap20 --instances tho40 --methods gpu pslp \
  --output-dir runs/qap20-check
```

Add `--dry-run` to preview the commands. Gurobi preflight still runs for previews
that select it, and skip reasons are written to stderr so stdout remains JSON.
`--datasets` and `--instances` filter only benchmarks; use `--suite benchmarks`
to exclude the QAP scalability group. For scalability, omit `--datasets`
and `--instances` and select the scale with `--suite scalability --sizes 20`.
Each run needs a new output directory. Generate its table rows with:

```sh
RUN_DIR="$PWD/runs/qap20-check" bash scripts/reproduce.sh results
```

The shared [experiment configuration](../scripts/experiment_config.json)
contains command arguments, output paths, and recorded environment settings.
For timing definitions, aggregation, and solver status, see
[statistics and provenance](STATISTICS_AND_PROVENANCE.md).
