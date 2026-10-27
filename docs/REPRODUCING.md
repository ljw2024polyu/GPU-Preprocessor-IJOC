# Reproducing the experiments

Follow these steps on a Linux x86-64 machine with an available NVIDIA GPU.
After cloning, run all commands from the repository root.

## Requirements

Install the following dependencies yourself before building or running:

| Dependency | Requirement |
| --- | --- |
| CUDA | CUDA Toolkit 12 or newer, including nvcc and CUDA libraries, with a compatible NVIDIA driver |
| Build tools | Git, Bash, Make, CMake 3.24 or newer, CUDA-compatible C/C++17 compilers, and zlib development headers |
| Input generators | GNU Fortran (`gfortran`), Julia 1.10 or newer, `curl`, and `gzip` |
| Python | Python 3.10 or newer |
| Python packages | `matplotlib==3.10.7` and `gurobipy==12.0.2`, as listed in [requirements.txt](../requirements.txt) |

The paper experiments used CUDA Toolkit 13.3.33. The complete hardware and
software settings are recorded in [experiment_environment.json](experiment_environment.json).

Activate your Python environment before following the steps below. The scripts
use `python3` from that environment; they do not install dependencies or create
a virtual environment. Julia needs only its standard library. The sources for
HPR-LP-C v0.1.3 and PSLP 0.0.11 are bundled and compiled with this project.

## 1. Download the project

```sh
git clone git@github.com:ljw2024polyu/GPU-Preprocessor-IJOC.git
cd GPU-Preprocessor-IJOC
```

Access to the private repository through your GitHub account is required.

## 2. Build

```sh
bash scripts/reproduce.sh setup
```

This compiles GPU-Preprocessor, HPR-LP, PSLP, the command-line tools, and tests
into `build-experiments/`. The experiment executable is
`build-experiments/bin/hprlp_solve`. CMake discovers the installed compilers and
CUDA Toolkit and targets the visible GPU. Optional compiler settings are
listed in [solver integration](SOLVER_INTEGRATION.md#build).

If multiple CUDA Toolkits are installed, set `CUDACXX` to the `nvcc` executable
from the toolkit you want to use. After changing compilers or CUDA Toolkits,
reset the cached build configuration with:

```sh
bash scripts/reproduce.sh setup --fresh
```

This resets CMake's configuration without changing input data or experiment
results. Run `bash scripts/reproduce.sh test` after a successful build; it
reports an error if no C++/CUDA tests are registered.

## 3. Prepare the data

Allow at least 100 GB of free disk space. For the first full download and model
generation, budget approximately 2–4 hours. Actual time
depends on network speed, CPU performance, and disk I/O.

```sh
bash scripts/reproduce.sh data
```

This downloads or generates all 92 models under `runs/reproduction/inputs/`.

| Input group | Models | Preparation |
| --- | ---: | --- |
| Mittelmann (Hans) | 49 | Download MPS/EMPS files and convert them to standard MPS where needed |
| MIPLIB Large | 18 | Download original MIPs and export their LP relaxations |
| QAP20 | 20 | Download QAPLIB data and generate LPs with Julia |
| QAP scalability | 5 | Generate scaled wil100 LPs for N = 20, 40, 60, 80, 100 |

Data preparation uses `gurobipy` to read MIP inputs and export their LP
relaxations without optimization or presolve. The package is publicly
available from PyPI, and this conversion was verified with its bundled
size-limited license, without a separate full Gurobi license; see the
[official installation guide](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python).
See the [input guide](../data/README.md) for download URLs and generator details.
If preparation is interrupted, repeat the same command; completed models are
reused.

To reuse existing models, skip this step and set `INPUT_ROOT` to their root
directory. Both `.mps` and `.mps.gz` are accepted and read in place. The input
folders may use the prepared names (`Mittelmann`, `MIPLIB_Large`, `QAP20`,
`QAP_wil100_scale`) or the collection names (`Hans49_original`,
`MIP_large18_original`, `QAP20_original`, `QAP_wil100_nested/mps`).

## 4. Run the experiments

```sh
bash scripts/reproduce.sh run
```

The default run selects no preprocessing, folding only, GPU-Preprocessor, and
PSLP: 348 benchmark calls and 10 scalability calls, all solved with HPR-LP.
It uses one visible GPU (logical device 0), a tolerance of 1e-6, a solve time
limit of 1000 seconds per call, and a check interval of 150.
The shared [experiment configuration](../scripts/experiment_config.json)
defines the command arguments, output paths, and recorded environment settings.

Gurobi comparisons require its 12.0.2 C SDK and a full license. To include them,
set `WITH_GUROBI=ON` and `GUROBI_HOME` before step 2 and keep them set for step 4;
see [Gurobi setup](SOLVER_INTEGRATION.md#gurobi-optional). The five-method run has
450 calls. Gurobi comparisons are disabled by default.

A run writes to a new `runs/reproduction/experiment/` directory and can take
several hours. For another run, set `RUN_DIR` to a new directory before steps
4 and 5. For selected datasets, instances, or methods, see
[running selected experiments](SOLVER_INTEGRATION.md#running-selected-experiments).

## 5. Generate the results

```sh
bash scripts/reproduce.sh results
```

After all selected calls finish, this generates CSVs for Tables 3–6 and
Figure 2 for the selected methods, using only the new run's outputs.

| Output | Location under `runs/reproduction/experiment/` |
| --- | --- |
| Selected commands | `commands.json` |
| Solver logs and benchmark summaries | `results/raw/` |
| Per-instance measurements | `results/tables_3_4_5_instance_results.csv` |
| Table CSVs | `results/Table3_Ablation/`, `results/Table4_Preprocessing/`, `results/Table5_EndToEnd/`, `results/Table6_QAPScalability/` |
| Figure 2 data and PDF | `results/Figure2_QAPScalability/` |

Use the same `RUN_DIR` for running and generating results. New outputs are
separate from the archived paper measurements in the repository's `results/`.
For timing definitions, aggregation, and recovered-solution checks, see
[statistics and provenance](STATISTICS_AND_PROVENANCE.md).
