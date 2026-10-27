# Reproducing the experiments

This guide starts with a fresh clone and ends with new CSV tables and a PDF
figure. The default workflow uses the official HPR-LP-C v0.1.3 release, the
official PSLP v0.0.11 source, and GPU-Preprocessor. Gurobi comparisons are optional.

## 1. Install the tools and clone

Use Linux x86-64 with CUDA, an NVIDIA GPU, Python 3.10 or newer, CMake 3.24 or
newer, a CUDA-compatible C/C++17 compiler, GNU Fortran (`gfortran`), Julia 1.10
or newer, curl, gzip, and zlib
development headers. Set aside at least 100 GB of free disk space for downloads,
generated models, and temporary files. The large-input validation machine has
256 GB of host RAM and an A100 with 80 GB of GPU memory.

On Ubuntu, the non-CUDA build dependencies can be installed with:

```sh
sudo apt-get install git curl gzip build-essential gfortran \
  python3-venv zlib1g-dev
```

Install CUDA, CMake, and Julia separately if needed. A normal Julia installation
is sufficient; the QAP generator does not install Julia packages.

```sh
git clone git@github.com:ljw2024polyu/GPU-Preprocessor-IJOC.git
cd GPU-Preprocessor-IJOC

export WORK_ROOT="$PWD/runs/reproduction"
```

The private repository requires GitHub access for your own account.

### Build configuration

CMake discovers the system C/C++ compilers and CUDA Toolkit. The default GPU
architecture is `native`, which detects the visible GPU during configuration.
Data preparation uses `cc`, `c++`, `gfortran`, and `julia` from `PATH`.

Set these optional variables before `setup` or data preparation when using a
different toolchain:

| Variable | Purpose |
| --- | --- |
| `CC`, `CXX`, `FC` | C, C++, and Fortran compiler executables |
| `CUDACXX`, `CUDAHOSTCXX` | CUDA compiler and its host C++ compiler |
| `CUDAToolkit_ROOT` | CUDA Toolkit root if automatic discovery selects another installation |
| `CUDA_ARCH` | Explicit GPU architecture or semicolon-separated target list |
| `CUDAARCHS` | Standard CMake architecture setting, used when `CUDA_ARCH` is unset |
| `BUILD_JOBS` | Explicit build parallelism; otherwise CMake uses `CMAKE_BUILD_PARALLEL_LEVEL` or the build tool's default |
| `JULIA` | Julia executable for input generation |

For a build without a visible GPU or for another target machine, set
`CUDA_ARCH` or `CUDAARCHS` to the target architecture before configuration.
For direct CMake builds, use `-DCMAKE_CUDA_ARCHITECTURES` instead of `CUDA_ARCH`.
CMake caches toolchain selections; use a new `BUILD_DIR` when changing the
compiler or CUDA installation. No compiler version, CUDA installation path, or
GPU model is forced by the wrapper.

## 2. Build and check one example from every input group

```sh
bash scripts/reproduce.sh setup
bash scripts/reproduce.sh test
RUN_DIR="$WORK_ROOT/smoke" bash scripts/reproduce.sh smoke
```

Use your site's allocation procedure before the GPU stages. The solver uses
logical device 0 by default. `setup` installs the pinned Python
packages in `.venv/` and builds the integrated solver, library tools, examples,
and numerical tests in `build-experiments/`.

`smoke` downloads and prepares FOME13, the LP relaxation of scpn2, tho40, and
wil100 at N=20. It runs the four benchmark pipelines (no preprocessing, folding
only, GPU-Preprocessor, and PSLP), plus GPU-Preprocessor and PSLP on the scale
model: 14 solver calls. It then reconstructs the selected table rows and plots
the two preprocessing measurements. These are a workflow check on a subset,
not the complete paper experiment.

For a second smoke run choose a new `RUN_DIR`. Completed input files are reused;
experiment outputs are never silently overwritten.

## 3. Download and generate all paper inputs

```sh
bash scripts/reproduce.sh data
```

This command prepares all 92 models and reads them with the solver's CPU MPS
reader to check their row, column, and matrix-nonzero counts against the paper
inputs. No GPU is used in this stage. The counts are a structural check; they
are not a claim of identical solver timings.

| Input group | Models | What the script does |
| --- | ---: | --- |
| Mittelmann (Hans) | 49 | Download the exact source files, expand EMPS where needed, and gzip standard MPS files. |
| MIPLIB Large | 18 | Download original MIPs and export their LP relaxations using Gurobi 12.0.2. |
| QAP20 | 20 | Download QAPLIB DAT files and generate the full LP with Julia. |
| QAP scalability | 5 | Generate the scaled symmetry-reduced wil100 LPs at N=20,40,60,80,100. |

Gurobi's Python package is used only to read, relax, and write the MIP inputs;
this step performs no optimization or presolve. It was checked with the
size-limited license included in `gurobipy==12.0.2`. A full Gurobi license is
needed only for large-model Gurobi presolve comparisons, which are disabled by
default. See [data/README.md](../data/README.md) for exact URLs, formulations,
source references, and individual download/generation commands.

The input preparation scripts keep completed models when rerun with `--resume`
(the wrapper already supplies this option). Temporary outputs are renamed only
after successful generation. If generation is interrupted, rerun the `data`
stage to continue with the missing models.

## 4. Run the complete selected experiment set

```sh
bash scripts/reproduce.sh run
bash scripts/reproduce.sh results
```

The default plan has 348 benchmark solver calls and 10 scale solver calls.
The solve tolerance is 1e-6, the HPR-LP time limit is 1000 seconds per call, and
the check interval is 150. This is a long run; allow several hours in addition
to data preparation. The machine's GPU and CPU affect the measured times.

`run` requires a new `$WORK_ROOT/experiment` directory. Set `RUN_DIR` explicitly
for another run and use the same value for the `results` stage. If a run fails,
inspect its last log before choosing a new output directory. Reconstruction
requires every solver call selected in that run's `commands.json` to finish;
it does not fill missing measurements with archived values.

After setup, the complete sequence can also be launched with one command:

```sh
bash scripts/reproduce.sh all
```

## 5. Locate the outputs

```text
runs/reproduction/
  inputs/
    Mittelmann/             # 49 prepared .mps.gz files
    MIPLIB_Large/            # 18 LP relaxations
    QAP20/                  # 20 generated LPs
    QAP_wil100_scale/        # 5 generated LPs
    sources/                # original benchmark downloads
    .tools/                 # downloaded generators and compiled CPU utilities
  experiment/
    commands.json           # exact selected solver commands and environment
    results/
      raw/                  # solver logs and benchmark summary CSVs
      Table3_Ablation/       # new table CSV
      Table4_Preprocessing/  # selected preprocessing methods
      Table5_EndToEnd/       # selected solver pipelines
      Table6_QAPScalability/ # rounded and full-precision CSVs, including status
      Figure2_QAPScalability/# plot CSV and PDF
```

The repository's top-level `results/` contains the original paper outputs.
To check and plot those saved measurements without running a solver:

```sh
.venv/bin/python scripts/rebuild_results.py
.venv/bin/python scripts/plot_figure2.py --output-dir runs/archive-figure
```

The verification command compares recomputed table values with the archived
CSVs without modifying the saved results or logs.

New measurements use the official release solver with external preprocessing
adapters. They must be compared with the archived measurements before being
used to replace manuscript values. See [statistical definitions](STATISTICS_AND_PROVENANCE.md)
for aggregation and the distinction between a reduced-space stopping criterion
and validation after recovery in the original model.

## Optional Gurobi comparison

Install the Gurobi 12.0.2 C SDK and a valid full license, then set
`WITH_GUROBI=ON` and `GUROBI_HOME` before setup, tests, and runs. The complete
five-method plan has 450 solver calls. [SOLVER_INTEGRATION.md](SOLVER_INTEGRATION.md#gurobi-optional)
gives the license and interface details. No Gurobi executable, library, or
license file is distributed in this repository.
