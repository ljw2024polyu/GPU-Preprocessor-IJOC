# GPU-Accelerated Preprocessing for Large-Scale Linear Programming

This repository contains the GPU-Preprocessor C++/CUDA library, its HPR-LP experiment driver, and experimental outputs for the paper by Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, and Xinyuan Zhao. See [AUTHORS](AUTHORS) for contact details.

Unless otherwise noted, first-party code and accompanying materials are
distributed under the [MIT license](LICENSE). Bundled third-party components
retain their original licenses. Benchmark sources and usage terms are
described in the [input guide](data/README.md#data-use-and-references).

The experiment driver combines [HPR-LP-C v0.1.3](https://github.com/PolyU-IOR/HPR-LP-C/releases/tag/v0.1.3)
with GPU-Preprocessor, PSLP 0.0.11, or optional Gurobi presolve.
See [solver integration](docs/SOLVER_INTEGRATION.md) for setup and usage.

## Cite

Please cite the [paper](https://arxiv.org/abs/2610.19840) and this [repository](https://github.com/ljw2024polyu/GPU-Preprocessor-IJOC). The citations below identify the preprint and working repository. Once IJOC assigns the archive identifier and DOIs, update this section with both DOI links and the archive BibTeX in the [journal's format](https://informsjoc.github.io/InstructionsForAuthors.html#citing-the-repo-in-your-paper), including `publisher`, `doi`, and the official archive URL in `url` and `note`.

BibTeX for the paper:

~~~bibtex
@misc{chen2026gpuacceleratedpreprocessing,
  author = {Kaihuang Chen and Jiawei Lu and Defeng Sun and Yancheng Yuan
            and Guojun Zhang and Xiangyu Zhang and Xinyuan Zhao},
  title = {{GPU}-Accelerated Preprocessing for Large-Scale Linear Programming},
  year = {2026},
  eprint = {2610.19840},
  archivePrefix = {arXiv},
  primaryClass = {math.OC},
  url = {https://arxiv.org/abs/2610.19840}
}
~~~

BibTeX for this repository, pending the official IJOC archive and DOI:

~~~bibtex
@misc{gpuPreprocessorIJOC,
  author = {Kaihuang Chen and Jiawei Lu and Defeng Sun and Yancheng Yuan
            and Guojun Zhang and Xiangyu Zhang and Xinyuan Zhao},
  title = {{GPU}-Accelerated Preprocessing for Large-Scale Linear Programming},
  year = {2026},
  url = {https://github.com/ljw2024polyu/GPU-Preprocessor-IJOC},
  note = {Available for download at https://github.com/ljw2024polyu/GPU-Preprocessor-IJOC}
}
~~~

## Description

GPU-Preprocessor applies folding and presolve to continuous linear programs before an external solver solves the reduced problem. Postsolve and unfolding recover a solution in the original model space. Both preprocessing stages are enabled by default.

| Directory | Contents |
| --- | --- |
| `cpp/` | C++/CUDA library, MPS tools, and regression tests |
| `cmake/` | Build and installation configuration |
| `examples/` | C++ integration example and a small MPS model |
| `integrations/hprlp/` | Preprocessing adapters, experiment driver, build rules, and tests that use the solver in `third_party/hprlp/` |
| `third_party/hprlp/` | Official HPR-LP-C v0.1.3 solver source, with its bundled dependencies and licenses |
| `third_party/pslp/` | PSLP 0.0.11 source and license for the comparison backend |
| `data/` | Benchmark input inventory and data sources |
| `scripts/` | Input preparation, experiment execution, and reconstruction of result tables and Figure 2 |
| `docs/` | Experimental settings, statistical definitions, and log mappings |
| `results/` | Raw logs, summary CSVs, CSV tables, and Figure 2 |

All methods in the experiment driver use the same solver in `third_party/hprlp/`.
The official HPR-LP source is kept unchanged. Before each solve, the driver
disables HPR-LP's built-in presolve by setting `presolver = HPRLP_PRESOLVER_NONE`
and `use_presolve = false`. Preprocessing is handled by the selected external
backend, or skipped for the no-preprocessing baseline.

## Building

The library requires Linux, Make, CMake 3.24 or newer, a C++17 compiler, CUDA Toolkit 12 or newer with nvcc and cuSPARSE, and an NVIDIA GPU. From the repository root:

~~~sh
make
~~~

This builds the library, MPS tools, and example, selecting the local GPU architecture. To build and run the existing numerical tests on a CUDA machine:

~~~sh
make BUILD_DIR=build-tests CMAKE_ARGS=-DGPUPREPROCESSOR_BUILD_TESTS=ON
ctest --test-dir build-tests --output-on-failure
~~~

The [MPS reader tests](cpp/mps_reader/README.md) can also be built independently without CUDA.

The integrated HPR-LP/PSLP experiment driver is built by the reproduction
workflow below. See [REPRODUCING.md](docs/REPRODUCING.md) for dependencies and
[SOLVER_INTEGRATION.md](docs/SOLVER_INTEGRATION.md) for manual CMake options,
HDF5 support, and the optional Gurobi adapter.

## Usage

From the repository root, run the minimal API example or preprocess and export
the bundled uncompressed MPS model:

~~~sh
./build/bin/gpu_preprocessor_example
./build/bin/gpu_preprocessor_mps examples/model.mps
./build/bin/gpu_preprocessor_mps examples/model.mps --folding false
./build/bin/gpu_preprocessor_export_reduced examples/model.mps reduced/
~~~

The MPS tools perform preprocessing and export; they do not solve the LP.
Integer-marked inputs are treated as LP relaxations. The exporter writes
`meta.txt` and binary LP data for an external solver. Use `--help` with either
MPS tool to inspect its command-line options.

To preprocess the bundled model with GPU-Preprocessor and solve it with HPR-LP,
first build the experiment driver using the `setup` step in the
[reproduction guide](docs/REPRODUCING.md), then run:

~~~sh
./build-experiments/bin/hprlp_solve -i examples/model.mps --presolver gpu
~~~

The [public API](cpp/include/gpu_preprocessor/gpu_preprocessor.hpp) accepts a
GPU-resident `DeviceLp`. Call `preprocess`, solve the reduced LP with an external
solver, then call `postprocess` with the GPU-resident reduced solution arrays
`x` (primal variables), `y` (row duals), and `z` (reduced costs).
See [presolve_example.cpp](examples/presolve_example.cpp) for a minimal API
example using an analytic solution to a toy LP. Validate primal feasibility,
dual feasibility, and the duality gap on the original LP after recovery; see
[recovery checks and solver status](docs/STATISTICS_AND_PROVENANCE.md#table-6-and-figure-2)
for the interpretation of these checks.

To link the library from another C++17/CUDA project:

~~~cmake
add_subdirectory(path/to/GPU-Preprocessor-IJOC gpu-preprocessor-build EXCLUDE_FROM_ALL)
target_link_libraries(YOUR_TARGET PRIVATE GPUPreprocessor::core)
~~~

## Results

| Paper item | Files |
| --- | --- |
| Table 3: folding and presolve ablation | [CSV](results/Table3_Ablation/table_3_ablation_results.csv) |
| Table 4: preprocessing time and reduction | [CSV](results/Table4_Preprocessing/table_4_preprocessing_results.csv) |
| Table 5: end-to-end solver performance | [CSV](results/Table5_EndToEnd/table_5_end_to_end_results.csv) |
| Table 6: QAP scalability | [CSV](results/Table6_QAPScalability/table_6_scalability_results.csv) |
| Figure 2: preprocessing time by QAP dimension | [CSV](results/Figure2_QAPScalability/figure_2_preprocessing_times.csv), [PDF](results/Figure2_QAPScalability/figure_2_preprocessing_time.pdf) |

The raw outputs comprise 435 benchmark logs, 15 benchmark summary CSVs, and 15 scalability logs. Logs used by multiple tables are stored once. Per-instance measurements are in the [benchmark CSV](results/tables_3_4_5_instance_results.csv) and [full-precision scalability CSV](results/Table6_QAPScalability/table_6_full_precision_results.csv), which identify their source logs.

See [statistical definitions](docs/STATISTICS_AND_PROVENANCE.md) for timing, aggregation, reduction ratios, and the meaning of solver status. The [environment record](docs/experiment_environment.json) contains the hardware, software versions, and stopping settings used in the paper.

## Replicating

Follow [REPRODUCING.md](docs/REPRODUCING.md) for prerequisites, input preparation,
and output locations. From a fresh clone on a CUDA machine:

~~~sh
bash scripts/reproduce.sh setup
bash scripts/reproduce.sh test
RUN_DIR="$PWD/runs/smoke" bash scripts/reproduce.sh smoke

bash scripts/reproduce.sh data
bash scripts/reproduce.sh run
bash scripts/reproduce.sh results
~~~

`setup` discovers the system compilers and CUDA Toolkit through CMake and
detects the visible GPU architecture. See the
[configuration options](docs/REPRODUCING.md#build-configuration) to select a
different toolchain or target architecture, including builds without a visible
GPU. `setup` installs the pinned
Python packages from `requirements.txt` and builds the solver, tools, and tests.
`smoke` prepares one input from each group and executes 14 solver calls.
The full default experiment prepares 92 models and executes 358 calls using
no preprocessing, folding only, GPU-Preprocessor, and PSLP. Gurobi comparisons
require its SDK and a full license; see [Gurobi setup](docs/SOLVER_INTEGRATION.md#gurobi-optional).

Input files are stored under `runs/reproduction/inputs/`; new logs, CSVs, and
Figure 2 are written under `runs/reproduction/experiment/`. Set `WORK_ROOT` to
change the root, or `RUN_DIR` to select a new experiment directory. Existing
experiment outputs are not overwritten. `bash scripts/reproduce.sh all` runs
the full sequence without the smoke stage.

The repository's `results/` directory contains the archived paper measurements.
To verify the saved table CSVs against raw outputs and redraw their figure,
without running a solver:

~~~sh
.venv/bin/python scripts/rebuild_results.py
.venv/bin/python scripts/plot_figure2.py --output-dir runs/archive-figure
~~~

The verification command does not modify the archived CSVs or logs.

For individual datasets, instances, methods, or sizes, see
[running Section 4](docs/SOLVER_INTEGRATION.md#running-section-4).
The [benchmark index](docs/tables_3_4_5_batch_index.csv) and
[scalability index](docs/table_6_figure_2_log_index.csv) record the commands;
`{solver_executable}`, `{input_mps}`, and `{input_dir}` are path placeholders.
Exact download URLs and generator details are in the [input guide](data/README.md).

New runs use the official release solver with external preprocessing. Their
numerical results and timings should be checked against the archived
measurements before replacing manuscript values.
