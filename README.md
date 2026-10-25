# GPU-Accelerated Preprocessing for Large-Scale Linear Programming

This repository contains the GPU-Preprocessor C++/CUDA library, its HPR-LP experiment driver, and experimental outputs for the paper by Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, and Xinyuan Zhao. See [AUTHORS](AUTHORS) for contact details and [LICENSE](LICENSE) for the MIT license. Bundled HPR-LP and PSLP code retains its own [MIT](third_party/hprlp/LICENSE) and [Apache-2.0](third_party/pslp/LICENSE) licenses, respectively.

The GPU library source is imported from the [main-branch snapshot](https://github.com/PolyU-IOR/GPU-Preprocessor/commit/febe85d0bde835d8e9c2798ff02296fd207cbbb3) of [PolyU-IOR/GPU-Preprocessor](https://github.com/PolyU-IOR/GPU-Preprocessor). The solver source comes from the official [HPR-LP-C v0.1.3 release](https://github.com/PolyU-IOR/HPR-LP-C/releases/tag/v0.1.3), commit `89d3cc06806b4d31d029f7b5c8591db58219ff6d`, and is unmodified. GPU-Preprocessor, PSLP 0.0.11, and the optional Gurobi adapter are connected outside the release source tree. See [solver integration](docs/SOLVER_INTEGRATION.md) for the build and comparison backends.

## Cite

Please cite the [paper](https://arxiv.org/abs/2610.19840) and this [repository](https://github.com/ljw2024polyu/GPU-Preprocessor-IJOC). The official IJOC archive identifier and paper/repository DOIs will be added when assigned.

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
| `integrations/hprlp/` | External preprocessing adapters, experiment driver, build configuration, and integration tests |
| `third_party/` | Official HPR-LP-C v0.1.3 source and PSLP 0.0.11, with their licenses |
| `data/` | Benchmark input inventory and data sources |
| `scripts/` | Input preparation, experiment execution, and reconstruction of result tables and Figure 2 |
| `docs/` | Experimental settings, statistical definitions, and log mappings |
| `results/` | Raw logs, summary CSVs, table data, LaTeX tables, and Figure 2 |

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

Run the bundled example or preprocess an uncompressed MPS file:

~~~sh
./build/bin/gpu_preprocessor_example
./build/bin/gpu_preprocessor_mps examples/model.mps
./build/bin/gpu_preprocessor_mps model.mps --folding false
./build/bin/gpu_preprocessor_export_reduced model.mps reduced/
~~~

Integer-marked inputs are treated as LP relaxations. The exporter writes `meta.txt` and binary LP data for an external solver. Use `--help` to inspect command-line options.

The [public API](cpp/include/gpu_preprocessor/gpu_preprocessor.hpp) accepts a GPU-resident `DeviceLp`. Call `preprocess`, solve the reduced LP, then call `postprocess` with the reduced primal and dual solution. See [presolve_example.cpp](examples/presolve_example.cpp) for an integration example. Folding and dual recovery have known numerical limitations; validate the recovered solution against the original LP.

To link the library from another C++17/CUDA project:

~~~cmake
add_subdirectory(path/to/GPU-Preprocessor-IJOC gpu-preprocessor-build EXCLUDE_FROM_ALL)
target_link_libraries(YOUR_TARGET PRIVATE GPUPreprocessor::core)
~~~

## Results

| Paper item | Data | Paper output |
| --- | --- | --- |
| Table 3: folding and presolve ablation | [CSV](results/Table3_Ablation/table_3_ablation_results.csv) | [LaTeX](results/Table3_Ablation/table_3_ablation.tex) |
| Table 4: preprocessing time and reduction | [CSV](results/Table4_Preprocessing/table_4_preprocessing_results.csv) | [LaTeX](results/Table4_Preprocessing/table_4_preprocessing.tex) |
| Table 5: end-to-end solver performance | [CSV](results/Table5_EndToEnd/table_5_end_to_end_results.csv) | [LaTeX](results/Table5_EndToEnd/table_5_end_to_end.tex) |
| Table 6: QAP scalability | [CSV](results/Table6_QAPScalability/table_6_scalability_results.csv) | [LaTeX](results/Table6_QAPScalability/table_6_scalability.tex) |
| Figure 2: preprocessing time by QAP dimension | [CSV](results/Figure2_QAPScalability/figure_2_preprocessing_times.csv) | [PDF](results/Figure2_QAPScalability/figure_2_preprocessing_time.pdf) |

The raw outputs comprise 435 benchmark logs, 15 benchmark summary CSVs, and 15 scalability logs. Logs used by multiple tables are stored once. Per-instance measurements are in the [benchmark CSV](results/tables_3_4_5_instance_results.csv) and [full-precision scalability CSV](results/Table6_QAPScalability/table_6_full_precision_results.csv), which identify their source logs.

See [statistical definitions](docs/STATISTICS_AND_PROVENANCE.md) for timing, aggregation, reduction ratios, and the meaning of solver status. The [environment record](docs/experiment_environment.json) contains the hardware, software versions, and stopping settings used in the paper.

## Replicating

Follow [REPRODUCING.md](docs/REPRODUCING.md) for prerequisites, input preparation,
and output locations. From a fresh clone on a CUDA machine:

~~~sh
export CC=gcc-11 CXX=g++-11 FC=gfortran-11
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc CUDAHOSTCXX=g++-11
export CUDA_ARCH=80                     # A100; use 90 for H100
export CUDA_VISIBLE_DEVICES=3           # choose an available physical GPU

bash scripts/reproduce.sh setup
bash scripts/reproduce.sh test
RUN_DIR="$PWD/runs/smoke" bash scripts/reproduce.sh smoke

bash scripts/reproduce.sh data
bash scripts/reproduce.sh run
bash scripts/reproduce.sh results
~~~

Adjust the compiler and CUDA paths for your machine. `setup` installs the pinned
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
To reconstruct and check those saved tables and redraw their figure, without
running a solver:

~~~sh
python3 scripts/rebuild_results.py
python3 scripts/plot_figure2.py
~~~

For individual datasets, instances, methods, or sizes, see
[running Section 4](docs/SOLVER_INTEGRATION.md#running-section-4).
The [benchmark index](docs/tables_3_4_5_batch_index.csv) and
[scalability index](docs/table_6_figure_2_log_index.csv) record the commands;
`{solver_executable}`, `{input_mps}`, and `{input_dir}` are path placeholders.
Exact download URLs and generator details are in the [input guide](data/README.md).

New runs use the official release solver with external preprocessing. Their
numerical results and timings should be checked against the archived
measurements before replacing manuscript values.
