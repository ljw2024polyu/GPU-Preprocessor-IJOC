# GPU-Accelerated Preprocessing for Large-Scale Linear Programming

This repository contains the GPU-Preprocessor C++/CUDA library, its HPR-LP experiment driver, and experimental outputs for the paper by Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, and Xinyuan Zhao. See [AUTHORS](AUTHORS) for contact details.

Unless otherwise noted, first-party code and accompanying materials are
distributed under the [MIT license](LICENSE). Bundled third-party components
retain their original licenses. Benchmark sources and usage terms are
described in the [input guide](data/README.md#data-use-and-references).

The experiment driver combines HPR-LP-C v0.1.3 with GPU-Preprocessor,
PSLP 0.0.11, or optional Gurobi presolve. The HPR-LP source needed for this
integration is bundled in [third_party/hprlp/](third_party/hprlp/) and built
locally; no separate HPR-LP download or installation is required. It comes
from the official [v0.1.3 release](https://github.com/PolyU-IOR/HPR-LP-C/releases/tag/v0.1.3)
and retains its original [MIT license](third_party/hprlp/LICENSE).
See [solver integration](docs/SOLVER_INTEGRATION.md) for setup and usage.

## Cite

Please cite the [paper](https://arxiv.org/abs/2610.19840) and this [repository](https://github.com/ljw2024polyu/GPU-Preprocessor-IJOC).

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

BibTeX for this repository:

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

## Building

The library requires Linux, Make, CMake 3.24 or newer, a C++17 compiler, CUDA Toolkit 12 or newer with nvcc and cuSPARSE, and an NVIDIA GPU. From the repository root:

~~~sh
make
~~~

## Usage

From the repository root, preprocess or export the bundled uncompressed MPS
model:

~~~sh
./build/bin/gpu_preprocessor_mps examples/model.mps
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

## Description

GPU-Preprocessor applies folding and presolve to continuous linear programs before an external solver solves the reduced problem. Postsolve and unfolding recover a solution in the original model space. Both preprocessing stages are enabled by default.

| Directory | Contents |
| --- | --- |
| `cpp/` | C++/CUDA library, MPS tools, and regression tests |
| `cmake/` | Build and installation configuration |
| `examples/` | C++ integration example and a small MPS model |
| `solver_interface/` | Preprocessing adapters, experiment driver, build rules, and tests that use the solver in `third_party/hprlp/` |
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

See [REPRODUCING.md](docs/REPRODUCING.md) for dependencies, data preparation, experiment execution, and generation of the result tables and Figure 2.
