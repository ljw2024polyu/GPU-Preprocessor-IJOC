# GPU-Accelerated Preprocessing for Large-Scale Linear Programming

This repository contains the GPU-Preprocessor C++/CUDA library and experimental outputs for the paper by Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, and Xinyuan Zhao. See [AUTHORS](AUTHORS) for contact details and [LICENSE](LICENSE) for the MIT license.

The library is imported from the [main-branch snapshot](https://github.com/PolyU-IOR/GPU-Preprocessor/commit/febe85d0bde835d8e9c2798ff02296fd207cbbb3) of [PolyU-IOR/GPU-Preprocessor](https://github.com/PolyU-IOR/GPU-Preprocessor). Its source and build files are unchanged. This snapshot provides the standalone preprocessor; it has not been established as the exact integrated executable used to produce the archived HPR-LP experiment results.

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
| `data/` | Benchmark input inventory and data sources |
| `scripts/` | Input download/generation and reconstruction of result tables and Figure 2 |
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

The result-processing scripts require Python 3.10 or newer. Figure generation requires the version listed in `requirements.txt`:

~~~sh
python3 -m pip install -r requirements.txt
~~~

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

Reconstruct the result CSVs from the saved outputs, then redraw Figure 2:

~~~sh
python3 scripts/rebuild_results.py
python3 scripts/plot_figure2.py
~~~

The first command reads the raw logs and summary CSVs and checks the reconstructed values against the LaTeX tables. The second writes the Figure 2 PDF; use `--output-dir PATH` to write the figure elsewhere.

The [benchmark index](docs/tables_3_4_5_batch_index.csv) and [scalability index](docs/table_6_figure_2_log_index.csv) record the experiment commands. Their placeholders `{solver_executable}`, `{input_mps}`, and `{input_dir}` denote the integrated solver executable and input paths. These commands target HPR-LP-C 0.1.3 with the corresponding preprocessing interfaces; they are not command-line options for the standalone MPS tool above.

The [input preparation guide](data/README.md) provides the Hans public entry points, exact MIPLIB/QAP download links, and commands to generate the MIPLIB Large, QAP20, and wil100 scalability inputs. Full solver reruns additionally require the integrated HPR-LP-C experiment implementation and the PSLP 0.0.11 and Gurobi 12.0.2 interfaces; those solver components remain to be incorporated. Gurobi must be installed separately under an appropriate license. Benchmark input models are not bundled.
