# Experiment inputs

[section_4_input_inventory.csv](section_4_input_inventory.csv) lists the 92 input models used in Section 4. Its byte counts describe the archived `.mps.gz` files; regenerated files can have different compression and whitespace. The benchmark inputs are downloaded or generated on the machine running the experiments and are not bundled in Git.

| Group | Instances | Preparation |
| --- | ---: | --- |
| Mittelmann (Hans) | 49 | Public MPS/EMPS downloads; use the fixed paper inventory. |
| MIPLIB Large | 18 | Download MIPLIB 2017 MPS files and export their original LP relaxations. |
| QAP20 | 20 | Download QAPLIB DAT files and generate the paper's full Adams–Johnson LP representation. |
| QAP scalability | 5 | Crop `wil100` to N = 20, 40, 60, 80, 100; generate the symmetry-reduced Adams–Johnson LPs and apply column scaling. |

## Mittelmann (Hans)

Keep the official [download directory](https://plato.asu.edu/ftp/lptestset/) and [LPfeas benchmark page](https://plato.asu.edu/ftp/lpfeas.html) as the public entry points. The benchmark page lists additional source collections. Select the 49 instances in the inventory, including files in the `fome/`, `pds/`, and `rail/` subdirectories. Some compressed files use EMPS and need conversion to standard MPS, as explained on the benchmark page. Local paper filenames may differ from the source names, for example `fhnw-bin1` versus `fhnw-binschedule1` and `set-cover` versus `set-cover-model`.

## Download and generation commands

Run the following commands from the repository root on the experiment machine. Set `INPUT_ROOT` to a directory with sufficient free space. The scripts create separate dataset subdirectories. Existing output models are never overwritten. `--list` only prints the exact URLs and output names; `--download-only` fetches the source files without generating LPs.

Requirements:

- All groups: Python 3.10 or newer and `curl`.
- MIPLIB Large: `gurobipy==12.0.2` and a valid Gurobi license.
- QAP20: Julia 1.10 or newer and `gzip`; the generator uses only Julia's standard library.
- QAP scalability: Linux x86-64, a Fortran compiler (`gfortran`), a C++17 compiler (`g++`), and `gzip`. CUDA and a solver license are not needed for QAP generation.

```sh
export INPUT_ROOT=/path/to/benchmark-inputs

python3 scripts/prepare_inputs.py miplib-large --list
python3 scripts/prepare_inputs.py qap20 --list
python3 scripts/prepare_inputs.py qap-scale --list

python3 -m pip install gurobipy==12.0.2
python3 scripts/prepare_inputs.py miplib-large --output-dir "$INPUT_ROOT"
python3 scripts/prepare_inputs.py qap20 --output-dir "$INPUT_ROOT"
python3 scripts/prepare_inputs.py qap-scale --output-dir "$INPUT_ROOT"
```

An initial one-instance run can be selected explicitly:

```sh
python3 scripts/prepare_inputs.py miplib-large --instances a2864-99blp_relax --output-dir "$INPUT_ROOT"
python3 scripts/prepare_inputs.py qap20 --instances tho40 --output-dir "$INPUT_ROOT"
python3 scripts/prepare_inputs.py qap-scale --instances qap-wil100-n020 --output-dir "$INPUT_ROOT"
```

Use a fresh output directory for a subsequent full run. To use previously downloaded original MPS/DAT files, add `--source-dir /path/to/source-files`. Use `--julia /path/to/julia`, `--fc /path/to/gfortran`, and `--cxx /path/to/g++` when the tools are not on `PATH`. Fortran flags can be set with `--fflags '...'`; the default is `-O3 -std=legacy -mcmodel=large`.

## MIPLIB Large: 18 LP relaxations

The source is [MIPLIB 2017](https://miplib.zib.de/). The exact original MPS download links are:

[a2864-99blp](https://miplib.zib.de/WebData/instances/a2864-99blp.mps.gz), [dlr1](https://miplib.zib.de/WebData/instances/dlr1.mps.gz), [ivu06-big](https://miplib.zib.de/WebData/instances/ivu06-big.mps.gz), [ivu59](https://miplib.zib.de/WebData/instances/ivu59.mps.gz), [kottenpark09](https://miplib.zib.de/WebData/instances/kottenpark09.mps.gz), [neos-2991472-kalu](https://miplib.zib.de/WebData/instances/neos-2991472-kalu.mps.gz), [neos-3208254-reiu](https://miplib.zib.de/WebData/instances/neos-3208254-reiu.mps.gz), [neos-3740487-motru](https://miplib.zib.de/WebData/instances/neos-3740487-motru.mps.gz), [neos-4332801-seret](https://miplib.zib.de/WebData/instances/neos-4332801-seret.mps.gz), [neos-4332810-sesia](https://miplib.zib.de/WebData/instances/neos-4332810-sesia.mps.gz), [neos-4535459-waipa](https://miplib.zib.de/WebData/instances/neos-4535459-waipa.mps.gz), [neos-4545615-waita](https://miplib.zib.de/WebData/instances/neos-4545615-waita.mps.gz), [nucorsav](https://miplib.zib.de/WebData/instances/nucorsav.mps.gz), [scpn2](https://miplib.zib.de/WebData/instances/scpn2.mps.gz), [square41](https://miplib.zib.de/WebData/instances/square41.mps.gz), [square47](https://miplib.zib.de/WebData/instances/square47.mps.gz), [t11nonreg](https://miplib.zib.de/WebData/instances/t11nonreg.mps.gz), [usafa](https://miplib.zib.de/WebData/instances/usafa.mps.gz)

[prepare_inputs.py](../scripts/prepare_inputs.py) reads each original MIP with Gurobi 12.0.2, calls `Model.relax()`, and writes `<instance>_relax.mps.gz`. It does not call `presolve()` or `optimize()`, add cuts, or change the LP formulation. The script rejects quadratic models and models containing SOS or general constraints. Bounds, linear constraints, objective sense, and objective constant are retained by the relaxation/export API.

Output directory: `$INPUT_ROOT/MIPLIB_Large/`. Downloaded originals are cached under `$INPUT_ROOT/sources/miplib-large/`.

## QAP20: 20 structured LPs

The source is [QAPLIB](https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/). Each DAT file contains N and the two N-by-N matrices. The exact downloads are:

[esc64a](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/esc64a.dat), [lipa40a](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/lipa40a.dat), [lipa40b](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/lipa40b.dat), [lipa50a](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/lipa50a.dat), [lipa50b](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/lipa50b.dat), [lipa60a](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/lipa60a.dat), [lipa60b](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/lipa60b.dat), [lipa70a](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/lipa70a.dat), [lipa70b](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/lipa70b.dat), [sko56](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/sko56.dat), [sko64](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/sko64.dat), [tai40a](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/tai40a.dat), [tai40b](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/tai40b.dat), [tai50a](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/tai50a.dat), [tai50b](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/tai50b.dat), [tai60a](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/tai60a.dat), [tai60b](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/tai60b.dat), [tai64c](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/tai64c.dat), [tho40](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/tho40.dat), [wil50](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/wil50.dat)

[generate_qap20.jl](../scripts/generate_qap20.jl) constructs the full LP in variables `x[i,j]` and `y[i,j,k,l]`, including assignment, marginalization, symmetry, and bound constraints. It preserves the column-major variable order and row ordering of the paper input files. The bounds `0 <= x <= 1` and `y >= 0` are explicit rows; MPS variables are free. No presolve or column scaling is applied. Empty self-symmetry equalities are retained.

The wrapper downloads the specified DAT files and invokes this generator. It can also be called directly:

```sh
julia scripts/generate_qap20.jl /path/to/tho40.dat /path/to/tho40.mps.gz
```

Output directory: `$INPUT_ROOT/QAP20/`. DAT files are cached under `$INPUT_ROOT/sources/qap20/`.

## QAP scalability: wil100 at five sizes

Download [wil100.dat](https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/data.d/wil100.dat). For each N in 20, 40, 60, 80, 100, the script takes the leading N-by-N principal submatrix of each matrix. It validates symmetry and zero diagonals, then writes the flow/distance input expected by the Fortran generator.

The upstream generator is [Hinder's large-scale LP test problems](https://github.com/ohinder/large-scale-LP-test-problems/tree/b44862990e19ddaeab5568a826bd4ced13f23819), pinned to commit `b44862990e19ddaeab5568a826bd4ced13f23819`. The script downloads and builds the unchanged [newlp-300.f](https://raw.githubusercontent.com/ohinder/large-scale-LP-test-problems/b44862990e19ddaeab5568a826bd4ced13f23819/newlp-300.f), together with its [MIT license](https://raw.githubusercontent.com/ohinder/large-scale-LP-test-problems/b44862990e19ddaeab5568a826bd4ced13f23819/LICENSE). It constructs the symmetry-reduced Adams–Johnson relaxation used in this experiment.

[rescale.cpp](../scripts/qap_scale/rescale.cpp) streams the same column-scaling rule as upstream [utils.jl](https://github.com/ohinder/large-scale-LP-test-problems/blob/b44862990e19ddaeab5568a826bd4ced13f23819/utils.jl): multiply column j and its objective coefficient by `1 / abs(c[j])` when `c[j] != 0`, and by 1 otherwise. These models have nonnegative variables with no finite upper bounds and equality right-hand sides in {0, 1}. The script writes the generic row/column order used in the paper and compresses the output with gzip.

All five sizes are generated from `wil100.dat`, including N = 100; no pre-existing large LP is needed. The full-size model has 1,980,200 rows, 49,015,000 columns, and 198,020,000 matrix nonzeros. The generator has O(N^4) work; run the larger sizes on a machine with sufficient memory and disk space. One uncompressed intermediate MPS is kept during each generation and removed after completion.

Output directory: `$INPUT_ROOT/QAP_wil100_scale/`, with filenames `qap-wil100-n020.mps.gz` through `qap-wil100-n100.mps.gz`. The original DAT is cached under `$INPUT_ROOT/sources/qap-scale/`; compiler outputs and the pinned upstream source/license are under `$INPUT_ROOT/.tools/`.

## Data use and references

External benchmark collections retain their own distribution terms. The small [examples/model.mps](../examples/model.mps) is an MIT-licensed usage example, not a Section 4 input.


- H. D. Mittelmann (1999). Benchmarking interior point LP/QP solvers. *Optimization Methods and Software*, 11(1–4), 655–670. [DOI](https://doi.org/10.1080/10556789908805767).
- A. Gleixner et al. (2021). MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library. *Mathematical Programming Computation*, 13, 443–490. [DOI](https://doi.org/10.1007/s12532-020-00194-3).
- R. E. Burkard, S. E. Karisch, and F. Rendl (1997). QAPLIB—A Quadratic Assignment Problem Library. *Journal of Global Optimization*, 10, 391–403. [DOI](https://doi.org/10.1023/A:1008293323270).
- O. Hinder (2024). *Large-Scale LP Test Problems*. [Repository](https://github.com/ohinder/large-scale-LP-test-problems).
